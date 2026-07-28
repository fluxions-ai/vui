"""LLM + ASR model HTTP routes.

Everything here goes through the LLMBackend abstraction rather than raw Ollama
endpoints, so the routes behave sensibly under vLLM (and any other
OpenAI-compatible server) instead of returning empty lists and 500s.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiohttp import web

from vui.serving.stream.asr_worker import ASR_MODELS
from vui.serving.stream.llm import llm_prefill_system
from vui.serving.stream.llm_backend import get_backend

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


async def handle_llm_models(srv: StreamServer, request):
    backend = get_backend()
    try:
        models = await backend.list_models()
    except Exception:
        models = [backend.model]
    return web.json_response(
        {
            "models": models,
            "current": srv.llm_model,
            "backend": backend.name,
            "can_switch": backend.supports_model_switch,
            "can_pull": backend.supports_pull,
        }
    )


async def handle_llm_set_model(srv: StreamServer, request):
    backend = get_backend()
    if not backend.supports_model_switch:
        return web.json_response(
            {"ok": False, "error": f"{backend.name} cannot switch models at runtime"},
            status=409,
        )
    data = await request.json()
    model = data.get("model", "").strip()
    if not model:
        return web.json_response({"ok": False, "error": "Model required"}, status=400)
    prev = backend.model
    await srv._block_ready("llm")
    try:
        # This is the call the old route was missing: it only ever moved a
        # label, so the switch appeared to work while the old model kept
        # answering.
        await backend.set_model(model)
        await srv._log(f"LLM model set to: {model}")
        try:
            await llm_prefill_system(srv.session.soul)
        except Exception as e:
            await backend.set_model(prev)
            await srv._log(f"Model failed to load, reverting to {prev}: {e}", "error")
            return web.json_response(
                {"ok": False, "error": f"Model failed to load: {e}"}
            )
        return web.json_response({"ok": True, "model": model})
    except ValueError as e:
        # e.g. VLLMBackend rejecting a model it doesn't serve.
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    finally:
        await srv._unblock_ready("llm")


async def handle_llm_pull(srv: StreamServer, request):
    backend = get_backend()
    if not backend.supports_pull:
        # 409, not 500 — an expected capability gap, not a server fault.
        return web.json_response(
            {
                "ok": False,
                "error": f"{backend.name} has no model registry to pull from",
            },
            status=409,
        )
    data = await request.json()
    model = data.get("model", "").strip()
    if not model:
        return web.json_response({"ok": False, "error": "Model required"}, status=400)
    await srv._block_ready("llm")
    try:
        await srv._log(f"Pulling model: {model}...")
        ws = srv.session.ws
        async for msg in backend.pull(model):
            status = msg.get("status", "")
            total = msg.get("total", 0)
            completed = msg.get("completed", 0)
            pct = int(completed / total * 100) if total > 0 else 0
            text = f"{status} {pct}%" if total > 0 else status
            if ws and not ws.closed:
                try:
                    await ws.send_json(
                        {
                            "type": "pull_progress",
                            "text": text,
                            "pct": pct,
                            "status": status,
                        }
                    )
                except Exception:
                    pass
        await srv._log(f"Model pulled: {model}")
        await backend.set_model(model)
        try:
            await llm_prefill_system(srv.session.soul)
        except Exception as e:
            await srv._log(f"LLM prefill failed: {e}", "warn")
        return web.json_response({"ok": True, "model": model})
    except Exception as e:
        await srv._log(f"Pull failed: {e}", "error")
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    finally:
        await srv._unblock_ready("llm")


async def handle_asr_models(srv: StreamServer, request):
    return web.json_response(
        {
            "models": list(ASR_MODELS.keys()),
            "current": srv.asr_model,
        }
    )


async def handle_asr_set_model(srv: StreamServer, request):
    data = await request.json()
    model = data.get("model", "").strip()
    if model not in ASR_MODELS:
        return web.json_response(
            {"ok": False, "error": f"Unknown ASR model: {model}"}, status=400
        )
    if model == srv.asr_model:
        return web.json_response({"ok": True, "model": model})
    await srv._block_ready("model")
    try:
        srv._asr_backend_set_event = asyncio.Event()
        srv._asr_backend_set_result = None
        srv.asr_cmd_queue.put({"cmd": "set_backend", "model": model})
        try:
            await asyncio.wait_for(srv._asr_backend_set_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            srv._asr_backend_set_event = None
            return web.json_response({"ok": False, "error": "timeout"}, status=500)
        srv._asr_backend_set_event = None
        msg = srv._asr_backend_set_result
        if msg and msg.get("ok"):
            srv.asr_model = model
            await srv._log(f"ASR model set to: {model}")
            return web.json_response({"ok": True, "model": model})
        return web.json_response(
            {"ok": False, "error": (msg or {}).get("error", "unknown")}, status=500
        )
    finally:
        await srv._unblock_ready("model")


def bind(cls):
    cls.handle_llm_models = lambda self, *a, **kw: handle_llm_models(self, *a, **kw)
    cls.handle_llm_set_model = lambda self, *a, **kw: handle_llm_set_model(
        self, *a, **kw
    )
    cls.handle_llm_pull = lambda self, *a, **kw: handle_llm_pull(self, *a, **kw)
    cls.handle_asr_models = lambda self, *a, **kw: handle_asr_models(self, *a, **kw)
    cls.handle_asr_set_model = lambda self, *a, **kw: handle_asr_set_model(
        self, *a, **kw
    )
