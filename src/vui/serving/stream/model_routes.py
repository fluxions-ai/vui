"""Ollama + ASR model HTTP routes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
from aiohttp import web

from vui.serving.stream.asr_worker import ASR_MODELS
from vui.serving.stream.llm import OLLAMA_URL, llm_prefill_system

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


async def handle_ollama_models(srv: StreamServer, request):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []
    return web.json_response({"models": models, "current": srv.ollama_model})


async def handle_ollama_set_model(srv: StreamServer, request):
    data = await request.json()
    model = data.get("model", "").strip()
    if not model:
        return web.json_response({"ok": False, "error": "Model required"}, status=400)
    prev = srv.ollama_model
    await srv._block_ready("ollama")
    try:
        if prev != model:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": prev, "keep_alive": 0},
                    )
            except Exception:
                pass
        srv.ollama_model = model
        await srv._log(f"LLM model set to: {model}")
        try:
            await llm_prefill_system(srv.session.soul, srv.ollama_model)
        except Exception as e:
            srv.ollama_model = prev
            await srv._log(f"Model failed to load, reverting to {prev}: {e}", "error")
            return web.json_response(
                {"ok": False, "error": f"Model failed to load: {e}"}
            )
        return web.json_response({"ok": True, "model": model})
    finally:
        await srv._unblock_ready("ollama")


async def handle_ollama_pull(srv: StreamServer, request):
    data = await request.json()
    model = data.get("model", "").strip()
    if not model:
        return web.json_response({"ok": False, "error": "Model required"}, status=400)
    await srv._block_ready("ollama")
    try:
        await srv._log(f"Pulling model: {model}...")
        ws = srv.session.ws
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/pull",
                json={"name": model, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    msg = json.loads(line)
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
        srv.ollama_model = model
        try:
            await llm_prefill_system(srv.session.soul, srv.ollama_model)
        except Exception as e:
            await srv._log(f"Ollama prefill failed: {e}", "warn")
        return web.json_response({"ok": True, "model": model})
    except Exception as e:
        await srv._log(f"Pull failed: {e}", "error")
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    finally:
        await srv._unblock_ready("ollama")


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
    cls.handle_ollama_models = lambda self, *a, **kw: handle_ollama_models(
        self, *a, **kw
    )
    cls.handle_ollama_set_model = lambda self, *a, **kw: handle_ollama_set_model(
        self, *a, **kw
    )
    cls.handle_ollama_pull = lambda self, *a, **kw: handle_ollama_pull(self, *a, **kw)
    cls.handle_asr_models = lambda self, *a, **kw: handle_asr_models(self, *a, **kw)
    cls.handle_asr_set_model = lambda self, *a, **kw: handle_asr_set_model(
        self, *a, **kw
    )
