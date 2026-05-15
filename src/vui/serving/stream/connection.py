"""WebSocket, WebRTC, and startup warmup handlers."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING

import httpx
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

import vui.serving.stream.server as _srv_mod
from vui.serving.stream._log import _slog, _spawn
from vui.serving.stream.frontend import get_html
from vui.serving.stream.llm import (
    GGUF_MODEL_NAME,
    OLLAMA_URL,
    ensure_mlx_model,
    llm_prefill_system,
)
from vui.serving.stream.playback import TTSPlaybackTrack
from vui.serving.stream.prompts import (
    SOUL,
    TASK_SERVER_URL,
    build_soul,
    fetch_task_server_capabilities,
    probe_task_server,
)

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


# Heuristic match for the user's name in freeform memory strings. Conservative
# — only fires on explicit self-introduction patterns to avoid grabbing third
# parties ("my friend Sarah", "my daughter Lily").
_USER_NAME_PATTERNS = [
    re.compile(r"\bmy name(?:'s| is)\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\b(?:i'?m|i am)\s+called\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Z][a-z]+)", re.IGNORECASE),
]


def _extract_user_name(memories: list[str]) -> str | None:
    for mem in memories:
        for pat in _USER_NAME_PATTERNS:
            m = pat.search(mem)
            if m:
                return m.group(1).capitalize()
    return None


def _greeting_text(srv: StreamServer) -> str:
    user_name = _extract_user_name(srv._memories)
    assistant = srv.session.assistant_name
    if user_name:
        return f"Hey {user_name}, it's {assistant}, what can I help with?"
    return f"Hey there, it's {assistant}, what can I help with?"


def try_fire_greeting(srv: StreamServer) -> None:
    """Fire the greeting iff all prerequisites are met. Idempotent — safe to
    call from every trigger point (WS connect, /offer, workers_ready); the
    greeting_pending flag guards against duplicates.

    Prereqs:
      - greeting_pending is True (set on WS connect)
      - session.ready (workers warmed, mic flowing)
      - playback_track exists (WebRTC handshake completed) — otherwise
        drain_tts_audio drops the audio frames silently
    """
    if not srv.session.greeting_pending:
        return
    if not srv.session.ready:
        return
    if srv.session.playback_track is None:
        return
    ws = srv.session.ws
    if ws is None or ws.closed:
        return

    srv.session.greeting_pending = False
    text = _greeting_text(srv)
    srv.session.cancel_generation = False
    srv._generates_sent = 1
    srv._generates_done = 0
    srv._turn_chunk_idx = 0
    srv._llm_streaming = False
    srv.tts_cmd_queue.put(
        {
            "cmd": "generate",
            "text": text,
            "is_voice": False,
            "settings": srv.session.settings,
        }
    )
    srv.session.conversation.append(
        {"role": "assistant", "content": text, "ts": time.time()}
    )
    asyncio.ensure_future(ws.send_json({"type": "reply", "text": text}))
    _slog(f"[connect] greeting: {text!r}")


# Backwards-compat alias for the old direct-fire helper.
fire_greeting = try_fire_greeting


async def handle_index(srv: StreamServer, request):
    return web.Response(text=get_html(), content_type="text/html")


async def handle_offer(srv: StreamServer, request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    srv.session.pc = pc

    playback = TTSPlaybackTrack()
    srv.session.playback_track = playback
    pc.addTrack(playback)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            sink = _srv_mod.AudioRecordingSink(
                track,
                srv.tts_cmd_queue,
                srv.asr_cmd_queue,
                srv.vad_queue,
                srv.asr_result_queue,
            )
            sink.vad_enabled = srv._server_vad
            srv.session.recording_sink = sink
            asyncio.ensure_future(sink.run())

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"[webrtc] State: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            if srv.session.pc is pc:
                srv.tts_cmd_queue.put({"cmd": "rewind"})

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # playback_track is now live — the greeting may have been deferred
    # waiting for it. (Other prereqs are re-checked inside.)
    try_fire_greeting(srv)

    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


async def handle_ws(srv: StreamServer, request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Per-tab client ID via `?cid=...` query param. sessionStorage on the
    # browser keeps this stable across reconnects/refresh in the same tab,
    # but different across tabs/browsers/devices. Same CID → it's the same
    # tab reconnecting, just rebind without kicking. Different (or unset)
    # CID → genuinely different client, fire the single-tenant kick.
    cid = request.query.get("cid") or ""
    existing_cid = getattr(srv.session, "client_id", None) or ""
    is_reconnect = bool(cid) and bool(existing_cid) and cid == existing_cid

    old_ws = srv.session.ws
    if not is_reconnect and old_ws is not None and old_ws is not ws and not old_ws.closed:
        try:
            await old_ws.send_json({"type": "session_taken"})
        except Exception:
            pass
        try:
            await old_ws.close()
        except Exception:
            pass
    if not is_reconnect:
        old_pc = srv.session.pc
        if old_pc is not None:
            try:
                asyncio.ensure_future(old_pc.close())
            except Exception:
                pass
            srv.session.pc = None
            srv.session.recording_sink = None
    # Always null the playback track on a fresh WS connect — the new /offer
    # will rebuild it. This is the gate try_fire_greeting waits on, so a
    # stale (or never-connected) track must not satisfy the check.
    srv.session.playback_track = None

    srv.session.ws = ws
    srv.session.client_id = cid
    print(f"[ws] Client connected (cid={cid[:8] or '-'}, reconnect={is_reconnect})")

    if srv.session.ready:
        await ws.send_json({"type": "workers_ready"})
        print("[ws] Sent workers_ready (already ready)")

    srv.tts_cmd_queue.put({"cmd": "get_state"})
    state_msg = await srv._wait_tts_response("state", timeout=5)
    if state_msg:
        srv.tts_T = state_msg.get("T", 0)
        srv.tts_max_T = state_msg.get("max_T", 0)
        if state_msg.get("has_prompt"):
            await ws.send_json(
                {
                    "type": "prompt_loaded",
                    "T": srv.tts_T,
                    "max_T": srv.tts_max_T,
                    "text": state_msg.get("prompt_text", ""),
                }
            )

    await ws.send_json({"type": "settings", "settings": srv.session.settings})
    await ws.send_json(
        {"type": "assistant_name", "name": srv.session.assistant_name}
    )
    await ws.send_json({"type": "listen_mode", "enabled": srv._listen_mode})
    await ws.send_json({"type": "memories", "memories": srv._memories})
    await ws.send_json(
        {"type": "task_server_status", "available": srv._task_server_available}
    )
    await ws.send_json({"type": "llm_status", "available": srv._llm_available})
    await ws.send_json(
        {
            "type": "ctx_status",
            "conv_ctx": srv.conv_ctx,
            "conv_ctx_max": srv.conv_ctx_max,
            "thoughts_ctx": srv.thoughts_ctx,
            "thoughts_ctx_max": srv.thoughts_ctx_max,
        }
    )
    await srv._send_worker_status()

    from vui.serving.stream.tasks import push_all_tasks

    push_all_tasks(srv)

    srv.session.greeting_pending = True
    try_fire_greeting(srv)

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)
            msg_type = data.get("type")

            if msg_type == "tts":
                tts_text = data.get("text", "").strip()
                if not tts_text or not srv.session.ready:
                    continue

                await srv._log(f"TTS: '{tts_text}'", "info")
                srv.session.cancel_generation = False
                srv._generates_sent = 1
                srv._generates_done = 0
                srv._turn_chunk_idx = 0
                srv._llm_streaming = False

                srv.tts_cmd_queue.put(
                    {
                        "cmd": "generate",
                        "text": tts_text,
                        "is_voice": False,
                        "settings": srv.session.settings,
                    }
                )

            elif msg_type == "listen_mode":
                enabled = bool(data.get("enabled", False))
                if enabled == srv._listen_mode:
                    continue
                srv._listen_mode = enabled
                await srv._log(
                    "Listen mode enabled" if enabled else "Listen mode disabled",
                    "info",
                )
                # Broadcast so the UI in any tab stays in sync.
                await ws.send_json({"type": "listen_mode", "enabled": enabled})

            elif msg_type == "floor_hold":
                hold = bool(data.get("hold", False))
                if hold == srv._floor_held:
                    continue
                srv._floor_held = hold
                if hold:
                    # Cancel any pending tiered-commit so it can't fire
                    # while the user is holding the floor.
                    if srv._endpointing_task and not srv._endpointing_task.done():
                        srv._endpointing_task.cancel()
                        srv._endpointing_task = None
                    await srv._log("[floor] held")
                else:
                    # If VAD stopped while held, fire the deferred commit now.
                    pending = srv._floor_pending_dur
                    srv._floor_pending_dur = None
                    if pending is not None and pending >= 0.3:
                        await srv._log(
                            f"[floor] released — running deferred commit ({pending:.2f}s)"
                        )
                        srv._endpointing_task = _spawn(
                            srv._tiered_commit(pending), "tiered_commit"
                        )
                    else:
                        await srv._log("[floor] released")

            elif msg_type == "vad_mode":
                enabled = data.get("enabled", False)
                if enabled == srv._server_vad:
                    continue
                srv._server_vad = enabled
                if srv.session.recording_sink:
                    srv.session.recording_sink.vad_enabled = enabled
                    print(
                        f"[main] Sink vad_enabled set to {enabled} (sink id={id(srv.session.recording_sink)})"
                    )
                if enabled:
                    srv.asr_cmd_queue.put({"cmd": "vad_enable"})
                    await srv._log("Server VAD enabled")
                else:
                    srv.asr_cmd_queue.put({"cmd": "vad_disable"})
                    if srv._endpointing_task and not srv._endpointing_task.done():
                        srv._endpointing_task.cancel()
                        srv._endpointing_task = None
                    if (
                        srv.session.recording_sink
                        and srv.session.recording_sink.recording
                    ):
                        _srv_mod._slog(
                            "[main] VAD disabled mid-turn, stopping recording"
                        )
                        srv.session.recording_sink.stop_recording()
                    if srv.session.recording_sink:
                        srv.session.recording_sink.speaking = False
                    srv._latest_partial = ""
                    srv._turn_best_asr = ""
                    srv._phase_transcript = ""
                    srv._committed_len = 0
                    srv._committed_text = ""
                    srv._carry = ""
                    srv._current_partial = ""
                    await srv._log("Server VAD disabled")

            elif msg_type == "settings":
                if data.get("reset"):
                    from vui.serving.stream.server import DEFAULT_SETTINGS

                    srv.session.settings = dict(DEFAULT_SETTINGS)
                    print("[settings] reset to defaults")
                    srv.asr_cmd_queue.put(
                        {
                            "cmd": "set_vad_stop_secs",
                            "secs": srv.session.settings.get("vad_stop_secs", 0.3),
                        }
                    )
                    await srv._send_ws(
                        {"type": "settings", "settings": srv.session.settings}
                    )
                    continue
                changed = []
                for key in (
                    "temperature",
                    "top_k",
                    "wps_score",
                    "rep_penalty",
                    "rep_window",
                    "max_duration",
                    "chunk_words",
                    "first_chunk_words",
                    "n_codebooks",
                    "eos_threshold",
                    "vad_stop_secs",
                    "asr_settle_s",
                    "trailing_off_delay",
                    "context_minutes",
                ):
                    if key in data and data[key] != srv.session.settings.get(key):
                        changed.append(f"{key}={data[key]}")
                        srv.session.settings[key] = data[key]
                if "vad_stop_secs" in data:
                    srv.asr_cmd_queue.put(
                        {
                            "cmd": "set_vad_stop_secs",
                            "secs": srv.session.settings.get("vad_stop_secs", 0.3),
                        }
                    )
                for key in (
                    "user_audio",
                    "keep_context",
                    "tool_check",
                ):
                    if key in data:
                        val = bool(data[key])
                        if val != srv.session.settings.get(key):
                            changed.append(f"{key}={val}")
                        srv.session.settings[key] = val
                if "sq_scores" in data:
                    new_sq = [float(v) for v in data["sq_scores"]]
                    if new_sq != srv.session.settings.get("sq_scores"):
                        changed.append(f"sq_scores={new_sq}")
                    srv.session.settings["sq_scores"] = new_sq
                if changed:
                    print(f"[settings] {', '.join(changed)}")
                if "n_codebooks" in data and any("n_codebooks" in c for c in changed):
                    srv.tts_cmd_queue.put(
                        {
                            "cmd": "reprefill",
                            "settings": srv.session.settings,
                        }
                    )
                    srv.session.conversation.clear()
                    await srv._log(
                        f"Re-prefilling prompt (n_codebooks changed)", "info"
                    )
                if "soul" in data:
                    srv.session.soul = data["soul"]
                    from vui.serving.stream.server import _save_soul

                    _save_soul(data["soul"])
                    if srv._warmup_done:

                        async def _prefill_safe():
                            try:
                                await llm_prefill_system(
                                    srv.session.soul, srv.ollama_model
                                )
                            except Exception as e:
                                await srv._log(f"Ollama prefill failed: {e}", "warn")

                        _spawn(_prefill_safe(), "ollama_prefill_after_soul")

                if "assistant_name" in data:
                    new_name = (data.get("assistant_name") or "").strip() or "Vui"
                    if new_name != srv.session.assistant_name:
                        from vui.serving.stream.server import (
                            SOUL_FILE,
                            _save_assistant_name,
                        )

                        srv.session.assistant_name = new_name
                        _save_assistant_name(new_name)
                        # If the user hasn't saved a custom soul, rebuild
                        # from the template with the new name. Otherwise
                        # their `.soul` file overrides the name placeholder
                        # and we leave it alone.
                        if not SOUL_FILE.exists():
                            srv.session.soul = build_soul(
                                with_claude=srv._task_server_available,
                                name=new_name,
                            )
                        if srv._warmup_done:

                            async def _prefill_name_change():
                                try:
                                    await llm_prefill_system(
                                        srv.session.soul, srv.ollama_model
                                    )
                                except Exception as e:
                                    await srv._log(
                                        f"Prefill after name change failed: {e}",
                                        "warn",
                                    )

                            _spawn(
                                _prefill_name_change(), "ollama_prefill_name_change"
                            )
                        await ws.send_json(
                            {"type": "assistant_name", "name": new_name}
                        )

        elif msg.type == web.WSMsgType.ERROR:
            break

    print("[ws] Client disconnected")
    srv.session.ws = None
    return ws


async def _apply_task_server_state(srv: StreamServer, available: bool):
    """Update prompt + UI for current task-server availability."""
    srv._task_server_available = available
    if available:
        srv._task_server_capabilities = await fetch_task_server_capabilities()
    else:
        srv._task_server_capabilities = []
    new_sys = build_soul(with_claude=available)
    prev_sys = build_soul(with_claude=not available)
    if srv.session.soul in (SOUL, prev_sys, new_sys):
        srv.session.soul = new_sys
    ws = srv.session.ws
    if ws and not ws.closed:
        try:
            await ws.send_json({"type": "task_server_status", "available": available})
        except Exception:
            pass


async def _task_server_poll_loop(srv: StreamServer):
    """Re-probe the task server periodically; flip tools when state changes."""
    fail_count = 0
    while True:
        if srv._task_server_available:
            await asyncio.sleep(30.0)
        else:
            fail_count += 1
            await asyncio.sleep(3.0 if fail_count < 4 else 10.0)

        try:
            available = await probe_task_server()
        except Exception:
            available = False

        if available == srv._task_server_available:
            continue

        print(
            f"[main] Task server transitioned: "
            f"{'DOWN → up' if available else 'up → DOWN'}"
        )
        await _apply_task_server_state(srv, available)
        fail_count = 0
        if available:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.post(f"{TASK_SERVER_URL}/tasks/clear")
                    print("[main] Cleared task server")
            except Exception:
                pass


async def probe_llm() -> bool:
    from vui.serving.stream.llm_backend import get_backend

    base_url = get_backend().base_url
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{base_url}/api/version")
            return r.status_code == 200
    except Exception:
        return False


async def _llm_poll_loop(srv: StreamServer):
    """Re-probe the LLM backend periodically; flip the client status pill."""
    fail_count = 0
    while True:
        if srv._llm_available:
            await asyncio.sleep(15.0)
        else:
            fail_count += 1
            await asyncio.sleep(3.0 if fail_count < 4 else 10.0)

        try:
            available = await probe_llm()
        except Exception:
            available = False

        if available == srv._llm_available:
            continue
        await srv._set_llm_available(available)
        if available:
            fail_count = 0


async def warmup(srv: StreamServer):
    """Background warmup: wait for workers, load prompt, pre-warm LLM."""
    srv._tasks.clear()

    available = await probe_task_server()
    print(
        f"[main] Task server probe ({TASK_SERVER_URL}): "
        f"{'up' if available else 'DOWN — disabling ask_claude'}"
    )
    await _apply_task_server_state(srv, available)

    if srv._task_server_available:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(f"{TASK_SERVER_URL}/tasks/clear")
                print("[main] Cleared task server")
        except Exception:
            pass

    llm_up = await probe_llm()
    srv._llm_available = llm_up
    print(f"[main] LLM probe: {'up' if llm_up else 'DOWN'}")
    await srv._send_llm_status()

    _spawn(_task_server_poll_loop(srv), "task_server_poll")
    _spawn(_llm_poll_loop(srv), "llm_poll")

    while not (srv.tts_ready and srv.asr_ready):
        await asyncio.sleep(0.2)

    srv.asr_cmd_queue.put({"cmd": "start"})
    srv._asr_session_started_t = time.monotonic()
    srv._asr_speaking_secs = 0.0
    print("[main] ASR session started (persistent)")

    # VAD is enabled in drain_asr_results as soon as the ASR worker reports
    # ready, independent of this warmup. The vad_start handler in drains.py
    # ignores speech detected before session.ready (so warmup-time speech is
    # safely dropped); the frontend gets a `mic_live` event from
    # vad_audio_ready so it can tell the user the mic pipeline is up while
    # the backend continues warming.
    await srv._send_worker_status()

    srv.tts_cmd_queue.put({"cmd": "get_state"})
    state_msg = await srv._wait_tts_response("state", timeout=10)
    if state_msg:
        srv.tts_max_T = state_msg.get("max_T", 0)

    last = _srv_mod._get_last_prompt()
    if last:
        # Worker resolves `{name}.{ckpt_id}.pt` and falls back to wav+txt
        # regen on first load against a new checkpoint; just need any
        # source to exist (wav, legacy .pt, keyed .pt, or .safetensors).
        any_src = (
            (_srv_mod.PROMPTS_DIR / f"{last}.wav").exists()
            or any(_srv_mod.PROMPTS_DIR.glob(f"{last}.*.pt"))
            or (_srv_mod.PROMPTS_DIR / f"{last}.pt").exists()
            or (_srv_mod.PROMPTS_DIR / f"{last}.safetensors").exists()
        )
        if not any_src:
            from vui.serving.stream.prompt_routes import _fetch_hf_prompt

            await srv._log(f"Fetching prompt '{last}' from Hugging Face...")
            fetched = await asyncio.to_thread(_fetch_hf_prompt, last)
            any_src = fetched is not None
            if not any_src:
                await srv._log(f"Could not fetch prompt '{last}'", "warn")

        async def _fetch_remaining_presets():
            from vui.serving.stream.prompt_routes import (
                PRESET_VOICES,
                ensure_preset_prompts,
            )

            missing = [
                s
                for s in PRESET_VOICES
                if not (_srv_mod.PROMPTS_DIR / f"{s}.wav").exists()
            ]
            if not missing:
                return
            ready = await asyncio.to_thread(ensure_preset_prompts)
            await srv._log(f"Preset voices ready: {', '.join(ready)}")

        _spawn(_fetch_remaining_presets(), "fetch_preset_prompts")

        if any_src:
            await srv._log(f"Loading prompt '{last}'...")
            srv.tts_cmd_queue.put({"cmd": "load_kv", "file": last})
            resp = await srv._wait_tts_response("kv_loaded", timeout=60)
            if resp and resp.get("ok"):
                srv.tts_T = resp["T"]
                from vui.telemetry import record_voice_load

                record_voice_load(resp.get("name", last))
                await srv._log(
                    f"Auto-loaded prompt '{resp.get('name', last)}' (T={resp['T']})"
                )

    if _srv_mod.IS_APPLE_SILICON:
        try:
            await asyncio.to_thread(ensure_mlx_model)
        except Exception as e:
            await srv._log(f"MLX model setup failed, falling back to GGUF: {e}", "warn")
            srv.ollama_model = GGUF_MODEL_NAME

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/ps")
            loaded = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
            if loaded:
                srv.ollama_model = loaded[0]
                await srv._log(f"Using loaded Ollama model: {srv.ollama_model}")
    except Exception:
        pass

    async def _prefill_system():
        try:
            await llm_prefill_system(srv.session.soul, srv.ollama_model)
        except Exception as e:
            await srv._log(f"Ollama prefill failed: {e}", "warn")

    async def _prefill_thoughts():
        try:
            thoughts_prompt = srv._thoughts._build_system_prompt()
            await llm_prefill_system(thoughts_prompt, srv.ollama_model)
            await srv._log("Thoughts stream KV warmed")
        except Exception as e:
            await srv._log(f"Thoughts prefill failed: {e}", "warn")

    await asyncio.gather(_prefill_system(), _prefill_thoughts())

    srv._warmup_done = True
    await srv._send_worker_status()
    await srv._unblock_ready("warmup")


def bind(cls):
    cls.handle_index = lambda self, *a, **kw: handle_index(self, *a, **kw)
    cls.handle_offer = lambda self, *a, **kw: handle_offer(self, *a, **kw)
    cls.handle_ws = lambda self, *a, **kw: handle_ws(self, *a, **kw)
    cls._warmup = lambda self: warmup(self)
