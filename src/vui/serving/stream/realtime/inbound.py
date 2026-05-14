"""Inbound translation: client OpenAI Realtime events → vui pipeline calls.

Called once per parsed client event. Each top-level event type has a
free-function handler that drives the existing pipeline (worker queues,
voice_respond, ASR session reset, ...).
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch
from julius.resample import resample_frac

import vui.serving.stream.server as _srv_mod
from vui.serving.stream._log import _slog
from vui.serving.stream.llm import llm_prefill_system

from .adapter import RealtimeAdapter
from .audio import (
    SAMPLE_RATE_ASR,
    SAMPLE_RATE_OUT,
    list_voices,
    new_item_id,
    pcm16_b64_to_float32,
)


async def handle_client_event(adapter: RealtimeAdapter, event: dict):
    t = event.get("type")
    eid = event.get("event_id")
    try:
        if t == "session.update":
            await _on_session_update(adapter, event)
        elif t == "input_audio_buffer.append":
            _on_input_audio_append(adapter, event)
        elif t == "input_audio_buffer.commit":
            _on_input_audio_commit(adapter, event)
        elif t == "input_audio_buffer.clear":
            _on_input_audio_clear(adapter)
        elif t == "response.create":
            await _on_response_create(adapter, event)
        elif t == "response.cancel":
            await _on_response_cancel(adapter, event)
        elif t == "conversation.item.create":
            await _on_item_create(adapter, event)
        elif t == "conversation.item.delete":
            _on_item_delete(adapter, event)
        elif t == "conversation.item.truncate":
            pass  # no-op: pipeline doesn't model per-item audio rewind
        else:
            await adapter.send_error("unknown_event", f"Unhandled type: {t}", eid)
    except Exception as e:
        _slog(f"[realtime] handle_client_event error: {e}")
        await adapter.send_error("internal_error", str(e), eid)


# --- session.update ---------------------------------------------------------


async def _on_session_update(adapter: RealtimeAdapter, event: dict):
    sess = event.get("session") or {}
    voice = sess.get("voice")
    if voice and voice != adapter.voice:
        await _load_voice(adapter, voice)
    if "instructions" in sess and sess["instructions"]:
        adapter.instructions = sess["instructions"]
        adapter.srv.session.soul = sess["instructions"]
        asyncio.create_task(_safe_prefill(adapter.srv, sess["instructions"]))
    if "turn_detection" in sess:
        td = sess.get("turn_detection")
        if td is None:
            adapter.srv._server_vad = False
            if adapter.srv.session.recording_sink:
                adapter.srv.session.recording_sink.vad_enabled = False
            adapter.srv.asr_cmd_queue.put({"cmd": "vad_disable"})
        elif isinstance(td, dict) and td.get("type") == "server_vad":
            adapter.srv._server_vad = True
            if adapter.srv.session.recording_sink:
                adapter.srv.session.recording_sink.vad_enabled = True
            adapter.srv.asr_cmd_queue.put({"cmd": "vad_enable"})
    if "tools" in sess:
        adapter.tools = list(sess.get("tools") or [])
    if "temperature" in sess:
        adapter.srv.session.settings["temperature"] = float(sess["temperature"])
    await adapter.send_session_updated()


async def _load_voice(adapter: RealtimeAdapter, voice: str):
    path = _srv_mod.PROMPTS_DIR / f"{voice}.pt"
    if not path.exists():
        await adapter.send_error(
            "voice_not_found",
            f"Voice '{voice}' not found. Available: {list_voices()}",
        )
        return
    adapter.srv.tts_cmd_queue.put({"cmd": "load_kv", "file": voice})
    resp = await adapter.srv._wait_tts_response("kv_loaded", timeout=30)
    if resp and resp.get("ok"):
        adapter.srv.tts_T = resp["T"]
        adapter.voice = voice
        _slog(f"[realtime] loaded voice '{voice}' (T={resp['T']})")
    else:
        await adapter.send_error("voice_load_failed", f"Could not load '{voice}'")


async def _safe_prefill(srv, prompt: str):
    try:
        await llm_prefill_system(prompt, srv.ollama_model)
    except Exception as e:
        _slog(f"[realtime] system prompt prefill failed: {e}")


# --- input_audio_buffer.* ---------------------------------------------------


def _on_input_audio_append(adapter: RealtimeAdapter, event: dict):
    b64 = event.get("audio")
    if not b64:
        return
    audio_24k = pcm16_b64_to_float32(b64)
    if audio_24k.size == 0:
        return

    a24_t = torch.from_numpy(audio_24k)
    a16_t = resample_frac(a24_t, SAMPLE_RATE_OUT, SAMPLE_RATE_ASR)
    audio_16k = a16_t.numpy().astype(np.float32)

    sink = adapter.srv.session.recording_sink
    if sink is None or not sink.vad_enabled:
        return

    adapter.srv.vad_queue.put(audio_16k)
    adapter.srv.asr_cmd_queue.put(
        {"cmd": "feed", "audio": audio_16k, "sample_rate": SAMPLE_RATE_ASR}
    )

    if sink.recording:
        sink._samples_recorded += audio_16k.size
        sink._audio_16k_chunks.append(audio_16k.copy())
        adapter.srv.tts_cmd_queue.put({"cmd": "stream_feed", "audio": audio_24k})


def _on_input_audio_commit(adapter: RealtimeAdapter, event: dict):
    # Manual commit (turn_detection: none). Phase 1 supports server VAD only;
    # log so misbehaving clients are visible.
    _slog("[realtime] input_audio_buffer.commit ignored (server VAD only)")


def _on_input_audio_clear(adapter: RealtimeAdapter):
    from vui.serving.stream.drains import _hard_reset_asr

    _hard_reset_asr(adapter.srv)


# --- response.* + conversation.item.* ---------------------------------------


async def _on_response_cancel(adapter: RealtimeAdapter, event: dict):
    if not adapter.response_id:
        await adapter.send_error(
            "no_active_response", "No active response", event.get("event_id")
        )
        return
    srv = adapter.srv
    srv.session.cancel_generation = True
    try:
        srv.tts_cancel_event.set()
    except Exception:
        pass
    rewind_T = getattr(srv, "_pre_turn_T", 0)
    srv.tts_cmd_queue.put({"cmd": "cancel", "rewind_to": rewind_T})
    if srv.session.playback_track:
        srv.session.playback_track.flush()
    while (
        srv.session.conversation and srv.session.conversation[-1]["role"] == "assistant"
    ):
        srv.session.conversation.pop()
    await adapter.end_response(status="cancelled")


async def _on_item_create(adapter: RealtimeAdapter, event: dict):
    item = event.get("item") or {}
    itype = item.get("type")
    if itype == "message":
        await _on_message_item_create(adapter, event, item)
    elif itype == "function_call_output":
        # Phase 3 wiring point.
        _slog("[realtime] function_call_output (Phase 3 — appending raw)")
        adapter.srv.session.conversation.append(
            {"role": "tool", "content": item.get("output", "")}
        )
    else:
        await adapter.send_error(
            "unsupported_item_type",
            f"item.type={itype} not supported",
            event.get("event_id"),
        )


async def _on_message_item_create(adapter: RealtimeAdapter, event: dict, item: dict):
    role = item.get("role", "user")
    content_parts = item.get("content") or []
    text_parts = []
    for c in content_parts:
        ct = c.get("type")
        if ct in ("input_text", "text"):
            text_parts.append(c.get("text", ""))
        elif ct == "input_audio" and c.get("transcript"):
            text_parts.append(c["transcript"])
    text = " ".join(p for p in text_parts if p).strip()
    if not text:
        await adapter.send_error(
            "empty_item",
            "Message item has no text content",
            event.get("event_id"),
        )
        return
    item_id = item.get("id") or new_item_id()
    adapter.srv.session.conversation.append({"role": role, "content": text})
    if role == "user":
        adapter.user_item_id = item_id
    await adapter.send(
        {
            "type": "conversation.item.created",
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": role,
                "status": "completed",
                "content": [{"type": "input_text", "text": text}],
            },
        }
    )


def _on_item_delete(adapter: RealtimeAdapter, event: dict):
    item_id = event.get("item_id")
    conv = adapter.srv.session.conversation
    if item_id and adapter.user_item_id == item_id and conv:
        if conv[-1]["role"] == "user":
            conv.pop()
            adapter.user_item_id = None


async def _on_response_create(adapter: RealtimeAdapter, event: dict):
    if adapter.response_id:
        await adapter.send_error(
            "response_in_progress",
            "A response is already active — send response.cancel first",
            event.get("event_id"),
        )
        return
    params = event.get("response") or {}
    if "instructions" in params and params["instructions"]:
        adapter.srv.session.soul = params["instructions"]
    last_user_text = ""
    for m in reversed(adapter.srv.session.conversation):
        if m["role"] == "user":
            last_user_text = m["content"]
            break
        if m["role"] == "assistant":
            break
    adapter.srv.session.ready = True
    await adapter.begin_response()
    asyncio.create_task(_run_voice_respond(adapter, last_user_text))


async def _run_voice_respond(adapter: RealtimeAdapter, text: str):
    try:
        await adapter.srv._voice_respond(text, None, None)
    except Exception as e:
        _slog(f"[realtime] voice_respond error: {e}")
        await adapter.end_response(status="failed")
