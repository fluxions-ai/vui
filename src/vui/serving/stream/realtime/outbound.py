"""Outbound translation: internal pipeline events → OpenAI Realtime events.

Called via `RealtimeWSProxy.send_json` whenever the existing pipeline emits
an event for what was previously the browser frontend (vad_start, partial_asr,
transcription, generating, turn_done, rollback_user, ...).
"""

from __future__ import annotations

from .adapter import RealtimeAdapter

# Internal event types that have no realtime-protocol analogue and should
# be silently dropped: log, status, settings, memories, mic_live,
# workers_ready, prompt_loaded, busy, worker_status, reply, generating.
# (We DO act on `generating` to emit transcript deltas — handled below.)


async def handle_internal_event(adapter: RealtimeAdapter, msg: dict):
    t = msg.get("type")
    if t == "vad_start":
        await adapter.send({"type": "input_audio_buffer.speech_started"})
    elif t == "vad_stop":
        await adapter.send({"type": "input_audio_buffer.speech_stopped"})
    elif t in ("partial_asr", "committed_asr"):
        await adapter.emit_transcription_delta(msg.get("text") or "")
    elif t == "transcription":
        await _on_transcription(adapter, msg)
    elif t == "generating":
        await _on_generating(adapter, msg)
    elif t == "turn_done":
        await adapter.end_response(status="completed")
    elif t == "rollback_user":
        await _on_rollback_user(adapter)


async def _on_transcription(adapter: RealtimeAdapter, msg: dict):
    text = (msg.get("text") or "").strip()
    if not text or text.startswith("("):
        return
    await adapter.ensure_user_item(text)
    await adapter.send(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": adapter.user_item_id,
            "content_index": 0,
            "transcript": text,
        }
    )
    await adapter.send(
        {
            "type": "input_audio_buffer.committed",
            "item_id": adapter.user_item_id,
        }
    )
    adapter.last_partial_emitted = ""
    await adapter.begin_response()


async def _on_generating(adapter: RealtimeAdapter, msg: dict):
    if not adapter.response_id:
        return
    text = msg.get("text") or ""
    await adapter.send(
        {
            "type": "response.audio_transcript.delta",
            "response_id": adapter.response_id,
            "item_id": adapter.assistant_item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }
    )
    adapter.transcript_buffer += text


async def _on_rollback_user(adapter: RealtimeAdapter):
    if not adapter.user_item_id:
        return
    await adapter.send(
        {
            "type": "conversation.item.deleted",
            "item_id": adapter.user_item_id,
        }
    )
    adapter.user_item_id = None
    adapter.last_partial_emitted = ""
