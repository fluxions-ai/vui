"""POST /v1/voice-note — one-shot audio→audio endpoint.

Multipart `audio` field in (any format torchaudio can decode). Returns a
JSON envelope with the rendered WAV (base64) + transcript + reply text.
Same ASR → LLM → TTS pipeline as the streaming server, but synchronous —
designed for voice-note style clients where latency matters less than
getting a complete reply in one HTTP call.

Pipeline (per request):
  1. Codec-encode the user audio in one batch (`cmd: encode_full`) AND
     run faster-whisper ASR — both run in parallel on independent workers.
  2. Prefill the user turn (text + codes) into the TTS KV cache.
  3. Get a full LLM reply from Ollama.
  4. Render the reply (engine chunks the text internally via `chunk_text`).
  5. Rewind the KV cache so the next caller — voice-note, WebRTC, or
     /v1/realtime — starts from the prefilled voice-prompt boundary.

Single-tenant: refuses with 409 if another voice-note is in flight, OR if
a WebRTC `/offer` / `/v1/realtime` client is currently bound (would
clobber its TTS engine + persistent ASR session).
"""

from __future__ import annotations

import asyncio
import base64

from aiohttp import web

from vui.serving.stream.test_routes import (
    _decode_audio,
    _llm_reply,
    _run_asr,
    _speak,
)

_active_lock = asyncio.Lock()


async def _encode_full(server, a24) -> object | None:
    """Batch codec-encode (one shot) — much cheaper than stream_start/feed/stop."""
    server.tts_cmd_queue.put({"cmd": "encode_full", "audio": a24})
    msg = await server._wait_tts_response("encoded", timeout=15)
    return msg.get("codes") if msg else None


def register_voice_note_routes(app, server):
    async def voice_note(request):
        if not (server.tts_ready and server.asr_ready):
            return web.json_response({"error": "workers not ready"}, status=503)
        if _active_lock.locked():
            return web.json_response(
                {"error": "another voice-note in flight"}, status=409
            )
        # The TTS engine + persistent ASR session are shared with the WS /
        # WebRTC pipelines. If anything is bound, refuse — running through
        # would rewind their KV mid-conversation and rotate their ASR.
        if server.session.ws is not None:
            return web.json_response(
                {"error": "another session is active (WebRTC or /v1/realtime)"},
                status=409,
            )

        try:
            reader = await request.multipart()
        except Exception as e:
            return web.json_response(
                {"error": f"multipart parse failed: {e}"}, status=400
            )
        audio_bytes: bytes | None = None
        async for part in reader:
            if part.name == "audio":
                audio_bytes = await part.read()
        if not audio_bytes:
            return web.json_response({"error": "audio required"}, status=400)

        try:
            a16, a24 = _decode_audio(audio_bytes)
        except Exception as e:
            return web.json_response({"error": f"decode failed: {e}"}, status=400)

        async with _active_lock:
            try:
                server.tts_cmd_queue.put({"cmd": "rewind"})
                await server._wait_tts_response("done", timeout=5)

                # Codec encode (TTS worker) || ASR (ASR worker) — independent workers,
                # safe to overlap.
                user_codes, asr_text = await asyncio.gather(
                    _encode_full(server, a24),
                    _run_asr(server, a16),
                )
                if not asr_text:
                    return web.json_response(
                        {"error": "asr produced no text"}, status=500
                    )

                server.tts_cmd_queue.put(
                    {
                        "cmd": "prefill_user_turn",
                        "text": asr_text,
                        "codes": user_codes,
                        "audio_16k": a16,
                        "settings": server.session.settings,
                    }
                )
                await server._wait_tts_response("user_prefilled", timeout=10)

                reply_text = await _llm_reply(server, asr_text)
                if not reply_text:
                    return web.json_response(
                        {"error": "llm empty reply", "asr_text": asr_text},
                        status=500,
                    )

                spoken = await _speak(server, reply_text)
                if not spoken.get("ok"):
                    return web.json_response(
                        {
                            "error": spoken.get("error", "tts failed"),
                            "asr_text": asr_text,
                            "reply_text": reply_text,
                        },
                        status=500,
                    )
            finally:
                # Rewind KV back to the prefilled-voice-prompt boundary so the
                # next session (voice-note, WebRTC, /v1/realtime) doesn't inherit
                # this turn's user codes + agent reply.
                server.tts_cmd_queue.put({"cmd": "rewind"})
                try:
                    await server._wait_tts_response("done", timeout=5)
                except Exception:
                    pass

        return web.json_response(
            {
                "ok": True,
                "asr_text": asr_text,
                "reply_text": reply_text,
                "duration_sec": float(spoken.get("duration", 0)),
                "audio": base64.b64encode(spoken["audio_bytes"]).decode("ascii"),
                "audio_format": "wav",
                "sample_rate": 24000,
            }
        )

    app.router.add_post("/v1/voice-note", voice_note)
