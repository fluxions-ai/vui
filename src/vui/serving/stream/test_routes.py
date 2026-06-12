"""HTTP test routes — OpenAI-style surface for scripted WER/regression tests.

Installed via `register_test_routes(app, server)`. No WebRTC, no browser.

Routes:
  POST /test/speech       — {input: text}    → {audio_bytes, ...}  (text → audio)
  POST /test/transcribe   — multipart audio  → {text}              (audio → text)
  POST /test/respond      — multipart audio  → {asr, reply, ...}   (audio → audio)
  GET  /test/settings     — current session.settings
  POST /test/settings     — merge-update session.settings

`_speak` returns the rendered audio inline as `audio_bytes` (in-memory WAV) —
no disk persistence. Callers that need a file save it themselves.

All routes use the StreamServer reference passed to register_test_routes().
The `_test_capture_sink` and `_test_done_event` attributes on StreamServer
— plus the audio-frame hook in `drain_tts_audio` — remain in server.py
because they sit inside the main audio-queue drain loop.
"""

from __future__ import annotations

import asyncio
import io

import numpy as np
import torch
import torchaudio
from aiohttp import web

from vui.serving.stream.server import DEFAULT_SETTINGS

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a float32 mono tensor to a PCM16 WAV in memory.

    Hand-rolled because torchaudio's torchcodec backend can't write to BytesIO
    (it needs a file extension to pick a muxer).
    """
    pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    n_bytes = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write((36 + n_bytes).to_bytes(4, "little"))
    buf.write(b"WAVEfmt ")
    buf.write((16).to_bytes(4, "little"))      # fmt chunk size
    buf.write((1).to_bytes(2, "little"))       # PCM
    buf.write((1).to_bytes(2, "little"))       # mono
    buf.write(sample_rate.to_bytes(4, "little"))
    buf.write((sample_rate * 2).to_bytes(4, "little"))  # byte rate
    buf.write((2).to_bytes(2, "little"))       # block align
    buf.write((16).to_bytes(2, "little"))      # bits per sample
    buf.write(b"data")
    buf.write(n_bytes.to_bytes(4, "little"))
    buf.write(pcm)
    return buf.getvalue()


def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """bytes → (audio_16k, audio_24k) float32 mono numpy arrays."""
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    wav = wav.mean(0)
    a16 = torchaudio.functional.resample(wav, sr, 16000).numpy().astype(np.float32)
    a24 = torchaudio.functional.resample(wav, sr, 24000).numpy().astype(np.float32)
    return a16, a24


async def _encode_audio(server, a24: np.ndarray):
    """Feed audio through StreamingCodecEncoder → codes tensor (or None)."""
    server._last_stream_codes = None
    server.tts_cmd_queue.put({"cmd": "stream_start"})
    chunk = int(0.02 * 24000)
    for i in range(0, len(a24), chunk):
        server.tts_cmd_queue.put({"cmd": "stream_feed", "audio": a24[i : i + chunk]})
    server.tts_cmd_queue.put({"cmd": "stream_stop"})
    # codes_final is handled by drain_tts_audio (not routed to response queue),
    # so poll _last_stream_codes instead of _wait_tts_response
    for _ in range(100):
        if server._last_stream_codes is not None:
            return server._last_stream_codes
        await asyncio.sleep(0.1)
    return None


async def _run_asr(server, a16: np.ndarray) -> str:
    """Feed audio through ASR worker → final text.

    The live server keeps a persistent ASR session running across turns.
    For test routes we rotate that session (stop -> wait final -> start)
    so the test's audio is transcribed in isolation but the persistent
    session stays alive afterwards.
    """
    from vui.serving.stream.drains import _rotate_asr_session

    # Reset transcript state so the captured final is just for this audio.
    server._phase_transcript = ""
    server._committed_len = 0
    server._carry = ""
    server._current_partial = ""
    server._last_asr_text = None

    # Tear down current persistent session.
    await _rotate_asr_session(server)
    # Now session is fresh — feed the test audio, then rotate again to get final.
    server.asr_cmd_queue.put({"cmd": "feed", "audio": a16, "sample_rate": 16000})
    # Brief wait so the worker actually consumes the feed before stop.
    await asyncio.sleep(0.2)
    server._last_asr_text = None
    await _rotate_asr_session(server)
    # The rotation's final updates _carry; the text we want is _carry.
    text = (server._carry or "").strip()
    server._carry = ""
    server._phase_transcript = ""
    return text


async def _ingest_audio(server, a16: np.ndarray, a24: np.ndarray) -> tuple[str, object]:
    """Codec-encode + ASR. Returns (asr_text, user_codes)."""
    user_codes = await _encode_audio(server, a24)
    asr_text = await _run_asr(server, a16)
    return asr_text, user_codes


async def _llm_reply(server, user_text: str) -> str:
    """Get the LLM's full reply in a single call.

    The live server chunks via `llm_next_chunk` with num_predict=15 to hide
    latency behind TTS, but some models (e.g. glm-4.7-flash) don't continue
    an assistant-role message cleanly and produce duplicated/looping output
    when the chunk loop passes assistant_so_far back in. For test-route
    benchmarking, fidelity > latency — so we just ask for the full reply.
    """
    import httpx

    from vui.serving.stream.server import OLLAMA_URL

    conv = list(server.session.conversation) + [{"role": "user", "content": user_text}]
    messages = [{"role": "system", "content": server.session.soul}] + conv
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": server.ollama_model,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
                "think": False,
                "options": {"num_ctx": 8192, "num_predict": 512},
            },
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
    return server._clean_llm_text(text)


async def _speak(server, text: str) -> dict:
    """Dispatch TTS generate, capture audio via _test_capture_sink, save wav."""
    server._test_capture_sink = []
    server._test_done_event = asyncio.Event()
    server._test_first_frame_ms = None
    server._test_gen_info = None

    T_before = server.tts_T
    server._generates_sent = 1
    server._generates_done = 0
    server._llm_streaming = False
    server.tts_cmd_queue.put(
        {
            "cmd": "generate",
            "text": text,
            "is_voice": False,
            "new_turn": True,
            "is_final": True,
            "settings": server.session.settings,
            "context": "",
        }
    )
    try:
        await asyncio.wait_for(server._test_done_event.wait(), timeout=60)
    except asyncio.TimeoutError:
        server._test_capture_sink = None
        server._test_done_event = None
        return {"ok": False, "error": "generate timeout", "text": text}

    chunks = server._test_capture_sink or []
    info = server._test_gen_info or {}
    first_ms = server._test_first_frame_ms or 0
    server._test_capture_sink = None
    server._test_done_event = None
    if not chunks:
        return {"ok": False, "error": "no audio produced", "text": text, **info}

    audio = torch.cat([c.flatten() for c in chunks]).float().cpu()
    return {
        "ok": True,
        "text": text,
        "audio_bytes": _to_wav_bytes(audio, sample_rate=24000),
        "duration": float(audio.shape[-1] / 24000),
        "T_before": T_before,
        "T_after": info.get("T", server.tts_T),
        "first_frame_ms": first_ms,
        "total_gen_time": info.get("total_gen_time", 0),
        "total_frames": info.get("total_frames", 0),
        "terminal": info.get("type", "unknown"),
    }


async def _maybe_rewind(server, request):
    """Honor ?reset=true — rewind KV back to prompt_T before the test."""
    if request.query.get("reset", "true").lower() == "true":
        server.tts_cmd_queue.put({"cmd": "rewind"})
        await server._wait_tts_response("done", timeout=5)


async def _read_audio_part(request) -> bytes | None:
    reader = await request.multipart()
    async for part in reader:
        if part.name == "audio":
            return await part.read()
    return None


# ----------------------------------------------------------------------------
# Route registration
# ----------------------------------------------------------------------------


def _spoken_to_json(spoken: dict) -> dict:
    """Strip raw audio bytes for JSON-safe response (keep text + metadata)."""
    return {k: v for k, v in spoken.items() if k != "audio_bytes"}


def register_test_routes(app, server):
    """Install /test/* routes on `app`, closing over `server` for state."""

    async def speech(request):
        if not server.tts_ready:
            return web.json_response(
                {"ok": False, "error": "tts not ready"}, status=503
            )
        data = await request.json()
        text = (data.get("input") or "").strip()
        if not text:
            return web.json_response(
                {"ok": False, "error": "input required"}, status=400
            )
        await _maybe_rewind(server, request)
        spoken = await _speak(server, text)
        if not spoken.get("ok"):
            return web.json_response(_spoken_to_json(spoken), status=500)
        # Return the rendered WAV directly (was previously a `wav_path` to disk).
        return web.Response(
            body=spoken["audio_bytes"],
            content_type="audio/wav",
            headers={"X-Duration-Sec": f"{spoken.get('duration', 0):.2f}"},
        )

    async def transcribe(request):
        if not server.asr_ready:
            return web.json_response(
                {"ok": False, "error": "asr not ready"}, status=503
            )
        audio_bytes = await _read_audio_part(request)
        if not audio_bytes:
            return web.json_response(
                {"ok": False, "error": "audio required"}, status=400
            )
        try:
            a16, _ = _decode_audio(audio_bytes)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": f"decode failed: {e}"}, status=400
            )
        text = await _run_asr(server, a16)
        if text:
            return web.json_response({"ok": True, "text": text})
        return web.json_response({"ok": False, "error": "asr timeout"}, status=500)

    async def respond(request):
        if not (server.tts_ready and server.asr_ready):
            return web.json_response(
                {"ok": False, "error": "workers not ready"}, status=503
            )
        audio_bytes = await _read_audio_part(request)
        if not audio_bytes:
            return web.json_response(
                {"ok": False, "error": "audio required"}, status=400
            )
        try:
            a16, a24 = _decode_audio(audio_bytes)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": f"decode failed: {e}"}, status=400
            )

        await _maybe_rewind(server, request)
        asr_text, user_codes = await _ingest_audio(server, a16, a24)
        if user_codes is not None:
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
                {"ok": False, "error": "LLM empty reply", "asr_text": asr_text},
                status=500,
            )
        spoken = await _speak(server, reply_text)
        spoken["asr_text"] = asr_text
        spoken["reply_text"] = reply_text
        return web.json_response(_spoken_to_json(spoken))

    async def prefill(request):
        """Prefill a user turn into the KV cache.

        JSON:      POST /test/prefill  {text: "..."}           — text only
        Multipart: POST /test/prefill  text=...  audio=<wav>   — text + audio codes

        When audio is provided, it's codec-encoded and the resulting codes
        are written into the KV alongside the text (matching the live mic path).
        """
        if not server.tts_ready:
            return web.json_response(
                {"ok": False, "error": "tts not ready"}, status=503
            )

        audio_bytes = None
        if request.content_type and "multipart" in request.content_type:
            text = ""
            reader = await request.multipart()
            async for part in reader:
                if part.name == "audio":
                    audio_bytes = await part.read()
                elif part.name == "text":
                    text = (await part.text()).strip()
        else:
            data = await request.json()
            text = (data.get("text") or "").strip()

        if not text:
            return web.json_response(
                {"ok": False, "error": "text required"}, status=400
            )

        await _maybe_rewind(server, request)

        user_codes = None
        a16 = None
        if audio_bytes:
            try:
                a16, a24 = _decode_audio(audio_bytes)
            except Exception as e:
                return web.json_response(
                    {"ok": False, "error": f"decode failed: {e}"}, status=400
                )
            user_codes = await _encode_audio(server, a24)

        server.tts_cmd_queue.put(
            {
                "cmd": "prefill_user_turn",
                "text": text,
                "codes": user_codes,
                "audio_16k": a16,
                "settings": server.session.settings,
            }
        )
        msg = await server._wait_tts_response("user_prefilled", timeout=10)
        if msg:
            return web.json_response(
                {
                    "ok": True,
                    "T": msg.get("T", 0),
                    "has_codes": user_codes is not None,
                }
            )
        return web.json_response({"ok": False, "error": "prefill timeout"}, status=500)

    async def settings(request):
        if request.method == "GET":
            return web.json_response(server.session.settings)
        data = await request.json()
        if data.get("reset"):
            server.session.settings = dict(DEFAULT_SETTINGS)
        else:
            server.session.settings.update(data)
        # Cross-process push for any setting that lives outside main.
        if data.get("reset") or "vad_stop_secs" in data:
            server.asr_cmd_queue.put(
                {
                    "cmd": "set_vad_stop_secs",
                    "secs": server.session.settings.get("vad_stop_secs", 0.3),
                }
            )
        return web.json_response(server.session.settings)

    app.router.add_post("/test/speech", speech)
    app.router.add_post("/test/transcribe", transcribe)
    app.router.add_post("/test/respond", respond)
    app.router.add_post("/test/prefill", prefill)
    app.router.add_get("/test/settings", settings)
    app.router.add_post("/test/settings", settings)
