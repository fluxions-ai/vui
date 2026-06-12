"""ASR worker process: backend-agnostic streaming speech recognition + Silero VAD.

Backend is selected via the `VUI_ASR` env var (see `vui/serving/stream/asr/__init__.py`):
  - `moonshine` (default, CPU, ONNX)
  - `fwhisper`  (GPU, faster-whisper)

Additional tuning:
  - `VUI_MOONSHINE_ARCH`   — 0|2|4|5  (default 4 = small-streaming)
  - `VUI_FWHISPER_MODEL`   — distil-small.en|distil-large-v3|turbo|...
  - `VUI_FWHISPER_DEVICE`  — cuda|cpu  (default cuda)
"""

import gc
import os
import time
import traceback
from multiprocessing import Queue
from queue import Empty

import numpy as np

from vui.serving.stream.asr import make_backend
from vui.serving.stream.vad import SileroVAD, VADState

ASR_MODELS = {
    "moonshine.tiny": ("moonshine", {"arch": 2}),
    "moonshine.small": ("moonshine", {"arch": 4}),
    "moonshine.medium": ("moonshine", {"arch": 5}),
    "fwhisper.distil-small.en": (
        "fwhisper",
        {"model": "distil-small.en"},
    ),
    "fwhisper.small.en": ("fwhisper", {"model": "small.en"}),
    "fwhisper.distil-medium.en": (
        "fwhisper",
        {"model": "distil-medium.en"},
    ),
    "fwhisper.medium.en": ("fwhisper", {"model": "medium.en"}),
    "fwhisper.distil-large-v3": (
        "fwhisper",
        {"model": "distil-large-v3"},
    ),
    "fwhisper.turbo": ("fwhisper", {"model": "turbo"}),
    "mlx-whisper.small": ("mlx_whisper", {"model": "small"}),
    "mlx-whisper.turbo": ("mlx_whisper", {"model": "turbo"}),
}

import platform as _platform
import sys as _sys

_IS_APPLE_SILICON = _sys.platform == "darwin" and _platform.machine() == "arm64"
DEFAULT_ASR_MODEL = "moonshine.small" if _IS_APPLE_SILICON else "fwhisper.distil-small.en"


def _build_backend(model_key: str | None = None):
    if model_key and model_key in ASR_MODELS:
        name, kwargs = ASR_MODELS[model_key]
    else:
        name = os.environ.get("VUI_ASR", "moonshine").lower()
        kwargs = {}
        if name == "moonshine":
            kwargs["arch"] = int(os.environ.get("VUI_MOONSHINE_ARCH", "4"))
        elif name == "fwhisper":
            kwargs["model"] = os.environ.get("VUI_FWHISPER_MODEL", "distil-small.en")
            kwargs["device"] = os.environ.get("VUI_FWHISPER_DEVICE", "cuda")
    print(f"[ASR] Backend: {name}  kwargs={kwargs}")
    return make_backend(name, **kwargs)


def asr_process(
    cmd_queue: Queue,
    result_queue: Queue,
    vad_queue: Queue,
    model_arch: int = 4,  # kept for backward-compat with old callers; overridden by env
):
    """Main ASR worker loop.

    Polls cmd_queue (ASR commands) and vad_queue (raw audio for VAD) separately
    to avoid contention between high-frequency VAD audio and ASR commands.
    """
    # Ignore SIGINT — main process handles Ctrl+C and sends `shutdown` cmd.
    # Without this the worker dumps a multiprocessing traceback on every quit.
    import signal
    import threading

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Allow model_arch kwarg to seed Moonshine env var if env not explicitly set
    os.environ.setdefault("VUI_MOONSHINE_ARCH", str(model_arch))

    print("[ASR] Loading backend...")
    backend = _build_backend(DEFAULT_ASR_MODEL)
    print(f"[ASR] Loaded backend: {backend.name}")

    print("[ASR] Loading Silero VAD...")
    vad = SileroVAD(sample_rate=16000, stop_secs=0.3)
    vad.process(np.zeros(512, dtype=np.float32))
    print("[ASR] VAD ready!")

    print("[ASR] Ready!")
    result_queue.put({"type": "ready"})

    session = None
    vad_enabled = False
    vad_chunks = 0
    vad_audio_signalled = False

    def _drain_vad():
        nonlocal vad_chunks, vad_audio_signalled
        drained = 0
        while True:
            try:
                audio = vad_queue.get_nowait()
            except Empty:
                break
            vad_chunks += 1
            drained += 1
            if not vad_audio_signalled:
                vad_audio_signalled = True
                print("[ASR] First VAD audio chunk received — mic pipeline live")
                result_queue.put({"type": "vad_audio_ready"})
            try:
                transition = vad.process(audio)
                if transition == VADState.SPEAKING:
                    print("[ASR] VAD: speech start")
                    result_queue.put({"type": "vad_start"})
                elif transition == VADState.QUIET:
                    print("[ASR] VAD: speech stop")
                    result_queue.put({"type": "vad_stop"})
            except Exception:
                traceback.print_exc()
            # Yield GIL periodically so other threads (TTS) can run
            if drained % 4 == 0:
                time.sleep(0)

    while True:
        if vad_enabled:
            _drain_vad()

        try:
            msg = cmd_queue.get(timeout=0.05)
        except Empty:
            continue
        except Exception:
            break

        cmd = msg.get("cmd")

        if cmd == "shutdown":
            if session is not None:
                try:
                    session.stop()
                except Exception:
                    pass
            break

        elif cmd == "start":
            try:
                session = backend.make_session(result_queue)
                session.start()
                # Sentinel so main can drop stale msgs queued by the dying
                # session's interim threads (joined inside `session.stop()`).
                result_queue.put({"type": "session_ready"})
            except Exception:
                traceback.print_exc()
                session = None

        elif cmd == "feed":
            if session is None:
                continue
            try:
                session.feed(msg["audio"], msg.get("sample_rate", 16000))
            except Exception:
                traceback.print_exc()

        elif cmd == "stop":
            if session is None:
                result_queue.put({"type": "final", "text": ""})
                continue
            try:
                session.stop()
            except Exception:
                traceback.print_exc()
                result_queue.put({"type": "final", "text": ""})
            session = None

        elif cmd == "vad_enable":
            vad_enabled = True
            vad.state = VADState.QUIET
            vad._counter = 0
            print("[ASR] VAD enabled (state reset)")

        elif cmd == "vad_disable":
            vad_enabled = False
            print("[ASR] VAD disabled")

        elif cmd == "set_vad_stop_secs":
            # SileroVAD reads stop_secs every frame to derive stop_frames,
            # so live mutation takes effect on the next chunk without state
            # rebuild. Lower values = snappier endpointing.
            try:
                secs = float(msg.get("secs", vad.stop_secs))
                vad.stop_secs = max(0.05, secs)
                print(f"[ASR] VAD stop_secs -> {vad.stop_secs:.2f}")
            except Exception:
                traceback.print_exc()

        elif cmd == "set_backend":
            model_key = msg.get("model", "")
            if model_key not in ASR_MODELS:
                print(f"[ASR] Unknown model: {model_key}")
                result_queue.put(
                    {
                        "type": "backend_set",
                        "ok": False,
                        "error": f"unknown: {model_key}",
                    }
                )
                continue
            if session is not None:
                try:
                    session.stop()
                except Exception:
                    pass
                session = None
            print(f"[ASR] Switching to {model_key}...")
            t0 = time.perf_counter()
            old_name = getattr(backend, "name", "?")
            backend = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"[ASR] Unloaded {old_name}")
            try:
                backend = _build_backend(model_key)
                t_ms = (time.perf_counter() - t0) * 1000
                print(f"[ASR] Switched to {backend.name} ({t_ms:.0f}ms)")
                result_queue.put(
                    {
                        "type": "backend_set",
                        "ok": True,
                        "model": model_key,
                        "name": backend.name,
                    }
                )
            except Exception as e:
                traceback.print_exc()
                print(f"[ASR] Failed to switch: {e}")
                result_queue.put({"type": "backend_set", "ok": False, "error": str(e)})
            # Flush stale VAD audio that accumulated during model load
            if vad_enabled:
                stale = 0
                while not vad_queue.empty():
                    try:
                        vad_queue.get_nowait()
                        stale += 1
                    except Empty:
                        break
                if stale:
                    print(f"[ASR] Flushed {stale} stale VAD chunks after model switch")
                vad.state = VADState.QUIET
                vad._counter = 0

        elif cmd == "transcribe_full":
            try:
                audio = msg["audio"]
                t0 = time.perf_counter()
                text = backend.transcribe_once(audio, 16000)
                t_ms = (time.perf_counter() - t0) * 1000
                print(f"[ASR] Transcribe: '{text}' ({t_ms:.0f}ms)")
                result_queue.put({"type": "transcribed", "text": text})
            except Exception:
                traceback.print_exc()
                result_queue.put({"type": "transcribed", "text": ""})

    print("[ASR] Shutting down")
