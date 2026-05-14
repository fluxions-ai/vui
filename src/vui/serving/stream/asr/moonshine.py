"""Moonshine streaming ASR backend (local CPU ONNX).

Wraps the existing Moonshine Transcriber in the ASRBackend/ASRSession API.
"""

from __future__ import annotations

import ctypes
import time
import traceback
from multiprocessing import Queue

import numpy as np
from moonshine_voice import TranscriptEventListener
from moonshine_voice.moonshine_api import ModelArch

from vui.serving.stream.asr.base import ASRBackend, ASRSession

FORCE_UPDATE = 1 << 0


def _add_audio_np(stream, audio_np: np.ndarray, sample_rate: int = 16000):
    """Feed audio to a moonshine stream without .tolist() copy."""
    from moonshine_voice.errors import check_error

    arr = np.ascontiguousarray(audio_np, dtype=np.float32)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    error = stream._lib.moonshine_transcribe_add_audio_to_stream(
        stream._transcriber._handle,
        stream._handle,
        ptr,
        len(arr),
        sample_rate,
        0,
    )
    check_error(error)
    stream._stream_time += len(arr) / sample_rate
    if stream._stream_time - stream._last_update_time >= stream._update_interval:
        stream.update_transcription(0)
        stream._last_update_time = stream._stream_time


def _transcribe_np(transcriber, audio_np: np.ndarray, sample_rate: int = 16000):
    from moonshine_voice.errors import check_error
    from moonshine_voice.moonshine_api import TranscriptC

    arr = np.ascontiguousarray(audio_np, dtype=np.float32)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out = ctypes.POINTER(TranscriptC)()
    error = transcriber._lib.moonshine_transcribe_without_streaming(
        transcriber._handle, ptr, len(arr), sample_rate, 0, ctypes.byref(out)
    )
    check_error(error)
    return transcriber._parse_transcript(out)


def _transcript_text(transcript) -> str:
    return " ".join(line.text.strip() for line in transcript.lines if line.text.strip())


class _ASRListener(TranscriptEventListener):
    """Moonshine event listener — forwards to result_queue + keeps local text."""

    def __init__(self, result_queue: Queue):
        self._queue = result_queue
        self._completed: list[str] = []
        self._current = ""
        self._callback_count = 0

    @property
    def partial_text(self) -> str:
        parts = self._completed[:]
        if self._current:
            parts.append(self._current)
        return " ".join(parts).strip()

    def on_line_text_changed(self, event):
        self._callback_count += 1
        self._current = event.line.text
        if self._callback_count == 1:
            print(f"[ASR] First callback: '{event.line.text}'")
        self._queue.put({"type": "partial", "text": self.partial_text})

    def on_line_completed(self, event):
        if event.line.text.strip():
            self._completed.append(event.line.text.strip())
            self._queue.put(
                {
                    "type": "line_completed",
                    "text": event.line.text.strip(),
                    "start_time": event.line.start_time,
                    "duration": event.line.duration,
                }
            )
        self._current = ""


class MoonshineSession(ASRSession):
    def __init__(self, transcriber, result_queue: Queue):
        super().__init__(result_queue)
        self._tr = transcriber
        self._stream = None
        self._listener: _ASRListener | None = None
        self._feed_count = 0
        self._total_samples = 0
        self._max_abs = 0.0

    def start(self) -> None:
        self._listener = _ASRListener(self.result_queue)
        self._stream = self._tr.create_stream(update_interval=0.3)
        self._stream.add_listener(self._listener)
        self._stream.start()
        print("[ASR] Stream started")

    def feed(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        if self._stream is None:
            return
        self._feed_count += 1
        self._total_samples += len(audio)
        self._max_abs = max(self._max_abs, float(np.abs(audio).max()))
        if self._feed_count == 1:
            print(
                f"[ASR] First feed: shape={audio.shape}, dtype={audio.dtype}, "
                f"range=[{audio.min():.4f}, {audio.max():.4f}]"
            )
        _add_audio_np(self._stream, audio, sample_rate)

    def stop(self) -> None:
        if self._stream is None:
            self.result_queue.put({"type": "final", "text": ""})
            return
        try:
            print(
                f"[ASR] Stopping stream (stream_time={self._stream._stream_time:.2f}s, "
                f"feeds={self._feed_count}, samples={self._total_samples}, "
                f"max_abs={self._max_abs:.4f}, "
                f"callbacks={self._listener._callback_count if self._listener else 0}, "
                f"listener='{self._listener.partial_text if self._listener else ''}')"
            )
            # Pad 1s of silence so moonshine flushes any in-flight partial
            _add_audio_np(self._stream, np.zeros(16000, dtype=np.float32), 16000)
            self._stream.update_transcription(FORCE_UPDATE)
            transcript = self._stream.stop()
            n_lines = len(transcript.lines) if transcript else 0
            text = _transcript_text(transcript) if transcript else ""
            if not text and self._listener:
                text = self._listener.partial_text
            print(f"[ASR] Final: '{text}' (transcript_lines={n_lines})")
            self.result_queue.put({"type": "final", "text": text})
        except Exception:
            traceback.print_exc()
            self.result_queue.put({"type": "final", "text": ""})
        self._stream = None
        self._listener = None


class MoonshineBackend(ASRBackend):
    def __init__(self, arch: int = 4, **_):
        from moonshine_voice import Transcriber, get_model_for_language

        mp, ma = get_model_for_language("en", ModelArch(arch))
        self.name = f"moonshine:{mp.split('/')[-2] if '/' in mp else mp}"
        t0 = time.perf_counter()
        self._tr = Transcriber(
            model_path=mp, model_arch=ma, options={"transcription_interval": 0.2}
        )
        print(f"[ASR] Loaded in {(time.perf_counter()-t0)*1000:.0f}ms ({self.name})")
        # Warm
        s = self._tr.create_stream(update_interval=0.1)
        s.start()
        s.stop()

    def make_session(self, result_queue: Queue) -> ASRSession:
        return MoonshineSession(self._tr, result_queue)

    def transcribe_once(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        transcript = _transcribe_np(self._tr, audio, sample_rate)
        return _transcript_text(transcript)
