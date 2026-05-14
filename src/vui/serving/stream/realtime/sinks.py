"""Stub sinks that mimic the WebRTC track interfaces the pipeline expects.

`RealtimePlaybackSink` replaces `TTSPlaybackTrack` (forwards 24k frames out
as response.audio.delta). `RealtimeRecordingSink` replaces
`AudioRecordingSink` for the codec stream lifecycle (no MediaStreamTrack —
audio enters via input_audio_buffer.append events instead).
"""

from __future__ import annotations

import numpy as np

from vui.serving.stream._log import _slog

from .audio import SAMPLE_RATE_ASR


class RealtimePlaybackSink:
    can_pause = False

    def __init__(self, on_audio):
        self._on_audio = on_audio
        self._target_rms: float | None = None

    def enqueue_audio(self, audio_24k):
        try:
            self._on_audio(audio_24k)
        except Exception as e:
            _slog(f"[realtime] enqueue_audio error: {e}")

    def flush(self):
        pass

    async def wait_drained(self):
        return

    def set_target_rms(self, rms: float):
        self._target_rms = rms

    def pause(self):
        pass

    def resume(self):
        pass


class RealtimeRecordingSink:
    def __init__(self, srv):
        self.srv = srv
        self.recording = False
        self.vad_enabled = True
        self._speaking = False
        self._samples_recorded = 0
        self._audio_16k_chunks: list[np.ndarray] = []

    def start_recording(self):
        self.recording = True
        self._samples_recorded = 0
        self._audio_16k_chunks = []
        self.srv.tts_cmd_queue.put({"cmd": "stream_start"})

    def stop_recording(self) -> float:
        self.recording = False
        self.srv.tts_cmd_queue.put({"cmd": "stream_stop"})
        return self._samples_recorded / SAMPLE_RATE_ASR

    @property
    def rec_duration(self) -> float:
        return self._samples_recorded / SAMPLE_RATE_ASR

    @property
    def speaking(self) -> bool:
        return self._speaking

    @speaking.setter
    def speaking(self, value: bool):
        self._speaking = value

    def get_audio_16k(self) -> np.ndarray | None:
        if not self._audio_16k_chunks:
            return None
        return np.concatenate(self._audio_16k_chunks)
