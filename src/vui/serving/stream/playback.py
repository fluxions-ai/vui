"""TTS playback track for WebRTC audio output.

Runs in the main process event loop. Serves TTS audio frames at 24kHz, paced at real-time.
"""

import asyncio
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

_ATTACK = 0.5
_RELEASE = 0.08
_MAX_GAIN = 4.0
_MIN_GAIN = 0.25
_DEFAULT_TARGET_RMS = 0.15
_LIMITER_HEADROOM = 2.5  # limiter kicks in at target_rms * headroom
_LIMITER_RATIO = 4.0


class TTSPlaybackTrack(MediaStreamTrack):
    """Serves TTS audio as WebRTC frames at 24kHz (native Qwen codec rate), paced at real-time."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[torch.Tensor] = asyncio.Queue()
        self._remainder: torch.Tensor | None = None
        self._pts = 0
        self._sample_rate = 48000
        self._samples_per_frame = 960  # 20ms at 48kHz
        self._frame_duration = self._samples_per_frame / self._sample_rate
        self._start_time: float | None = None
        self._frame_count = 0
        self._paused = False
        self._target_rms: float = _DEFAULT_TARGET_RMS
        self._norm_gain: float | None = None

    def set_target_rms(self, rms: float):
        self._target_rms = max(rms, 1e-4)
        self._norm_gain = None

    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        rms = audio.pow(2).mean().sqrt().item()
        if rms < 1e-6:
            return audio
        desired = max(_MIN_GAIN, min(_MAX_GAIN, self._target_rms / rms))
        if self._norm_gain is None:
            self._norm_gain = desired
        else:
            alpha = _ATTACK if desired < self._norm_gain else _RELEASE
            self._norm_gain += alpha * (desired - self._norm_gain)
        return audio * self._norm_gain

    def _limit(self, audio: torch.Tensor) -> torch.Tensor:
        thresh = self._target_rms * _LIMITER_HEADROOM
        mag = audio.abs()
        over = (mag - thresh).clamp(min=0)
        reduced = thresh + over / _LIMITER_RATIO
        return torch.where(mag > thresh, audio.sign() * reduced, audio)

    def enqueue_audio(self, audio_24k: torch.Tensor):
        # Per-frame normalize+limit was causing pumping/stutter — gain is
        # recomputed from each 80ms RMS and the asymmetric smoothing
        # (attack=0.5, release=0.08) creates audible level swells across
        # frames. Bypass for now and see if the live server still stutters.
        # audio_24k = self._limit(self._normalize(audio_24k))
        # Resample 24kHz -> 48kHz for Opus compatibility (simple linear interpolation)
        audio_48k = (
            torch.nn.functional.interpolate(
                audio_24k.unsqueeze(0).unsqueeze(0),
                scale_factor=2,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        self._queue.put_nowait(audio_48k)

    async def wait_drained(self):
        """Wait until all queued audio has been served as WebRTC frames."""
        while not self._queue.empty() or self._remainder is not None:
            await asyncio.sleep(0.02)

    @property
    def can_pause(self) -> bool:
        return not self._queue.empty() or self._remainder is not None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def flush(self):
        self._paused = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._remainder = None

    async def recv(self) -> AudioFrame:
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()

        self._frame_count += 1
        target_time = self._start_time + self._frame_count * self._frame_duration
        now = asyncio.get_event_loop().time()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        samples_needed = self._samples_per_frame

        if self._paused:
            audio = torch.zeros(samples_needed)
        else:
            chunks = []
            have = 0

            if self._remainder is not None and len(self._remainder) > 0:
                chunks.append(self._remainder)
                have += len(self._remainder)
                self._remainder = None

            while have < samples_needed:
                try:
                    chunk = self._queue.get_nowait()
                    chunks.append(chunk)
                    have += len(chunk)
                except asyncio.QueueEmpty:
                    break

            if chunks:
                audio = torch.cat(chunks)
                if len(audio) > samples_needed:
                    self._remainder = audio[samples_needed:]
                    audio = audio[:samples_needed]
                elif len(audio) < samples_needed:
                    audio = F.pad(audio, (0, samples_needed - len(audio)))
            else:
                audio = torch.zeros(samples_needed)

        audio_np = (audio.clamp(-1, 1).numpy() * 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(
            audio_np.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, 48000)
        self._pts += samples_needed

        return frame
