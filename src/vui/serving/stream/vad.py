"""Silero VAD with hysteresis state machine. Adapted from pipecat."""

import time
from enum import IntEnum
from pathlib import Path

import numpy as np
import onnxruntime

MODEL_PATH = Path(__file__).parent / "silero_vad.onnx"
MODEL_RESET_INTERVAL = 120.0


class VADState(IntEnum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class SileroVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        confidence: float = 0.7,
        min_volume: float = 0.0,
        start_secs: float = 0.2,
        stop_secs: float = 0.5,
    ):
        assert sample_rate in (8000, 16000)
        self.sample_rate = sample_rate
        self.confidence = confidence
        self.min_volume = min_volume
        self.start_secs = start_secs
        self.stop_secs = stop_secs

        # ONNX model
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._state_h = np.zeros((2, 1, 128), dtype="float32")
        self._context = np.zeros((1, 0), dtype="float32")
        self._last_reset = time.monotonic()

        # State machine
        self.state = VADState.QUIET
        self._counter = 0
        self._smoothed_volume = 0.0

        # Buffering: silero needs exactly 512 samples @ 16kHz
        self._num_samples = 512 if sample_rate == 16000 else 256
        self._buf = np.array([], dtype=np.float32)

        # Warm up: run a few silent frames so the RNN hidden state is initialized
        warmup = np.zeros(self._num_samples, dtype=np.float32)
        for _ in range(5):
            self._model_call(warmup)

    def _model_call(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = x[np.newaxis, :]
        ctx_size = 64 if self.sample_rate == 16000 else 32
        if self._context.shape[1] == 0:
            self._context = np.zeros((1, ctx_size), dtype="float32")
        x = np.concatenate((self._context, x), axis=1)
        ort_inputs = {
            "input": x,
            "state": self._state_h,
            "sr": np.array(self.sample_rate, dtype="int64"),
        }
        out, state = self._session.run(None, ort_inputs)
        self._state_h = state
        self._context = x[..., -ctx_size:]
        # Periodic reset (only during silence to avoid breaking mid-speech detection)
        now = time.monotonic()
        if (
            now - self._last_reset >= MODEL_RESET_INTERVAL
            and self.state == VADState.QUIET
        ):
            self._state_h = np.zeros((2, 1, 128), dtype="float32")
            self._context = np.zeros((1, 0), dtype="float32")
            self._last_reset = now
        return float(out[0].item())

    def _volume(self, audio: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio**2)))
        self._smoothed_volume = 0.2 * rms + 0.8 * self._smoothed_volume
        return self._smoothed_volume

    def process(self, audio: np.ndarray) -> VADState | None:
        """Feed float32 audio. Returns new state only on QUIET<->SPEAKING transitions."""
        self._buf = np.concatenate((self._buf, audio))

        result = None
        while len(self._buf) >= self._num_samples:
            chunk = self._buf[: self._num_samples]
            self._buf = self._buf[self._num_samples :]

            conf = self._model_call(chunk)
            vol = self._volume(chunk)
            speaking = conf >= self.confidence and vol >= self.min_volume

            self.state
            frame_dur = self._num_samples / self.sample_rate
            start_frames = max(1, int(self.start_secs / frame_dur))
            stop_frames = max(1, int(self.stop_secs / frame_dur))

            if self.state == VADState.QUIET:
                if speaking:
                    self.state = VADState.STARTING
                    self._counter = 1
            elif self.state == VADState.STARTING:
                if speaking:
                    self._counter += 1
                    if self._counter >= start_frames:
                        self.state = VADState.SPEAKING
                        result = VADState.SPEAKING
                else:
                    self.state = VADState.QUIET
                    self._counter = 0
            elif self.state == VADState.SPEAKING:
                if not speaking:
                    self.state = VADState.STOPPING
                    self._counter = 1
            elif self.state == VADState.STOPPING:
                if speaking:
                    self.state = VADState.SPEAKING
                    self._counter = 0
                else:
                    self._counter += 1
                    if self._counter >= stop_frames:
                        self.state = VADState.QUIET
                        result = VADState.QUIET
                        self._counter = 0

        return result
