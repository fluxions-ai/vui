"""ASR backend + session protocol.

All backends emit the same message shape on `result_queue` so the worker
loop in `asr_worker.py` is backend-agnostic:

  {"type": "partial",        "text": "..."}               # running transcript updates
  {"type": "line_completed", "text": "...",
   "start_time": float, "duration": float}                # sentence-completed events
  {"type": "final",          "text": "..."}               # session stopped, final text
  {"type": "transcribed",    "text": "..."}               # non-streaming transcribe_full
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from multiprocessing import Queue

import numpy as np


class ASRSession(ABC):
    """A single streaming ASR session (start -> feed -> stop)."""

    def __init__(self, result_queue: Queue):
        self.result_queue = result_queue

    @abstractmethod
    def start(self) -> None:
        """Begin a new streaming session."""

    @abstractmethod
    def feed(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Push an audio chunk (float32, mono)."""

    @abstractmethod
    def stop(self) -> None:
        """Finalize the session. MUST emit {"type": "final", "text": ...}."""


class ASRBackend(ABC):
    """Loads the ASR model, creates sessions on demand."""

    name: str = "<unset>"

    @abstractmethod
    def make_session(self, result_queue: Queue) -> ASRSession:
        """Create a new streaming session bound to this backend."""

    @abstractmethod
    def transcribe_once(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Non-streaming batch transcription. Used for prompt audio uploads."""
