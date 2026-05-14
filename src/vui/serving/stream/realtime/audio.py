"""Audio encoding helpers + voice discovery for the /v1/realtime endpoint."""

from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

SAMPLE_RATE_OUT = 24000
SAMPLE_RATE_ASR = 16000


def list_voices() -> list[str]:
    pdir = Path("prompts")
    if not pdir.exists():
        return []
    return sorted(p.stem for p in pdir.glob("*.pt"))


def audio_to_pcm16_b64(data: Any) -> str:
    if isinstance(data, torch.Tensor):
        a = data.detach().to(torch.float32).cpu().numpy()
    else:
        a = np.asarray(data, dtype=np.float32)
    a = a.flatten()
    a = np.clip(a, -1.0, 1.0)
    pcm16 = (a * 32767.0).astype("<i2").tobytes()
    return base64.b64encode(pcm16).decode("ascii")


def pcm16_b64_to_float32(b64: str) -> np.ndarray:
    pcm = base64.b64decode(b64)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:16]}"


def new_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:16]}"


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:16]}"


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:24]}"
