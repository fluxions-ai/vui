"""Load moonshine streaming models and transcribe audio."""

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from vui.mlx.asr.model import MoonshineStreaming, StreamingConfig

MODELS = {
    "tiny": "UsefulSensors/moonshine-streaming-tiny",
    "small": "UsefulSensors/moonshine-streaming-small",
    "medium": "UsefulSensors/moonshine-streaming-medium",
}


def load_model(name_or_path: str = "small") -> MoonshineStreaming:
    from huggingface_hub import snapshot_download

    repo = MODELS.get(name_or_path, name_or_path)
    local_path = Path(snapshot_download(repo))

    with open(local_path / "config.json") as f:
        config = StreamingConfig.from_dict(json.load(f))

    model = MoonshineStreaming(config)

    weights = {}
    for wf in sorted(local_path.glob("*.safetensors")):
        weights.update(mx.load(str(wf)))

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    try:
        from transformers import AutoTokenizer

        model._tokenizer = AutoTokenizer.from_pretrained(str(local_path))
    except Exception:
        pass

    return model


def load_audio(path: str, sr: int = 16000) -> mx.array:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != sr:
        from scipy.signal import resample_poly

        gcd = math.gcd(sample_rate, sr)
        audio = resample_poly(audio, sr // gcd, sample_rate // gcd).astype(np.float32)
    return mx.array(audio)


def transcribe(
    audio: mx.array | str, model: MoonshineStreaming | None = None, **kwargs
) -> str:
    if model is None:
        model = load_model("small")
    if isinstance(audio, str):
        audio = load_audio(audio)
    return model.transcribe(audio, **kwargs)
