"""Pluggable streaming ASR backends.

Backend selected by env var `VUI_ASR`:
  - `moonshine` (default): local CPU, Moonshine streaming models
  - `fwhisper`: local GPU, faster-whisper (distil-small.en etc.)

Additional tuning (backend-specific):
  - Moonshine: `VUI_MOONSHINE_ARCH=0|2|4|5` (0=tiny, 2=tiny-streaming, 4=small-streaming, 5=medium-streaming)
  - fwhisper:  `VUI_FWHISPER_MODEL=distil-small.en|distil-large-v3|turbo|...`
               `VUI_FWHISPER_DEVICE=cuda|cpu`
"""

from vui.serving.stream.asr.base import ASRBackend, ASRSession


def make_backend(name: str, **kwargs) -> ASRBackend:
    name = name.lower()
    if name == "moonshine":
        from vui.serving.stream.asr.moonshine import MoonshineBackend

        return MoonshineBackend(**kwargs)
    if name == "fwhisper":
        from vui.serving.stream.asr.fwhisper import FWhisperBackend

        return FWhisperBackend(**kwargs)
    if name == "mlx_whisper":
        from vui.serving.stream.asr.mlx_whisper import MLXWhisperBackend

        return MLXWhisperBackend(**kwargs)
    raise ValueError(
        f"Unknown ASR backend: {name!r} (options: moonshine, fwhisper, mlx_whisper)"
    )


__all__ = ["ASRBackend", "ASRSession", "make_backend"]
