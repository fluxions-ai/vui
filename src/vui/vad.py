import torch

_model = None


def _load():
    global _model
    if _model is not None:
        return _model

    _model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        onnx=True,
        force_onnx_cpu=True,
        trust_repo=True,
    )
    if hasattr(_model, "eval"):
        _model.eval()
    return _model


def _get_speech_timestamps(
    waveform: torch.Tensor,
    model,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
) -> list[dict]:
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000

    audio = waveform.flatten()
    speeches = []
    current_speech = {}
    triggered = False
    neg_threshold = threshold - 0.15

    model.reset_states()

    for i in range(0, len(audio), window_size_samples):
        chunk = audio[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
        prob = model(chunk, sampling_rate).item()

        if prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = i
        elif prob < neg_threshold and triggered:
            if i - current_speech["start"] >= min_speech_samples:
                current_speech["end"] = i
                # check silence
                if speeches and current_speech["start"] - speeches[-1]["end"] < min_silence_samples:
                    speeches[-1]["end"] = current_speech["end"]
                else:
                    speeches.append(current_speech)
            current_speech = {}
            triggered = False

    if triggered and len(audio) - current_speech["start"] >= min_speech_samples:
        current_speech["end"] = len(audio)
        speeches.append(current_speech)

    return speeches


@torch.autocast("cuda", enabled=False)
def detect_voice_activity(waveform: torch.Tensor, sr: int = 16000) -> list[tuple[float, float]]:
    """Returns list of (start_seconds, end_seconds) for speech segments. Input: 16kHz mono."""
    waveform = waveform.flatten().float().cpu()
    model = _load()
    segments = _get_speech_timestamps(waveform, model, sampling_rate=sr)
    return [(s["start"] / sr, s["end"] / sr) for s in segments]
