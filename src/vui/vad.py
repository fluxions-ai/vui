import torch

_model = None
_JIT_PATH = None


def _load():
    global _model, _JIT_PATH
    if _model is not None:
        return _model

    import os
    hub_dir = os.path.join(torch.hub.get_dir(), "snakers4_silero-vad_master")
    jit_path = os.path.join(hub_dir, "src", "silero_vad", "data", "silero_vad.jit")

    if not os.path.exists(jit_path):
        torch.hub.download_url_to_file(
            "https://github.com/snakers4/silero-vad/zipball/master",
            os.path.join(torch.hub.get_dir(), "master.zip"),
        )
        import zipfile
        with zipfile.ZipFile(os.path.join(torch.hub.get_dir(), "master.zip")) as z:
            z.extractall(torch.hub.get_dir())

    _model = torch.jit.load(jit_path, map_location="cpu")
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
