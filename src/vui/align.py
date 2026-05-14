"""Forced alignment using wav2vec2 via torchaudio. Word-level timing extraction."""

from dataclasses import dataclass

import torch
import torchaudio

SAMPLE_RATE = 16000

_model = None
_meta = None


def _load(device="cuda"):
    global _model, _meta
    target = torch.device(device)
    if _model is not None:
        cur = next(_model.parameters()).device
        if cur != target:
            _model = _model.to(target)
        return _model, _meta
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    _model = bundle.get_model().to(target)
    labels = bundle.get_labels()
    _meta = {c.lower(): i for i, c in enumerate(labels)}
    return _model, _meta


def unload():
    global _model, _meta
    if _model is not None:
        del _model
        _model = None
        _meta = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_trellis(emission, tokens, blank_id=0):
    T, N = emission.size(0), len(tokens)
    trellis = torch.empty((T + 1, N + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -N:] = -float("inf")
    trellis[-N:, 0] = float("inf")
    for t in range(T):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class _Point:
    token_index: int
    time_index: int
    score: float


def _backtrack(trellis, emission, tokens, blank_id=0):
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        path.append(_Point(j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


@dataclass
class _Seg:
    label: str
    start: int
    end: int
    score: float


def _merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            _Seg(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def align_words(audio_16k: torch.Tensor, text: str, device="cuda") -> list[dict] | None:
    """Align text to 16kHz audio, return word-level timings.

    Returns list of {"word": str, "start": float, "end": float} or None on failure.
    """
    model, dictionary = _load(device)

    if audio_16k.ndim == 1:
        audio_16k = audio_16k.unsqueeze(0)

    duration = audio_16k.shape[-1] / SAMPLE_RATE

    # Clean text for alignment
    text_clean = "".join(c.lower() for c in text if c.lower() in dictionary or c == " ")
    text_clean = (
        text_clean.replace(" ", "|")
        if "|" in dictionary
        else text_clean.replace(" ", "")
    )
    if not text_clean:
        return None

    tokens = [dictionary[c] for c in text_clean]

    # Pad short audio
    wav = audio_16k
    if wav.shape[-1] < 400:
        wav = torch.nn.functional.pad(wav, (0, 400 - wav.shape[-1]))

    with torch.inference_mode():
        emissions, _ = model(wav.to(device).float())
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu()

    # Find blank id
    blank_id = 0
    for char, code in dictionary.items():
        if char in ("[pad]", "<pad>"):
            blank_id = code

    try:
        trellis = _get_trellis(emission, tokens, blank_id)
        path = _backtrack(trellis, emission, tokens, blank_id)
        if path is None:
            return None

        segments = _merge_repeats(path, text_clean)
        ratio = duration / emission.shape[0]

        words = text.split()
        word_segments = []
        seg_idx = 0

        for word in words:
            word_start, word_end = None, None
            chars_remaining = len([c for c in word.lower() if c in dictionary])
            while chars_remaining > 0 and seg_idx < len(segments):
                seg = segments[seg_idx]
                if seg.label != "|":
                    if word_start is None:
                        word_start = seg.start * ratio
                    word_end = seg.end * ratio
                    chars_remaining -= 1
                seg_idx += 1
            if word_start is not None and word_end is not None:
                word_segments.append(
                    {"word": word, "start": word_start, "end": word_end}
                )

        return word_segments
    except Exception:
        return None
