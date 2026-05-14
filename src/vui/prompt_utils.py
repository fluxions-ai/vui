"""Shared prompt preparation utilities for CUDA and MLX inference paths.

Builds multi-segment prompts from 16kHz audio:
  1. Encode full audio to codec codes once (caller provides encoder).
  2. Transcribe via ASR (caller provides transcriber).
  3. Wav2Vec2 forced alignment -> word timings.
  4. Trim trailing words past last sentence terminator.
  5. Split into ~target_seg segments at sentence boundaries.
  6. Slice pre-encoded codes tensor (no re-encoding).

Returns list of (text, codes) where codes is (T_seg, Q).
"""

import math
from typing import Callable

import torch

CODEC_HZ = 12.5


def _split_indices(word_timings: list[dict], target_seg: float) -> list[int]:
    out: list[int] = []
    seg_start = word_timings[0]["start"]
    for i, wt in enumerate(word_timings):
        seg_dur = wt["end"] - seg_start
        is_last = i == len(word_timings) - 1
        is_term = wt["word"].rstrip().endswith((".", "!", "?"))
        next_too_long = (
            not is_last and (word_timings[i + 1]["end"] - seg_start) > target_seg * 1.5
        )
        if is_last or (seg_dur >= target_seg and is_term) or next_too_long:
            out.append(i)
            if not is_last:
                seg_start = word_timings[i + 1]["start"]
    # Merge a too-short trailing segment back into the previous one
    if len(out) > 1:
        last_start = out[-2] + 1
        last_dur = word_timings[out[-1]]["end"] - word_timings[last_start]["start"]
        if last_dur < target_seg * 0.3:
            out.pop(-2)
    return out


def _trim_to_last_sentence(word_timings: list[dict]) -> list[dict]:
    last_term = None
    for i, wt in enumerate(word_timings):
        if wt["word"].rstrip().endswith((".", "!", "?")):
            last_term = i
    if last_term is not None and last_term < len(word_timings) - 1:
        return word_timings[: last_term + 1]
    return word_timings


def build_prompt_segments(
    audio_16k: torch.Tensor,
    encode_codes: Callable[[torch.Tensor], torch.Tensor],
    transcribe: Callable[[torch.Tensor], str],
    align_device: str = "cpu",
    target_seg: float = 10.0,
    return_timings: bool = False,
):
    """Build multi-segment prompt.

    encode_codes(audio_16k) -> (T, Q) codec codes.
    transcribe(audio_16k) -> full transcript text.

    return_timings=False (default): returns list[(text, codes)].
    return_timings=True: returns list[(text, codes, word_timings)] where
        word_timings is list[{"word", "start", "end", "fs"}] — absolute
        audio-time word boundaries plus `fs`, the segment's frame-start
        offset into `full_codes`. Callers can trim a segment to N words via
        `new_fe = round(words[N-1]["end"] * CODEC_HZ) + 2; new_T = new_fe - fs`.
    """
    from vui.align import align_words
    from vui.align import unload as unload_align

    total_secs = audio_16k.shape[-1] / 16000

    # Short audio: single segment, no alignment needed
    if total_secs <= target_seg * 1.5:
        text = transcribe(audio_16k)
        codes = encode_codes(audio_16k)
        print(f"  Prompt (single): '{text[:60]}' ({codes.shape[0]} frames)")
        if return_timings:
            return [(text, codes, [])]  # no per-word timings on the short path
        return [(text, codes)]

    # 1. Encode full audio -> codes ONCE
    full_codes = encode_codes(audio_16k)
    T_full = full_codes.shape[0]
    print(f"  Prompt: encoded {T_full} frames ({total_secs:.1f}s)")

    # 2. ASR transcript
    full_text = transcribe(audio_16k).strip()
    print(f"  ASR: '{full_text[:100]}...'")

    # 3. Wav2Vec2 forced alignment
    word_timings = align_words(audio_16k, full_text, device=align_device)
    unload_align()
    if not word_timings:
        print("  Alignment failed, using single segment")
        return [(full_text, full_codes)]

    # 4. Trim past last sentence terminator (clean cache boundary)
    before = len(word_timings)
    word_timings = _trim_to_last_sentence(word_timings)
    if len(word_timings) < before:
        print(
            f"  Trimmed {before - len(word_timings)} trailing words after last sentence end"
        )

    # 5. Segment at sentence boundaries
    splits = _split_indices(word_timings, target_seg)

    # 6. Slice pre-encoded codes by frame index.
    # Last segment ends on a completed sentence (thanks to _trim_to_last_sentence)
    # — round to the nearest frame and pad +2 frames of trailing silence/breath
    # so the prompt doesn't end mid-frame with a clipped consonant.
    segments: list[tuple[str, torch.Tensor]] = []
    seg_word_timings: list[list[dict]] = []
    prev_end = 0
    for i, split_i in enumerate(splits):
        is_last = i == len(splits) - 1
        seg_wts = word_timings[prev_end : split_i + 1]
        seg_text = " ".join(w["word"] for w in seg_wts)
        fs = max(0, int(seg_wts[0]["start"] * CODEC_HZ))
        if is_last:
            fe = min(T_full, round(seg_wts[-1]["end"] * CODEC_HZ) + 2)
        else:
            fe = min(T_full, math.ceil(seg_wts[-1]["end"] * CODEC_HZ))
        if fe > fs:
            seg_codes = full_codes[fs:fe]
            segments.append((seg_text, seg_codes))
            seg_word_timings.append([{**w, "fs": fs} for w in seg_wts])
            seg_dur = seg_wts[-1]["end"] - seg_wts[0]["start"]
            print(
                f"  Segment {len(segments)}: '{seg_text[:60]}' "
                f"({seg_codes.shape[0]} frames, {seg_dur:.1f}s)"
            )
        prev_end = split_i + 1

    if return_timings:
        return [(t, c, ws) for (t, c), ws in zip(segments, seg_word_timings)]
    return segments
