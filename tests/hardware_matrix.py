"""Render the same text under every dtype/attention combination and compare.

The dtype and attention decisions in `vui.hardware` / `vui.flash_compat` are
unit-tested against faked capabilities, but that only pins the *decision*. It
says nothing about whether the model is numerically stable in fp16, or whether
the SDPA fallback actually sounds like the flash kernel. Those need a GPU.

This renders one fixed line under each combination, then checks the audio three
ways: no NaNs and not silence, transcribes it back and scores WER against the
input, and correlates the waveform against the bf16+flash baseline. WER is the
one that matters — a render can be finite, non-silent, and still garbage.

Each combination runs in its own subprocess, because both decisions are
resolved once and cached (`lru_cache`, and flash_compat's one-way switch), so
they cannot be varied within a process.

Usage:
    python tests/hardware_matrix.py                     # all combinations
    python tests/hardware_matrix.py --dtype fp16        # just one
    python tests/hardware_matrix.py --keep              # leave the wavs

Note fp16 and SDPA can be forced on *any* CUDA card, so an Ampere box can
exercise the paths a Turing box would take — everything except the absence of
sm_75 kernels itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TEXT = (
    "So the thing about this is, it's not really what you'd expect at first. "
    "Give it a moment and it starts to make sense."
)

COMBINATIONS = [
    ("bf16", "flash"),
    ("bf16", "torch"),
    ("fp16", "flash"),
    ("fp16", "torch"),
]


# --------------------------------------------------------------- child side


def _render_one(dtype: str, attn: str, prompt: str, out_wav: str) -> dict:
    """Runs in the subprocess: render TEXT once and report metrics."""
    import torch
    import vui  # noqa: F401  — preloads ffmpeg before torchcodec
    from julius.resample import resample_frac
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder

    from vui import hardware
    from vui.engine import Engine, GenConfig, Segment
    from vui.inference import asr
    from vui.qwen_codec import SAMPLE_RATE as SR
    from vui.qwen_codec import QwenCodecEncoder

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    engine = Engine()

    wav_16k = (
        AudioDecoder(prompt, sample_rate=16000, num_channels=1)
        .get_all_samples()
        .data.squeeze(0)
    )
    wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SR)
    codec_enc = QwenCodecEncoder.from_pretrained().cuda().float().eval()
    with torch.inference_mode():
        codes = codec_enc.encode(wav_24k.float().cuda().unsqueeze(0))
    prompt_codes = codes[0, : engine.Q].T.long()
    prompt_text = asr(wav_16k)

    t0 = time.perf_counter()
    with engine.new_row() as row:
        row.prefill([Segment(prompt_text, prompt_codes)])
        _, audio = row.render(TEXT, GenConfig(temperature=0.7, max_secs=20))
    elapsed = time.perf_counter() - t0

    a = audio.squeeze().detach().cpu().float()
    secs = a.numel() / SR
    AudioEncoder(a.unsqueeze(0), sample_rate=SR).to_file(out_wav)

    from vui.flash_compat import _impl, _sdpa_attn_with_kvcache

    return {
        "dtype_requested": dtype,
        "attn_requested": attn,
        "dtype_resolved": str(hardware.dtype()).replace("torch.", ""),
        "attn_resolved": (
            "sdpa" if _impl is _sdpa_attn_with_kvcache else "flash-attn"
        ),
        "gpu": hardware.gpu_name(),
        "compute_capability": hardware.summary()["compute_capability"],
        "seconds_audio": round(secs, 3),
        "seconds_wall": round(elapsed, 3),
        # >1 is faster than realtime.
        "rtf": round(secs / elapsed, 2) if elapsed > 0 else None,
        "has_nan": bool(a.isnan().any()),
        "has_inf": bool(a.isinf().any()),
        "peak": round(float(a.abs().max()), 5),
        "rms": round(float(a.pow(2).mean().sqrt()), 5),
        "wav": out_wav,
    }


# -------------------------------------------------------------- parent side


def _run_child(dtype: str, attn: str, prompt: str, outdir: Path) -> dict:
    out_wav = str(outdir / f"{dtype}_{attn}.wav")
    env = {**os.environ, "VUI_DTYPE": dtype}
    # VUI_ATTN=torch forces SDPA; anything else leaves the normal dispatch,
    # which picks flash-attn when the card and wheel allow it.
    if attn == "torch":
        env["VUI_ATTN"] = "torch"
    else:
        env.pop("VUI_ATTN", None)

    code = (
        "import json,sys;"
        "sys.path.insert(0, 'tests');"
        "from hardware_matrix import _render_one;"
        f"print('@@' + json.dumps(_render_one({dtype!r},{attn!r},{prompt!r},{out_wav!r})))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    return {
        "dtype_requested": dtype,
        "attn_requested": attn,
        "error": (proc.stderr.strip().splitlines() or ["no output"])[-1],
        "wav": None,
    }


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate — Levenshtein over words, normalised by reference length."""
    import re

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

    r, h = norm(reference), norm(hypothesis)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / len(r)


def _transcribe(path: str) -> str | None:
    """Transcribe on CPU with Moonshine — never judge audio by its statistics."""
    try:
        import moonshine_voice
    except ImportError:
        return None
    try:
        model = _transcribe._model
    except AttributeError:
        model = _transcribe._model = moonshine_voice.load_model("moonshine/small")
    try:
        return moonshine_voice.transcribe(path, model=model)[0]
    except Exception as e:
        return f"<transcribe failed: {e}>"


def _correlate(a_path: str, b_path: str) -> float | None:
    """Pearson correlation of two renders, on the shorter common length."""
    from torchcodec.decoders import AudioDecoder

    try:
        a = AudioDecoder(a_path).get_all_samples().data.squeeze().float()
        b = AudioDecoder(b_path).get_all_samples().data.squeeze().float()
    except Exception:
        return None
    n = min(a.numel(), b.numel())
    if n == 0:
        return None
    a, b = a[:n], b[:n]
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return None if denom == 0 else round(float((a * b).sum() / denom), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="prompts/abraham.wav")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], help="only this dtype")
    ap.add_argument("--attn", choices=["flash", "torch"], help="only this attention")
    ap.add_argument("--out", default="/tmp/vui-hwmatrix")
    ap.add_argument("--keep", action="store_true", help="keep the rendered wavs")
    args = ap.parse_args()

    if not Path(args.prompt).exists():
        print(f"prompt not found: {args.prompt}", file=sys.stderr)
        return 2

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    combos = [
        (d, a)
        for d, a in COMBINATIONS
        if (args.dtype is None or d == args.dtype)
        and (args.attn is None or a == args.attn)
    ]

    print(f"prompt: {args.prompt}")
    print(f"text:   {TEXT!r}\n")

    results = []
    for dtype, attn in combos:
        print(f"  rendering {dtype:<5} / {attn:<5} ...", end="", flush=True)
        r = _run_child(dtype, attn, args.prompt, outdir)
        results.append(r)
        print(" error" if "error" in r else f" {r['seconds_audio']}s @ RTF {r['rtf']}")

    print("\ntranscribing (moonshine, CPU) ...")
    for r in results:
        if r.get("wav") and Path(r["wav"]).exists():
            text = _transcribe(r["wav"])
            r["transcript"] = text
            r["wer"] = None if text is None else round(_wer(TEXT, text), 3)

    baseline = next(
        (r for r in results if r.get("wav") and r["dtype_requested"] == "bf16"
         and r["attn_requested"] == "flash"),
        None,
    )
    if baseline and baseline.get("wav"):
        for r in results:
            if r.get("wav") and r is not baseline:
                r["corr_vs_baseline"] = _correlate(baseline["wav"], r["wav"])

    hdr = f"{'dtype':<6} {'attn':<6} {'resolved':<20} {'RTF':>6} {'WER':>6} {'corr':>7}  notes"
    print("\n" + hdr)
    print("-" * len(hdr))
    problems = []
    for r in results:
        if "error" in r:
            print(f"{r['dtype_requested']:<6} {r['attn_requested']:<6} {'FAILED':<20} "
                  f"{'':>6} {'':>6} {'':>7}  {r['error'][:60]}")
            problems.append(f"{r['dtype_requested']}/{r['attn_requested']}: {r['error'][:80]}")
            continue
        resolved = f"{r['dtype_resolved']}+{r['attn_resolved']}"
        notes = []
        if r["has_nan"] or r["has_inf"]:
            notes.append("NaN/Inf!")
            problems.append(f"{r['dtype_requested']}/{r['attn_requested']}: NaN or Inf in output")
        if r["rms"] < 1e-4:
            notes.append("near-silent!")
            problems.append(f"{r['dtype_requested']}/{r['attn_requested']}: near-silent output")
        if r.get("wer") is not None and r["wer"] > 0.25:
            notes.append("high WER")
            problems.append(f"{r['dtype_requested']}/{r['attn_requested']}: WER {r['wer']}")
        wer = "-" if r.get("wer") is None else f"{r['wer']:.3f}"
        corr = r.get("corr_vs_baseline")
        corr_s = "baseline" if r is baseline else ("-" if corr is None else f"{corr:.3f}")
        print(f"{r['dtype_requested']:<6} {r['attn_requested']:<6} {resolved:<20} "
              f"{r['rtf']:>6} {wer:>6} {corr_s:>7}  {' '.join(notes)}")

    for r in results:
        if r.get("transcript"):
            print(f"\n  {r['dtype_requested']}/{r['attn_requested']}: {r['transcript']!r}")

    if any(r.get("wer") is None for r in results if "error" not in r):
        print("\nNote: moonshine not installed — WER unavailable, which is the "
              "check that actually catches garbage audio. `uv sync --extra moonshine`.")

    print(f"\nwavs: {outdir}" if args.keep else "")
    if not args.keep:
        for r in results:
            if r.get("wav"):
                Path(r["wav"]).unlink(missing_ok=True)

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll combinations rendered cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
