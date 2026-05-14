"""Test that the code examples in docs/python-api.md actually run.

Exercises: minimal example, chunked prompt, streaming, continuous batching,
codes-only decode, cleanup. Substitutes prompts/rhian.wav (30.8s) for the
chunked-prompt path; falls back to abraham.wav for the short-prompt path.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from julius.resample import resample_frac

from vui.engine import Engine, GenConfig, RenderRequest, Segment
from vui.inference import asr
from vui.prompt_utils import build_prompt_segments
from vui.qwen_codec import SAMPLE_RATE as SR
from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder

CKPT = Path.home() / ".cache" / "vui" / "last_checkpoint"
OUT_DIR = Path("outputs/api_doc_test")
SHORT_WAV = Path("prompts/abraham.wav")
LONG_WAV = Path("prompts/rhian.wav")

OUT_DIR.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str]] = []


def check(name: str, fn):
    print(f"\n=== {name} ===")
    t0 = time.perf_counter()
    try:
        fn()
        dt = time.perf_counter() - t0
        print(f"PASS ({dt:.1f}s)")
        results.append((name, "PASS"))
    except Exception as e:
        traceback.print_exc()
        results.append((name, f"FAIL: {type(e).__name__}: {e}"))


# Load engine once via from_checkpoint (block 1: load)
ckpt_path = CKPT.read_text().strip()
print(f"Loading engine from {ckpt_path}...")
engine = Engine.from_checkpoint(ckpt_path)  # max_rows=1 (default) — required for .render()/.stream()
codec_enc = QwenCodecEncoder.from_pretrained().cuda().float().eval()


# --- Block: minimal example (short prompt, single Segment) ---
def test_minimal():
    wav_16k = AudioDecoder(str(SHORT_WAV), sample_rate=16000, num_channels=1) \
        .get_all_samples().data.squeeze(0)
    wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SR)
    with torch.inference_mode():
        codes = codec_enc.encode(wav_24k.float().cuda().unsqueeze(0))
    prompt_codes = codes[0, : engine.Q].T.long()
    prompt_text = asr(wav_16k)
    print(f"prompt_text: {prompt_text[:60]}, prompt_codes: {prompt_codes.shape}")

    with engine.new_row() as row:
        row.prefill([Segment(prompt_text, prompt_codes)])
        codes, audio = row.render(
            "So [breath] the thing about this is, it's not what you'd expect.",
            GenConfig(temperature=0.7, max_secs=10),
        )
    assert codes.dim() == 2 and codes.shape[1] == engine.Q, f"codes shape: {codes.shape}"
    assert audio.dim() == 3 and audio.shape[0] == 1, f"audio shape: {audio.shape}"

    AudioEncoder(audio.squeeze().cpu().float().unsqueeze(0), sample_rate=SR) \
        .to_file(str(OUT_DIR / "minimal.wav"))
    print(f"codes: {codes.shape}, audio: {audio.shape} -> minimal.wav")


# --- Block: chunked prompt via build_prompt_segments ---
def test_chunked_prompt():
    wav_16k = AudioDecoder(str(LONG_WAV), sample_rate=16000, num_channels=1) \
        .get_all_samples().data.squeeze(0)

    def _encode(audio_16k):
        audio_24k = resample_frac(audio_16k.unsqueeze(0), 16000, SR)
        with torch.inference_mode():
            codes = codec_enc.encode(audio_24k.float().cuda().unsqueeze(0))
        return codes[0, : engine.Q].T.long()

    segments = build_prompt_segments(
        wav_16k,
        encode_codes=_encode,
        transcribe=asr,
        align_device="cuda",
        target_seg=10.0,
    )
    print(f"{len(segments)} segments, "
          f"{sum(c.shape[0] for _, c in segments)} frames total")
    assert len(segments) > 1, f"expected multi-segment for {wav_16k.shape[-1]/16000:.1f}s audio"

    with engine.new_row() as row:
        row.prefill([Segment(t, c) for t, c in segments])
        codes, audio = row.render(
            "Right, so a quick test of the multi-segment prompt path.",
            GenConfig(temperature=0.7, max_secs=10),
        )
    AudioEncoder(audio.squeeze().cpu().float().unsqueeze(0), sample_rate=SR) \
        .to_file(str(OUT_DIR / "chunked.wav"))
    print(f"codes: {codes.shape}, audio: {audio.shape} -> chunked.wav")


# --- Block: streaming (verify the generator, skip sounddevice) ---
def test_streaming():
    # Load a short prompt
    wav_16k = AudioDecoder(str(SHORT_WAV), sample_rate=16000, num_channels=1) \
        .get_all_samples().data.squeeze(0)
    wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SR)
    with torch.inference_mode():
        codes = codec_enc.encode(wav_24k.float().cuda().unsqueeze(0))
    prompt_codes = codes[0, : engine.Q].T.long()
    prompt_text = asr(wav_16k)

    # Reuse the main engine (already max_rows=1)
    chunks = []
    with engine.new_row() as row:
        row.prefill([Segment(prompt_text, prompt_codes)])
        for audio_frame in row.stream(
            "Streaming reply.", GenConfig(temperature=0.7, max_secs=5)
        ):
            chunks.append(audio_frame.detach().cpu())
            if len(chunks) > 200:
                break
    assert chunks, "no frames yielded"
    print(f"streamed {len(chunks)} frames, each shape: {chunks[0].shape}")

    full = torch.cat(chunks, dim=-1).squeeze().float().unsqueeze(0)
    AudioEncoder(full, sample_rate=SR).to_file(str(OUT_DIR / "streamed.wav"))


# --- Block: continuous batching ---
def test_continuous_batching():
    wav_16k = AudioDecoder(str(SHORT_WAV), sample_rate=16000, num_channels=1) \
        .get_all_samples().data.squeeze(0)
    wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SR)
    with torch.inference_mode():
        codes = codec_enc.encode(wav_24k.float().cuda().unsqueeze(0))
    prompt_codes = codes[0, : engine.Q].T.long()
    prompt_text = asr(wav_16k)

    requests = [
        RenderRequest(
            segments=[Segment(prompt_text, prompt_codes)],
            text=f"Batch line number {i+1}.",
        )
        for i in range(4)
    ]

    # render_continuous wants max_rows>=2 worth of parallel slots; build a
    # second engine for this test since the main one is max_rows=1.
    batch_engine = Engine.from_checkpoint(ckpt_path, max_rows=2)
    results = batch_engine.render_continuous(requests, cfg=GenConfig(max_secs=8))
    print(f"got {len(results)} results, types: {set(type(r).__name__ for r in results)}")
    assert len(results) == len(requests)
    for i, r in enumerate(results):
        # The doc claims tuples (codes, audio) — verify reality
        if isinstance(r, tuple):
            codes, audio = r
        else:
            audio = r
        assert audio.dim() == 3, f"audio shape: {audio.shape}"
        AudioEncoder(audio.squeeze().cpu().float().unsqueeze(0), sample_rate=SR) \
            .to_file(str(OUT_DIR / f"batch_{i}.wav"))
    print(f"wrote {len(results)} batch wavs")


# --- Block: codes-only decode via internal API + manual decode ---
def test_codes_only_decode():
    wav_16k = AudioDecoder(str(SHORT_WAV), sample_rate=16000, num_channels=1) \
        .get_all_samples().data.squeeze(0)
    wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SR)
    with torch.inference_mode():
        codes = codec_enc.encode(wav_24k.float().cuda().unsqueeze(0))
    prompt_codes = codes[0, : engine.Q].T.long()
    prompt_text = asr(wav_16k)

    with engine.new_row() as row:
        row.prefill([Segment(prompt_text, prompt_codes)])
        gen_codes = engine._render_row(row, "Codes only test.", GenConfig(max_secs=5))
    assert gen_codes.dim() == 2, f"codes shape: {gen_codes.shape}"
    print(f"got codes: {gen_codes.shape}")

    codec_dec = QwenCodecDecoder.from_pretrained().cuda().float().eval()
    c = gen_codes.T.unsqueeze(0).cuda()
    with torch.inference_mode():
        audio = codec_dec.decode_chunked(c, ctx=6)
    assert audio.dim() == 3, f"audio shape: {audio.shape}"
    AudioEncoder(audio.squeeze().cpu().float().unsqueeze(0), sample_rate=SR) \
        .to_file(str(OUT_DIR / "codes_only.wav"))
    print(f"decoded audio: {audio.shape}")


# --- Block: doc no longer claims teardown() exists; verify our claim is consistent ---
def test_no_teardown_in_doc():
    has = hasattr(engine, "teardown")
    print(f"engine.teardown exists: {has}")
    doc = Path("docs/python-api.md").read_text()
    if "engine.teardown()" in doc:
        raise AssertionError("doc still mentions engine.teardown() but it does not exist")


# --- Block: render_all signature matches doc ---
def test_render_all_signature_matches_doc():
    import inspect
    sig = inspect.signature(engine.render_all)
    params = list(sig.parameters.keys())
    print(f"render_all params: {params}")
    doc = Path("docs/python-api.md").read_text()
    # Doc now says (rows, texts, cfg) — verify reality matches
    assert params == ["rows", "texts", "cfg"], f"unexpected signature: {params}"
    assert "render_all(rows, texts, cfg)" in doc, "doc should describe (rows, texts, cfg) signature"


check("minimal_example", test_minimal)
check("chunked_prompt", test_chunked_prompt)
check("streaming", test_streaming)
check("continuous_batching", test_continuous_batching)
check("codes_only_decode", test_codes_only_decode)
check("no_teardown_in_doc", test_no_teardown_in_doc)
check("render_all_signature_matches_doc", test_render_all_signature_matches_doc)


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, status in results:
    print(f"  {name}: {status}")
sys.exit(0 if all(s == "PASS" for _, s in results) else 1)
