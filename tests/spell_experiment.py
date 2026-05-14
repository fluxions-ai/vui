"""A/B test the `|spell|` byte-level escape against the bare BPE form.

Renders matched pairs to outputs/spell_ab/ so you can listen to whether
wrapping hard words in pipes actually improves pronunciation.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from vui.engine import Engine, GenConfig, Segment
from vui.hf import download
from vui.model import Vui
from vui.qwen_codec import SAMPLE_RATE as QWEN_SR
from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder

CKPT = Path.home() / ".cache" / "vui" / "last_checkpoint"
PROMPT_WAV = Path("prompts/abraham.wav")
PROMPT_TXT = Path("prompts/abraham.txt")
OUT_DIR = Path("outputs/spell_ab")

PAIRS: list[tuple[str, str, str]] = [
    ("saoirse", "Have you seen the new Saoirse Ronan film?",
                "Have you seen the new |Saoirse| Ronan film?"),
    ("kubectl", "Just run kubectl apply on the manifest.",
                "Just run |kubectl| apply on the manifest."),
    ("nginx",   "I'm restarting nginx now.",
                "I'm restarting |nginx| now."),
    ("xochitl", "Xochitl asked about the deploy schedule.",
                "|Xochitl| asked about the deploy schedule."),
    ("siobhan", "Siobhan said she'd be in by ten.",
                "|Siobhan| said she'd be in by ten."),
    ("psql",    "Open psql and run the migration.",
                "Open |psql| and run the migration."),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    ckpt_path = download(CKPT.read_text().strip())
    print(f"Loading {ckpt_path}...")
    model = Vui.from_pretrained_inf(ckpt_path).cuda()

    print("Loading codec...")
    codec_enc = QwenCodecEncoder.from_pretrained().cuda().half().eval()
    codec_dec = QwenCodecDecoder.from_pretrained().cuda().float().eval()
    Q = model.config.model.n_quantizers

    print(f"Encoding prompt {PROMPT_WAV}...")
    wav = AudioDecoder(str(PROMPT_WAV), sample_rate=16000, num_channels=1).get_all_samples()
    audio_16k = wav.data.squeeze(0)
    from julius.resample import resample_frac
    audio_24k = resample_frac(audio_16k.unsqueeze(0), 16000, QWEN_SR)
    with torch.inference_mode():
        codes = codec_enc.encode(audio_24k.half().cuda().unsqueeze(0))
        prompt_codes = codes[0, :Q].T.long()
    prompt_text = PROMPT_TXT.read_text().strip()
    print(f"Prompt: '{prompt_text[:60]}' ({prompt_codes.shape[0]} frames)")

    print("Building engine...")
    engine = Engine(model, codec_dec, max_rows=1, vocoder_ctx=25)
    with engine.new_row() as row:
        row.render("Warmup.", GenConfig(max_secs=2, temperature=0.7))

    cfg = GenConfig(
        temperature=0.7,
        max_secs=8.0,
        eos_threshold=0.45,
        n_codebooks=Q,
        sentence_only=True,
    )

    for name, bare, piped in PAIRS:
        for tag, text in (("a_bare", bare), ("b_piped", piped)):
            with engine.new_row() as row:
                row.prefill([Segment(prompt_text, prompt_codes)], spk_emb=None)
                t0 = time.perf_counter()
                gen_codes, _ = row.render(text, cfg)
                dt = time.perf_counter() - t0

            if gen_codes is None or gen_codes.shape[0] == 0:
                print(f"  [skip] {name} {tag}: no audio")
                continue

            c = gen_codes.T.unsqueeze(0).cuda()
            ctx = prompt_codes.T.unsqueeze(0).cuda()[:, :, -6:]
            full = torch.cat([ctx, c], dim=2)
            with torch.inference_mode():
                audio = codec_dec.decode_chunked(full, ctx=6)
            audio = audio[..., 6 * 1920:]
            wav_t = audio[0, 0].detach().float().cpu()
            out = OUT_DIR / f"{name}_{tag}.wav"
            AudioEncoder(wav_t.unsqueeze(0), sample_rate=int(QWEN_SR)).to_file(str(out))
            dur = wav_t.shape[-1] / QWEN_SR
            print(f"  [{name} {tag}] {dur:.1f}s in {dt:.2f}s -> {out}  ::  {text}")

    print(f"\nDone. Compare pairs in {OUT_DIR}/<name>_a_bare.wav vs <name>_b_piped.wav")


if __name__ == "__main__":
    main()
