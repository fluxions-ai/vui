"""Bake the release voice prompts against a checkpoint.

Writes `<out>/<voice>.safetensors` for each voice, holding:

  codes          (T, Q) int64   codec codes of the reference clip — checkpoint-agnostic
  cond_bias      (1,1,d) bf16   the SQ/WPS conditioning bias  — checkpoint-specific
  spk_token_emb  (1,1,d) bf16   the projected speaker token   — checkpoint-specific

plus a JSON `config` in the safetensors metadata: name, the exact transcript
(`text`), T, n_q, d_model, sample_rate and the checkpoint it was baked for.
No audio copy and no sidecar `.txt` — the transcript lives in the metadata.

The Python engine and the streaming server only need `codes` + the transcript,
so they work with any checkpoint's folder. The `cpu/` C engine
(`prepare_prompt_official.py`, `export_full.py`) and the MLX/iOS prebake read
the baked pair, so a new checkpoint needs a new prompt folder.

Usage (CUDA):

    uv run scripts/build_prompts.py --ckpt vui-nano-1.1 --out prompts/vui-nano-1.1
    uv run scripts/build_prompts.py --ckpt vui-nano-1.1 --src ./my_voices --voices alice,bob

Source audio is `<name>.wav`; the transcript comes from a sibling
`<name>.safetensors` (metadata `text`), else `<name>.txt`. With no `--src` the
release voices are pulled from the `prompts/` folder of the fluxions/vui Hub
repo. `--ckpt` takes an `Engine.NAMES` key, a Hub filename or a local path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file
import torch
from torchcodec.decoders import AudioDecoder

from vui.engine import Engine, Segment
from vui.hf import download
from vui.model import Vui
from vui.prompt_files import prompt_transcript
from vui.qwen_codec import SAMPLE_RATE as SR, QwenCodecDecoder, QwenCodecEncoder
from vui.qwen_spk_enc import QwenSpeakerEncoder

RELEASE_VOICES = ["maeve", "abraham", "rhian", "harry"]
# Speech-quality conditioning used for the shipped prompts. vui-nano takes 6
# channels (as does vui-nano-1.1); vui-190k takes 7 (the extra channel stays at 0).
SQ_6 = (0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
SQ_7 = (3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0)


def _load_24k(path: str) -> torch.Tensor:
    audio = AudioDecoder(path, sample_rate=SR, num_channels=1).get_all_samples().data
    audio = audio.squeeze(0).float()
    peak = audio.abs().max()
    return audio / peak if peak > 0 else audio


def _source(src: str | None, voice: str) -> tuple[str, str]:
    """(wav path, transcript) for a voice, from --src or the Hub prompts/ folder."""
    if src:
        wav = Path(src) / f"{voice}.wav"
    else:
        wav = Path(download(f"prompts/{voice}.wav"))
        # Sibling files live next to the wav in the Hub cache snapshot.
        for ext in ("safetensors", "txt"):
            try:
                download(f"prompts/{voice}.{ext}")
            except Exception:  # noqa: BLE001 — optional sidecars
                pass
    text = prompt_transcript(wav)
    if not text:
        raise SystemExit(
            f"{voice}: no transcript found beside {wav} (.safetensors metadata or .txt)"
        )
    return str(wav), text


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt", default="vui-nano-1.1", help="Engine.NAMES key, Hub filename or local path"
    )
    ap.add_argument("--out", required=True, help="output folder for <voice>.safetensors")
    ap.add_argument(
        "--src",
        default=None,
        help="folder with <voice>.wav (+ .safetensors/.txt transcript); default: Hub prompts/",
    )
    ap.add_argument(
        "--voices", default=",".join(RELEASE_VOICES), help="comma-separated voice names"
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "build_prompts.py needs a CUDA device (the codec encoder + engine run on GPU)"
        )
    dev = torch.device("cuda")
    ckpt_path = download(Engine.NAMES.get(args.ckpt, args.ckpt))
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_prompts] checkpoint: {ckpt_path}")
    model = Vui.from_pretrained_inf(ckpt_path).to(dev)
    codec_dec = QwenCodecDecoder.from_pretrained().to(dev).float().eval()
    codec_enc = QwenCodecEncoder.from_pretrained().to(dev).half().eval()
    spk_enc = QwenSpeakerEncoder.from_pretrained() if model.spk_proj is not None else None

    engine = Engine(model=model, codec=codec_dec, max_rows=1)
    n_sq = getattr(model.config.model, "sq_input_dim", 6)
    engine.set_conditioning(sq_scores=SQ_7 if n_sq == 7 else SQ_6, wps_score=0.0)
    n_q = model.config.model.n_quantizers
    print(
        f"[build_prompts] d_model={model.config.model.d_model} n_q={n_q} "
        f"sq_dim={n_sq} spk_proj={spk_enc is not None}"
    )

    for voice in voices:
        wav_path, text = _source(args.src, voice)
        audio_24k = _load_24k(wav_path)

        with torch.inference_mode():
            codes = codec_enc.encode(audio_24k.to(torch.float16).to(dev).reshape(1, 1, -1))
            codes = codes[0, :n_q].T.long().cpu()  # (T, Q)

            spk_emb = spk_enc.embed(audio_24k[: 30 * SR], sr=SR) if spk_enc is not None else None

            with engine.new_row() as row:
                row.prefill([Segment(text, codes)], spk_emb=spk_emb)
                cond_bias = engine.model._cond_bias.detach().cpu().clone()
                spk_token = (
                    row._spk_token.detach().cpu().clone() if row._spk_token is not None else None
                )
                T = int(row.offset)

        tensors = {"codes": codes.contiguous(), "cond_bias": cond_bias.contiguous()}
        if spk_token is not None:
            tensors["spk_token_emb"] = spk_token.contiguous()
        cfg = {
            "name": voice,
            "text": text,
            "T": T,
            "n_q": int(codes.shape[-1]),
            "d_model": int(model.config.model.d_model),
            "sample_rate": SR,
            "checkpoint": Path(ckpt_path).name,
        }
        out_path = out_dir / f"{voice}.safetensors"
        save_file(tensors, str(out_path), metadata={"config": json.dumps(cfg)})
        print(
            f"  {voice:8s} T={T:4d} codes={tuple(codes.shape)} "
            f"audio={audio_24k.numel() / SR:5.1f}s -> {out_path}"
        )


if __name__ == "__main__":
    main()
