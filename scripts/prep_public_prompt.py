"""Convert a prompts/{name}.pt + .wav pair into a portable safetensors file.

Drops:
  - kv (checkpoint-specific KV cache slabs)
  - checkpoint (path identifier)

Keeps (tensors):
  - codes, cond_bias, spk_token_emb, audio (int16 PCM samples)
Keeps (metadata, JSON in `config`):
  - name, text, T, n_q, d_model, sample_rate

Usage:
  uv run scripts/prep_public_prompt.py prompts/harry
  uv run scripts/prep_public_prompt.py prompts/harry prompts/abraham prompts/irish2
"""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file


def convert(stem: Path) -> Path:
    pt_path = stem.with_suffix(".pt")
    wav_path = stem.with_suffix(".wav")
    out_path = stem.with_suffix(".safetensors")

    print(f"Loading {pt_path}...")
    saved = torch.load(pt_path, map_location="cpu", weights_only=False)

    samples, sr = sf.read(wav_path, dtype="int16", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype("int16")
    audio = torch.from_numpy(samples)

    tensors: dict[str, torch.Tensor] = {
        "codes": saved["codes"].contiguous(),
        "cond_bias": saved["cond_bias"].contiguous(),
        "audio": audio.contiguous(),
    }
    spk = saved.get("spk_token_emb")
    if spk is not None:
        tensors["spk_token_emb"] = spk.contiguous()

    cfg = {
        "name": saved.get("name", stem.name),
        "text": saved["text"],
        "T": int(saved["T"]),
        "n_q": int(saved.get("n_q") or saved["codes"].shape[-1]),
        "d_model": int(saved["d_model"]),
        "sample_rate": int(sr),
    }

    save_file(tensors, out_path, metadata={"config": json.dumps(cfg)})

    in_kb = (pt_path.stat().st_size + wav_path.stat().st_size) / 1e3
    out_kb = out_path.stat().st_size / 1e3
    print(f"  tensors: {sorted(tensors)}")
    print(f"  config:  {cfg}")
    print(f"  {in_kb:.0f}KB (.pt + .wav) -> {out_kb:.0f}KB  {out_path}")
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/prep_public_prompt.py <prompt_stem> [<stem> ...]")
        print("  e.g. python scripts/prep_public_prompt.py prompts/harry")
        sys.exit(1)
    for arg in sys.argv[1:]:
        convert(Path(arg).with_suffix(""))


if __name__ == "__main__":
    main()
