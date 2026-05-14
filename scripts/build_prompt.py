"""Build a portable prompt safetensors from a fresh checkpoint + audio + transcript.

Drops the KV slabs and checkpoint identifier so the output loads against any
compatible checkpoint via the server's reprefill path. The saved cond_bias
and spk_token_emb are computed against the given checkpoint.

Usage:
  uv run scripts/build_prompt.py <ckpt> <audio.wav> <transcript> [<out.safetensors>]

Example:
  uv run scripts/build_prompt.py vui-nano.safetensors prompts/harry.wav \\
      "Hello, this is Harry..." prompts/harry.safetensors
"""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file

from vui.engine import Engine, Segment
from vui.model import Vui
from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder
from vui.qwen_spk_enc import QwenSpeakerEncoder

QWEN_SR = 24000
DEFAULT_SQ = (0.0, 0.0, 0.0, 0.0, 0.0, 5.0)


def _load_audio_24k(wav_path: str) -> torch.Tensor:
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    audio = torch.from_numpy(samples).float()
    if sr != QWEN_SR:
        from julius.resample import resample_frac

        audio = resample_frac(audio.unsqueeze(0), sr, QWEN_SR).squeeze(0)
    if audio.abs().max() > 0:
        audio = audio / audio.abs().max()
    return audio


def build_prompt(
    ckpt_path: str,
    wav_path: str,
    text: str,
    out_path: str,
    name: str | None = None,
    sq_scores: tuple[float, ...] = DEFAULT_SQ,
    wps_score: float = 0.0,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_24k = _load_audio_24k(wav_path)

    print(f"Loading {ckpt_path}...")
    model = Vui.from_pretrained_inf(ckpt_path).to(device)
    codec_dec = QwenCodecDecoder.from_pretrained().to(device).float().eval()
    codec_enc = QwenCodecEncoder.from_pretrained().to(device).half().eval()
    spk_enc = (
        QwenSpeakerEncoder.from_pretrained() if model.spk_proj is not None else None
    )

    engine = Engine(model, codec_dec, max_rows=1)
    engine.set_conditioning(sq_scores=sq_scores, wps_score=wps_score, pq_score=0.0)

    n_q = model.config.model.n_quantizers
    with torch.inference_mode():
        audio_in = audio_24k.to(torch.float16).to(device).reshape(1, 1, -1)
        codes = codec_enc.encode(audio_in)
        codes = codes[0, :n_q].T.long().cpu()

        spk_emb = None
        if spk_enc is not None:
            spk_emb = spk_enc.embed(audio_24k[: 30 * QWEN_SR], sr=QWEN_SR)

        row = engine.new_row()
        row.prefill([Segment(text=text, codes=codes)], spk_emb=spk_emb)

        cond_bias = engine.model._cond_bias.detach().cpu().clone()
        spk_token = (
            row._spk_token.detach().cpu().clone()
            if row._spk_token is not None
            else None
        )
        T = row.offset

    audio_int16 = (audio_24k * 32767).clamp(-32768, 32767).to(torch.int16).contiguous()

    tensors: dict[str, torch.Tensor] = {
        "codes": codes.contiguous(),
        "cond_bias": cond_bias.contiguous(),
        "audio": audio_int16,
    }
    if spk_token is not None:
        tensors["spk_token_emb"] = spk_token.contiguous()

    cfg = {
        "name": name or Path(wav_path).stem,
        "text": text,
        "T": int(T),
        "n_q": int(codes.shape[-1]),
        "d_model": int(model.config.model.d_model),
        "sample_rate": QWEN_SR,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out, metadata={"config": json.dumps(cfg)})
    out_kb = out.stat().st_size / 1e3
    print(
        f"Wrote {out}  T={T} codes={tuple(codes.shape)} "
        f"audio={audio_int16.shape[0] / QWEN_SR:.1f}s  ({out_kb:.0f}KB)"
    )
    return out


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python scripts/build_prompt.py "
            "<ckpt> <audio.wav> <transcript> [<out.safetensors>]"
        )
        sys.exit(1)
    ckpt, wav, text = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else f"prompts/{Path(wav).stem}.safetensors"
    build_prompt(ckpt, wav, text, out)


if __name__ == "__main__":
    main()
