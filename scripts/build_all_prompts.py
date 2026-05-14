"""Build the 4 README voices (maeve, abraham, rhian, harry) against vui-nano.safetensors.

Replaces 4× separate model-loads with a single shared load. Reads each
transcript from prompts/<name>.txt; writes prompts/<name>.safetensors.
"""

from __future__ import annotations

import json
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
VOICES = ["maeve", "abraham", "rhian", "harry"]
CKPT = "vui-nano.safetensors"


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


device = torch.device("cuda")
print(f"Loading {CKPT}...")
model = Vui.from_pretrained_inf(CKPT).to(device)
codec_dec = QwenCodecDecoder.from_pretrained().to(device).float().eval()
codec_enc = QwenCodecEncoder.from_pretrained().to(device).half().eval()
spk_enc = QwenSpeakerEncoder.from_pretrained() if model.spk_proj is not None else None

engine = Engine(model, codec_dec, max_rows=1)
engine.set_conditioning(sq_scores=DEFAULT_SQ, wps_score=0.0)
n_q = model.config.model.n_quantizers

for voice in VOICES:
    wav_path = f"prompts/{voice}.wav"
    txt_path = f"prompts/{voice}.txt"
    out_path = f"prompts/{voice}.safetensors"
    text = Path(txt_path).read_text().strip()

    audio_24k = _load_audio_24k(wav_path)
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
        row.close()

    audio_int16 = (audio_24k * 32767).clamp(-32768, 32767).to(torch.int16).contiguous()
    tensors: dict[str, torch.Tensor] = {
        "codes": codes.contiguous(),
        "cond_bias": cond_bias.contiguous(),
        "audio": audio_int16,
    }
    if spk_token is not None:
        tensors["spk_token_emb"] = spk_token.contiguous()
    cfg = {
        "name": voice,
        "text": text,
        "T": int(T),
        "n_q": int(codes.shape[-1]),
        "d_model": int(model.config.model.d_model),
        "sample_rate": QWEN_SR,
    }
    save_file(tensors, out_path, metadata={"config": json.dumps(cfg)})
    print(f"  {voice}: T={T} codes={tuple(codes.shape)} "
          f"audio={audio_int16.shape[0]/QWEN_SR:.1f}s -> {out_path}")

print("Done.")
