"""Build a C-engine KV cache from an OFFICIAL vui prompt safetensors (preloaded voice).
Uses the shipped exact transcript + pre-encoded codes + spk_token_emb (no whisper, no re-encode).
Usage: prepare_prompt_official.py <prompt.safetensors> <prompt.txt> <out_cache.bin> <checkpoint>

This is how the shipped prompt_<voice>_official.bin caches were built. Inputs are the
official vui release prompt files (prompts/<voice>.safetensors holding `codes` +
`spk_token_emb`, prompts/<voice>.txt holding the exact transcript with disfluencies).
Prefer this over prepare_prompt.py for the release voices: no whisper transcription, no
audio re-encode, and it uses the official pre-projected speaker token.
"""

import struct
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from vui.model import Vui

st_path, txt_path, out_path, ckpt = sys.argv[1:5]
P = load_file(st_path)
prompt_text = open(txt_path).read().strip()
codes = P["codes"].long()  # (T, 16)
spk_token = P["spk_token_emb"].float()  # (1,1,768) already spk_proj'd
print(f"official prompt: text='{prompt_text[:60]}...' codes={tuple(codes.shape)}")

model = Vui.from_pretrained(ckpt).float().eval()
cfg = model.config.model
n_layers = cfg.n_layers
n_kv_heads = cfg.n_kv_heads or cfg.n_heads
head_dim = cfg.d_model // cfg.n_heads
max_seq = model.decoder.max_seqlen
Q = cfg.n_quantizers
model.decoder.allocate_inference_cache(1, "cpu", torch.float32)
pos = 0


def prefill(emb, n):
    global pos
    with torch.inference_mode():
        model.decoder.forward(emb, torch.arange(pos, pos + n))
    pos += n


# 1. official speaker token (already projected)
prefill(spk_token, 1)
# 2. exact prompt text + [SC]
tok = model.text_tokenizer
ids = torch.cat([tok.encode(prompt_text), torch.tensor([tok.special_to_id["[SC]"]])])
prefill(model.token_emb(ids).unsqueeze(0), len(ids))
print(f"text tokens: {len(ids)}")
# 3. official pre-encoded audio codes
for t in range(codes.shape[0]):
    prefill(model.audio_emb(codes[t, :Q].unsqueeze(0)).unsqueeze(0), 1)
print(f"KV cache position: {pos}")

kv_dim = n_kv_heads * head_dim
kc = np.zeros((n_layers, max_seq, kv_dim), np.float32)
vc = np.zeros((n_layers, max_seq, kv_dim), np.float32)
for l in range(n_layers):
    kv = model.decoder.blocks[l].attn.kv_cache
    k = kv.k_cache[0, :, :pos, :].detach().float().numpy()
    v = kv.v_cache[0, :, :pos, :].detach().float().numpy()
    kc[l, :pos] = k.transpose(1, 0, 2).reshape(pos, kv_dim)
    vc[l, :pos] = v.transpose(1, 0, 2).reshape(pos, kv_dim)
with open(out_path, "wb") as f:
    f.write(struct.pack("i", pos))
    f.write(kc.tobytes())
    f.write(vc.tobytes())
print(f"Saved {out_path} pos={pos}")
