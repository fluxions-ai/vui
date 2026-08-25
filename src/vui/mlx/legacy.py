"""MLX backend for the original-release (pre-1.0) 100M checkpoints.

Hybrid split: the 12-layer decoder (virtually all the compute in the
autoregressive loop) runs on MLX; embeddings, audio heads, the delayed
codebook pattern, sampling, and the Fluac codec stay in the proven torch
code from vui.legacy — those are per-step lookups and a 9x1008x768 matmul,
too small to matter. `MLXDecoderAdapter` plugs into vui.legacy.generate /
render via their `decoder=` override.

    from vui.mlx.legacy import load_legacy_mlx
    model, adapter = load_legacy_mlx("vui-cohost-100m.pt")
    audio = render(model, "Hello!", decoder=adapter)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from vui.mlx.tts.model import KVCache


def _rotate_half(x: mx.array) -> mx.array:
    """GPT-J interleaved: pairs (x0,x1) -> (-x1,x0)."""
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return mx.stack([-x2, x1], axis=-1).reshape(shape)


class _LegacyMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, cos, sin, cache):
        B, T, _ = x.shape
        qkv = self.Wqkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # cos/sin: (T, head_dim) for these positions, broadcast over B, heads
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        k, v = cache.update_and_fetch(k, v)

        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(k.shape[2])
            mask = mask[-T:]
        else:
            mask = None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class _LegacyMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class _LegacyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = _LegacyMHA(d_model, n_heads)
        self.mlp_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mlp = _LegacyMLP(d_model, hidden)

    def __call__(self, x, cos, sin, cache):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class LegacyDecoderMLX(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, hidden: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.blocks = [_LegacyBlock(d_model, n_heads, hidden) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.kv_caches: list[KVCache] = []
        self.max_seqlen = 0
        self._cos: mx.array | None = None
        self._sin: mx.array | None = None

    def make_cache(self):
        self.kv_caches = [
            KVCache(self.n_heads, self.head_dim, self.max_seqlen)
            for _ in self.blocks
        ]

    def __call__(self, x):
        offset = self.kv_caches[0].offset
        T = x.shape[1]
        cos = self._cos[offset : offset + T]
        sin = self._sin[offset : offset + T]
        for block, cache in zip(self.blocks, self.kv_caches):
            x = block(x, cos, sin, cache)
        return self.norm(x)


class MLXDecoderAdapter:
    """Drop-in for the legacy torch Decoder in vui.legacy.generate(decoder=...).

    Takes/returns torch tensors; positions are tracked by the KV cache offset,
    so `input_pos` is validated against it rather than used."""

    def __init__(self, torch_decoder):
        cfg_blocks = torch_decoder.blocks
        d_model = cfg_blocks[0].attn.dim
        n_heads = cfg_blocks[0].n_heads
        hidden = cfg_blocks[0].mlp.w1.out_features
        dec = LegacyDecoderMLX(len(cfg_blocks), d_model, n_heads, hidden)
        dec.max_seqlen = torch_decoder.max_seqlen

        # Identical RoPE values: reuse the torch-precomputed freqs buffer
        freqs = torch_decoder.freqs_cis.detach().float().cpu().numpy()
        dec._cos = mx.array(np.cos(freqs))
        dec._sin = mx.array(np.sin(freqs))

        sd = {k: v.detach().float().cpu().numpy()
              for k, v in torch_decoder.state_dict().items()}
        for i, block in enumerate(dec.blocks):
            p = f"blocks.{i}"
            block.attn_norm.weight = mx.array(sd[f"{p}.attn_norm.weight"])
            block.attn.Wqkv.weight = mx.array(sd[f"{p}.attn.Wqkv.weight"])
            block.attn.out_proj.weight = mx.array(sd[f"{p}.attn.out_proj.weight"])
            block.mlp_norm.weight = mx.array(sd[f"{p}.mlp_norm.weight"])
            block.mlp.w1.weight = mx.array(sd[f"{p}.mlp.w1.weight"])
            block.mlp.w3.weight = mx.array(sd[f"{p}.mlp.w3.weight"])
            block.mlp.w2.weight = mx.array(sd[f"{p}.mlp.w2.weight"])
        dec.norm.weight = mx.array(sd["norm.weight"])
        mx.eval(dec.parameters())
        self.decoder = dec

    def allocate_inference_cache(self, batch_size: int, device, dtype=None):
        assert batch_size == 1, "MLX legacy decoder is B=1"
        self.decoder.make_cache()

    def __call__(self, embeddings: torch.Tensor, input_pos: torch.Tensor):
        offset = self.decoder.kv_caches[0].offset
        assert int(input_pos[0]) == offset, (
            f"non-sequential input_pos {int(input_pos[0])} != cache offset {offset}"
        )
        x = mx.array(embeddings.detach().float().cpu().numpy())
        out = self.decoder(x)
        mx.eval(out)
        return torch.from_numpy(np.array(out))


def load_legacy_mlx(checkpoint_path: str = "vui-cohost-100m.pt"):
    """Returns (torch legacy Vui on CPU, MLXDecoderAdapter). Render with
    vui.legacy.render(model, text, decoder=adapter)."""
    from vui.legacy import Vui

    model = Vui.from_pretrained(checkpoint_path).eval()
    return model, MLXDecoderAdapter(model.decoder)
