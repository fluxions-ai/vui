"""MLX model definitions for Vui TTS."""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn


class KVCache:
    """Fixed-size pre-allocated KV cache. No dynamic resizing."""

    def __init__(self, n_kv_heads: int, head_dim: int, max_seqlen: int):
        self.keys = mx.zeros((1, n_kv_heads, max_seqlen, head_dim))
        self.values = mx.zeros((1, n_kv_heads, max_seqlen, head_dim))
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        return self.keys, self.values

    def reset(self):
        self.offset = 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RoPE:
    """Interleaved RoPE using mx.fast.rope (fused Metal kernel)."""

    def __init__(self, dim: int, max_seqlen: int = 8192, theta: float = 10000.0):
        self.dims = dim
        self.theta = theta

    def __call__(self, q, k, offset=0):
        # mx.fast.rope expects (B, n_heads, T, head_dim) with traditional=True for interleaved
        q = mx.fast.rope(
            q, self.dims, traditional=True, base=self.theta, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, self.dims, traditional=True, base=self.theta, scale=1.0, offset=offset
        )
        return q, k


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        qkv_dim = (n_heads + 2 * n_kv_heads) * self.head_dim
        self.Wqkv = nn.Linear(d_model, qkv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, rope, cache=None):
        B, T, _ = x.shape
        qkv = self.Wqkv(x)
        q, k, v = mx.split(
            qkv,
            [
                self.head_dim * self.n_heads,
                self.head_dim * self.n_heads + self.head_dim * self.n_kv_heads,
            ],
            axis=-1,
        )
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q, k = rope(q, k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q, k = rope(q, k)

        L, S = T, k.shape[2]
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(S)
            mask = mask[-L:]
        else:
            mask = None

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


@partial(mx.compile, shapeless=True)
def _swiglu(gate, x):
    return nn.silu(gate) * x


class LlamaMLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int | None = None):
        super().__init__()
        if intermediate_size is None:
            hidden = int(2 * 4 * d_model / 3)
            hidden = 256 * ((hidden + 255) // 256)
        else:
            hidden = intermediate_size
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.w2(_swiglu(self.w1(x), self.w3(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None, norm_eps=1e-5):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MHA(d_model, n_heads, n_kv_heads)
        self.mlp_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = LlamaMLP(d_model, intermediate_size)

    def __call__(self, x, rope, cache=None):
        x = x + self.attn(self.attn_norm(x), rope, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        max_seqlen=8192,
        rope_theta=10000.0,
        intermediate_size=None,
    ):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.max_seqlen = max_seqlen
        self.blocks = [
            Block(d_model, n_heads, n_kv_heads, intermediate_size)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(d_model, eps=1e-5)
        self.rope = RoPE(d_model // n_heads, max_seqlen, rope_theta)
        self.kv_caches: list[KVCache] = []

    def __call__(self, x, cache=None):
        if cache is None:
            cache = self.kv_caches
        for block, c in zip(self.blocks, cache):
            x = block(x, self.rope, c)
        return self.norm(x)

    def make_cache(self):
        self.kv_caches = [
            KVCache(self.n_kv_heads, self.head_dim, self.max_seqlen)
            for _ in self.blocks
        ]

    def reset_cache(self):
        for c in self.kv_caches:
            c.reset()
        self.kv_caches = []

    @property
    def cache_T(self) -> int:
        if self.kv_caches:
            return self.kv_caches[0].offset
        return 0


class AudioEmbedding(nn.Module):
    def __init__(self, n_quantizers, codebook_size, d_model):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(n_quantizers * codebook_size, d_model)

    def __call__(self, codes):
        Q = codes.shape[-1]
        offsets = mx.arange(Q) * self.codebook_size
        return self.embedding(codes + offsets).sum(axis=-2)


class ScalarCondProjector(nn.Module):
    def __init__(self, input_dim, d_model, n_freq=32):
        super().__init__()
        self.freqs = mx.exp(mx.linspace(0, math.log(1000), n_freq))
        self.proj0 = nn.Linear(input_dim * n_freq * 2, d_model, bias=False)
        self.proj2 = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x):
        if x.ndim == 1:
            x = x[None, :]
        xf = mx.expand_dims(x, -1) * self.freqs
        features = mx.concatenate([mx.sin(xf), mx.cos(xf)], axis=-1).reshape(
            x.shape[0], -1
        )
        return self.proj2(nn.silu(self.proj0(features)))


class RQBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.attn_norm = RMSNorm(d_model)
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = RMSNorm(d_model)
        hidden = 64 * ((int(d_model * 8 / 3) + 63) // 64)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x, cache=None):
        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.Wqkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            k_c, v_c = cache
            k = mx.concatenate([k_c, k], axis=2)
            v = mx.concatenate([v_c, v], axis=2)

        L, S = T, k.shape[2]
        if L > 1:
            mask = mx.triu(mx.full((S, S), -1e9), k=1)
            mask = mask[-L:]
        else:
            mask = None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.out_proj(out)
        h = self.mlp_norm(x)
        x = x + self.w2(nn.silu(self.w1(h)) * self.w3(h))
        return x, (k, v)

    def forward_kv(self, x, k_cache, v_cache, pos):
        """Single-token forward with pre-allocated KV cache. pos = current write position."""
        B = x.shape[0]
        h = self.attn_norm(x)
        qkv = self.Wqkv(h).reshape(B, 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # Write into pre-allocated cache
        k_cache[:, :, pos:pos+1, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, pos:pos+1, :] = v.transpose(0, 2, 1, 3)
        q = q.transpose(0, 2, 1, 3)
        k_all = k_cache[:, :, :pos+1, :]
        v_all = v_cache[:, :, :pos+1, :]
        out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        x = x + self.out_proj(out)
        h = self.mlp_norm(x)
        x = x + self.w2(nn.silu(self.w1(h)) * self.w3(h))
        return x


class RQTransformer(nn.Module):
    def __init__(
        self, backbone_dim, rq_dim, n_layers, n_heads, n_quantizers, codebook_size
    ):
        super().__init__()
        self.rq_dim = rq_dim
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.backbone_proj = (
            nn.Linear(backbone_dim, rq_dim, bias=False)
            if backbone_dim != rq_dim
            else None
        )
        self.code_emb = AudioEmbedding(n_quantizers - 1, codebook_size, rq_dim)
        self.pos_emb = nn.Embedding(n_quantizers, rq_dim)
        self.blocks = [RQBlock(rq_dim, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(rq_dim)
        self.head_W = mx.zeros((n_quantizers - 1, codebook_size, rq_dim))

    def _forward_blocks(self, x, caches):
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            x, cache = block(x, cache)
            new_caches.append(cache)
        return self.norm(x), new_caches

    def generate(
        self,
        backbone_hidden,
        code_0,
        temperature=0.7,
        top_k=None,
        logit_bias=None,
        max_q: int = 0,
    ):
        Q = min(max_q, self.n_quantizers) if max_q > 0 else self.n_quantizers
        n_blocks = len(self.blocks)
        proj = (
            self.backbone_proj(backbone_hidden)
            if self.backbone_proj
            else backbone_hidden
        )

        tok0 = (proj + self.pos_emb.weight[0])[:, None, :]
        tok1 = (self.code_emb.embedding(code_0) + self.pos_emb.weight[1])[:, None, :]
        prefill = mx.concatenate([tok0, tok1], axis=1)

        caches = [None] * n_blocks
        h, caches = self._forward_blocks(prefill, caches)

        logits_0 = h[:, 1] @ self.head_W[0].T
        if logit_bias is not None:
            logits_0 = logits_0 + logit_bias[0:1]
        logits_0 = logits_0 / temperature
        if top_k is not None:
            top_vals = mx.topk(logits_0, top_k, axis=-1)
            logits_0 = mx.where(logits_0 < top_vals[:, -1:], mx.array(-1e9), logits_0)
        next_code = mx.random.categorical(logits_0)
        codes = [code_0, next_code]

        for i in range(1, Q - 1):
            offset = i * self.codebook_size
            tok = (
                self.code_emb.embedding(next_code + offset) + self.pos_emb.weight[i + 1]
            )[:, None, :]
            h, caches = self._forward_blocks(tok, caches)
            logits_i = h[:, 0] @ self.head_W[i].T
            if logit_bias is not None:
                logits_i = logits_i + logit_bias[i : i + 1]
            logits_i = logits_i / temperature
            if top_k is not None:
                top_vals = mx.topk(logits_i, top_k, axis=-1)
                logits_i = mx.where(
                    logits_i < top_vals[:, -1:], mx.array(-1e9), logits_i
                )
            next_code = mx.random.categorical(logits_i)
            codes.append(next_code)

        return mx.stack(codes, axis=1)

    def generate_kv(
        self,
        backbone_hidden,
        code_0,
        temperature=0.7,
        top_k=None,
        logit_bias=None,
        max_q: int = 0,
    ):
        """Generate with pre-allocated KV caches (no concatenation)."""
        Q = min(max_q, self.n_quantizers) if max_q > 0 else self.n_quantizers
        n_blocks = len(self.blocks)
        n_h = self.blocks[0].n_heads
        h_d = self.blocks[0].head_dim
        D = self.rq_dim
        proj = (
            self.backbone_proj(backbone_hidden)
            if self.backbone_proj
            else backbone_hidden
        )

        # Pre-allocate KV caches for all blocks: (1, n_heads, Q, head_dim)
        k_caches = [mx.zeros((1, n_h, Q, h_d)) for _ in range(n_blocks)]
        v_caches = [mx.zeros((1, n_h, Q, h_d)) for _ in range(n_blocks)]

        # Prefill positions 0 (backbone) and 1 (code_0)
        tok0 = (proj + self.pos_emb.weight[0])[:, None, :]
        tok1 = (self.code_emb.embedding(code_0) + self.pos_emb.weight[1])[:, None, :]
        seq2 = mx.concatenate([tok0, tok1], axis=1)  # (1, 2, D)

        for li, block in enumerate(self.blocks):
            h = block.attn_norm(seq2)
            B, T2 = 1, 2
            qkv = block.Wqkv(h).reshape(B, T2, 3, n_h, h_d)
            q = qkv[:, :, 0].transpose(0, 2, 1, 3)
            k = qkv[:, :, 1].transpose(0, 2, 1, 3)
            v = qkv[:, :, 2].transpose(0, 2, 1, 3)
            k_caches[li][:, :, :2, :] = k
            v_caches[li][:, :, :2, :] = v
            mask = mx.triu(mx.full((2, 2), -1e9), k=1)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(B, T2, D)
            seq2 = seq2 + block.out_proj(out)
            h2 = block.mlp_norm(seq2)
            seq2 = seq2 + block.w2(nn.silu(block.w1(h2)) * block.w3(h2))

        # Predict code 1 from position 1 hidden
        h_out = self.norm(seq2[:, 1])
        logits = h_out @ self.head_W[0].T
        if logit_bias is not None:
            logits = logits + logit_bias[0:1]
        logits = logits / temperature
        if top_k is not None:
            top_vals = mx.topk(logits, top_k, axis=-1)
            logits = mx.where(logits < top_vals[:, -1:], mx.array(-1e9), logits)
        next_code = mx.random.categorical(logits)
        codes = [code_0, next_code]

        # Autoregressive single-token steps with pre-allocated KV
        for i in range(1, Q - 1):
            offset = i * self.codebook_size
            tok = (
                self.code_emb.embedding(next_code + offset) + self.pos_emb.weight[i + 1]
            )[:, None, :]  # (1, 1, D)
            for li, block in enumerate(self.blocks):
                tok = block.forward_kv(tok, k_caches[li], v_caches[li], i + 1)
            h_out = self.norm(tok[:, 0])
            logits = h_out @ self.head_W[i].T
            if logit_bias is not None:
                logits = logits + logit_bias[i : i + 1]
            logits = logits / temperature
            if top_k is not None:
                top_vals = mx.topk(logits, top_k, axis=-1)
                logits = mx.where(logits < top_vals[:, -1:], mx.array(-1e9), logits)
            next_code = mx.random.categorical(logits)
            codes.append(next_code)

        return mx.stack(codes, axis=1)

    def compile_forward(self):
        self._forward_blocks = mx.compile(self._forward_blocks)


def _compute_max_seq(data, cfg):
    max_secs = data.get("max_secs", 30)
    codec_hz = cfg.get("codec_hz", 12.5)
    audio_tokens = int(max_secs * codec_hz)
    text_tokens = int(audio_tokens * 2.5)
    total = audio_tokens + text_tokens
    return 64 * ((total + 63) // 64)


class VuiMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        data = config.get("data", {})
        n_kv_heads = cfg.get("n_kv_heads") or cfg["n_heads"]

        from vui.tokenizer import TokenizerConfig, VuiTokenizer

        tok_cfg = TokenizerConfig(**(data.get("tokenizer", {})))
        self.text_tokenizer = VuiTokenizer(tok_cfg)

        self.token_emb = nn.Embedding(self.text_tokenizer.vocab_size, cfg["d_model"])
        self.audio_emb = AudioEmbedding(
            cfg["n_quantizers"], cfg["codebook_size"], cfg["d_model"]
        )
        self.codec_head = nn.Linear(cfg["d_model"], cfg["codebook_size"], bias=False)
        self.eos_head = nn.Linear(cfg["d_model"], 1)

        self.rq_transformer = RQTransformer(
            cfg["d_model"],
            cfg.get("rq_d_model", cfg["d_model"]),
            cfg.get("rq_n_layers", 6),
            cfg.get("rq_n_heads", 8),
            cfg["n_quantizers"],
            cfg["codebook_size"],
        )

        max_seq = _compute_max_seq(data, cfg)
        self.decoder = Decoder(
            cfg["n_layers"],
            cfg["d_model"],
            cfg["n_heads"],
            n_kv_heads,
            max_seqlen=max_seq + cfg["n_quantizers"],
            rope_theta=cfg.get("rope_theta", 10000.0),
            intermediate_size=cfg.get("intermediate_size"),
        )

        self.sq_proj = (
            ScalarCondProjector(6, cfg["d_model"])
            if data.get("prob_sq", 0) > 0 and cfg.get("sinusoidal_cond", False)
            else None
        )
        self.wps_proj = (
            ScalarCondProjector(1, cfg["d_model"])
            if data.get("prob_wps", 0) > 0 and cfg.get("sinusoidal_cond", False)
            else None
        )
        self.spk_proj = (
            nn.Linear(cfg.get("spk_emb_dim", 0), cfg["d_model"], bias=False)
            if cfg.get("spk_emb_dim", 0) > 0
            else None
        )
        self.sc_id = self.text_tokenizer.special_to_id["[SC]"]
        self.d_model = cfg["d_model"]
