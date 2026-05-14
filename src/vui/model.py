import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from vui.config import Config, VuiConfig
from vui.rope import apply_rotary_emb, precompute_freqs_cis
from vui.tokenizer import VuiTokenizer


def load_what_you_can(checkpoint: dict, model: nn.Module):
    """Load as many weights from `checkpoint` as possible.

    Same-shape params copy directly; mismatched-rank params are skipped;
    same-rank but mismatched-shape params copy the overlapping prefix
    (useful when vocab / embed dims have grown across runs).
    """
    model_state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name not in model_state_dict:
            print(f"Ignoring parameter '{name}' because it is not found in the model")
            continue
        model_state = model_state_dict[name]
        mshape = model_state.shape
        pshape = param.shape
        if pshape == mshape:
            model_state.copy_(param)
            continue
        if len(pshape) != len(mshape):
            continue
        min_shape = [min(pshape[i], mshape[i]) for i in range(len(pshape))]
        print(name, "model:", mshape, "chkpt:", pshape, "loading:", min_shape)
        slices = tuple(slice(0, s) for s in min_shape)
        model_state[slices].copy_(param[slices])
    return model.load_state_dict(model_state_dict)


class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_seqlen: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        cache_shape = (batch_size, n_kv_heads, max_seqlen, head_dim)

        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: (T,), k_val: (B, nh, T, d)
        torch._assert(input_pos.size(0) == k_val.size(-2), "pos/kv size mismatch")
        dtype = self.k_cache.dtype

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(dtype)
        v_out[:, :, input_pos] = v_val.to(dtype)

        return k_out, v_out


class FlashKVCache(nn.Module):
    """KV cache for flash_attn_with_kvcache. Shape: [BS, max_seq, n_heads, head_dim]."""

    def __init__(
        self,
        batch_size: int,
        max_seqlen: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
    ):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros(
                batch_size, max_seqlen, n_kv_heads, head_dim, dtype=dtype, device=device
            ),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                batch_size, max_seqlen, n_kv_heads, head_dim, dtype=dtype, device=device
            ),
        )
        self.register_buffer(
            "seq_lens",
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )


def repeat_kv(x: torch.Tensor, n_reps: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, T, head_dim = x.shape

    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_reps, T, head_dim)
        .reshape(bs, n_kv_heads * n_reps, T, head_dim)
    )


class MHA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        block_idx: int,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        use_rotary_emb: bool = True,
        window_size: int | None = None,
    ):
        super().__init__()

        head_dim = dim // n_heads

        self.use_rotary_emb = use_rotary_emb
        self.block_idx = block_idx
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size
        self.n_reps = n_heads // n_kv_heads
        qkv_dim = (n_heads + 2 * n_kv_heads) * head_dim
        self.Wqkv = nn.Linear(dim, qkv_dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.kv_cache = None

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor | None = None,
        input_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        B, T, d = x.size()
        dtype = x.dtype

        dropout_p = self.dropout if self.training else 0.0

        qkv = self.Wqkv(x)
        if self.n_heads == self.n_kv_heads:
            qkv = rearrange(
                qkv, "B T (three h d) -> B three h T d", three=3, h=self.n_heads
            )
            q, k, v = qkv.unbind(dim=1)  # (B, h, T, d)
        else:
            q, k, v = torch.split(
                qkv,
                [
                    self.head_dim * self.n_heads,
                    self.head_dim * self.n_kv_heads,
                    self.head_dim * self.n_kv_heads,
                ],
                dim=-1,
            )
            q = rearrange(q, "B T (h d) -> B h T d", h=self.n_heads)
            k = rearrange(k, "B T (h d) -> B h T d", h=self.n_kv_heads)
            v = rearrange(v, "B T (h d) -> B h T d", h=self.n_kv_heads)

        if self.use_rotary_emb:
            q = apply_rotary_emb(freqs_cis, q)
            k = apply_rotary_emb(freqs_cis, k)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if self.n_reps > 1:
            k = repeat_kv(k, self.n_reps)
            v = repeat_kv(v, self.n_reps)

        is_causal = self.causal and self.kv_cache is None

        out = F.scaled_dot_product_attention(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )

        out = self.out_proj(rearrange(out, "B h T d -> B T (h d)"))

        return out

    def forward_flash(
        self,
        x: Tensor,
        kv_cache: FlashKVCache,
        freqs_cis: Tensor,
        per_sample_freqs: bool = False,
        cache_batch_idx: Tensor | None = None,
    ) -> Tensor:
        """Forward using flash_attn_with_kvcache. Fuses cache update + attention.
        x: (B, T, d). Updates kv_cache.seq_lens by T after call.
        If per_sample_freqs=True, freqs_cis is (B, rot_dim, 2) for T=1 decode.
        If cache_batch_idx is provided, maps B query rows to arbitrary KV cache slots.
        """
        from flash_attn import flash_attn_with_kvcache

        B, T, _ = x.shape
        qkv = self.Wqkv(x)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.n_heads,
                self.head_dim * self.n_kv_heads,
                self.head_dim * self.n_kv_heads,
            ],
            dim=-1,
        )
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        if self.use_rotary_emb:
            if per_sample_freqs and T == 1:
                q = apply_rotary_emb(freqs_cis, q.squeeze(1)).unsqueeze(1)
                k = apply_rotary_emb(freqs_cis, k.squeeze(1)).unsqueeze(1)
            else:
                q = apply_rotary_emb(freqs_cis, q.transpose(1, 2)).transpose(1, 2)
                k = apply_rotary_emb(freqs_cis, k.transpose(1, 2)).transpose(1, 2)

        if cache_batch_idx is not None:
            cache_seqlens = kv_cache.seq_lens[cache_batch_idx.long()]
        else:
            cache_seqlens = kv_cache.seq_lens[:B]

        out = flash_attn_with_kvcache(
            q,
            kv_cache.k_cache,
            kv_cache.v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            causal=self.causal,
            window_size=((self.window_size, 0) if self.window_size else (-1, -1)),
        )
        return self.out_proj(out.reshape(B, T, self.dim))


class MLP(nn.Module):
    def __init__(
        self, *, d_model: int, bias: bool, dropout: float, act=nn.GELU, **kwargs
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.act = act()
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class LlamaMLP(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        intermediate_size: int | None = None,
        multiple_of: int = 256,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if intermediate_size is not None:
            hidden_dim = intermediate_size
        else:
            hidden_dim = 4 * d_model
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Block(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        block_idx: int,
        bias: bool,
        dropout: float,
        norm_eps: float = 1e-5,  # use 1e-6 for rms
        use_rotary_emb: bool = True,
        intermediate_size: int | None = None,
        window_size: int | None = None,
    ):
        super().__init__()

        self.block_idx = block_idx
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MHA(
            d_model,
            n_heads,
            n_kv_heads,
            block_idx=block_idx,
            bias=bias,
            dropout=dropout,
            causal=True,
            use_rotary_emb=use_rotary_emb,
            window_size=window_size,
        )
        self.mlp_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = LlamaMLP(
            d_model=d_model,
            intermediate_size=intermediate_size,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor | None = None,
        input_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        x = x + self.attn(
            self.attn_norm(x),
            freqs_cis=freqs_cis,
            input_pos=input_pos,
            attn_mask=attn_mask,
        )
        x = x + self.mlp(self.mlp_norm(x))

        return x

    def forward_flash(
        self,
        x: Tensor,
        kv_cache: FlashKVCache,
        freqs_cis: Tensor,
        per_sample_freqs: bool = False,
        cache_batch_idx: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn.forward_flash(
            self.attn_norm(x),
            kv_cache,
            freqs_cis,
            per_sample_freqs=per_sample_freqs,
            cache_batch_idx=cache_batch_idx,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        bias: bool,
        dropout: float,
        max_seqlen: int = 4096,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
        norm_eps: float = 1e-5,
        use_rotary_emb: bool = True,
        rope_dim: int | None = None,
        intermediate_size: int | None = None,
        window_size: int | None = None,
        global_every: int | None = None,
        global_rope_dim: int | None = None,
        global_rope_theta: float | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.use_rotary_emb = use_rotary_emb
        self.max_seqlen = max_seqlen
        self.window_size = window_size

        def _is_global(idx: int) -> bool:
            if window_size is None:
                return True
            if global_every is not None:
                if idx == n_layers - 1 or (idx + 1) % global_every == 0:
                    return True
            return False

        def _layer_window(idx: int) -> int | None:
            return None if _is_global(idx) else window_size

        self.block_is_global: list[bool] = [_is_global(i) for i in range(n_layers)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    block_idx=block_idx,
                    bias=bias,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    use_rotary_emb=use_rotary_emb,
                    window_size=_layer_window(block_idx),
                    intermediate_size=intermediate_size,
                )
                for block_idx in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.attn_mask = None
        head_dim = d_model // n_heads
        rope_dim = rope_dim or head_dim
        assert rope_dim <= head_dim
        freqs_cis = precompute_freqs_cis(
            rope_dim,
            max_seqlen,
            theta=rope_theta,
            theta_rescale_factor=rope_theta_rescale_factor,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Separate freqs for global attention layers (p-RoPE).
        # When None, global layers reuse self.freqs_cis (backward compatible).
        # Auto-derive global theta: scale by max_seq/window ratio.
        _g_dim = global_rope_dim or rope_dim
        if global_rope_theta is not None:
            _g_theta = global_rope_theta
        elif window_size is not None and any(self.block_is_global):
            _g_theta = rope_theta * (max_seqlen / window_size)
        else:
            _g_theta = rope_theta
        if _g_dim != rope_dim or _g_theta != rope_theta:
            freqs_cis_global = precompute_freqs_cis(
                _g_dim,
                max_seqlen,
                theta=_g_theta,
                theta_rescale_factor=rope_theta_rescale_factor,
            )
            self.register_buffer("freqs_cis_global", freqs_cis_global, persistent=False)
        else:
            self.freqs_cis_global = None
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.flash_kv_caches: list[FlashKVCache] | None = None

    def allocate_inference_cache(
        self, batch_size: int, device: str, dtype=torch.bfloat16
    ):
        for block in self.blocks:
            block.attn.kv_cache = KVCache(
                batch_size, self.max_seqlen, block.n_kv_heads, block.head_dim, dtype
            ).to(device)
        self.attn_mask = torch.tril(
            torch.ones(
                self.max_seqlen, self.max_seqlen, dtype=torch.bool, device=device
            )
        )

    def allocate_flash_kv_cache(
        self,
        batch_size: int,
        device,
        dtype=torch.bfloat16,
        max_seqlen: int | None = None,
    ):
        seqlen = max_seqlen or self.max_seqlen
        self.flash_kv_caches = [
            FlashKVCache(
                batch_size,
                seqlen,
                self.n_kv_heads,
                self.head_dim,
                dtype,
                device,
            )
            for _ in self.blocks
        ]
        # Share a single seq_lens tensor across all layers
        shared_seq_lens = self.flash_kv_caches[0].seq_lens
        for kv in self.flash_kv_caches[1:]:
            kv.seq_lens = shared_seq_lens

    def deallocate_flash_kv_cache(self):
        self.flash_kv_caches = None

    def reset_kv_cache(self):
        for block in self.blocks:
            if block.attn.kv_cache is not None:
                block.attn.kv_cache.k_cache.zero_()
                block.attn.kv_cache.v_cache.zero_()

    def deallocate_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = None
        self.attn_mask = None
        if self.flash_kv_caches is not None:
            self.flash_kv_caches = None

    def _get_freqs(self, positions: Tensor, block_idx: int) -> Tensor | None:
        if not self.use_rotary_emb:
            return None
        if self.freqs_cis_global is not None and self.block_is_global[block_idx]:
            return self.freqs_cis_global[positions]
        return self.freqs_cis[positions]

    def forward(self, x: Tensor, input_pos: Tensor, padding_mask: Tensor | None = None):
        B = x.shape[0]

        attn_mask = (
            self.attn_mask[None, None, input_pos].to(torch.bool)
            if self.attn_mask is not None
            else None
        )
        if padding_mask is not None:
            attn_mask = attn_mask.repeat(B, 1, 1, 1)
            attn_mask = (
                attn_mask[:, :, : padding_mask.shape[1]]
                & padding_mask[:, None, :, None].bool()
            )

        for i, block in enumerate(self.blocks):
            freqs_cis = self._get_freqs(input_pos, i)
            x = block(x, freqs_cis=freqs_cis, input_pos=input_pos, attn_mask=attn_mask)

        x = self.norm(x)

        return x

    def forward_flash(
        self,
        x: Tensor,
        positions: Tensor,
        per_sample_positions: bool = False,
    ) -> Tensor:
        """Forward using flash_attn_with_kvcache. x: (B, T, d).
        positions: (T,) shared, or (B,) per-sample when per_sample_positions=True.
        Automatically updates shared seq_lens after all layers."""
        B, T, _ = x.shape
        per_sample = per_sample_positions and T == 1
        for i, (block, kv_cache) in enumerate(zip(self.blocks, self.flash_kv_caches)):
            freqs_cis = self._get_freqs(positions, i)
            x = block.forward_flash(x, kv_cache, freqs_cis, per_sample_freqs=per_sample)
        self.flash_kv_caches[0].seq_lens[:B] += T
        return self.norm(x)


class MetricProjector(nn.Module):
    def __init__(self, n_metrics=3, d_model=1024):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_metrics)]
        )
        self.dropout = nn.Dropout(0.1)

        # Better initialization for [0,1] inputs
        for proj in self.projections:
            nn.init.normal_(proj.weight, mean=0.0, std=0.01)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, metrics: Tensor):
        projections = []
        for i in range(len(self.projections)):
            proj = self.projections[i](metrics[:, i : i + 1])
            proj = self.dropout(proj)
            projections.append(proj)

        return torch.stack(projections, dim=1)


class AudioEmbedding(nn.Module):
    def __init__(self, n_quantizers: int, codebook_size: int, d_model: int):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(n_quantizers * codebook_size, d_model)

    def forward(self, codes: Tensor) -> Tensor:
        # codes: (*, Q) long -> (*, d_model), Q may be <= n_quantizers
        Q = codes.shape[-1]
        offsets = torch.arange(Q, device=codes.device) * self.codebook_size
        return self.embedding(codes + offsets).sum(dim=-2)


class AudioHead(nn.Module):
    def __init__(self, n_quantizers: int, codebook_size: int, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_quantizers, codebook_size, d_model))
        nn.init.normal_(self.W, std=0.02)

    def forward(self, hidden: Tensor) -> Tensor:
        # hidden: (N, d_model) -> (N, Q, n_codes)
        return torch.einsum("nd,qcd->nqc", hidden, self.W)


class RQBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.attn_norm = RMSNorm(d_model)
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = RMSNorm(d_model)
        hidden = int(d_model * 8 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        # Causal mask for short-sequence math attention (avoids flash kernel overhead)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq, max_seq), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.Wqkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.causal_mask[:T, :T]
        out = F.softmax(attn, dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        x = x + self.out_proj(out)
        h = self.mlp_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x

    def forward_kv(
        self,
        x: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        pos: int,
    ) -> Tensor:
        """Forward single token (B, 1, D) with KV cache. pos = current position."""
        B = x.shape[0]
        h = self.attn_norm(x)
        qkv = self.Wqkv(h).reshape(B, 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        k_cache[:, pos] = k[:, 0]
        v_cache[:, pos] = v[:, 0]
        q = q.transpose(1, 2)  # (B, n_heads, 1, head_dim)
        k_all = k_cache[:, : pos + 1].transpose(1, 2)
        v_all = v_cache[:, : pos + 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
        out = out.transpose(1, 2).reshape(B, 1, -1)
        x = x + self.out_proj(out)
        h = self.mlp_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class RQTransformer(nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        rq_dim: int,
        n_layers: int,
        n_heads: int,
        n_quantizers: int,
        codebook_size: int,
    ):
        super().__init__()
        self.rq_dim = rq_dim
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size

        self.backbone_proj = (
            nn.Linear(backbone_dim, rq_dim, bias=False)
            if backbone_dim != rq_dim
            else nn.Identity()
        )
        self.code_emb = AudioEmbedding(n_quantizers - 1, codebook_size, rq_dim)
        self.pos_emb = nn.Embedding(n_quantizers, rq_dim)
        self.blocks = nn.ModuleList([RQBlock(rq_dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(rq_dim)
        self.head_W = nn.Parameter(torch.empty(n_quantizers - 1, codebook_size, rq_dim))
        nn.init.normal_(self.head_W, std=0.02)

    def forward(self, backbone_hidden: Tensor, codes: Tensor) -> Tensor:
        Q = codes.shape[1]  # may be < self.n_quantizers due to codebook dropout

        # codes[:, :Q-1] are the teacher-forced inputs for quantizers 1..Q-1
        offsets = torch.arange(Q - 1, device=codes.device) * self.codebook_size
        code_embs = self.code_emb.embedding(
            codes[:, : Q - 1] + offsets
        )  # (N, Q-1, rq_dim)

        seq = torch.cat(
            [self.backbone_proj(backbone_hidden).unsqueeze(1), code_embs], dim=1
        )  # (N, Q, rq_dim)
        seq = seq + self.pos_emb.weight[:Q]

        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)

        logits = torch.einsum("nqd,qcd->nqc", seq[:, 1:], self.head_W[: Q - 1])
        return logits  # (N, Q-1, codebook_size)

    @torch.inference_mode()
    def generate(
        self,
        backbone_hidden: Tensor,
        code_0: Tensor,
        temperature: float = 0.7,
        top_k: int | None = None,
        logit_bias: Tensor | None = None,
    ) -> Tensor:
        B = backbone_hidden.shape[0]
        device = backbone_hidden.device
        dtype = backbone_hidden.dtype
        Q = self.n_quantizers

        seq = torch.zeros(B, Q, self.rq_dim, device=device, dtype=dtype)
        seq[:, 0] = self.backbone_proj(backbone_hidden) + self.pos_emb.weight[0]
        offset_0 = 0 * self.codebook_size
        seq[:, 1] = self.code_emb.embedding(code_0 + offset_0) + self.pos_emb.weight[1]
        codes = [code_0]

        for i in range(Q - 1):
            h = seq
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)
            logits_i = F.linear(h[:, i + 1], self.head_W[i])
            if logit_bias is not None:
                logits_i = logits_i + logit_bias[i : i + 1]
            logits_i = logits_i.float() / temperature
            if top_k is not None:
                v, _ = logits_i.topk(top_k)
                logits_i[logits_i < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits_i, dim=-1)
            next_code = torch.multinomial(probs, 1).squeeze(-1)
            codes.append(next_code)
            if i + 2 < Q:
                offset = (i + 1) * self.codebook_size
                seq[:, i + 2] = (
                    self.code_emb.embedding(next_code + offset)
                    + self.pos_emb.weight[i + 2]
                )

        return torch.stack(codes, dim=1)

    def generate_kv(
        self,
        backbone_hidden: Tensor,
        code_0: Tensor,
        temperature: float | Tensor = 0.7,
        top_k: int | Tensor = 0,
        logit_bias: Tensor | None = None,
    ) -> Tensor:
        B = backbone_hidden.shape[0]
        device = backbone_hidden.device
        dtype = backbone_hidden.dtype
        Q = self.n_quantizers
        n_h = self.blocks[0].n_heads
        h_d = self.blocks[0].head_dim
        D = n_h * h_d
        n_layers = len(self.blocks)
        k_caches = torch.zeros(n_layers, B, Q, n_h, h_d, device=device, dtype=dtype)
        v_caches = torch.zeros(n_layers, B, Q, n_h, h_d, device=device, dtype=dtype)

        # Prefill positions 0 (backbone) and 1 (code_0)
        tok0 = self.backbone_proj(backbone_hidden) + self.pos_emb.weight[0]
        tok1 = self.code_emb.embedding(code_0) + self.pos_emb.weight[1]
        seq2 = torch.stack([tok0, tok1], dim=1)
        for li, block in enumerate(self.blocks):
            h = block.attn_norm(seq2)
            qkv = block.Wqkv(h).reshape(B, 2, 3, n_h, h_d)
            q, k, v = qkv.unbind(2)
            k_caches[li, :, :2] = k
            v_caches[li, :, :2] = v
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, 2, D)
            seq2 = seq2 + block.out_proj(out)
            h2 = block.mlp_norm(seq2)
            seq2 = seq2 + block.w2(F.silu(block.w1(h2)) * block.w3(h2))

        codes = [code_0]
        # Predict code 1 from position 1 hidden
        h_out = self.norm(seq2[:, 1])
        logits = F.linear(h_out, self.head_W[0]).float() / temperature
        if logit_bias is not None:
            logits = logits + logit_bias[0]
        if top_k > 0:
            v_top, _ = logits.topk(top_k)
            logits[logits < v_top[:, -1:]] = float("-inf")
        next_code = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        codes.append(next_code)

        # Autoregressive single-token steps
        for i in range(1, Q - 1):
            offset = i * self.codebook_size
            tok = (
                self.code_emb.embedding(next_code + offset) + self.pos_emb.weight[i + 1]
            ).unsqueeze(1)
            for li, block in enumerate(self.blocks):
                tok = block.forward_kv(tok, k_caches[li], v_caches[li], i + 1)
            h_out = self.norm(tok[:, 0])
            logits = F.linear(h_out, self.head_W[i]).float() / temperature
            if logit_bias is not None:
                logits = logits + logit_bias[i]
            if top_k > 0:
                v_top, _ = logits.topk(top_k)
                logits[logits < v_top[:, -1:]] = float("-inf")
            next_code = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            codes.append(next_code)

        return torch.stack(codes, dim=1)

    def generate_kv_compilable(
        self,
        backbone_hidden: Tensor,
        code_0: Tensor,
        temperature: Tensor,
        logit_bias: Tensor,
        top_k: int = 100,
    ) -> Tensor:
        """torch.compile-friendly generate_kv: no graph breaks.

        Unlike generate_kv, requires: top_k as Python int (not tensor),
        logit_bias always provided (use zeros), temperature as tensor.
        Inlines forward_kv to keep pos as compile-time constants.
        """
        device = backbone_hidden.device
        dtype = backbone_hidden.dtype
        Q = self.n_quantizers
        n_h = self.blocks[0].n_heads
        h_d = self.blocks[0].head_dim
        D = n_h * h_d
        n_layers = len(self.blocks)
        NEG_INF = torch.tensor(float("-inf"), device=device)
        k_c = torch.zeros(n_layers, 1, Q, n_h, h_d, device=device, dtype=dtype)
        v_c = torch.zeros(n_layers, 1, Q, n_h, h_d, device=device, dtype=dtype)

        # Prefill positions 0 (backbone) and 1 (code_0)
        tok0 = self.backbone_proj(backbone_hidden) + self.pos_emb.weight[0]
        tok1 = self.code_emb.embedding(code_0) + self.pos_emb.weight[1]
        seq = torch.stack([tok0, tok1], dim=1)
        for li in range(n_layers):
            block = self.blocks[li]
            h = block.attn_norm(seq)
            qkv = block.Wqkv(h).reshape(1, 2, 3, n_h, h_d)
            q, k, v = qkv.unbind(2)
            k_c[li, :, :2] = k
            v_c[li, :, :2] = v
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(1, 2, D)
            seq = seq + block.out_proj(out)
            h2 = block.mlp_norm(seq)
            seq = seq + block.w2(F.silu(block.w1(h2)) * block.w3(h2))

        # Code 1 from position 1
        h_out = self.norm(seq[:, 1])
        logits = F.linear(h_out, self.head_W[0]).float() / temperature
        logits = logits + logit_bias[0]
        v_top, _ = logits.topk(top_k)
        logits = torch.where(logits < v_top[:, -1:], NEG_INF, logits)
        codes = [code_0, torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)]

        # Autoregressive steps (inlined forward_kv, pos is compile-time constant)
        for i in range(1, Q - 1):
            pos = i + 1
            offset = i * self.codebook_size
            tok = (
                self.code_emb.embedding(codes[-1] + offset) + self.pos_emb.weight[pos]
            ).unsqueeze(1)
            for li in range(n_layers):
                block = self.blocks[li]
                h = block.attn_norm(tok)
                qkv = block.Wqkv(h).reshape(1, 1, 3, n_h, h_d)
                q, k, v = qkv.unbind(2)
                k_c[li, :, pos] = k[:, 0]
                v_c[li, :, pos] = v[:, 0]
                q = q.transpose(1, 2)
                k_all = k_c[li, :, : pos + 1].transpose(1, 2)
                v_all = v_c[li, :, : pos + 1].transpose(1, 2)
                out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
                out = out.transpose(1, 2).reshape(1, 1, D)
                tok = tok + block.out_proj(out)
                h2 = block.mlp_norm(tok)
                tok = tok + block.w2(F.silu(block.w1(h2)) * block.w3(h2))
            h_out = self.norm(tok[:, 0])
            logits = F.linear(h_out, self.head_W[i]).float() / temperature
            logits = logits + logit_bias[i]
            v_top, _ = logits.topk(top_k)
            logits = torch.where(logits < v_top[:, -1:], NEG_INF, logits)
            codes.append(torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1))

        return torch.stack(codes, dim=1)

    def setup_cuda_graph_batched(self, batch_size: int, device, dtype=torch.bfloat16):
        """Record CUDA graph for batched RQ generation at fixed batch size."""
        Q = self.n_quantizers
        self._bgraph_bs = batch_size
        self._bgraph_hidden = torch.randn(
            batch_size, self.rq_dim, device=device, dtype=dtype
        )
        self._bgraph_code0 = torch.randint(
            0, self.codebook_size, (batch_size,), device=device
        )
        self._bgraph_temp = torch.tensor(0.9, device=device, dtype=torch.float32)
        self._bgraph_seq = torch.zeros(
            batch_size, Q, self.rq_dim, device=device, dtype=dtype
        )
        self._bgraph_codes = torch.zeros(batch_size, Q, device=device, dtype=torch.long)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.autocast("cuda", dtype):
            for _ in range(3):
                self._batched_graph_body()
        torch.cuda.current_stream().wait_stream(s)
        self._bgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._bgraph), torch.autocast("cuda", dtype):
            self._batched_graph_body()

    def _batched_graph_body(self):
        Q = self.n_quantizers
        self._bgraph_bs
        seq = self._bgraph_seq
        seq.zero_()
        seq[:, 0].copy_(
            self.backbone_proj(self._bgraph_hidden) + self.pos_emb.weight[0]
        )
        seq[:, 1].copy_(
            self.code_emb.embedding(self._bgraph_code0) + self.pos_emb.weight[1]
        )
        self._bgraph_codes[:, 0].copy_(self._bgraph_code0)

        for i in range(Q - 1):
            h = seq[:, :Q]
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)
            logits_i = F.linear(h[:, i + 1], self.head_W[i]) / self._bgraph_temp
            probs = F.softmax(logits_i, dim=-1)
            next_code = torch.multinomial(probs, 1).squeeze(-1)
            self._bgraph_codes[:, i + 1].copy_(next_code)
            if i + 2 < Q:
                offset = (i + 1) * self.codebook_size
                seq[:, i + 2].copy_(
                    self.code_emb.embedding(next_code + offset)
                    + self.pos_emb.weight[i + 2]
                )

    def generate_batched_graph(
        self,
        backbone_hidden: Tensor,
        code_0: Tensor,
        temperature: float = 0.9,
    ) -> Tensor:
        self._bgraph_hidden.copy_(backbone_hidden)
        self._bgraph_code0.copy_(code_0)
        self._bgraph_temp.fill_(temperature)
        self._bgraph.replay()
        return self._bgraph_codes.clone()

    def setup_cuda_graph(self, device, dtype=torch.bfloat16, top_k: int = 100):
        """Record CUDA graphs for each possible Q value (2..n_quantizers)."""
        Q_max = self.n_quantizers
        self._graph_top_k = top_k
        self._graph_backbone_hidden = torch.zeros(
            1, self.rq_dim, device=device, dtype=dtype
        )
        self._graph_code0 = torch.zeros(1, device=device, dtype=torch.long)
        self._graph_temperature = torch.tensor(0.7, device=device, dtype=torch.float32)
        self._graph_seq = torch.zeros(1, Q_max, self.rq_dim, device=device, dtype=dtype)
        self._graph_logit_bias = torch.zeros(
            Q_max - 1, self.codebook_size, device=device, dtype=torch.float32
        )
        self._graph_codes_out = torch.zeros(1, Q_max, device=device, dtype=torch.long)
        self._cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        saved_Q = self.n_quantizers
        for Q in range(2, Q_max + 1):
            self.n_quantizers = Q
            for _ in range(3):
                self._generate_graph_body()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._generate_graph_body()
            self._cuda_graphs[Q] = graph
        self.n_quantizers = saved_Q
        self._cuda_graph = self._cuda_graphs[saved_Q]

    def _generate_graph_body(self):
        Q = self.n_quantizers
        seq = self._graph_seq
        seq.zero_()
        seq[:, 0].copy_(
            self.backbone_proj(self._graph_backbone_hidden) + self.pos_emb.weight[0]
        )
        seq[:, 1].copy_(
            self.code_emb.embedding(self._graph_code0 + 0 * self.codebook_size)
            + self.pos_emb.weight[1]
        )
        self._graph_codes_out[:, 0].copy_(self._graph_code0)

        for i in range(Q - 1):
            h = seq[:, :Q]
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)
            logits_i = F.linear(h[:, i + 1], self.head_W[i])
            logits_i = logits_i + self._graph_logit_bias[i : i + 1]
            logits_i = logits_i / self._graph_temperature
            if self._graph_top_k > 0:
                v, _ = logits_i.topk(self._graph_top_k)
                logits_i[logits_i < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits_i, dim=-1)
            next_code = torch.multinomial(probs, 1).squeeze(-1)
            self._graph_codes_out[:, i + 1].copy_(next_code)
            if i + 2 < Q:
                offset = (i + 1) * self.codebook_size
                seq[:, i + 2].copy_(
                    self.code_emb.embedding(next_code + offset)
                    + self.pos_emb.weight[i + 2]
                )

    def generate_cuda_graph(
        self,
        backbone_hidden: Tensor,
        code_0: Tensor,
        temperature: float = 0.7,
        logit_bias: Tensor | None = None,
        n_quantizers: int = 0,
    ) -> Tensor:
        Q = n_quantizers if n_quantizers > 0 else self.n_quantizers
        self._graph_backbone_hidden.copy_(backbone_hidden)
        self._graph_code0.copy_(code_0)
        self._graph_temperature.fill_(temperature)
        self._graph_logit_bias.zero_()
        if logit_bias is not None:
            self._graph_logit_bias[: logit_bias.size(0)].copy_(logit_bias)
        self._cuda_graphs[Q].replay()
        return self._graph_codes_out[:, :Q].clone()

    def setup_cuda_graph_kv(
        self,
        device,
        dtype=torch.bfloat16,
        top_k: int = 100,
        pool=None,
        batch_sizes: tuple[int, ...] = (1,),
    ):
        """Capture CUDA graphs for the RQ autoregressive body at one or more
        batch sizes. Pass `batch_sizes=(1, 2, 4, 8)` to support dynamic dispatch.

        The highest batch size also sets the storage for the static buffers
        (`_gkv_backbone`, `_gkv_k`, etc.) — smaller-B graphs capture views
        into the top-B buffers (B=1 at row 0, B=2 at rows 0..1, etc.).
        At runtime, `generate_cuda_graph_kv(hidden, code0, ...)` infers B
        from `hidden.shape[0]` and replays the matching graph.
        """
        Q = self.n_quantizers
        n_h = self.blocks[0].n_heads
        h_d = self.blocks[0].head_dim
        D = n_h * h_d
        n_layers = len(self.blocks)
        self._gkv_top_k = top_k
        self._gkv_batch_sizes = tuple(sorted(set(batch_sizes)))
        B_max = max(self._gkv_batch_sizes)

        # Allocate static buffers at B_max. Smaller B graphs capture views [:B].
        self._gkv_backbone = torch.zeros(B_max, self.rq_dim, device=device, dtype=dtype)
        self._gkv_code0 = torch.zeros(B_max, device=device, dtype=torch.long)
        self._gkv_temp = torch.tensor(0.7, device=device, dtype=torch.float32)
        self._gkv_logit_bias = torch.zeros(
            Q - 1, self.codebook_size, device=device, dtype=torch.float32
        )
        self._gkv_codes_out = torch.zeros(B_max, Q, device=device, dtype=torch.long)
        # KV caches: (n_layers, B_max, Q, n_heads, head_dim)
        self._gkv_k = torch.zeros(
            n_layers, B_max, Q, n_h, h_d, device=device, dtype=dtype
        )
        self._gkv_v = torch.zeros(
            n_layers, B_max, Q, n_h, h_d, device=device, dtype=dtype
        )
        # Pre-split Wqkv weights into views for zero-copy K,V writes
        self._gkv_Wq = [b.Wqkv.weight[:D] for b in self.blocks]
        self._gkv_Wk = [b.Wqkv.weight[D : 2 * D] for b in self.blocks]
        self._gkv_Wv = [b.Wqkv.weight[2 * D :] for b in self.blocks]

        # One graph per (batch_size, Q) pair.
        # Key: (batch_size, quantizer_count) → CUDAGraph
        self._gkv_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        saved_Q = self.n_quantizers
        prev_batch = getattr(self, "_gkv_current_batch", None)
        with torch.inference_mode():
            for B in self._gkv_batch_sizes:
                self._gkv_current_batch = B
                for Qi in range(2, Q + 1):
                    self.n_quantizers = Qi
                    for _ in range(3):
                        self._generate_graph_body_kv()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        self._generate_graph_body_kv()
                    self._gkv_graphs[(B, Qi)] = graph
        self.n_quantizers = saved_Q
        if prev_batch is not None:
            self._gkv_current_batch = prev_batch
        else:
            self._gkv_current_batch = B_max

    def _gkv_step(self, li: int, x: Tensor, pos: int) -> Tensor:
        """Single-token forward through block li with zero-copy KV cache write.
        x: (B, 1, D). Reads B from self._gkv_current_batch."""
        B = self._gkv_current_batch
        block = self.blocks[li]
        n_h, h_d = block.n_heads, block.head_dim
        D = n_h * h_d
        h = block.attn_norm(x)  # (B, 1, D)
        h_flat = h.view(B, D)
        q = torch.mm(h_flat, self._gkv_Wq[li].T).view(B, n_h, 1, h_d)
        # K,V written directly into cache[:, :, pos] which is a (B, n_h, h_d) slice
        # contiguous over (B, n_h*h_d=D) → view as (B, D) target for the mm
        k_slot = self._gkv_k[li, :B, pos]  # (B, n_h, h_d)
        v_slot = self._gkv_v[li, :B, pos]
        torch.mm(h_flat, self._gkv_Wk[li].T, out=k_slot.view(B, D))
        torch.mm(h_flat, self._gkv_Wv[li].T, out=v_slot.view(B, D))
        k_all = self._gkv_k[li, :B, : pos + 1].transpose(1, 2)  # (B, n_h, pos+1, h_d)
        v_all = self._gkv_v[li, :B, : pos + 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
        out = out.transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)
        x = x + block.out_proj(out)
        h2 = block.mlp_norm(x)
        x = x + block.w2(F.silu(block.w1(h2)) * block.w3(h2))
        return x

    def _generate_graph_body_kv(self):
        """Autoregressive RQ body. Reads B from self._gkv_current_batch,
        writes into views [:B] of the static buffers."""
        Q = self.n_quantizers
        B = self._gkv_current_batch
        n_h = self.blocks[0].n_heads
        h_d = self.blocks[0].head_dim
        D = n_h * h_d

        bb = self._gkv_backbone[:B]  # (B, rq_dim)
        c0 = self._gkv_code0[:B]  # (B,)

        # Prefill positions 0 and 1 (token from backbone + token from code0)
        tok0 = self.backbone_proj(bb) + self.pos_emb.weight[0]  # (B, D)
        tok1 = (
            self.code_emb.embedding(c0 + 0 * self.codebook_size)
            + self.pos_emb.weight[1]
        )  # (B, D)
        seq2 = torch.stack([tok0, tok1], dim=1)  # (B, 2, D)
        for li, block in enumerate(self.blocks):
            h = block.attn_norm(seq2)  # (B, 2, D)
            h_flat = h.reshape(B * 2, D)
            q = (
                torch.mm(h_flat, self._gkv_Wq[li].T)
                .view(B, 2, n_h, h_d)
                .transpose(1, 2)  # (B, n_h, 2, h_d)
            )
            # Compute K,V to fresh (contiguous) buffers then copy_ into the cache.
            # The slice `_gkv_k[li, :B, :2]` is non-contiguous at B>1 (stride Q*D
            # on the batch dim, not 2*D), so `reshape(B*2, D)` would silently copy
            # and any `mm(out=...)` would write to the discarded copy — previously
            # leaving the cache at zeros and garbling every multi-row decode.
            k_new = torch.mm(h_flat, self._gkv_Wk[li].T).view(B, 2, n_h, h_d)
            v_new = torch.mm(h_flat, self._gkv_Wv[li].T).view(B, 2, n_h, h_d)
            self._gkv_k[li, :B, :2].copy_(k_new)
            self._gkv_v[li, :B, :2].copy_(v_new)
            k = self._gkv_k[li, :B, :2].transpose(1, 2)  # (B, n_h, 2, h_d)
            v = self._gkv_v[li, :B, :2].transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, 2, D)  # (B, 2, D)
            seq2 = seq2 + block.out_proj(out)
            h2 = block.mlp_norm(seq2)
            seq2 = seq2 + block.w2(F.silu(block.w1(h2)) * block.w3(h2))
        self._gkv_codes_out[:B, 0].copy_(c0)

        # Predict code 1 from position 1 hidden
        h_out = self.norm(seq2[:, 1:2])  # (B, 1, D)
        logits = F.linear(h_out.squeeze(1), self.head_W[0]) + self._gkv_logit_bias[0:1]
        # logits: (B, CS)  (bias broadcasts from (1, CS))
        logits = logits / self._gkv_temp
        if self._gkv_top_k > 0:
            v_top, _ = logits.topk(self._gkv_top_k)
            logits[logits < v_top[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_code = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
        self._gkv_codes_out[:B, 1].copy_(next_code)

        # Autoregressive single-token steps with zero-copy KV
        for i in range(1, Q - 1):
            offset = i * self.codebook_size
            tok = (
                self.code_emb.embedding(next_code + offset) + self.pos_emb.weight[i + 1]
            ).unsqueeze(
                1
            )  # (B, 1, D)
            for li in range(len(self.blocks)):
                tok = self._gkv_step(li, tok, i + 1)
            h_out = self.norm(tok)  # (B, 1, D)
            logits = (
                F.linear(h_out.squeeze(1), self.head_W[i])
                + self._gkv_logit_bias[i : i + 1]
            )
            logits = logits / self._gkv_temp
            if self._gkv_top_k > 0:
                v_top, _ = logits.topk(self._gkv_top_k)
                logits[logits < v_top[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_code = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            self._gkv_codes_out[:B, i + 1].copy_(next_code)

    def generate_cuda_graph_kv(
        self,
        backbone_hidden: Tensor,
        code_0: Tensor,
        temperature: float = 0.7,
        logit_bias: Tensor | None = None,
        n_quantizers: int = 0,
    ) -> Tensor:
        """Sample Q codes for a batch of B rows. B is inferred from
        `backbone_hidden.shape[0]` and dispatched to the graph captured at
        that batch size (must be one of `setup_cuda_graph_kv(batch_sizes=...)`).
        Returns `(B, Q)` long tensor — a VIEW into the static codes_out buffer."""
        B = backbone_hidden.shape[0]
        Q = n_quantizers if n_quantizers > 0 else self.n_quantizers
        key = (B, Q)
        if key not in self._gkv_graphs:
            raise KeyError(
                f"No RQ CUDA graph captured for batch_size={B}, Q={Q}. "
                f"Call setup_cuda_graph_kv(batch_sizes=(1,2,4,...)) to capture."
            )
        # Write inputs into the views [:B] of the static buffers
        self._gkv_backbone[:B].copy_(backbone_hidden)
        self._gkv_code0[:B].copy_(code_0)
        self._gkv_temp.fill_(temperature)
        self._gkv_logit_bias.zero_()
        if logit_bias is not None:
            self._gkv_logit_bias[: logit_bias.size(0)].copy_(logit_bias)
        self._gkv_graphs[key].replay()
        return self._gkv_codes_out[:B, :Q]


class ScalarProjector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.proj(x).unsqueeze(1)


class ScalarCondProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_freq: int = 32):
        super().__init__()
        freqs = torch.exp(torch.linspace(0, math.log(1000), n_freq))
        self.register_buffer("freqs", freqs)
        feat_dim = input_dim * n_freq * 2
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        nn.init.normal_(self.proj[0].weight, std=0.01)
        nn.init.normal_(self.proj[2].weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        # x: (B, input_dim) -> sinusoidal features -> (B, d_model)
        xf = x.unsqueeze(-1) * self.freqs
        features = torch.cat([xf.sin(), xf.cos()], dim=-1).flatten(1)
        return self.proj(features)


def _expand_kv_heads(weight: Tensor, n_kv_heads: int, n_heads: int) -> Tensor:
    """Repeat GQA KV heads to full MHA. (n_kv * head_dim, d) -> (n_heads * head_dim, d)."""
    if n_kv_heads == n_heads:
        return weight
    repeats = n_heads // n_kv_heads
    head_dim = weight.shape[0] // n_kv_heads
    # (n_kv, head_dim, d) -> repeat -> (n_heads, head_dim, d) -> (n_heads * head_dim, d)
    w = weight.view(n_kv_heads, head_dim, -1)
    w = w.repeat_interleave(repeats, dim=0)
    return w.reshape(n_heads * head_dim, -1)


def convert_hf_llama_state_dict(
    hf_sd: dict[str, Tensor],
    n_heads: int | None = None,
    n_kv_heads_src: int | None = None,
) -> dict[str, Tensor]:
    """Convert HuggingFace LlamaForCausalLM weights to our Vui decoder format.
    If n_heads != n_kv_heads_src, expands GQA KV weights to full MHA."""
    sd = {}
    sd["decoder.norm.weight"] = hf_sd["model.norm.weight"]
    if "model.embed_tokens.weight" in hf_sd:
        sd["token_emb.weight"] = hf_sd["model.embed_tokens.weight"]
    expand = n_heads and n_kv_heads_src and n_kv_heads_src != n_heads

    i = 0
    while f"model.layers.{i}.self_attn.q_proj.weight" in hf_sd:
        prefix_hf = f"model.layers.{i}"
        prefix_ours = f"decoder.blocks.{i}"

        q = hf_sd[f"{prefix_hf}.self_attn.q_proj.weight"]
        k = hf_sd[f"{prefix_hf}.self_attn.k_proj.weight"]
        v = hf_sd[f"{prefix_hf}.self_attn.v_proj.weight"]
        if expand:
            k = _expand_kv_heads(k, n_kv_heads_src, n_heads)
            v = _expand_kv_heads(v, n_kv_heads_src, n_heads)
        sd[f"{prefix_ours}.attn.Wqkv.weight"] = torch.cat([q, k, v], dim=0)

        sd[f"{prefix_ours}.attn.out_proj.weight"] = hf_sd[
            f"{prefix_hf}.self_attn.o_proj.weight"
        ]
        sd[f"{prefix_ours}.mlp.w1.weight"] = hf_sd[f"{prefix_hf}.mlp.gate_proj.weight"]
        sd[f"{prefix_ours}.mlp.w3.weight"] = hf_sd[f"{prefix_hf}.mlp.up_proj.weight"]
        sd[f"{prefix_ours}.mlp.w2.weight"] = hf_sd[f"{prefix_hf}.mlp.down_proj.weight"]
        sd[f"{prefix_ours}.attn_norm.weight"] = hf_sd[
            f"{prefix_hf}.input_layernorm.weight"
        ]
        sd[f"{prefix_ours}.mlp_norm.weight"] = hf_sd[
            f"{prefix_hf}.post_attention_layernorm.weight"
        ]
        i += 1

    return sd


class Vui(nn.Module):
    VUI2 = "vui-nano.safetensors"
    # Legacy vui1 100M checkpoints (also on the same HF repo)
    BASE = "vui-100m-base.pt"
    COHOST = "vui-cohost-100m.pt"
    ABRAHAM = "vui-abraham-100m.pt"

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.codec = None
        cfg = config.model
        self.use_rotary_emb = cfg.use_rotary_emb

        n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
        max_seqlen = (
            config.max_seq_len
            if config.max_seq_len > 0
            else cfg.max_text_tokens + cfg.max_audio_tokens
        )

        self.text_tokenizer = VuiTokenizer(config.data.tokenizer)
        self.token_emb = nn.Embedding(self.text_tokenizer.vocab_size, cfg.d_model)
        self.audio_emb = AudioEmbedding(
            cfg.n_quantizers,
            cfg.codebook_size,
            cfg.d_model,
        )
        if cfg.use_rq_transformer:
            self.codec_head = nn.Linear(cfg.d_model, cfg.codebook_size, bias=False)
            self.rq_transformer = RQTransformer(
                backbone_dim=cfg.d_model,
                rq_dim=cfg.rq_d_model,
                n_layers=cfg.rq_n_layers,
                n_heads=cfg.rq_n_heads,
                n_quantizers=cfg.n_quantizers,
                codebook_size=cfg.codebook_size,
            )
            self.audio_head = None
        else:
            self.audio_head = AudioHead(
                cfg.n_quantizers, cfg.codebook_size, cfg.d_model
            )
            self.codec_head = None
            self.rq_transformer = None
        self.eos_head = nn.Linear(cfg.d_model, 1)
        if cfg.sinusoidal_cond:
            self.sq_proj = (
                ScalarCondProjector(6, cfg.d_model) if cfg.has_sq_proj else None
            )
            self.wps_proj = (
                ScalarCondProjector(1, cfg.d_model) if cfg.has_wps_proj else None
            )
        else:
            self.sq_proj = None
            if cfg.has_sq_proj:
                self.sq_proj = nn.Linear(6, cfg.d_model, bias=False)
                nn.init.normal_(self.sq_proj.weight, std=0.01)
            self.wps_proj = ScalarProjector(cfg.d_model) if cfg.has_wps_proj else None
        if cfg.has_spk_proj:
            self.spk_proj = nn.Linear(cfg.spk_emb_dim, cfg.d_model, bias=False)
            nn.init.normal_(self.spk_proj.weight, std=0.01)
        else:
            self.spk_proj = None
        self.noisy_emb = nn.Parameter(torch.randn(cfg.d_model) * 0.01)

        self.decoder = Decoder(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=n_kv_heads,
            bias=cfg.bias,
            dropout=cfg.dropout,
            max_seqlen=max_seqlen + cfg.n_quantizers,
            rope_dim=cfg.rope_dim,
            rope_theta=cfg.rope_theta,
            rope_theta_rescale_factor=cfg.rope_theta_rescale_factor,
            intermediate_size=cfg.intermediate_size,
            window_size=cfg.window_size,
            global_every=cfg.global_every,
            global_rope_dim=cfg.global_rope_dim,
            global_rope_theta=cfg.global_rope_theta,
        )

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def from_pretrained(
        checkpoint_path: str | dict,
        **config_kwargs,
    ):
        if isinstance(checkpoint_path, dict):
            checkpoint = checkpoint_path
            config = {**checkpoint["config"], **config_kwargs}
            state_dict = checkpoint["model"]
        else:
            if not os.path.exists(checkpoint_path):
                from vui.hf import download

                checkpoint_path = download(checkpoint_path)

            if checkpoint_path.endswith(".safetensors"):
                import json

                from safetensors.torch import load_file, safe_open

                with safe_open(checkpoint_path, framework="pt") as f:
                    metadata = f.metadata()
                config = json.loads(metadata["config"])
                config = {**config, **config_kwargs}
                state_dict = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                config = {**checkpoint["config"], **config_kwargs}
                state_dict = checkpoint["model"]

        from vui.config import infer_optional_modules

        config = infer_optional_modules(config, state_dict.keys())
        config = Config(**config)

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {
            k.replace("text_embedding.", "token_emb."): v for k, v in state_dict.items()
        }
        model = Vui(config)

        # Migrate old checkpoints trained before byte tokens were added to the
        # tokenizer. Old layout: [base | specials], new: [base | 256 bytes | specials].
        tok = model.text_tokenizer
        emb_key = "token_emb.weight"
        if emb_key in state_dict:
            ckpt_vocab = state_dict[emb_key].shape[0]
            expected_vocab = tok.vocab_size
            if ckpt_vocab == expected_vocab - tok.NUM_BYTES:
                old_specials = state_dict[emb_key][tok._base_vocab_size :]
                new_emb = torch.zeros(expected_vocab, state_dict[emb_key].shape[1])
                new_emb[: tok._base_vocab_size] = state_dict[emb_key][
                    : tok._base_vocab_size
                ]
                new_emb[
                    tok.special_offset : tok.special_offset + old_specials.shape[0]
                ] = old_specials
                state_dict[emb_key] = new_emb
                print(
                    f"Migrated token_emb: {ckpt_vocab} -> {expected_vocab} (inserted 256 byte token slots)"
                )

        load_what_you_can(state_dict, model)
        return model

    @staticmethod
    def from_hf_llama(repo_id: str, config: Config, n_kv_heads_src: int | None = None):
        """Load decoder + text embedding weights from a HuggingFace LlamaForCausalLM.
        n_kv_heads_src: KV heads in the source model (for GQA->MHA expansion).
        Auto-detected from HF config if not provided."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if n_kv_heads_src is None:
            from transformers import AutoConfig

            hf_cfg = AutoConfig.from_pretrained(repo_id)
            n_kv_heads_src = getattr(hf_cfg, "num_key_value_heads", None)

        path = hf_hub_download(repo_id, "model.safetensors")
        hf_sd = load_file(path)
        cfg = config.model
        sd = convert_hf_llama_state_dict(
            hf_sd,
            n_heads=cfg.n_heads,
            n_kv_heads_src=n_kv_heads_src,
        )
        model = Vui(config)
        load_what_you_can(sd, model)
        return model

    @staticmethod
    def from_pretrained_inf(
        checkpoint_path: str | dict,
        **config_kwargs,
    ):
        return Vui.from_pretrained(checkpoint_path, **config_kwargs).bfloat16().eval()

    def setup_decode_graph(self, pool=None):
        device = self.device
        dtype = self.dtype
        cfg = self.config.model

        self.decoder.allocate_flash_kv_cache(1, device, dtype)

        self._decode_code_in = torch.zeros(
            cfg.n_quantizers, device=device, dtype=torch.long
        )
        self._decode_pos = torch.zeros(1, 1, device=device, dtype=torch.long)
        self._decode_hidden_out = torch.zeros(
            1, cfg.d_model, device=device, dtype=dtype
        )
        self._decode_codec_logits = torch.zeros(
            1, cfg.codebook_size, device=device, dtype=dtype
        )
        self._decode_eos_logit = torch.zeros(1, 1, device=device, dtype=dtype)
        self._cond_bias = torch.zeros(1, 1, cfg.d_model, device=device, dtype=dtype)

        for _ in range(3):
            self._decode_step_inner()
        torch.cuda.synchronize()

        self.decoder.flash_kv_caches[0].seq_lens.zero_()

        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph, pool=pool):
            self._decode_step_inner()

    def set_cond_bias(
        self,
        sq_scores: tuple[float, ...] | None = None,
        wps_score: float = 0.0,
    ):
        """Compute and cache SQ/WPS conditioning bias for inference."""
        self._cond_bias.zero_()
        device = self.device
        dtype = self._cond_bias.dtype
        if sq_scores is not None and self.sq_proj is not None:
            sq_val = torch.tensor([sq_scores], device=device, dtype=torch.bfloat16)
            self._cond_bias.add_(self.sq_proj(sq_val).reshape(1, 1, -1).to(dtype))
        if wps_score > 0 and self.wps_proj is not None:
            wps_val = torch.tensor([wps_score], device=device, dtype=torch.bfloat16)
            self._cond_bias.add_(
                self.wps_proj(wps_val).squeeze(1).reshape(1, 1, -1).to(dtype)
            )

    def embed_speaker(self, spk_emb: Tensor) -> Tensor:
        """Project speaker embedding to a single (1, 1, d_model) token for prefill."""
        return self.spk_proj(spk_emb.to(self.device, torch.bfloat16)).reshape(1, 1, -1)

    def _decode_step_inner(self):
        codes = self._decode_code_in  # (Q,)
        emb = self.audio_emb(codes.unsqueeze(0)).unsqueeze(0)  # (1, 1, d)
        h = self.decoder.forward_flash(emb, self._decode_pos[:, 0])
        hidden = h[:, 0]
        self._decode_hidden_out.copy_(hidden)
        self._decode_codec_logits.copy_(self.codec_head(hidden))
        self._decode_eos_logit.copy_(self.eos_head(hidden))

    @torch.inference_mode()
    def decode_step(self, codes: Tensor, pos: int) -> tuple[Tensor, Tensor, Tensor]:
        self._decode_code_in.copy_(codes)
        self._decode_pos[:, 0].fill_(pos)
        # flash_attn_with_kvcache writes at cache_seqlens, then forward_flash increments
        self.decoder.flash_kv_caches[0].seq_lens[:1].fill_(pos)
        self._decode_graph.replay()
        return (
            self._decode_hidden_out,
            self._decode_codec_logits,
            self._decode_eos_logit,
        )

    def embed_audio(self, codes: Tensor) -> Tensor:
        return self.audio_emb(codes)

    def embed_text(self, token_ids: Tensor) -> Tensor:
        return self.token_emb(token_ids)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
