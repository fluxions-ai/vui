import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from transformers import AutoTokenizer

from vui.legacy.fluac import Fluac
from vui.model import load_what_you_can

from .config import Config
from .patterns import DelayedPatternProvider
from .rope import apply_rotary_emb, precompute_freqs_cis


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
        assert input_pos.size(0) == k_val.size(-2)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


def repeat_kv(x: torch.Tensor, n_reps: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, n_kv_heads, T, head_dim = x.shape

    return (
        x[:, :, :, None, :]
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
        self.n_reps = n_kv_heads // n_heads
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
        x.dtype

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
                dim=1,
            )
            q, k, v = map(lambda t: rearrange(t, "B T (h d) -> B h T d"), (q, k, v))

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
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )

        out = self.out_proj(rearrange(out, "B h T d -> B T (h d)"))

        return out


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
        self, *, d_model: int, multiple_of: int = 256, bias: bool = False, **kwargs
    ) -> None:
        super().__init__()
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
        )
        self.mlp_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = LlamaMLP(d_model=d_model, bias=bias, dropout=dropout)

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
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.use_rotary_emb = use_rotary_emb

        self.max_seqlen = max_seqlen
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
                )
                for block_idx in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps=norm_eps)

        self.attn_mask = None

        head_dim = d_model // n_heads

        rope_dim = rope_dim or head_dim

        assert rope_dim <= head_dim  # apply RoPE to a fraction of embeddings

        freqs_cis = precompute_freqs_cis(
            rope_dim,
            max_seqlen,
            theta=rope_theta,
            theta_rescale_factor=rope_theta_rescale_factor,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def allocate_inference_cache(
        self, batch_size: int, device: str, dtype=torch.bfloat16
    ):
        for block in self.blocks:
            block.attn.kv_cache = KVCache(
                batch_size, self.max_seqlen, block.n_kv_heads, block.head_dim, dtype
            ).to(device)

        # I don't understand why this is needed
        self.attn_mask = torch.tril(
            torch.ones(
                self.max_seqlen, self.max_seqlen, dtype=torch.bool, device=device
            )
        )

    def deallocate_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = None

        self.attn_mask = None

    def forward(self, x: Tensor, input_pos: Tensor):
        if self.use_rotary_emb:
            freqs_cis = self.freqs_cis[input_pos]
        else:
            freqs_cis = None

        attn_mask = (
            self.attn_mask[None, None, input_pos]
            if self.attn_mask is not None
            else None
        )

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, input_pos=input_pos, attn_mask=attn_mask)

        x = self.norm(x)

        return x


class Vui(nn.Module):
    BASE = "vui-100m-base.pt"
    COHOST = "vui-cohost-100m.pt"
    ABRAHAM = "vui-abraham-100m.pt"

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.codec = Fluac.from_pretrained()
        self.config = config
        cfg = config.model
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        self.use_rotary_emb = cfg.use_rotary_emb
        self.token_emb = nn.Embedding(self.tokenizer.vocab_size, cfg.d_model)

        self.pattern_provider = DelayedPatternProvider(n_q=cfg.n_quantizers)

        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(cfg.codebook_size + 8, cfg.d_model)
                for _ in range(cfg.n_quantizers)
            ]
        )

        n_kv_heads = cfg.n_heads

        max_seqlen = cfg.max_text_tokens + cfg.max_audio_tokens
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
        )

        self.audio_heads = nn.ModuleList(
            [
                nn.Linear(cfg.d_model, cfg.codebook_size + 8, bias=cfg.bias)
                for _ in range(cfg.n_quantizers)
            ]
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
        checkpoint_path: str | dict = ABRAHAM,
        **config_kwargs,
    ):
        if isinstance(checkpoint_path, dict):
            checkpoint = checkpoint_path
        else:
            if not os.path.exists(checkpoint_path):
                from huggingface_hub import hf_hub_download

                checkpoint_path = hf_hub_download(
                    "fluxions/vui",
                    checkpoint_path,
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        config = {**checkpoint["config"], **config_kwargs}
        config = Config(**config)
        state_dict = checkpoint["model"]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {
            k.replace("text_embedding.", "token_emb."): v for k, v in state_dict.items()
        }
        model = Vui(config)
        load_what_you_can(state_dict, model)
        return model

    @staticmethod
    def from_pretrained_inf(
        checkpoint_path: str | dict,
        **config_kwargs,
    ):
        return Vui.from_pretrained(checkpoint_path, **config_kwargs).eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
