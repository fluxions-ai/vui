import torch
from einops import repeat
from torch import Tensor
from torch.amp import autocast

# RoPE
# copied out of lucidrains' rotary_embedding_torch for hackability
#
# Round and Round We Go! What makes Rotary Positional Encodings useful?
#   https://arxiv.org/abs/2410.06205
#   to me p-RoPE looks like the `rotary_emb_fraction` which has been around for a long time in the flash-attention repo? What am I missing?
#   other than the observations that you might want some proportion of dims to be NoPE to carry semantic information - i.e. stuff that's not
#   relative position dependent


def rotate_half(x):
    """Also known as "interleaved" style or GPT-J style."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs: Tensor, t: Tensor, start_index: int = 0, scale: float = 1.0
):
    dtype = t.dtype

    rot_dim = freqs.shape[-2]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # freqs: (seq, rot_dim, 2) — already position-indexed by caller
    # t: (B, heads, seq, head_dim) or (seq, heads, head_dim)
    # Unsqueeze freqs to broadcast over batch/heads dims
    if t.ndim == 3:
        # varlen: t is (total, heads, head_dim), freqs is (total, rot_dim, 2)
        freqs_cos = freqs[:, :, 0].unsqueeze(1)  # (total, 1, rot_dim)
        freqs_sin = freqs[:, :, 1].unsqueeze(1)  # (total, 1, rot_dim)
    else:
        # standard: t is (B, heads, T, head_dim), freqs is (T, rot_dim, 2)
        freqs_cos = freqs[..., 0]  # (T, rot_dim)
        freqs_sin = freqs[..., 1]  # (T, rot_dim)

    if start_index == 0 and end_index >= t.shape[-1]:
        # Full rotation — skip slicing and cat
        t = (t * freqs_cos * scale) + (rotate_half(t) * freqs_sin * scale)
        return t.to(dtype)

    t_left = t[..., :start_index]
    t_mid = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_mid = (t_mid * freqs_cos * scale) + (rotate_half(t_mid) * freqs_sin * scale)
    return torch.cat((t_left, t_mid, t_right), dim=-1).to(dtype)


def precompute_freqs_cis(
    dim: int,
    max_seqlen: int,
    theta: float = 10_000.0,
    theta_rescale_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
):
    theta *= theta_rescale_factor ** (dim / (dim - 2))

    # some good comments on numerical precision from the flash-attention repo
    pos = torch.arange(max_seqlen, dtype=dtype)
    inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))

    # outer product - torch.outer would be better
    freqs = torch.einsum("..., f -> ... f", pos.to(inv_freqs.dtype), inv_freqs)
    freqs = repeat(freqs, "... n -> ... (n r)", r=2)

    return torch.stack((freqs.cos(), freqs.sin()), dim=-1)
