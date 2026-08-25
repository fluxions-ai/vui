import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.amp import autocast


def rotate_half(x):
    """Also known as "interleaved" style or GPT-J style."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs: Tensor, t: Tensor, start_index: int = 0, scale: float = 1.0, seq_dim=-2
):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)
    return out.to(dtype)


def precompute_freqs_cis(
    dim: int,
    max_seqlen: int,
    theta: float = 10_000.0,
    theta_rescale_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    pos = torch.arange(max_seqlen, dtype=dtype)
    inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    freqs = torch.einsum("..., f -> ... f", pos.to(inv_freqs.dtype), inv_freqs)
    freqs = repeat(freqs, "... n -> ... (n r)", r=2)
    return freqs
