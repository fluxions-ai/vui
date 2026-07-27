"""flash-attn compatibility shim.

The model needs exactly one entry point from `flash_attn` —
`flash_attn_with_kvcache` — and there are hosts where it isn't usable:

* CPU / macOS installs, where the package isn't installed at all;
* Jetson-class ARM64 (Orin sm_87, Thor sm_110), where the published aarch64
  wheel imports fine but has no kernels for the device (it is built for
  sm_80/90/100/120 with no PTX fallback) — that surfaces as a launch-time
  "no kernel image is available for execution on the device".

Server ARM64 (Grace-Hopper sm_90, Grace-Blackwell sm_100) *is* covered by the
aarch64 wheel pinned in pyproject.toml, and uses the real kernel.

So this module dispatches: real kernel when importable, and a pure-PyTorch SDPA
implementation with the same semantics otherwise — including a one-way switch
if the first real call fails on an unsupported architecture. The fallback is
CUDA-graph safe (no host syncs, no data-dependent shapes), so the engine's
captured decode graphs work unchanged. It is slower: it attends over the full
pre-allocated cache length instead of just the filled prefix, so cap `max_seq`
when running on it.

Set `VUI_ATTN=torch` to force the fallback (used by the equivalence tests).
"""

import os

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["HAS_FLASH_ATTN", "flash_attn_with_kvcache"]


def _sdpa_attn_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Tensor | None = None,
    v: Tensor | None = None,
    cache_seqlens: Tensor | int | None = None,
    cache_batch_idx: Tensor | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
    **unsupported,
) -> Tensor:
    """Pure-PyTorch stand-in for `flash_attn.flash_attn_with_kvcache`.

    q: (B, Tq, H, D). k/v: (B, Tq, Hkv, D), appended to the cache in place at
    `cache_seqlens`. k_cache/v_cache: (Bc, S, Hkv, D). cache_batch_idx maps the
    B query rows onto arbitrary cache slots. Queries are right-aligned to the
    end of the key sequence, as in flash-attn: query i attends up to key index
    `cache_seqlens + Tnew - Tq + i`.

    Returns (B, Tq, H, D) in q's dtype.
    """
    if unsupported:
        # Better to fail loudly than to silently ignore e.g. rotary_cos/softcap.
        raise NotImplementedError(
            f"flash_compat: unsupported flash_attn_with_kvcache arguments: "
            f"{sorted(unsupported)}"
        )

    B, Tq, H, D = q.shape
    S, Hkv = k_cache.shape[1], k_cache.shape[2]
    device = q.device

    if cache_seqlens is None:
        seqlens = torch.zeros(B, dtype=torch.long, device=device)
    elif isinstance(cache_seqlens, int):
        seqlens = torch.full((B,), cache_seqlens, dtype=torch.long, device=device)
    else:
        seqlens = cache_seqlens.to(device=device, dtype=torch.long)
        if seqlens.numel() == 1:
            seqlens = seqlens.expand(B)

    slots = None if cache_batch_idx is None else cache_batch_idx.to(torch.long)

    # Append the new K/V into the cache, exactly like the fused kernel does.
    if k is not None:
        Tnew = k.shape[1]
        pos = seqlens[:, None] + torch.arange(Tnew, device=device)  # (B, Tnew)
        rows = (torch.arange(B, device=device) if slots is None else slots)[:, None]
        k_cache[rows, pos] = k.to(k_cache.dtype)
        v_cache[rows, pos] = v.to(v_cache.dtype)
        klen = seqlens + Tnew
    else:
        klen = seqlens

    if slots is None:
        keys, vals = k_cache[:B], v_cache[:B]  # view, no copy
    else:
        keys, vals = k_cache.index_select(0, slots), v_cache.index_select(0, slots)

    keys = keys.permute(0, 2, 1, 3).to(q.dtype)  # (B, Hkv, S, D)
    vals = vals.permute(0, 2, 1, 3).to(q.dtype)
    if H != Hkv:
        n_reps = H // Hkv
        keys = keys[:, :, None].expand(B, Hkv, n_reps, S, D).reshape(B, H, S, D)
        vals = vals[:, :, None].expand(B, Hkv, n_reps, S, D).reshape(B, H, S, D)

    # Last key index each query may attend to (right-aligned queries).
    j = torch.arange(S, device=device)
    j_max = (klen - Tq)[:, None] + torch.arange(Tq, device=device)  # (B, Tq)
    if causal:
        allowed = j <= j_max[..., None]  # (B, Tq, S)
    else:
        allowed = j < klen[:, None, None]
    left, right = window_size
    if left is not None and left >= 0:
        allowed = allowed & (j >= (j_max[..., None] - left))
    if not causal and right is not None and right >= 0:
        allowed = allowed & (j <= (j_max[..., None] + right))

    # An empty key set would make softmax produce NaNs; flash returns zeros.
    empty = ~allowed.any(dim=-1, keepdim=True)  # (B, Tq, 1)
    allowed = allowed | (empty & (j == 0))

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),  # (B, H, Tq, D)
        keys,
        vals,
        attn_mask=allowed[:, None],  # (B, 1, Tq, S)
        scale=softmax_scale,
    )
    out = out.masked_fill(empty[:, None], 0)  # (B, 1, Tq, 1) broadcasts over H, D
    return out.transpose(1, 2).contiguous()  # (B, Tq, H, D)


def _load_flash_attn():
    if os.environ.get("VUI_ATTN", "").lower() in ("torch", "sdpa"):
        return None
    try:
        from flash_attn import flash_attn_with_kvcache as _fn

        return _fn
    except ImportError:
        return None


_flash_fn = _load_flash_attn()
HAS_FLASH_ATTN = _flash_fn is not None  # False also when VUI_ATTN forces SDPA

# Launch-time failures that mean "this build has no kernels for this GPU" —
# Jetson Orin (sm_87) / Thor (sm_110) against an sm_80/90/100/120 wheel.
_ARCH_ERRORS = (
    "no kernel image is available",
    "invalid device function",
)


def _fallback_notice(reason: str) -> None:
    print(
        f"[vui] {reason} — using the PyTorch SDPA attention fallback. "
        "Correct but slower; pass max_seq= to cap the KV cache."
    )


_impl = _flash_fn or _sdpa_attn_with_kvcache

if not HAS_FLASH_ATTN:
    _fallback_notice("flash_attn unavailable on this platform")


def flash_attn_with_kvcache(*args, **kwargs):
    """flash-attn's kernel, degrading to SDPA if this GPU has no kernels for it.

    The switch is one-way and happens on the first failing call — in practice
    during engine warm-up, before any CUDA graph capture, so the captured
    graphs stay consistent.
    """
    global _impl
    if _impl is _sdpa_attn_with_kvcache:
        return _sdpa_attn_with_kvcache(*args, **kwargs)
    try:
        return _impl(*args, **kwargs)
    except RuntimeError as e:
        if not any(m in str(e).lower() for m in _ARCH_ERRORS):
            raise
        _impl = _sdpa_attn_with_kvcache
        _fallback_notice(f"flash_attn has no kernels for this GPU ({e})")
        return _sdpa_attn_with_kvcache(*args, **kwargs)
