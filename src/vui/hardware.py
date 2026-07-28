"""Hardware capability detection — one place that decides dtype and attention.

The model was written for Ampere-and-newer (bf16 + FlashAttention-2). Neither is
available everywhere, and the failure modes are unhelpful: bf16 on a card
without native support is silently slow or raises deep in a kernel, and the
pinned flash-attn wheel carries cubins for sm_80/90/100/120 with no PTX, so an
older card gets "no kernel image is available" at the first decode step.

Compute capability floors that matter here:

    >= 8.0  Ampere+   bf16 native, FlashAttention-2 works
    7.5     Turing    no native bf16 -> fp16; no FA2 -> SDPA fallback
    7.0     Volta     as Turing, and needs a cu12 torch build (the cu130
                      wheels carry no sm_70 kernels)
    none    CPU/MPS   fp32

Override with `VUI_DTYPE=bf16|fp16|fp32` when the automatic choice is wrong.
`python -m vui.doctor` prints everything this module resolves.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from functools import lru_cache

import torch

__all__ = [
    "compute_capability",
    "gpu_name",
    "supports_bf16",
    "dtype",
    "autocast",
    "torch_supports_this_gpu",
]

_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "full": torch.float32,
}


@lru_cache(maxsize=1)
def compute_capability() -> tuple[int, int] | None:
    """(major, minor) of the current CUDA device, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return None


@lru_cache(maxsize=1)
def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name()
    except Exception:
        return None


@lru_cache(maxsize=1)
def supports_bf16() -> bool:
    """Native bf16 — Ampere (sm_80) and up.

    `torch.cuda.is_bf16_supported()` returns True on some older cards where
    bf16 is emulated rather than native, which is exactly the slow path we're
    trying to avoid, so go by compute capability instead.
    """
    cap = compute_capability()
    return cap is not None and cap >= (8, 0)


@lru_cache(maxsize=1)
def torch_supports_this_gpu() -> bool | None:
    """Does the installed torch carry kernels for this device?

    None when there's no CUDA device to judge. False means the wheel was built
    without this architecture — e.g. a cu130 build on Volta — which surfaces at
    the first kernel launch rather than at import, so it's worth saying up
    front.
    """
    cap = compute_capability()
    if cap is None:
        return None
    try:
        arches = torch.cuda.get_arch_list()
    except Exception:
        return None
    if not arches:
        return None
    return f"sm_{cap[0]}{cap[1]}" in arches


@lru_cache(maxsize=1)
def dtype() -> torch.dtype:
    """The dtype to run the model in.

    bf16 where it's native, fp16 on older CUDA cards, fp32 off-GPU. fp16 has a
    much smaller exponent range than bf16, so if you see NaNs on a pre-Ampere
    card, `VUI_DTYPE=fp32` is the escape hatch.
    """
    override = os.environ.get("VUI_DTYPE", "").strip().lower()
    if override:
        if override not in _DTYPES:
            raise ValueError(
                f"VUI_DTYPE={override!r} not recognised "
                f"(expected one of: bf16, fp16, fp32)"
            )
        return _DTYPES[override]

    cap = compute_capability()
    if cap is None:
        return torch.float32
    if cap >= (8, 0):
        return torch.bfloat16
    return torch.float16


def autocast(enabled: bool = True):
    """`torch.autocast` in the resolved dtype, or a no-op for fp32/CPU.

    Autocasting to fp32 is meaningless, and on a CPU-only box there's no CUDA
    autocast to enter at all, so both collapse to nullcontext.
    """
    if not enabled:
        return torch.autocast("cuda", enabled=False)
    d = dtype()
    if d is torch.float32 or not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast("cuda", d, True)


def summary() -> dict:
    """Everything resolved above, for `vui.doctor` and log lines."""
    cap = compute_capability()
    return {
        "gpu": gpu_name(),
        "compute_capability": None if cap is None else f"{cap[0]}.{cap[1]}",
        "torch_version": torch.__version__,
        "torch_arch_list": (
            list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []
        ),
        "torch_supports_gpu": torch_supports_this_gpu(),
        "dtype": str(dtype()).replace("torch.", ""),
        "dtype_forced": bool(os.environ.get("VUI_DTYPE", "").strip()),
        "bf16_native": supports_bf16(),
    }
