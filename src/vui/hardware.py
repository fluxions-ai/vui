"""Hardware capability detection — one place that decides dtype and attention.

The model was written for Ampere-and-newer (bf16 + FlashAttention-2). Neither is
available everywhere, and the failure modes are unhelpful: bf16 on a card
without native support is silently slow or raises deep in a kernel, and the
pinned flash-attn wheel carries cubins for sm_80/90/100/120 with no PTX, so an
older card gets "no kernel image is available" at the first decode step.

Compute capability floors that matter here:

    >= 8.0  Ampere+   bf16 native, FlashAttention-2 works
    7.5     Turing    no native bf16 -> fp32; no FA2 -> SDPA fallback
    7.0     Volta     as Turing, and needs a cu12 torch build (the cu130
                      wheels carry no sm_70 kernels)
    none    CPU/MPS   fp32

fp32 rather than fp16 below Ampere — see `dtype()` for why that isn't the
obvious choice it looks like.

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


def _parse_arch(a: str) -> tuple[int, int] | None:
    """'sm_86' -> (8, 6); 'sm_100' -> (10, 0). Last digit is the minor."""
    digits = a.removeprefix("sm_").removeprefix("compute_")
    if not digits.isdigit() or len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


@lru_cache(maxsize=1)
def torch_supports_this_gpu() -> bool | None:
    """Does the installed torch carry kernels this device can run?

    Not an exact match against `get_arch_list()`: CUDA cubins are
    forward-compatible *within* a major generation, so an sm_86 kernel runs on
    an sm_89 device (which is why a 4090 is fine on a build that never names
    sm_89). Embedded PTX for an equal-or-lower capability also works, via a
    JIT on first launch.

    None when there's no CUDA device to judge. False means the first kernel
    launch will fail — the cu130-on-Volta case — which is worth saying up front
    rather than discovering mid-decode.
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

    for a in arches:
        parsed = _parse_arch(a)
        if parsed is None:
            continue
        major, minor = parsed
        if a.startswith("compute_"):
            # PTX: JIT-able onto anything at least this capable.
            if (major, minor) <= cap:
                return True
        elif major == cap[0] and minor <= cap[1]:
            return True
    return False


@lru_cache(maxsize=1)
def dtype() -> torch.dtype:
    """The dtype to run the model in: bf16 where native, else fp32.

    Not fp16 below Ampere, even though that would be the obvious choice. fp16
    keeps bf16's precision but loses most of its exponent range (max 65504),
    and this model's activations exceed it: a decode in fp16 produces
    out-of-range sampled tokens and dies in `scatter_add_` with a device-side
    assert. Measured on a 4090 with `VUI_DTYPE=fp16` forced; the weights
    themselves fit fine (max |w| ~16), so it's activation overflow.

    fp32 is correct everywhere and about 7x slower than bf16+flash on the same
    card, which is the price of running at all on pre-Ampere. `VUI_DTYPE=fp16`
    is still accepted if you want to experiment with fixing it.
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
    return torch.float32


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
