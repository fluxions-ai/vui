"""Hardware-capability policy: dtype selection and attention dispatch.

Capabilities are faked, so this runs on any box — including CPU-only CI, which
is the point: the decisions being tested are exactly the ones nobody can
exercise without a shelf of old GPUs.
"""

import importlib
import sys
from unittest import mock

import pytest
import torch

from vui import hardware


def _fake_gpu(cap):
    """Patch torch.cuda to look like a device of the given capability.

    cap=None means no CUDA at all.
    """
    return (
        mock.patch.object(torch.cuda, "is_available", lambda: cap is not None),
        mock.patch.object(torch.cuda, "get_device_capability", lambda *a: cap),
    )


def _resolve_dtype(cap, env=None):
    avail, devcap = _fake_gpu(cap)
    with avail, devcap, mock.patch.dict("os.environ", env or {}, clear=False):
        for fn in (
            hardware.compute_capability,
            hardware.supports_bf16,
            hardware.dtype,
            hardware.gpu_name,
        ):
            fn.cache_clear()
        try:
            return hardware.dtype()
        finally:
            for fn in (
                hardware.compute_capability,
                hardware.supports_bf16,
                hardware.dtype,
                hardware.gpu_name,
            ):
                fn.cache_clear()


# ------------------------------------------------------------------- dtype


@pytest.mark.parametrize(
    "cap,expected",
    [
        ((9, 0), torch.bfloat16),  # Hopper H100
        ((8, 6), torch.bfloat16),  # RTX 3090
        ((8, 0), torch.bfloat16),  # Ampere A100 — the bf16 floor
        ((7, 5), torch.float16),  # Turing T4: no native bf16
        ((7, 0), torch.float16),  # Volta V100
        (None, torch.float32),  # CPU
    ],
)
def test_dtype_follows_compute_capability(cap, expected):
    assert _resolve_dtype(cap) is expected


@pytest.mark.parametrize("forced,expected", [
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
    ("fp32", torch.float32),
])
def test_vui_dtype_overrides_detection(forced, expected):
    """The override wins even where the hardware disagrees — it's the escape
    hatch for fp16 NaNs and for testing the other paths."""
    assert _resolve_dtype((7, 0), {"VUI_DTYPE": forced}) is expected


def test_unknown_vui_dtype_is_rejected_loudly():
    with pytest.raises(ValueError, match="bf16, fp16, fp32"):
        _resolve_dtype((8, 0), {"VUI_DTYPE": "float8"})


def test_bf16_native_only_from_ampere():
    for cap, want in [((7, 5), False), ((8, 0), True), ((9, 0), True), (None, False)]:
        avail, devcap = _fake_gpu(cap)
        with avail, devcap:
            hardware.compute_capability.cache_clear()
            hardware.supports_bf16.cache_clear()
            assert hardware.supports_bf16() is want, cap
    hardware.compute_capability.cache_clear()
    hardware.supports_bf16.cache_clear()


# --------------------------------------------------------------- attention


def _attention_impl(cap, flash_installed):
    """Which implementation flash_compat settles on, for a given device."""
    sys.modules.pop("vui.flash_compat", None)
    import vui.flash_compat as fc

    importlib.reload(fc)
    # Force the "is flash-attn importable" answer both ways — this venv may or
    # may not actually have it, and the test needs to pin both branches.
    if flash_installed:
        fc._impl = mock.Mock(name="flash_kernel")
        fc.HAS_FLASH_ATTN = True
    else:
        fc._impl = fc._sdpa_attn_with_kvcache
        fc.HAS_FLASH_ATTN = False
    fc._checked_capability = False

    avail, devcap = _fake_gpu(cap)
    with avail, devcap:
        uses_flash = fc._impl is not fc._sdpa_attn_with_kvcache
        if uses_flash and not fc._capability_permits_flash():
            fc._impl = fc._sdpa_attn_with_kvcache
        return "flash" if fc._impl is not fc._sdpa_attn_with_kvcache else "sdpa"


@pytest.mark.parametrize(
    "cap,flash_installed,expected",
    [
        # FlashAttention-2 has no pre-Ampere kernels: don't even try, rather
        # than provoking "no kernel image is available" at the first decode.
        ((7, 0), True, "sdpa"),
        ((7, 5), True, "sdpa"),
        # Ampere and up use the real kernel when it's installed.
        ((8, 0), True, "flash"),
        ((8, 6), True, "flash"),
        ((9, 0), True, "flash"),
        # ...and fall back cleanly when it isn't.
        ((8, 0), False, "sdpa"),
        ((9, 0), False, "sdpa"),
        (None, True, "sdpa"),
        (None, False, "sdpa"),
    ],
)
def test_attention_dispatch(cap, flash_installed, expected):
    assert _attention_impl(cap, flash_installed) == expected


def test_capability_check_is_permissive_when_it_cannot_tell():
    """If the capability query raises, proceed and let the launch-error catch
    handle it — that's the Jetson path, where capability looks fine anyway."""
    sys.modules.pop("vui.flash_compat", None)
    import vui.flash_compat as fc

    importlib.reload(fc)
    def boom():
        raise RuntimeError("driver said no")

    with mock.patch.object(torch.cuda, "is_available", boom):
        assert fc._capability_permits_flash() is True
