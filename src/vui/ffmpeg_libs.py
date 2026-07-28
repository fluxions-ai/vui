"""User-local ffmpeg shared libs for torchcodec (rootless installs).

torchcodec ships `libtorchcodec_core{4..8}.so` — one per supported ffmpeg major
— and at import dlopen()s them in turn, keeping whichever loads. Each variant
NEEDEDs exactly one ffmpeg generation's sonames (core7 -> libavcodec.so.61,
core6 -> .so.60, ...), so what has to resolve is ordinary NEEDED linking. On a
box with no root there is no system ffmpeg for the loader to find, so
install.sh drops an LGPL shared build under $VUI_FFMPEG_DIR
(default ~/.cache/vui/ffmpeg).

Stdlib only and torch-free: this is imported from vui/__init__.py before
anything else. Nothing here downloads — install.sh does the fetching.
"""

import ctypes
import os
import sys
from pathlib import Path

# Dependency order matters: the vendored build may carry no RUNPATH, so each
# lib's NEEDED entries have to already be in the loader map when it loads.
_LIBS = (
    "avutil",
    "swresample",
    "swscale",
    "avcodec",
    "avformat",
    "avfilter",
    "avdevice",
)


def lib_dir() -> Path | None:
    """The vendored lib dir, or None if install.sh never fetched one."""
    root = Path(
        os.environ.get("VUI_FFMPEG_DIR") or Path.home() / ".cache" / "vui" / "ffmpeg"
    )
    d = root / "lib"
    return d if any(d.glob("libavcodec.so.*")) else None


def preload() -> bool:
    """CDLL(RTLD_GLOBAL) the vendored libs so libtorchcodec_core*.so links.

    glibc satisfies a NEEDED "libavcodec.so.61" from objects already in the
    link map, matching on DT_SONAME — so loading them by absolute path first is
    enough, and no LD_LIBRARY_PATH is required. Same trick as
    `vui._preload_nvidia_npp`. No-op when there's no vendored copy (system
    ffmpeg, or macOS).
    """
    if sys.platform != "linux":
        return False
    d = lib_dir()
    if d is None:
        return False
    ok = False
    for stem in _LIBS:
        # Prefer libfoo.so.61 (the soname) over libfoo.so.61.19.101 or libfoo.so,
        # so the SONAME identity dlopen matches on is unambiguous.
        cands = sorted(d.glob(f"lib{stem}.so.[0-9]*"), key=lambda p: len(p.name))
        if not cands:
            continue
        try:
            ctypes.CDLL(str(cands[0]), mode=ctypes.RTLD_GLOBAL)
            ok = True
        except OSError as e:
            print(f"[ffmpeg] preload failed for {cands[0].name}: {e}", file=sys.stderr)
    return ok


def selftest() -> int:
    """Exit 0 iff torchcodec is usable. install.sh runs this after `uv sync`.

    The only thing that proves ffmpeg is present is torchcodec importing, so
    that's the check. Invoke as:

        python -c 'import vui.ffmpeg_libs as m; raise SystemExit(m.selftest())'

    (rather than `python -m`, which warns about the double import — vui's
    package __init__ has already imported this module by then.)
    """
    preload()
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        from torchcodec.encoders import AudioEncoder  # noqa: F401
    except Exception as exc:
        print(f"torchcodec unusable: {exc}", file=sys.stderr)
        return 1
    print(f"torchcodec OK (ffmpeg: {lib_dir() or 'system'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
