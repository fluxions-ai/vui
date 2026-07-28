__version__ = "1.0.0"


def _preload_nvidia_npp() -> None:
    """Preload NVIDIA NPP libs so torchcodec's core*.so dlopen chain resolves.

    torchcodec 0.10.0+cu128's `libtorchcodec_core7.so` has `libnppicc.so.12`
    as a NEEDED dependency but no RPATH pointing at `nvidia/npp/lib`. The
    symbol isn't in libtorch's RPATH either. Pulling the libs into the process
    address space via ctypes.CDLL(RTLD_GLOBAL) BEFORE torchcodec imports makes
    dlopen find them via the already-loaded-libs cache.

    Installing `nvidia-npp-cu12` alone isn't enough — its .so files sit in
    `nvidia/npp/lib/` which no other package's RPATH covers.
    """
    import ctypes
    import glob
    import os
    import sysconfig

    site_packages = sysconfig.get_paths()["purelib"]
    npp_dir = os.path.join(site_packages, "nvidia", "npp", "lib")
    if not os.path.isdir(npp_dir):
        # nvidia-npp-cu12 not installed — skip silently.
        return
    for so in sorted(glob.glob(os.path.join(npp_dir, "libnpp*.so.12"))):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_nvidia_npp()
del _preload_nvidia_npp


def _preload_ffmpeg() -> None:
    """Preload a user-local ffmpeg so torchcodec's runtime dlopen resolves.

    Rootless installs have no system ffmpeg to find; install.sh caches an LGPL
    shared build under ~/.cache/vui/ffmpeg. See vui.ffmpeg_libs. No-op when the
    system supplies ffmpeg (nothing cached) or on macOS.
    """
    try:
        from vui.ffmpeg_libs import preload
    except Exception:
        # A preload helper must never break `import vui`.
        return
    preload()


_preload_ffmpeg()
del _preload_ffmpeg
