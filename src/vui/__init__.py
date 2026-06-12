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
