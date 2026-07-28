"""Preflight check — `python -m vui.doctor`.

Answers "will this box run Vui, and if not, what do I do about it" in one
command, because the natural failure modes are all late and cryptic: a cu130
torch on Volta dies at the first kernel launch, flash-attn dies at the first
decode step, and a missing ffmpeg dies on the first audio decode.

Exit code is 0 when nothing is blocking, 1 when something is.
"""

from __future__ import annotations

import os
import shutil
import sys

# Status markers. Plain ASCII so this stays readable over ssh and in CI logs.
OK, WARN, FAIL = "ok  ", "warn", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.blocking = False

    def add(self, status: str, item: str, detail: str, remedy: str = "") -> None:
        if status == FAIL:
            self.blocking = True
        self.rows.append((status, item, detail, remedy))

    def render(self) -> str:
        w_item = max((len(r[1]) for r in self.rows), default=0)
        w_detail = max((len(r[2]) for r in self.rows), default=0)
        lines = []
        for status, item, detail, remedy in self.rows:
            line = f"  [{status}] {item:<{w_item}}  {detail:<{w_detail}}"
            lines.append(line.rstrip())
            if remedy:
                for rl in remedy.split("\n"):
                    lines.append(f"         {' ' * w_item}  -> {rl}")
        return "\n".join(lines)


def _check_torch(rep: Report) -> None:
    try:
        import torch
    except Exception as e:
        rep.add(FAIL, "torch", f"import failed: {e}", "uv sync")
        return

    rep.add(OK, "torch", torch.__version__)

    from vui import hardware

    cap = hardware.compute_capability()
    if cap is None:
        rep.add(
            WARN,
            "gpu",
            "no CUDA device",
            "The streaming server needs a GPU. CPU-only inference lives in cpu/.",
        )
        rep.add(OK, "dtype", str(hardware.dtype()).replace("torch.", ""))
        return

    cap_str = f"{cap[0]}.{cap[1]}"
    rep.add(OK, "gpu", f"{hardware.gpu_name()} (compute {cap_str})")

    supported = hardware.torch_supports_this_gpu()
    arches = ", ".join(torch.cuda.get_arch_list())
    if supported is False:
        rep.add(
            FAIL,
            "torch kernels",
            f"this torch has no sm_{cap[0]}{cap[1]} kernels",
            f"built for: {arches}\n"
            "Reinstall against an older CUDA, e.g.:\n"
            "  UV_TORCH_BACKEND=cu126 uv sync",
        )
    else:
        rep.add(OK, "torch kernels", f"sm_{cap[0]}{cap[1]} in {arches}")

    d = hardware.dtype()
    forced = os.environ.get("VUI_DTYPE", "").strip()
    note = f"{str(d).replace('torch.', '')}"
    if forced:
        rep.add(OK, "dtype", f"{note} (forced by VUI_DTYPE={forced})")
    elif hardware.supports_bf16():
        rep.add(OK, "dtype", f"{note} (native bf16)")
    else:
        rep.add(
            WARN,
            "dtype",
            f"{note} — no native bf16 below compute 8.0",
            "fp16 has a smaller exponent range; if you see NaNs, try VUI_DTYPE=fp32.",
        )


def _check_attention(rep: Report) -> None:
    try:
        from vui.flash_compat import HAS_FLASH_ATTN
    except Exception as e:
        rep.add(FAIL, "attention", f"vui.flash_compat import failed: {e}")
        return

    from vui import hardware

    cap = hardware.compute_capability()
    if cap is None:
        # No device, so neither path is exercised — saying "ok, flash-attn"
        # here would imply a working setup that hasn't been tested at all.
        rep.add(OK, "attention", "n/a without a CUDA device")
        return
    if HAS_FLASH_ATTN:
        if cap < (8, 0):
            rep.add(
                WARN,
                "attention",
                "flash-attn installed but unusable on compute "
                f"{cap[0]}.{cap[1]}",
                "It will fall back to SDPA on the first call. To skip the\n"
                "detour: VUI_ATTN=torch, or `uv sync` without --extra flash.",
            )
        else:
            rep.add(OK, "attention", "flash-attn")
    else:
        why = (
            "forced off by VUI_ATTN"
            if os.environ.get("VUI_ATTN", "").lower() in ("torch", "sdpa")
            else "not installed"
        )
        detail = f"PyTorch SDPA fallback ({why})"
        if cap >= (8, 0):
            rep.add(
                WARN,
                "attention",
                detail,
                "This GPU supports flash-attn and would be faster with it:\n"
                "  uv sync --extra flash",
            )
        else:
            rep.add(OK, "attention", detail)


def _check_ffmpeg(rep: Report) -> None:
    from vui import ffmpeg_libs

    try:
        import torchcodec.decoders  # noqa: F401  — importing it *is* the check
    except Exception as e:
        rep.add(
            FAIL,
            "ffmpeg",
            f"torchcodec cannot load: {e}",
            "./install.sh --native fetches shared libs into ~/.cache/vui/ffmpeg.\n"
            "See docs/rootless-install.md (no root required).",
        )
        return
    where = ffmpeg_libs.lib_dir()
    rep.add(OK, "ffmpeg", f"torchcodec OK ({where or 'system libs'})")


def _check_llm(rep: Report) -> None:
    import asyncio

    try:
        from vui.serving.stream.llm_backend import get_backend

        backend = get_backend()
    except ValueError as e:
        rep.add(FAIL, "llm", str(e), "Set VUI_LLM_BACKEND to 'ollama' or 'vllm'.")
        return
    except Exception as e:
        rep.add(WARN, "llm", f"backend not resolvable: {e}")
        return

    try:
        up = asyncio.run(backend.health())
    except Exception:
        up = False

    where = f"{backend.name} {backend.model} @ {backend.base_url}"
    if up:
        rep.add(OK, "llm", where)
    else:
        # Not blocking: the server serves TTS and ASR without an LLM, and the
        # UI's llm pill goes green on its own once one appears.
        rep.add(
            WARN,
            "llm",
            f"{where} — unreachable",
            "TTS/ASR work without it. Start a backend, or point\n"
            "VUI_VLLM_URL / VUI_OLLAMA_URL somewhere that's up.",
        )


def _check_disk(rep: Report) -> None:
    home = os.path.expanduser("~")
    try:
        free_gb = shutil.disk_usage(home).free / 1e9
    except Exception:
        return
    # Weights + CUDA wheels + an LLM land in $HOME; quotas bite before root does
    # on the shared boxes this tends to run on.
    if free_gb < 15:
        rep.add(
            WARN,
            "disk",
            f"{free_gb:.0f} GB free in {home}",
            "Weights and CUDA wheels need ~10 GB. Redirect with HF_HOME,\n"
            "UV_CACHE_DIR, VUI_FFMPEG_DIR if this is a quota'd home.",
        )
    else:
        rep.add(OK, "disk", f"{free_gb:.0f} GB free in {home}")


def main() -> int:
    rep = Report()
    for check in (_check_torch, _check_attention, _check_ffmpeg, _check_llm, _check_disk):
        try:
            check(rep)
        except Exception as e:
            rep.add(FAIL, check.__name__.removeprefix("_check_"), f"check crashed: {e}")

    print("vui doctor")
    print(rep.render())
    if rep.blocking:
        print("\nBlocking problems above — see the -> lines.")
        return 1
    print("\nNo blocking problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
