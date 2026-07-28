"""Run the hardware checks on a Turing T4 via Modal.

The dtype and attention policies are unit-tested against faked capabilities and
were measured on an Ampere card by forcing the paths a pre-Ampere card would
take. This runs them on a card that actually *is* pre-Ampere — a T4 (sm_75) —
which is the last untested variable for that class of GPU.

A T4 exercises, for real rather than by simulation:
  * flash-attn genuinely absent for the arch (its wheels are sm_80+),
  * no native bf16, so the fp32 path,
  * torch cu130 on sm_75, which *is* in its arch list (unlike Volta sm_70).

The image deliberately has no ffmpeg, so the rootless `vui.ffmpeg_libs` fetch is
exercised too.

    uv run --no-project --with modal modal run tests/modal_t4.py

Roughly 10 minutes on a T4, most of it `uv sync` and the checkpoint download.
"""

import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    # curl for the ffmpeg fetch, git because uv wants it for some sources.
    # Deliberately NOT ffmpeg: the rootless fetch is part of what's under test.
    .apt_install("curl", "git", "xz-utils")
    .pip_install("uv>=0.9")
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local", "HF_HOME": "/cache/hf"})
    .add_local_dir(
        REPO,
        remote_path="/vui",
        ignore=[".venv", ".git", "__pycache__", ".pytest_cache", "*.pyc", "prompts"],
        copy=True,
    )
    .run_commands(
        # moonshine for WER scoring, pytest for the unit tests. No --extra
        # flash: a T4 can't run those kernels, and install.sh would make the
        # same choice from the detected capability.
        "cd /vui && uv sync --extra moonshine",
    )
)

app = modal.App("vui-hw-t4", image=image)
cache = modal.Volume.from_name("vui-hw-cache", create_if_missing=True)


def _sh(cmd: str, **kw) -> int:
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd="/vui", **kw).returncode


@app.function(gpu="T4", timeout=3600, volumes={"/cache": cache})
def hardware_check():
    print("=" * 72, flush=True)
    _sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv")

    print("\n" + "=" * 72 + "\nDOCTOR (before any ffmpeg fetch)", flush=True)
    doctor_rc = _sh("python -m vui.doctor")

    # The image has no ffmpeg, so this is the rootless fetch under test.
    print("\n" + "=" * 72 + "\nROOTLESS FFMPEG FETCH", flush=True)
    check = "import vui.ffmpeg_libs as m; raise SystemExit(m.selftest())"
    if _sh(f'python -c "{check}"') != 0:
        _sh(
            "mkdir -p /root/.cache/vui/ffmpeg && "
            "curl -fL --retry 3 https://github.com/BtbN/FFmpeg-Builds/releases/download/"
            "latest/ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz "
            "| tar -xJ --strip-components=1 -C /root/.cache/vui/ffmpeg"
        )
        rc = _sh(f'python -c "{check}"')
        print(f"\nffmpeg usable after fetch: {rc == 0}", flush=True)

    print("\n" + "=" * 72 + "\nDOCTOR (after)", flush=True)
    doctor_rc = _sh("python -m vui.doctor")

    print("\n" + "=" * 72 + "\nRESOLVED POLICY", flush=True)
    _sh(
        'python -c "'
        "import json; from vui import hardware; "
        "import vui.flash_compat as fc; "
        "s = hardware.summary(); s['flash_importable'] = fc.HAS_FLASH_ATTN; "
        'print(json.dumps(s, indent=2))"'
    )

    print("\n" + "=" * 72 + "\nBF16 ON THIS DEVICE", flush=True)
    _sh(
        'python -c "'
        "import torch; "
        "print('is_bf16_supported:', torch.cuda.is_bf16_supported()); "
        "a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16); "
        "b = (a @ a).float(); "
        "print('bf16 matmul finite:', bool(b.isfinite().all()), 'max', float(b.abs().max()))"
        '"'
    )

    print("\n" + "=" * 72 + "\nUNIT TESTS", flush=True)
    _sh("python -m pytest tests/test_hardware.py tests/test_llm_backend.py -q")

    print("\n" + "=" * 72 + "\nRENDER MATRIX (real audio + WER)", flush=True)
    # One short reference clip, fetched rather than shipped (prompts/ is gitignored).
    _sh(
        "python -c \"from vui.hf import download; "
        "import shutil, os; os.makedirs('/vui/prompts', exist_ok=True); "
        "shutil.copy(download('prompts/abraham.wav'), '/vui/prompts/abraham.wav')\" "
        "|| echo 'prompt fetch failed — matrix will be skipped'"
    )
    if Path("/vui/prompts/abraham.wav").exists():
        # Twice: one WER reading per config is not evidence, sampling is
        # stochastic at temperature 0.7.
        matrix_rc = _sh("python tests/hardware_matrix.py --prompt prompts/abraham.wav")
        _sh("python tests/hardware_matrix.py --prompt prompts/abraham.wav")
    else:
        print("no prompt available; skipping render matrix", flush=True)
        matrix_rc = None

    print("\n" + "=" * 72, flush=True)
    print(f"doctor exit={doctor_rc}  matrix exit={matrix_rc}", flush=True)
    return {"doctor": doctor_rc, "matrix": matrix_rc}


@app.local_entrypoint()
def main():
    result = hardware_check.remote()
    print(f"\nresult: {result}")
    if result.get("matrix") not in (0, None):
        sys.exit(1)
