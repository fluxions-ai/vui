"""HuggingFace + s3 download helper for model checkpoints."""

import os
import shutil
import subprocess
from pathlib import Path

HF_REPO = "fluxions/vui"
S3_CACHE = Path.home() / ".cache" / "vui" / "s3"


def download(filename: str, repo_id: str = HF_REPO) -> str:
    """Return a local path for `filename`.

    Behavior:
      - If `filename` is an existing local path, return it as-is.
      - If `filename` starts with s3:// / r2:// / sj://, fetch via s5cmd
        into ~/.cache/vui/s3/ and return the cached path.
      - Otherwise download from `huggingface.co/{repo_id}/{filename}` and
        return the cached path.
    """
    if os.path.exists(filename):
        return filename
    if filename.startswith(("s3://", "r2://", "sj://")):
        return _download_s3(filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename)


def _download_s3(url: str) -> str:
    if not shutil.which("s5cmd"):
        raise RuntimeError(
            f"s5cmd not found on PATH — install s5cmd or pre-download {url} locally"
        )
    S3_CACHE.mkdir(parents=True, exist_ok=True)
    key = url.split("://", 1)[1]
    parts = Path(key).parts
    local_name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    local = S3_CACHE / local_name
    if local.exists():
        return str(local)
    print(f"[hf] Fetching {url} -> {local}")
    subprocess.run(["s5cmd", "cp", url, str(local)], check=True)
    return str(local)
