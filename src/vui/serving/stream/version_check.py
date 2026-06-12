import os
import subprocess

import httpx

from vui.serving.stream._log import _slog

GITHUB_API = "https://api.github.com/repos/{repo}/commits/{ref}"


def _local_sha() -> str | None:
    sha = os.environ.get("VUI_GIT_SHA")
    if sha and sha.strip():
        return sha.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


async def check_for_updates() -> None:
    if os.environ.get("VUI_UPDATE_CHECK", "1") == "0":
        return
    repo = os.environ.get("VUI_UPDATE_REPO", "fluxions-ai/vui")
    ref = os.environ.get("VUI_UPDATE_REF", "main")
    try:
        local = _local_sha()
        url = GITHUB_API.format(repo=repo, ref=ref)
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                url, headers={"Accept": "application/vnd.github+json"}
            )
            resp.raise_for_status()
            data = resp.json()
        remote = data.get("sha")
        if not remote:
            raise ValueError("no sha in GitHub response")
        if local is None:
            _slog(f"[update] Latest {ref} is {remote[:8]} (local SHA unknown)")
        elif local == remote or local.startswith(remote) or remote.startswith(local):
            _slog(f"[update] Up to date ({remote[:8]})")
        else:
            msg = (data.get("commit") or {}).get("message", "").splitlines()
            headline = msg[0] if msg else ""
            _slog(
                f"[update] New version available: {remote[:8]} (local {local[:8]}). "
                f"Run ./install.sh --upgrade"
            )
            if headline:
                _slog(f"[update]   latest: {headline}")
    except Exception as e:
        _slog(f"[update] Failed to check for updates: {e}")
