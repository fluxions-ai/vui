"""Where voice-prompt files live and how to read them.

A prompt is `<voice>.safetensors` holding codec `codes` (checkpoint-agnostic)
plus a baked `cond_bias` / `spk_token_emb` pair (checkpoint-specific), with the
exact transcript in the safetensors metadata (`config.text`). The Python engine
and the streaming server only need codes + transcript; the `cpu/` C engine and
the MLX engine also consume the baked pair, so each checkpoint has its own
folder on the Hub:

    prompts/                 baked for vui-nano (also the .wav sources + legacy .txt)
    prompts/vui-nano-1.1/    baked for vui-nano-1.1

`scripts/build_prompts.py` regenerates a folder for any checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

HF_REPO = "fluxions/vui"

# Checkpoint file -> Hub folder holding prompts baked for it. Anything not
# listed (vui-nano, vui-190k, local paths) falls back to the root `prompts/`.
PROMPT_FOLDERS = {
    "vui-nano-1.1.safetensors": "prompts/vui-nano-1.1",
}


def prompt_folder(checkpoint: str | Path | None) -> str:
    """Hub folder of the prompt set baked for `checkpoint` (a name, filename or path)."""
    if not checkpoint:
        return "prompts"
    base = Path(str(checkpoint)).name
    if not base.endswith(".safetensors"):
        base = f"{base}.safetensors"
    return PROMPT_FOLDERS.get(base, "prompts")


def read_prompt_metadata(safetensors_path: str | Path) -> dict:
    """The JSON `config` stored in a prompt safetensors' metadata ({} if absent)."""
    from safetensors import safe_open

    with safe_open(str(safetensors_path), "pt") as f:
        meta = f.metadata() or {}
    try:
        return json.loads(meta.get("config", "{}"))
    except json.JSONDecodeError:
        return {}


def prompt_transcript(path: str | Path) -> str | None:
    """Exact transcript for a prompt file.

    `path` may be the `.safetensors`, the `.wav`, or a `.txt`; the transcript is
    taken from the sibling `.safetensors` metadata first, then a sibling `.txt`.
    Returns None if neither exists — callers fall back to ASR.
    """
    p = Path(path)
    st = p.with_suffix(".safetensors")
    if st.exists():
        text = read_prompt_metadata(st).get("text", "")
        if text.strip():
            return text.strip()
    txt = p.with_suffix(".txt")
    if txt.exists():
        return txt.read_text().strip()
    return None


def hub_prompt(voice: str, checkpoint: str | Path | None = None) -> str:
    """Download `<voice>.safetensors` for `checkpoint` from the Hub; returns the local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, f"{prompt_folder(checkpoint)}/{voice}.safetensors")


def hub_prompt_transcript(voice: str, local_st: str | Path) -> str:
    """Transcript for a Hub voice: metadata of the downloaded file, else the legacy root `.txt`."""
    text = prompt_transcript(local_st)
    if text:
        return text
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, f"prompts/{voice}.txt")).read_text().strip()
