"""Ollama LLM calling: prefill + streaming completion.

Kept separate from the aiohttp plumbing in `server.py` so prompt logic can
be tweaked without touching the WebRTC/queue code.

Public API:
  - `llm_prefill_system`, `llm_prefill_user` \u2014 warm Ollama's KV cache
  - `llm_next_chunk` \u2014 legacy num_predict-based polling (retained; see note)
  - `llm_stream_chunks` \u2014 preferred streaming chunker that yields sentence-
    bounded text + an is_final flag, guaranteeing terminal punctuation on
    the last chunk.

`llm_next_chunk` with the assistant-content continuation loop is the
source of the duplication bug observed with models like glm-4.7-flash that
don't cleanly continue an assistant-role message (see project memory).
Prefer `llm_stream_chunks` for any new caller.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import httpx

from vui.serving.stream.llm_backend import get_backend
from vui.serving.stream.prompts import SOUL

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
MLX_MODEL_NAME = "qwen3.5-4b-mlx"
MLX_MODEL_HF = "Qwen/Qwen3.5-4B"
MLX_MODEL_DIR = Path.home() / "models" / "qwen3.5-4b"
GGUF_MODEL_NAME = "qwen3.5:4b"
DEFAULT_OLLAMA_MODEL = MLX_MODEL_NAME if IS_APPLE_SILICON else GGUF_MODEL_NAME

LLM_CHUNK_TOKENS = 15  # ~5 words per chunk (legacy llm_next_chunk only)
_SENT_END_RE = re.compile(r'[.!?]+(?:["\'\)\]]+)?(?=\s|\[|$)')


def _ollama_running() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/version", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _ollama_model_exists(name: str) -> bool:
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/show", json={"name": name}, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_mlx_model() -> None:
    """Download + create the MLX safetensors model if it doesn't exist."""
    if not _ollama_running():
        raise RuntimeError(
            "Ollama is not running \u2014 start it first (brew services start ollama or open Ollama.app)"
        )
    if _ollama_model_exists(MLX_MODEL_NAME):
        return
    print(f"[mlx] Setting up {MLX_MODEL_NAME} (first run only)...")
    if not (MLX_MODEL_DIR / "config.json").exists():
        print(f"[mlx] Downloading {MLX_MODEL_HF}...")
        MLX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["hf", "download", MLX_MODEL_HF, "--local-dir", str(MLX_MODEL_DIR)],
            check=True,
        )
    modelfile = MLX_MODEL_DIR / "Modelfile"
    modelfile.write_text(
        f"FROM {MLX_MODEL_DIR}\n"
        "RENDERER qwen3.5\n"
        "PARSER qwen3\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_k 20\n"
        "PARAMETER top_p 0.95\n"
    )
    print(f"[mlx] Creating {MLX_MODEL_NAME} (int4 quantization)...")
    subprocess.run(
        [
            "ollama",
            "create",
            MLX_MODEL_NAME,
            "--experimental",
            "--quantize",
            "int4",
            "-f",
            str(modelfile),
        ],
        check=True,
        cwd=str(MLX_MODEL_DIR),
    )
    print(f"[mlx] {MLX_MODEL_NAME} ready")


async def llm_prefill_system(
    system_prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,  # kept for signature compat; backend owns model
) -> None:
    import time as _time

    del model
    backend = get_backend()
    t0 = _time.perf_counter()
    await backend.prefill([{"role": "system", "content": system_prompt}])
    print(
        f"[timing] System prompt prefilled ({backend.name}/{backend.model}): "
        f"{_time.perf_counter() - t0:.3f}s"
    )


async def llm_prefill_user(
    conversation: list[dict],
    system_prompt: str = SOUL,
    model: str = DEFAULT_OLLAMA_MODEL,  # kept for signature compat
) -> None:
    """Prefill backend KV cache with conversation so far (no generation)."""
    del model
    backend = get_backend()
    messages = [{"role": "system", "content": system_prompt}] + conversation
    await backend.prefill(messages)


async def llm_next_chunk(
    conversation: list[dict],
    system_prompt: str = SOUL,
    model: str = DEFAULT_OLLAMA_MODEL,
    num_predict: int = LLM_CHUNK_TOKENS,
) -> tuple[str, bool]:
    """LEGACY: get the next ~5 words from the LLM. Returns (text, is_done).

    Prefer `llm_stream_chunks` \u2014 this function's continuation pattern
    (pass assistant_so_far back in) duplicates output on models that don't
    cleanly continue assistant-role messages (e.g. glm-4.7-flash).
    """
    del model
    messages = [{"role": "system", "content": system_prompt}] + conversation
    backend = get_backend()
    res = await backend.complete(messages, max_tokens=num_predict)
    text = (res.get("content") or "").strip()
    is_done = res.get("done_reason") == "stop" or not text
    return text, is_done


async def llm_stream_chunks(
    conversation: list[dict],
    system_prompt: str = SOUL,
    model: str = DEFAULT_OLLAMA_MODEL,
    max_words: int = 20,
    stats: dict | None = None,
):
    """Stream Ollama's reply, yielding (chunk, is_final) at sentence boundaries.

    Splits on sentence ends ([.!?]) as soon as they appear. Falls back to
    a hard word-count split at `max_words` if no sentence boundary is
    found. The final chunk always ends with terminal punctuation.
    """
    messages = [{"role": "system", "content": system_prompt}] + conversation
    buf = ""
    pending: str | None = None
    # Stop sequences prevent the model from hallucinating the few-shot
    # tool-call format ([Results for: ...] / [You asked about: ...]) or
    # rolling into a fake follow-up turn (\nUser: / \nYou:). The actual
    # tool result is delivered as a SEPARATE LLM call from tasks.py
    # `deliver_pending_task_results`; the model must stop at the filler.
    # Inline tags like [laugh]/[sigh] still pass — they don't match these
    # capitalised prefixes.
    # `[Results / [You asked` block hallucination: the model parrots the
    # few-shot tool-call format inline. `\nUser: / \nYou:` stops mid-turn
    # role bleed. The "one sec." / "hold on." cuts after filler so the
    # model can't continue into hallucinated calendar/email data — the
    # actual data arrives via the separate relay LLM call.
    stop_seqs = [
        "[Results",
        "[You asked",
        "\nUser:",
        "\nYou:",
        "one sec.",
        "One sec.",
        "hold on.",
        "Hold on.",
        "give me a sec.",
        "Give me a sec.",
    ]
    del model  # backend owns the model id
    backend = get_backend()
    async for tok in backend.stream(
        messages, max_tokens=2048, stop=stop_seqs, stats=stats
    ):
        buf += tok
        while True:
            m = _SENT_END_RE.search(buf)
            if m is None:
                break
            chunk = buf[: m.end()].strip()
            buf = buf[m.end() :].lstrip()
            if not chunk:
                continue
            if pending is not None:
                yield pending, False
            pending = chunk
        if len(buf.split()) >= max_words:
            # Hard word-count fallback only — never break on commas (TTS
            # speaking a fragment ending mid-clause sounds disjointed).
            # Sentence boundaries (`.!?`) above are preferred; this only
            # fires when the LLM has emitted ≥max_words without one.
            words = buf.split()
            chunk = " ".join(words[:max_words]).rstrip(",;:")
            buf = " ".join(words[max_words:])
            if buf:
                buf = " " + buf
            if chunk:
                if pending is not None:
                    yield pending, False
                pending = chunk

    # Stream ended — finalise trailing text and emit final chunk.
    trailing = buf.strip()
    while trailing and trailing[-1] in ",;:- ":
        trailing = trailing[:-1].rstrip()
    if trailing and trailing[-1] not in ".!?\"')]":
        trailing = trailing + "."
    if pending is not None and trailing:
        yield pending, False
        yield trailing, True
    elif pending is not None:
        yield pending, True
    elif trailing:
        yield trailing, True
