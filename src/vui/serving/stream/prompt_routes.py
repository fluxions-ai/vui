"""Prompt HTTP routes: upload, re-align, load, save, list, decode audio."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from vui.telemetry import record_voice_load

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer

QWEN_SR = 24000
PROMPTS_DIR = Path("prompts")
HF_PROMPTS_REPO = "fluxions/vui"
HF_PROMPTS_PREFIX = "prompts/"


def _list_hf_prompts() -> list[str]:
    """Return safetensors filenames (stems) under prompts/ in the HF repo. Empty on failure."""
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(HF_PROMPTS_REPO)
    except Exception as e:
        print(f"[prompts] HF list failed: {e}")
        return []
    return [
        Path(f).stem
        for f in files
        if f.startswith(HF_PROMPTS_PREFIX) and f.endswith(".safetensors")
    ]


def _fetch_hf_prompt(stem: str) -> Path | None:
    """Download prompts/{stem}.safetensors from HF into PROMPTS_DIR. Returns local path or None."""
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            HF_PROMPTS_REPO, f"{HF_PROMPTS_PREFIX}{stem}.safetensors"
        )
    except Exception as e:
        print(f"[prompts] HF download {stem}: {e}")
        return None
    PROMPTS_DIR.mkdir(exist_ok=True)
    dst = PROMPTS_DIR / f"{stem}.safetensors"
    if not dst.exists():
        import shutil

        shutil.copy(cached, dst)
    return dst


async def handle_upload_prompt(srv: StreamServer, request):
    t_start = time.perf_counter()
    reader = await request.multipart()
    field = await reader.next()
    data = await field.read()

    from julius.resample import resample_frac
    from torchcodec.decoders import AudioDecoder

    t_decode = time.perf_counter()
    raw_bytes = bytes(data)
    decoder_24k = AudioDecoder(raw_bytes, sample_rate=QWEN_SR, num_channels=1)
    audio_24k = decoder_24k.get_all_samples().data.squeeze(0)
    audio_dur = len(audio_24k) / QWEN_SR

    max_samples_24k = 180 * QWEN_SR
    if len(audio_24k) > max_samples_24k:
        audio_24k = audio_24k[:max_samples_24k]

    if audio_24k.abs().max() > 0:
        audio_24k = audio_24k / audio_24k.abs().max()

    audio_16k = resample_frac(audio_24k.unsqueeze(0), QWEN_SR, 16000).squeeze(0)
    t_decoded = time.perf_counter()
    await srv._log(
        f"Audio decode: {(t_decoded - t_decode)*1000:.0f}ms ({audio_dur:.1f}s audio)",
        "timing",
    )

    audio_16k_np = audio_16k.numpy()
    audio_24k_np = audio_24k.numpy()

    srv._last_transcription = None
    srv.asr_cmd_queue.put({"cmd": "transcribe_full", "audio": audio_16k_np})

    prompt_text = None
    for _ in range(1200):
        if srv._last_transcription is not None:
            prompt_text = srv._last_transcription
            srv._last_transcription = None
            break
        await asyncio.sleep(0.05)

    if not prompt_text:
        return web.json_response({"ok": False, "error": "ASR failed"}, status=500)

    await srv._log(f"Prompt ASR: '{prompt_text[:80]}'", "timing")

    srv._prompt_audio_16k = audio_16k_np
    srv._prompt_audio_24k = audio_24k_np

    await srv._block_ready("prompt")
    await srv._reset_session_state()
    srv.tts_cmd_queue.put({"cmd": "reset"})
    srv.tts_cmd_queue.put(
        {
            "cmd": "allocate_and_prefill",
            "audio_16k": audio_16k_np,
            "text": prompt_text,
            "settings": srv.session.settings,
            "audio_24k": audio_24k_np,
        }
    )
    srv.session.cancel_generation = False

    try:
        resp_msg = await srv._wait_tts_response("prompt_loaded", timeout=30)
    finally:
        await srv._unblock_ready("prompt")

    t_total = time.perf_counter() - t_start
    await srv._log(f"Total upload: {t_total*1000:.0f}ms", "timing")

    if resp_msg:
        srv.tts_T = resp_msg["T"]
        record_voice_load("clone")
        return web.json_response(
            {"ok": True, "T": resp_msg["T"], "text": prompt_text or ""}
        )
    return web.json_response({"ok": False, "error": "Prefill timeout"}, status=500)


async def handle_update_prompt_text(srv: StreamServer, request):
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return web.json_response({"ok": False, "error": "Text required"}, status=400)
    if srv._prompt_audio_16k is None:
        return web.json_response(
            {"ok": False, "error": "No prompt audio uploaded yet"}, status=400
        )

    await srv._log(f"Re-aligning prompt: '{text[:80]}'", "info")
    await srv._block_ready("prompt")
    await srv._reset_session_state()
    srv.tts_cmd_queue.put({"cmd": "reset"})
    srv.tts_cmd_queue.put(
        {
            "cmd": "allocate_and_prefill",
            "audio_16k": srv._prompt_audio_16k,
            "text": text,
            "settings": srv.session.settings,
            "audio_24k": srv._prompt_audio_24k,
        }
    )
    srv.session.cancel_generation = False

    try:
        resp_msg = await srv._wait_tts_response("prompt_loaded", timeout=30)
    finally:
        await srv._unblock_ready("prompt")
    if resp_msg:
        srv.tts_T = resp_msg["T"]
        record_voice_load("clone")
        return web.json_response({"ok": True, "T": resp_msg["T"], "text": text})
    return web.json_response({"ok": False, "error": "Re-align timeout"}, status=500)


async def handle_save_prompt(srv: StreamServer, request):
    data = await request.json()
    name = data.get("name", "").strip()
    if not name:
        return web.json_response({"ok": False, "error": "Name required"}, status=400)
    srv.tts_cmd_queue.put({"cmd": "save_kv", "name": name})
    from vui.serving.stream.server import _save_last_prompt

    safe_name = re.sub(r"[^\w\-]", "_", name)
    _save_last_prompt(safe_name)
    return web.json_response({"ok": True, "name": name})


async def handle_list_prompts(srv: StreamServer, request):
    PROMPTS_DIR.mkdir(exist_ok=True)
    prompts: list[dict] = []
    seen: set[str] = set()

    for f in sorted(PROMPTS_DIR.glob("*.safetensors")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        try:
            import json as _json

            from safetensors import safe_open

            with safe_open(f, framework="pt") as sf:
                cfg = _json.loads((sf.metadata() or {}).get("config", "{}"))
            prompts.append(
                {
                    "name": cfg.get("name", f.stem),
                    "file": f.stem,
                    "text": cfg.get("text", ""),
                    "T": int(cfg.get("T", 0)),
                    "source": "local",
                }
            )
        except Exception:
            pass

    # Also list prompts that only have wav+txt source (no safetensors yet).
    # `load_kv_by_name` regenerates the keyed `.pt` on first load. Without
    # this scan, prompts like `rhian.wav`+`rhian.txt` never appear in the UI.
    for wav in sorted(PROMPTS_DIR.glob("*.wav")):
        stem = wav.stem
        if stem in seen:
            continue
        txt = PROMPTS_DIR / f"{stem}.txt"
        if not txt.exists():
            continue
        seen.add(stem)
        try:
            text = txt.read_text().strip()
        except Exception:
            text = ""
        prompts.append(
            {"name": stem, "file": stem, "text": text, "T": 0, "source": "local"}
        )

    # And prompts that only have a keyed `{name}.{ckpt_id}.pt` (legacy
    # caches from a different checkpoint — still loadable, will regen).
    for pt in sorted(PROMPTS_DIR.glob("*.*.pt")):
        stem = pt.name.split(".", 1)[0]
        if stem in seen:
            continue
        seen.add(stem)
        prompts.append(
            {"name": stem, "file": stem, "text": "", "T": 0, "source": "local"}
        )

    # Merge in HF-hosted prompts not yet downloaded so users see them in the UI.
    remote_stems = await asyncio.to_thread(_list_hf_prompts)
    for stem in remote_stems:
        if stem in seen:
            continue
        seen.add(stem)
        prompts.append({"name": stem, "file": stem, "text": "", "T": 0, "source": "hf"})

    return web.json_response({"prompts": prompts})


async def handle_load_prompt(srv: StreamServer, request):
    data = await request.json()
    file_name = data.get("file", "").strip()
    if not file_name:
        return web.json_response({"ok": False, "error": "File required"}, status=400)

    has_local = (
        (PROMPTS_DIR / f"{file_name}.safetensors").exists()
        or (PROMPTS_DIR / f"{file_name}.pt").exists()
        or (
            (PROMPTS_DIR / f"{file_name}.wav").exists()
            and (PROMPTS_DIR / f"{file_name}.txt").exists()
        )
        or any(PROMPTS_DIR.glob(f"{file_name}.*.pt"))
    )
    if not has_local:
        fetched = await asyncio.to_thread(_fetch_hf_prompt, file_name)
        if fetched is None:
            return web.json_response(
                {"ok": False, "error": f"Prompt {file_name!r} not found"}, status=404
            )

    await srv._block_ready("prompt")
    await srv._reset_session_state()
    srv._prompt_audio_16k = None
    srv._prompt_audio_24k = None
    srv.tts_cmd_queue.put({"cmd": "load_kv", "file": file_name})
    from vui.serving.stream.server import _save_last_prompt

    _save_last_prompt(file_name)
    srv.session.cancel_generation = False

    try:
        resp_msg = await srv._wait_tts_response("kv_loaded", timeout=30)
    finally:
        await srv._unblock_ready("prompt")

    if resp_msg and resp_msg.get("ok"):
        srv.tts_T = resp_msg["T"]
        record_voice_load(file_name)
        return web.json_response(
            {
                "ok": True,
                "name": resp_msg.get("name", file_name),
                "T": resp_msg["T"],
                "text": resp_msg.get("text", ""),
            }
        )
    return web.json_response({"ok": False, "error": "Load failed"}, status=500)


async def handle_prompt_audio(srv: StreamServer, request):
    srv.tts_cmd_queue.put({"cmd": "decode_prompt"})
    resp = await srv._wait_tts_response("prompt_audio", timeout=30)
    if resp and resp.get("audio") is not None:
        from torchcodec.encoders import AudioEncoder

        wav = resp["audio"]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        AudioEncoder(wav.float(), sample_rate=resp["sample_rate"]).to_file(tmp.name)
        audio_bytes = open(tmp.name, "rb").read()
        os.unlink(tmp.name)
        return web.Response(
            body=audio_bytes,
            content_type="audio/wav",
            headers={
                "X-Codes-Shape": "x".join(str(s) for s in resp.get("codes_shape", [])),
            },
        )
    return web.json_response({"error": "No prompt loaded"}, status=404)


def bind(cls):
    cls.handle_upload_prompt = lambda self, *a, **kw: handle_upload_prompt(
        self, *a, **kw
    )
    cls.handle_update_prompt_text = lambda self, *a, **kw: handle_update_prompt_text(
        self, *a, **kw
    )
    cls.handle_save_prompt = lambda self, *a, **kw: handle_save_prompt(self, *a, **kw)
    cls.handle_list_prompts = lambda self, *a, **kw: handle_list_prompts(self, *a, **kw)
    cls.handle_load_prompt = lambda self, *a, **kw: handle_load_prompt(self, *a, **kw)
    cls.handle_prompt_audio = lambda self, *a, **kw: handle_prompt_audio(self, *a, **kw)
