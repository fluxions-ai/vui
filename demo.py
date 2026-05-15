"""TTS demo with training-style text chunking.

Auto-detects platform: uses MLX on Apple Silicon, CUDA elsewhere.

Usage:
    python demo.py [checkpoint]                    # Gradio web UI
    python demo.py --render [checkpoint]           # Interactive CLI
    python demo.py --render [checkpoint] "text"    # Render text and exit
"""

import argparse
import json
import platform
import sys

# Serialize concurrent prompt preparation (gradio fires audio.change and
# text_to_speech in parallel; without this both race and encode twice).
import threading as _threading
import time
import warnings
from pathlib import Path

import gradio as gr
import torch

from vui.inference import chunk_text, simple_clean
from vui.qwen_codec import SAMPLE_RATE as QWEN_SR

_parser = argparse.ArgumentParser(description="VUI TTS Demo")
_parser.add_argument(
    "checkpoint", nargs="?", default=None, help="Checkpoint path or S3 URL"
)
_parser.add_argument("--render", action="store_true", help="CLI render mode")
_parser.add_argument(
    "--text", "-t", nargs="+", help="Text to render (non-interactive mode)"
)
_parser.add_argument(
    "--prompt", "-p", default="prompts/good_prompt3.wav", help="Voice prompt wav file"
)
_parser.add_argument(
    "--temperature", type=float, default=None, help="Sampling temperature"
)
_parser.add_argument(
    "--n-codebooks", type=int, default=None, help="Number of codebooks (2-16)"
)
_parser.add_argument(
    "--max-secs", type=float, default=None, help="Max audio duration in seconds"
)
_parser.add_argument(
    "--eos-threshold", type=float, default=None, help="EOS detection threshold"
)
_args = _parser.parse_args()

RENDER_MODE = _args.render

# CLI render: exit early before heavy imports
if RENDER_MODE and __name__ == "__main__":
    from vui.demo.cli import run as cli_run

    checkpoint_path = _args.checkpoint or "vui-nano.safetensors"
    render_text = " ".join(_args.text) if _args.text else None
    overrides = {}
    if _args.temperature is not None:
        overrides["temperature"] = _args.temperature
    if _args.n_codebooks is not None:
        overrides["n_codebooks"] = _args.n_codebooks
    if _args.max_secs is not None:
        overrides["max_secs"] = _args.max_secs
    if _args.eos_threshold is not None:
        overrides["eos_threshold"] = _args.eos_threshold
    cli_run(checkpoint_path, prompt_file=_args.prompt, text=render_text, **overrides)
    sys.exit(0)


warnings.filterwarnings("ignore", message="Online softmax is disabled")


# --- Platform detection ---
IS_APPLE_SILICON = platform.machine() == "arm64" and sys.platform == "darwin"
HAS_CUDA = not IS_APPLE_SILICON and torch.cuda.is_available()
USE_MLX = IS_APPLE_SILICON

if USE_MLX:
    print(f"Apple Silicon detected ({platform.machine()}), using MLX backend")
    import mlx.core as mx
    import mlx.nn as mnn
    import numpy as np

    from vui.mlx.tts.generate import CODEC_HZ, compute_cond_bias
    from vui.mlx.tts.generate import generate as mlx_generate
    from vui.mlx.tts.generate import prefill_prompt
    from vui.mlx.tts.model import VuiMLX
    from vui.mlx.tts.weights import load_quantized
else:
    print("Using CUDA backend")
    torch.set_float32_matmul_precision("high")

    from vui.model import Vui
    from vui.qwen_codec import FRAME_RATE as CODEC_HZ  # 12.5
    from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder


# --- Text chunking (imported from vui.inference) ---


# --- MLX generation (Apple Silicon) ---

if USE_MLX:

    KV_DISK_CACHE = Path.home() / ".cache" / "vui" / "kv"
    KV_DISK_CACHE.mkdir(parents=True, exist_ok=True)

    def _kv_cache_path(prompt_hash: str, ckpt_path: str, precision: str) -> Path:
        ckpt_key = Path(ckpt_path).stem
        return KV_DISK_CACHE / f"{prompt_hash}_{ckpt_key}_{precision}.safetensors"

    def _save_kv_to_disk(path: Path, offset: int):
        flat = {}
        for i, c in enumerate(model.decoder.kv_caches):
            # Only save the populated region [:offset]
            flat[f"k_{i}"] = c.keys[..., :offset, :]
            flat[f"v_{i}"] = c.values[..., :offset, :]
        mx.save_safetensors(str(path), flat, metadata={"offset": str(offset)})
        print(f"  KV cache saved to {path}")

    def _load_kv_from_disk(path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            flat, metadata = mx.load(str(path), return_metadata=True)
            offset = int(metadata["offset"])
            model.decoder.make_cache()
            for i, c in enumerate(model.decoder.kv_caches):
                c.keys[..., :offset, :] = flat[f"k_{i}"]
                c.values[..., :offset, :] = flat[f"v_{i}"]
                c.offset = offset
            mx.eval([c.state for c in model.decoder.kv_caches])
            print(f"  KV cache loaded from {path} (offset={offset})")
            return offset
        except Exception as e:
            print(f"  Failed to load KV cache: {e}")
            return None

    def _prefill_segments_mlx(model, prompt_segments, spk_emb):
        """Prefill multi-segment prompt: [spk] [text_i] [audio_i] per segment."""
        from vui.inference import simple_clean

        model.decoder.make_cache()
        for text_i, codes_i in prompt_segments:
            if spk_emb is not None and model.spk_proj is not None:
                spk_token = model.spk_proj(spk_emb).reshape(1, 1, -1)
                model.decoder(spk_token)
            ids = model.text_tokenizer.encode(simple_clean(text_i))
            ids_mx = mx.array(np.array(ids, dtype=np.int32))
            model.decoder(model.token_emb(ids_mx[None]))
            # codes_i: (T, Q) torch tensor -> (T, Q) mx
            pc = mx.array(codes_i.cpu().numpy().astype(np.int32))
            model.decoder(model.audio_emb(pc)[None])
        mx.eval([c.state for c in model.decoder.kv_caches])

    def generate_chunked_mlx(
        model: VuiMLX,
        text: str,
        cond_bias: mx.array,
        prompt_codes=None,
        prompt_text: str | None = None,
        prompt_segments: list[tuple[str, torch.Tensor]] | None = None,
        temperature: float = 0.8,
        top_k: int = 300,
        max_secs: float = 120,
        sq_scores: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
        wps_score: float = 0.0,
        rep_penalty: float = 1.4,
        rep_window: int = 24,
        spk_emb=None,
        prompt_hash: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, list[dict]]:
        """MLX chunked generation matching the CUDA generate_chunked interface."""
        # MLX path doesn't support a second speaker prompt — always single.
        chunks = chunk_text(text, sentence_only=True, single_speaker=True)
        if not chunks:
            return None, []

        max_frames = int(max_secs * CODEC_HZ)
        max_per_turn = int(_max_turn_secs * CODEC_HZ)

        # Recompute cond_bias with current settings
        cond_bias = compute_cond_bias(
            model,
            sq=list(sq_scores) if any(v > 0 for v in sq_scores) else None,
            wps=wps_score,
        )

        all_codes = []
        total_frames = 0

        # Prefill the prompt ONCE and snapshot the cache offset.
        # Try to load precomputed KV state from disk (keyed by prompt + ckpt + precision).
        # For each subsequent chunk we just rewind cache.offset to this value;
        # the prompt K/V stays valid and the next chunk's writes overwrite the
        # previous chunk's region (causal attention only reads up to offset).
        kv_path = (
            _kv_cache_path(prompt_hash, checkpoint_path, _mlx_precision)
            if prompt_hash
            else None
        )
        prompt_offset = None
        model.decoder.reset_cache()
        if kv_path is not None:
            prompt_offset = _load_kv_from_disk(kv_path)
        if prompt_offset is None:
            if prompt_segments:
                _prefill_segments_mlx(model, prompt_segments, spk_emb)
            elif prompt_text and prompt_codes is not None:
                prefill_prompt(model, prompt_text, prompt_codes, spk_emb=spk_emb)
            else:
                model.decoder.make_cache()
            prompt_offset = model.decoder.cache_T
            if kv_path is not None and prompt_offset > 0:
                _save_kv_to_disk(kv_path, prompt_offset)

        for turn_idx, chunk in enumerate(chunks):
            if total_frames >= max_frames:
                break

            # Rewind cache to end-of-prompt — no re-prefill needed.
            for c in model.decoder.kv_caches:
                c.offset = prompt_offset

            # Per-turn speaker token: training format puts [spk] before every
            # turn's text, so re-inject before the generated chunk text.
            if spk_emb is not None and model.spk_proj is not None:
                spk_token = model.spk_proj(spk_emb).reshape(1, 1, -1)
                model.decoder(spk_token)
                mx.eval([c.state for c in model.decoder.kv_caches])

            # Prepend [SC] for speaker changes if not first chunk
            gen_text = simple_clean(chunk["text"])

            remaining = min(max_per_turn, max_frames - total_frames)
            codes_list, n_frames, elapsed = mlx_generate(
                model,
                gen_text,
                cond_bias,
                temperature=temperature,
                top_k=top_k,
                rep_penalty=rep_penalty,
                rep_window=rep_window,
                max_frames=remaining,
                compile_rq=True,
            )

            chunk["frames"] = n_frames
            chunk["secs"] = n_frames / CODEC_HZ
            total_frames += n_frames

            if codes_list:
                # Convert MLX codes to PyTorch (T, Q) format
                codes_np = np.stack([np.array(c) for c in codes_list])
                all_codes.append(torch.from_numpy(codes_np).long())

            print(
                f"  Turn {turn_idx+1}: {n_frames} frames ({n_frames/CODEC_HZ:.1f}s) in {elapsed:.2f}s"
            )

        if not all_codes:
            return None, chunks
        return torch.cat(all_codes, dim=0), chunks


# Generation logic now lives in vui.engine.Engine.


# --- Setup ---

CACHE_DIR = Path.home() / ".cache" / "vui"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = CACHE_DIR / "demo_settings.json"

DEFAULTS = {
    "temperature": 0.9,
    "top_k": 50,
    "use_top_p": False,
    "top_p": 1.0,
    "max_duration": 120,
    "sq_dns_sig": 0.0,
    "sq_dns_bak": 0.0,
    "sq_nq_noi": 0.0,
    "sq_nq_disc": 0.0,
    "sq_nq_col": 0.0,
    "sq_nq_loud": 5.0,
    "wps_score": 0.0,
    "rep_penalty": 1.1,
    "rep_window": 0,
    "chunk_words": 20,
    "n_codebooks": 0,
    "eos_threshold": 0.45,
    "compile_rq": False,
}


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
        except Exception:
            pass
    return dict(DEFAULTS)


def _save_settings(**kwargs):
    s = _load_settings()
    s.update(kwargs)
    SETTINGS_FILE.write_text(json.dumps(s))


S = _load_settings()


checkpoint_path = _args.checkpoint or "vui-nano.safetensors"

from vui.hf import download

checkpoint_path = download(checkpoint_path)

codec_enc = codec_dec = None
spk_enc = None
_max_turn_secs = 15.0

_mlx_precision = "float32"  # "float32", "int8", or "int4"


def _load_mlx_model(ckpt_path, precision="float32"):
    global model, _mlx_precision, _max_turn_secs
    m, cfg = load_quantized(ckpt_path, precision)
    _max_turn_secs = cfg.get("data", {}).get("max_secs", 15.0)
    m.rq_transformer.compile_forward()
    model = m
    _mlx_precision = precision
    flat = mnn.utils.tree_flatten(m.parameters())
    mem = sum(v.nbytes for _, v in flat) / 1e6
    print(f"MLX model loaded ({precision}, {mem:.0f}MB)")
    return m


if USE_MLX:
    print(f"Loading MLX model from {checkpoint_path}...")
    model = _load_mlx_model(checkpoint_path, _mlx_precision)

    # codec_enc / codec_dec are lazy — only loaded on first prompt encode / decode.
    codec_enc = None
    codec_dec = None
    CODEC_SR = QWEN_SR
else:
    from vui.model import Vui

    print(f"Loading model from {checkpoint_path}...")
    model = Vui.from_pretrained_inf(checkpoint_path).cuda()
    _max_turn_secs = model.config.data.max_secs
    if model.rq_transformer is None:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has no RQ-Transformer head; "
            "the legacy STFT codec path has been removed."
        )
    print("Loading Qwen codec...")
    codec_enc = QwenCodecEncoder.from_pretrained().cuda().half().eval()
    codec_dec = QwenCodecDecoder.from_pretrained().cuda().float().eval()
    CODEC_SR = QWEN_SR

# spk_enc is lazy — only loaded on first prompt spk_emb computation.
spk_enc = None


def _get_codec_enc():
    global codec_enc
    if codec_enc is None:
        from vui.qwen_codec import QwenCodecEncoder

        print("Loading Qwen codec encoder...")
        device = "mps" if USE_MLX else "cuda"
        dtype = torch.float32 if USE_MLX else torch.float16
        codec_enc = QwenCodecEncoder.from_pretrained().to(device).to(dtype).eval()
    return codec_enc


def _get_codec_dec():
    global codec_dec
    if codec_dec is None:
        from vui.qwen_codec import QwenCodecDecoder

        print("Loading Qwen codec decoder...")
        device = "mps" if USE_MLX else "cuda"
        codec_dec = QwenCodecDecoder.from_pretrained().to(device).float().eval()
    return codec_dec


def _get_spk_enc():
    global spk_enc
    if spk_enc is None and model.spk_proj is not None:
        from vui.qwen_spk_enc import QwenSpeakerEncoder

        print("Loading speaker encoder...")
        spk_enc = QwenSpeakerEncoder.from_pretrained()
    return spk_enc


def _encode_audio_chunk(audio_16k: torch.Tensor) -> torch.Tensor:
    """Codec-encode a 16kHz audio chunk, returns codes (T, Q)."""
    n_q = (
        model.rq_transformer.n_quantizers
        if USE_MLX
        else model.config.model.n_quantizers
    )
    from julius.resample import resample_frac

    audio_24k = resample_frac(audio_16k, 16000, QWEN_SR)
    device = "mps" if USE_MLX else "cuda"
    dtype = torch.float32 if USE_MLX else torch.float16
    audio_in = audio_24k.to(dtype).to(device).reshape(1, 1, -1)
    codes = _get_codec_enc().encode(audio_in)
    return codes[0, :n_q].T.long()


def encode_prompt_segments(
    audio_16k: torch.Tensor,
) -> tuple[list[tuple[str, torch.Tensor]], list[list[dict]]]:
    """Build multi-segment prompt + per-segment word timings.

    Returns ((text, codes), word_timings) — segments unchanged for callers
    that don't care; word_timings is parallel list[list[{"word", "start",
    "end", "fs"}]] in absolute audio time, used by `_apply_text_edits` for
    word-level codes trim on user edits.
    """
    from vui.prompt_utils import build_prompt_segments

    def _transcribe(a16: torch.Tensor) -> str:
        dur = a16.shape[-1] / 16000
        if USE_MLX:
            import mlx_whisper

            a_np = a16.numpy() if isinstance(a16, torch.Tensor) else a16
            if a16.shape[-1] <= 15 * 16000:
                from vui.mlx.asr.load import transcribe as moonshine_transcribe

                model_name = "mlx-moonshine"
                text = moonshine_transcribe(mx.array(a_np))
            else:
                model_name = "mlx-whisper-large-v3-turbo"
                text = mlx_whisper.transcribe(
                    a_np,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    language="en",
                    verbose=False,
                    initial_prompt=_asr_initial_prompt(),
                )["text"]
            print(f"  [ASR:{model_name} {dur:.1f}s] {text}")
            return text
        if a16.shape[-1] <= 15 * 16000:
            from vui.inference import asr

            model_name = "whisper-turbo (decode)"
            text = asr(a16, prompt=_asr_initial_prompt())
            print(f"  [ASR:{model_name} {dur:.1f}s] {text}")
            return text
        import whisper

        from vui import inference

        model_name = "whisper-turbo (long)"
        if inference.wm is None:
            inference.wm = whisper.load_model("turbo", "cuda")
        text = inference.wm.transcribe(
            a16.numpy() if isinstance(a16, torch.Tensor) else a16,
            language="en",
            verbose=False,
            initial_prompt=_asr_initial_prompt(),
        )["text"]
        print(f"  [ASR:{model_name} {dur:.1f}s] {text}")
        return text

    align_device = "cuda" if torch.cuda.is_available() else "cpu"
    triples = build_prompt_segments(
        audio_16k,
        encode_codes=_encode_audio_chunk,
        transcribe=_transcribe,
        align_device=align_device,
        return_timings=True,
    )
    segments = [(t, c) for t, c, _ in triples]
    word_timings = [w for _, _, w in triples]
    return segments, word_timings


# --- Prompt & KV cache ---

_prompt_cache: dict = {
    "hash": None,
    "segments": None,  # list[(str, Tensor)] — (text, codes) per ~10s chunk
    "baseline_texts": None,  # ASR-original texts; used to detect real user edits
    "spk_emb": None,
}
_kv_state: dict = {"key": None, "T": 0, "B": 0}


_prompt_prep_lock = _threading.Lock()


def _audio_hash(prompt_audio) -> str | None:
    if prompt_audio is None:
        return None
    sr, audio = prompt_audio
    import hashlib

    h = hashlib.sha256()
    h.update(str(sr).encode())
    h.update(audio.tobytes()[: sr * audio.itemsize * 10])
    return h.hexdigest()[:16]


def _prep_audio(prompt_audio) -> torch.Tensor | None:
    """Convert gradio audio (sr, np_array) to normalized float tensor at 16kHz."""
    if prompt_audio is None:
        return None
    sr, audio = prompt_audio
    audio_t = torch.from_numpy(audio).float()
    if audio_t.numel() == 0:
        return None
    if audio_t.abs().max() > 0:
        audio_t = audio_t / audio_t.abs().max()
    if len(audio_t.shape) > 1:
        audio_t = audio_t.mean(1)
    max_samples = int(180 * sr)  # 3 minutes
    if len(audio_t) > max_samples:
        audio_t = audio_t[:max_samples]
    if sr != 16000:
        from julius.resample import resample_frac

        audio_t = resample_frac(audio_t, sr, 16000)
    return audio_t


def _get_spk_emb(audio_16k: torch.Tensor) -> torch.Tensor | None:
    enc = _get_spk_enc()
    if enc is None:
        return None
    from julius.resample import resample_frac

    audio_24k = resample_frac(audio_16k, 16000, 24000)
    with torch.no_grad():
        emb = enc.embed(audio_24k, sr=24000)
    print(f"Speaker embedding: {emb.shape}")
    return emb


PROMPT_DISK_CACHE = CACHE_DIR / "prompts"
PROMPT_DISK_CACHE.mkdir(exist_ok=True)


def _prompt_disk_path(h: str) -> Path:
    ckpt_key = Path(checkpoint_path).stem if checkpoint_path else "none"
    return PROMPT_DISK_CACHE / f"{h}_{ckpt_key}.pt"


def _load_prompt_from_disk(h: str) -> dict | None:
    p = _prompt_disk_path(h)
    if not p.exists():
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        return blob
    except Exception as e:
        print(f"Failed to load cached prompt {h}: {e}")
        return None


def _save_prompt_to_disk(h: str, segments, spk_emb, word_timings=None):
    try:
        blob = {"segments": segments, "spk_emb": spk_emb}
        if word_timings is not None:
            blob["word_timings"] = word_timings
        torch.save(blob, _prompt_disk_path(h))
        print(f"Cached prompt to {_prompt_disk_path(h)}")
    except Exception as e:
        print(f"Failed to save cached prompt: {e}")


def _get_prompt(prompt_audio, cache: dict, label: str = "Prompt"):
    """Return (segments, spk_emb) with caching.

    segments is list[(str, Tensor)] — (text, codes) per ~10s chunk.
    """
    h = _audio_hash(prompt_audio)
    if h is None:
        cache["hash"] = None
        return None, None
    if h == cache["hash"]:
        return cache["segments"], cache["spk_emb"]

    # Serialize concurrent prep for the same prompt
    with _prompt_prep_lock:
        if h == cache["hash"]:
            return cache["segments"], cache["spk_emb"]

        audio_t = _prep_audio(prompt_audio)
        if audio_t is None:
            return None, None

        # Try disk cache first (segments + spk_emb persist across sessions)
        disk = _load_prompt_from_disk(h)
        if disk is not None:
            segments = disk["segments"]
            word_timings = disk.get("word_timings") or [[] for _ in segments]
            spk_emb_t = disk.get("spk_emb")
            if spk_emb_t is None:
                spk_emb_t = _get_spk_emb(audio_t)
                _save_prompt_to_disk(h, segments, spk_emb_t, word_timings)
            print(
                f"{label}: loaded cached prompt {h} "
                f"({len(segments)} segments, {sum(c.shape[0] for _, c in segments)} frames)"
            )
            cache.update(
                hash=h,
                segments=segments,
                word_timings=word_timings,
                baseline_texts=[t for t, _ in segments],
                spk_emb=spk_emb_t,
            )
            return segments, spk_emb_t

        print(f"{label}: encoding {audio_t.shape[-1]/16000:.1f}s audio...")
        segments, word_timings = encode_prompt_segments(audio_t)
        spk_emb_t = _get_spk_emb(audio_t)

        total_frames = sum(codes.shape[0] for _, codes in segments)
        print(f"{label}: {len(segments)} segments, {total_frames} frames total")

        _save_prompt_to_disk(h, segments, spk_emb_t, word_timings)
        cache.update(
            hash=h,
            segments=segments,
            word_timings=word_timings,
            baseline_texts=[t for t, _ in segments],
            spk_emb=spk_emb_t,
        )

        # Unload ASR model to free VRAM
        from vui import inference

        if inference.wm is not None:
            del inference.wm
            inference.wm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Unloaded ASR model")

        return segments, spk_emb_t


def get_prompt(prompt_audio):
    return _get_prompt(prompt_audio, _prompt_cache, "Prompt")


def _invalidate_prompt_caches():
    _prompt_cache.update(hash=None, segments=None, baseline_texts=None, spk_emb=None)


def invalidate_kv():
    _kv_state.update(key=None, T=0, B=0)
    # Engine owns the flash KV cache + manual CUDA graphs; each text_to_speech
    # call creates a fresh Row so the cache is rewritten in place anyway.


CODEC_CTX_FRAMES = 6  # ~500ms at 12.5Hz — minimum decode context

from vui.serving.stream import prompts as _vui_prompts


def _asr_initial_prompt() -> str:
    return f"{_vui_prompts.DEFAULT_ASSISTANT_NAME}. Um, uh, hmm, right."


def decode_audio(
    codes: torch.Tensor, n_codebooks: int = 0, prompt_codes: torch.Tensor | None = None
) -> tuple[torch.Tensor, int]:
    from vui.qwen_codec import DOWNSAMPLE_RATE

    dec = _get_codec_dec()
    device = "mps" if USE_MLX else "cuda"
    c = codes.T.unsqueeze(0).to(device)
    if n_codebooks > 0:
        c = c[:, :n_codebooks]
    ctx_frames = 0
    if prompt_codes is not None:
        pc = prompt_codes.T.unsqueeze(0).to(device)
        if n_codebooks > 0:
            pc = pc[:, :n_codebooks]
        ctx_codes = pc[:, :, -CODEC_CTX_FRAMES:]
        ctx_frames = ctx_codes.shape[2]
        c = torch.cat([ctx_codes, c], dim=2)
    with torch.no_grad():
        audio = dec.decode_chunked(c, ctx=CODEC_CTX_FRAMES)
    if ctx_frames > 0:
        audio = audio[..., ctx_frames * DOWNSAMPLE_RATE :]
    return audio[0, 0].detach().float().cpu(), CODEC_SR


_engine = None  # vui.engine.Engine (CUDA only)
_engine_row = None  # cached Row; prompt prefilled on demand

if USE_MLX:
    print("MLX warmup...")
    cond_bias_warmup = compute_cond_bias(model, sq=[0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    warmup_codes, _ = generate_chunked_mlx(
        model, "Hello warmup.", cond_bias_warmup, max_secs=3
    )
    if warmup_codes is not None:
        decode_audio(warmup_codes)
else:
    from vui.engine import Engine, GenConfig, Segment

    print("Building Engine (manual CUDA graphs + vocoder graph)...")
    _engine = Engine(model, _get_codec_dec(), max_rows=1, vocoder_ctx=25)
    print("Warmup...")
    with _engine.new_row() as _warm_row:
        _warm_row.render(
            "Hello, this is a warmup.", GenConfig(max_secs=3, temperature=0.7)
        )
        _warm_row.rewind()
        _warm_row.render(
            "Testing one two three.", GenConfig(max_secs=3, temperature=0.7)
        )

print("Ready!")


KNOWN_CHECKPOINTS = [
    "vui-nano.safetensors",
]

SAMPLE_TEXTS_PATH = Path(__file__).parent / "sample_texts.json"


def load_sample_texts() -> dict[str, str]:
    try:
        return json.loads(SAMPLE_TEXTS_PATH.read_text())
    except FileNotFoundError:
        return {}


def text_to_speech(
    text,
    prompt_audio,
    prompt_text_1_value,
    temperature,
    top_k_val,
    use_top_p,
    top_p_val,
    max_duration,
    sq_dns_sig,
    sq_dns_bak,
    sq_nq_noi,
    sq_nq_disc,
    sq_nq_col,
    sq_nq_loud,
    wps_score,
    rep_penalty,
    rep_window,
    chunk_words,
    eos_threshold,
    n_codebooks,
):
    n_codebooks = int(n_codebooks)
    chunk_words = int(chunk_words)
    if not text.strip():
        return None, "Enter some text."

    _save_settings(
        temperature=temperature,
        top_k=top_k_val,
        use_top_p=use_top_p,
        top_p=top_p_val,
        max_duration=max_duration,
        sq_dns_sig=sq_dns_sig,
        sq_dns_bak=sq_dns_bak,
        sq_nq_noi=sq_nq_noi,
        sq_nq_disc=sq_nq_disc,
        sq_nq_col=sq_nq_col,
        sq_nq_loud=sq_nq_loud,
        wps_score=wps_score,
        rep_penalty=rep_penalty,
        rep_window=int(rep_window),
        chunk_words=chunk_words,
        eos_threshold=eos_threshold,
        n_codebooks=n_codebooks,
    )

    top_p = top_p_val if use_top_p else None

    prompt_segs, prompt_spk_emb = get_prompt(prompt_audio)

    # Apply edited segment texts from the prompt textboxes.
    # One line per segment. Behavior:
    # - line count == segments: edit each line in place.
    #   * If the line is a strict word-prefix of the baseline, trim the segment's
    #     codes to the timestamp of the last kept word (drops trailing hallucinations).
    # - line count <  segments: trim trailing segments (drop hallucinated tail).
    # - line count >  segments: stale textbox (left over from a prior prompt) — skip.
    CODEC_HZ = 12.5

    def _word_prefix_count(edited_words: list[str], baseline_words: list[str]) -> int:
        """Length of the longest word-prefix match (case-insensitive,
        punctuation-stripped). Returns 0 if not a prefix."""
        if len(edited_words) > len(baseline_words):
            return 0
        norm = lambda w: w.strip(".,!?;:\"'").lower()
        for i, (e, b) in enumerate(zip(edited_words, baseline_words)):
            if norm(e) != norm(b):
                return 0
        return len(edited_words)

    def _apply_text_edits(segs, baseline_texts, word_timings, raw_text, label):
        if not segs or not raw_text:
            return False
        lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
        if len(lines) > len(segs):
            print(
                f"  Skipping {label} edits (stale: {len(lines)} lines > {len(segs)} segments)"
            )
            return False
        edited_any = False
        for i, (edited, (orig_text, codes)) in enumerate(zip(lines, segs)):
            base = baseline_texts[i] if i < len(baseline_texts) else orig_text
            if edited == base:
                continue
            # Try word-level trim: edited is a strict prefix of baseline,
            # and we have word timings → slice codes at the last kept word's end.
            wts = word_timings[i] if i < len(word_timings) else []
            n_keep = _word_prefix_count(edited.split(), base.split()) if wts else 0
            if 0 < n_keep < len(wts):
                fs = wts[0].get("fs", 0)
                # +2 frames trailing slack — same as the last-segment treatment in
                # build_prompt_segments, avoids clipping the final consonant.
                new_fe = round(wts[n_keep - 1]["end"] * CODEC_HZ) + 2
                new_T = max(0, new_fe - fs)
                if 0 < new_T < codes.shape[0]:
                    new_codes = codes[:new_T].contiguous()
                    segs[i] = (edited, new_codes)
                    print(
                        f"  {label} seg {i+1} trimmed to {n_keep}/{len(wts)} words: "
                        f"'{edited[:40]}' ({codes.shape[0]} -> {new_T} frames)"
                    )
                    edited_any = True
                    continue
            # Fallback: text-only edit (no trim).
            segs[i] = (edited, codes)
            edited_any = True
            print(f"  {label} seg {i+1} text: '{orig_text[:40]}' -> '{edited[:40]}'")
        if len(lines) < len(segs):
            for i in range(len(lines), len(segs)):
                print(
                    f"  {label} seg {i+1} dropped (line removed): '{segs[i][0][:40]}'"
                )
            del segs[len(lines) :]
            del word_timings[len(lines) :]
            edited_any = True
        return edited_any

    edits_applied = False
    if prompt_segs:
        edits_applied |= _apply_text_edits(
            prompt_segs,
            _prompt_cache.get("baseline_texts") or [t for t, _ in prompt_segs],
            _prompt_cache.get("word_timings") or [[] for _ in prompt_segs],
            prompt_text_1_value,
            "S1",
        )
    if edits_applied:
        invalidate_kv()  # text changed, need to re-prefill

    chunks_preview = chunk_text(
        text,
        min_words=chunk_words,
        sentence_only=True,
        single_speaker=True,
    )
    print(f"--- Chunks ({len(chunks_preview)}) ---")
    for i, c in enumerate(chunks_preview):
        sc = "[SC] " if c["sc"] else ""
        print(f"  {i+1}. {sc}{c['text']}")
    print("---")

    t1 = time.perf_counter()
    sq_scores_tuple = (
        sq_dns_sig,
        sq_dns_bak,
        sq_nq_noi,
        sq_nq_disc,
        sq_nq_col,
        sq_nq_loud,
    )

    # MLX: pass prompt_segs directly to generate_chunked_mlx for proper
    # per-segment [spk] [text] [audio] prefilling.
    prompt_codes_mlx = None
    prompt_text_mlx = None

    spk_emb_mlx = None
    if USE_MLX and prompt_spk_emb is not None:
        spk_emb_mlx = mx.array(
            prompt_spk_emb.cpu().numpy()
            if hasattr(prompt_spk_emb, "cpu")
            else prompt_spk_emb.numpy()
        )

    if USE_MLX:
        codes, chunks = generate_chunked_mlx(
            model,
            text,
            None,
            prompt_codes=prompt_codes_mlx,
            prompt_text=prompt_text_mlx,
            prompt_segments=prompt_segs,
            temperature=temperature,
            top_k=top_k_val,
            max_secs=max_duration,
            sq_scores=sq_scores_tuple,
            wps_score=wps_score,
            rep_penalty=rep_penalty,
            rep_window=int(rep_window),
            spk_emb=spk_emb_mlx,
            prompt_hash=_prompt_cache.get("hash"),
        )
    else:
        _engine.set_conditioning(
            sq_scores=sq_scores_tuple,
            wps_score=wps_score,
        )
        cfg = GenConfig(
            temperature=temperature,
            top_k=int(top_k_val),
            top_p=top_p,
            rep_penalty=rep_penalty,
            rep_window=int(rep_window),
            eos_threshold=S["eos_threshold"],
            max_secs=max_duration,
            chunk_words=int(chunk_words),
            n_codebooks=int(n_codebooks),
            sentence_only=True,
        )
        seg_list = [Segment(t, c) for t, c in (prompt_segs or [])]
        with _engine.new_row() as _row:
            _row.prefill(
                seg_list,
                spk_emb=prompt_spk_emb,
            )
            codes, _ = _row.render(text, cfg)
        chunks = chunk_text(
            text,
            min_words=int(chunk_words),
            sentence_only=True,
            single_speaker=True,
        )

    if codes is None:
        return None, "No audio generated."

    # Use all prompt codes concatenated for codec context
    all_prompt_codes = None
    if prompt_segs:
        all_prompt_codes = torch.cat([c for _, c in prompt_segs], dim=0)
    waveform, sr = decode_audio(
        codes, n_codebooks=n_codebooks, prompt_codes=all_prompt_codes
    )
    gen_time = time.perf_counter() - t1
    audio_dur = waveform.shape[-1] / sr
    rtf = audio_dur / gen_time if gen_time > 0 else 0

    # Code and waveform diagnostics
    per_q_unique = torch.tensor(
        [codes[:, q].unique().shape[0] for q in range(codes.shape[1])]
    )
    abs_mean = waveform.abs().mean().item()

    diag = (
        f"codes: {codes.shape[0]} frames, per_q_unique={per_q_unique.float().mean():.1f}, "
        f"wav abs_mean={abs_mean:.6f}"
    )
    print(diag)

    info = (
        f"{audio_dur:.1f}s in {gen_time:.1f}s ({rtf:.1f}x RT) | "
        f"{len(chunks)} chunks, {codes.shape[0]} frames\n"
        f"{diag}"
    )
    print(info)
    wav_np = waveform.numpy()
    return (sr, wav_np), info


def _quantize_cuda_model(m, precision: str):
    """Apply torchao weight-only quantization to the CUDA model.
    Skips attention q/k/v/o projections to keep flash_attn compatible (needs bf16)."""
    if precision == "bfloat16":
        return
    import warnings

    warnings.filterwarnings("ignore", ".*Deprecation.*torchao.*")
    warnings.filterwarnings("ignore", ".*Config Deprecation.*")
    import torch.nn as nn
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    # Skip attention projections — flash_attn requires bf16/fp16 inputs
    attn_names = {"Wqkv", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"}

    def _not_attn(mod, fqn):
        return isinstance(mod, nn.Linear) and not any(n in fqn for n in attn_names)

    if precision == "int8":
        quantize_(m.decoder, Int8WeightOnlyConfig(), filter_fn=_not_attn)
        if m.rq_transformer is not None:
            quantize_(m.rq_transformer, Int8WeightOnlyConfig())
        print("Quantized to int8 (MLP + RQ, attn stays bf16)")
    elif precision == "int4":
        from torchao.quantization import Int4WeightOnlyConfig

        try:
            quantize_(
                m.decoder, Int4WeightOnlyConfig(group_size=64), filter_fn=_not_attn
            )
        except ImportError:
            # Fallback for older torchao without mslk
            from torchao.quantization import Int4WeightOnlyQuantizer

            for block in m.decoder.blocks:
                block.mlp = Int4WeightOnlyQuantizer(groupsize=64).quantize(block.mlp)
        if m.rq_transformer is not None:
            try:
                quantize_(m.rq_transformer, Int4WeightOnlyConfig(group_size=64))
            except (ImportError, NameError):
                from torchao.quantization import Int4WeightOnlyQuantizer

                m.rq_transformer = Int4WeightOnlyQuantizer(groupsize=64).quantize(
                    m.rq_transformer
                )
        print("Quantized to int4 (MLP + RQ, attn stays bf16)")


def load_new_checkpoint(ckpt_path):
    global model, _engine, codec_enc, codec_dec, CODEC_SR, _max_turn_secs, checkpoint_path
    if not ckpt_path.strip():
        return "Enter a checkpoint path."
    try:
        path = ckpt_path.strip()
        if path.startswith("s3://"):
            from vui.hf import download

            path = download(path)
        print(f"Loading {path}...")
        invalidate_kv()
        _invalidate_prompt_caches()

        if USE_MLX:
            checkpoint_path = path
            _load_mlx_model(path, _mlx_precision)
            cond_bias_w = compute_cond_bias(model, sq=[3.5, 4.0, 4.0, 4.0, 4.0, 0.0])
            generate_chunked_mlx(model, "Warmup.", cond_bias_w, max_secs=2)
        else:
            model = Vui.from_pretrained_inf(path).cuda()
            checkpoint_path = path
            _max_turn_secs = model.config.data.max_secs
            if model.rq_transformer is None:
                raise RuntimeError(
                    f"Checkpoint {path} has no RQ-Transformer head; the "
                    "legacy STFT codec path has been removed."
                )
            _engine = Engine(model, _get_codec_dec(), max_rows=1, vocoder_ctx=25)
            with _engine.new_row() as _wr:
                _wr.render("Warmup.", GenConfig(max_secs=2, temperature=0.7))

        return f"Loaded: {ckpt_path}"
    except Exception as e:
        return f"Error: {e}"


# --- Gradio UI ---

with gr.Blocks(
    title="Vui STFT",
    theme=gr.themes.Soft(),
    head="""
<script>
document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            var btn = document.querySelector('button[variant="primary"]');
            if (btn && !btn.disabled) btn.click();
        }
        if (e.ctrlKey && e.code === 'Space') {
            e.preventDefault();
            var a = document.querySelector('audio');
            if (a) a.paused ? a.play() : a.pause();
        }
    });
    new MutationObserver(function(ms) {
        ms.forEach(function(m) {
            if (m.type === 'childList')
                document.querySelectorAll('audio').forEach(function(a) {
                    if (a.src && !a.dataset.auto) {
                        a.dataset.auto = '1';
                        a.addEventListener('loadeddata', function() {
                            setTimeout(function() { a.play().catch(function(){}); }, 100);
                        });
                    }
                });
        });
    }).observe(document.body, {childList: true, subtree: true});
});
</script>
""",
) as demo:
    gr.Markdown("**Ctrl+Enter** generate | **Ctrl+Space** pause")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                ckpt_input = gr.Dropdown(
                    choices=KNOWN_CHECKPOINTS,
                    value=sys.argv[1] if len(sys.argv) > 1 else KNOWN_CHECKPOINTS[0],
                    label="Checkpoint",
                    allow_custom_value=True,
                    scale=3,
                )
                load_btn = gr.Button("Load", size="sm", scale=1)
            with gr.Row():
                ckpt_status = gr.Textbox(
                    value=f"Loaded: {sys.argv[1] if len(sys.argv) > 1 else ''} ({'MLX ' + _mlx_precision if USE_MLX else 'CUDA'})",
                    interactive=False,
                    lines=1,
                    label=None,
                    scale=3,
                )
                if USE_MLX:
                    precision_radio = gr.Radio(
                        choices=[
                            "float32 (quality)",
                            "int8 (balanced)",
                            "int4 (fastest)",
                        ],
                        value="float32 (quality)",
                        label="Precision",
                        scale=1,
                    )
                else:
                    precision_radio = gr.Radio(
                        choices=[
                            "bfloat16 (default)",
                            "int8 (smaller)",
                            "int4 (smallest)",
                        ],
                        value="bfloat16 (default)",
                        label="Precision",
                        scale=1,
                    )

            audio_input = gr.Audio(
                label="Voice Prompt (up to 3min)",
                type="numpy",
                format="wav",
            )
            prompt_text_1 = gr.Textbox(
                label="Prompt text (one segment per line — edit to fix ASR)",
                lines=4,
                max_lines=8,
                interactive=True,
                visible=False,
            )

            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter text...\nNewlines create separate turns.",
                lines=6,
                max_lines=12,
            )

            with gr.Row():
                sample_dropdown = gr.Dropdown(
                    choices=list(load_sample_texts().keys()),
                    label=f"Sample texts ({SAMPLE_TEXTS_PATH.name})",
                    value=None,
                    scale=10,
                )
                sample_refresh = gr.Button("Reload", scale=1)
            sample_dropdown.change(
                fn=lambda k: load_sample_texts().get(k, ""),
                inputs=sample_dropdown,
                outputs=text_input,
            )
            sample_refresh.click(
                fn=lambda: gr.update(choices=list(load_sample_texts().keys())),
                outputs=sample_dropdown,
            )

            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(
                    0.1, 1.0, S["temperature"], step=0.1, label="Temperature"
                )
                top_k = gr.Slider(1, 300, S["top_k"], step=1, label="Top-K")
                use_top_p = gr.Checkbox(label="Use Top-P", value=S["use_top_p"])
                top_p = gr.Slider(
                    0.1,
                    1.0,
                    S["top_p"],
                    step=0.05,
                    label="Top-P",
                    visible=S["use_top_p"],
                )
                max_duration = gr.Slider(
                    5, 120, S["max_duration"], step=5, label="Max Duration (s)"
                )
                gr.Markdown(
                    "**Speech Quality** (0 = off, 1-5 scale, each independently masked during training)\n\n"
                    "- **DNS Signal/Background**: DNSMOS signal clarity and background silence\n"
                    "- **NISQA Noise/Disc./Color.**: noise level, discontinuity artifacts, coloration\n"
                    "- **NISQA Loudness**: volume level"
                )
                with gr.Row():
                    sq_dns_sig_slider = gr.Slider(
                        0.0, 5.0, S["sq_dns_sig"], step=0.5, label="DNS Signal"
                    )
                    sq_dns_bak_slider = gr.Slider(
                        0.0, 5.0, S["sq_dns_bak"], step=0.5, label="DNS Background"
                    )
                with gr.Row():
                    sq_nq_noi_slider = gr.Slider(
                        0.0, 5.0, S["sq_nq_noi"], step=0.5, label="NISQA Noise"
                    )
                    sq_nq_disc_slider = gr.Slider(
                        0.0, 5.0, S["sq_nq_disc"], step=0.5, label="NISQA Disc."
                    )
                    sq_nq_col_slider = gr.Slider(
                        0.0, 5.0, S["sq_nq_col"], step=0.5, label="NISQA Color."
                    )
                    sq_nq_loud_slider = gr.Slider(
                        0.0, 5.0, S["sq_nq_loud"], step=0.5, label="NISQA Loudness"
                    )
                wps_slider = gr.Slider(
                    0.0,
                    6.0,
                    S["wps_score"],
                    step=0.1,
                    label="Words/sec (0 = off, typical ~2-4)",
                )
                rep_penalty_slider = gr.Slider(
                    1.0,
                    3.0,
                    S["rep_penalty"],
                    step=0.1,
                    label="Repetition Penalty (1.0 = off)",
                )
                rep_window_slider = gr.Slider(
                    0,
                    50,
                    S["rep_window"],
                    step=1,
                    label="Repetition Window (0 = full history)",
                )
                chunk_words_slider = gr.Slider(
                    5,
                    100,
                    S["chunk_words"],
                    step=1,
                    label="Chunk Words (min words per chunk)",
                )
                eos_threshold_slider = gr.Slider(
                    0.1, 0.95, S["eos_threshold"], step=0.05, label="EOS Threshold"
                )
                n_codebooks_slider = gr.Slider(
                    0,
                    16,
                    S["n_codebooks"],
                    step=1,
                    label="Codebooks (0 = all)",
                )
                compile_rq_checkbox = gr.Checkbox(
                    label="Compile RQ (~2x decode speed, ~80s warmup, may freeze)",
                    value=S.get("compile_rq", False),
                )
                use_top_p.change(
                    fn=lambda x: gr.update(visible=x), inputs=use_top_p, outputs=top_p
                )

                def _toggle_compile_rq(enabled):
                    global _engine
                    _save_settings(compile_rq=bool(enabled))
                    S["compile_rq"] = bool(enabled)
                    if USE_MLX:
                        return "MLX: ignored (CUDA-only)"
                    try:
                        _engine = Engine(
                            model,
                            _get_codec_dec(),
                            max_rows=1,
                            vocoder_ctx=25,
                        )
                        with _engine.new_row() as _wr:
                            _wr.render(
                                "Warmup.", GenConfig(max_secs=2, temperature=0.7)
                            )
                        return f"compile_rq={enabled} (rebuilt)"
                    except Exception as e:
                        return f"compile_rq toggle error: {e}"

                compile_rq_checkbox.change(
                    fn=_toggle_compile_rq,
                    inputs=compile_rq_checkbox,
                    outputs=ckpt_status,
                )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Speech", type="numpy", autoplay=True
            )
            info_output = gr.Textbox(label="Info", lines=2, interactive=False)

    load_btn.click(fn=load_new_checkpoint, inputs=ckpt_input, outputs=ckpt_status)

    if USE_MLX:

        def _switch_precision(choice):
            precision = (
                "int4"
                if "int4" in choice
                else ("int8" if "int8" in choice else "float32")
            )
            if precision == _mlx_precision:
                return f"Already {precision}"
            _load_mlx_model(checkpoint_path, precision)
            return f"Switched to {precision}"

        precision_radio.change(
            fn=_switch_precision, inputs=precision_radio, outputs=ckpt_status
        )
    else:

        def _switch_cuda_precision(choice):
            global model, _engine
            precision = (
                "int4"
                if "int4" in choice
                else ("int8" if "int8" in choice else "bfloat16")
            )
            try:
                invalidate_kv()
                _invalidate_prompt_caches()
                # Reload fresh model and quantize
                model = Vui.from_pretrained_inf(checkpoint_path).cuda()
                if precision != "bfloat16":
                    _quantize_cuda_model(model, precision)
                _engine = Engine(model, _get_codec_dec(), max_rows=1, vocoder_ctx=25)
                with _engine.new_row() as _wr:
                    _wr.render("Warmup.", GenConfig(max_secs=2, temperature=0.7))
                return f"Switched to {precision}"
            except Exception as e:
                return f"Error: {e}"

        precision_radio.change(
            fn=_switch_cuda_precision, inputs=precision_radio, outputs=ckpt_status
        )

    tts_inputs = [
        text_input,
        audio_input,
        prompt_text_1,
        temperature,
        top_k,
        use_top_p,
        top_p,
        max_duration,
        sq_dns_sig_slider,
        sq_dns_bak_slider,
        sq_nq_noi_slider,
        sq_nq_disc_slider,
        sq_nq_col_slider,
        sq_nq_loud_slider,
        wps_slider,
        rep_penalty_slider,
        rep_window_slider,
        chunk_words_slider,
        eos_threshold_slider,
        n_codebooks_slider,
    ]
    tts_outputs = [
        audio_output,
        info_output,
    ]

    generate_btn.click(fn=text_to_speech, inputs=tts_inputs, outputs=tts_outputs)
    text_input.submit(fn=text_to_speech, inputs=tts_inputs, outputs=tts_outputs)

    def _segments_to_text(segs) -> str:
        return "\n".join(text for text, _ in segs) if segs else ""

    def _prepare_prompt_1(audio):
        if audio is None:
            _prompt_cache.update(
                hash=None, segments=None, baseline_texts=None, spk_emb=None
            )
            return gr.update(visible=False, value="")
        get_prompt(audio)
        segs = _prompt_cache.get("segments")
        if not segs:
            return gr.update(visible=False, value="")
        return gr.update(visible=True, value=_segments_to_text(segs))

    audio_input.change(fn=_prepare_prompt_1, inputs=audio_input, outputs=prompt_text_1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
