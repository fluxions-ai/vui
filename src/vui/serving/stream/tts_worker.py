"""TTS+Audio worker process: model, codec encode/decode, generation.

Runs in its own process with exclusive GPU access. `TTSEngine` is a thin
adapter over `vui.engine.Engine` + a persistent single `Row`, layering on
the server-side concerns: KV save/load, conversation debug log, streaming
codec encoder, ASR-aligned prompt building, VAD trimming, and the
command-queue interface that `tts_process()` dispatches to.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import traceback
from multiprocessing import Queue
from pathlib import Path

import torch
import torch.nn.functional as F

from vui.engine import DEFAULT_WPS, Engine, GenConfig, Segment
from vui.inference import chunk_text, simple_clean
from vui.model import Vui
from vui.qwen_codec import QwenCodecDecoder, QwenCodecEncoder, StreamingCodecEncoder
from vui.qwen_spk_enc import QwenSpeakerEncoder

# VUI_DEBUG=1 (default): per-turn .pt dumps + seq log. Set to "0" for fast mode
# (skips disk writes and seq-log construction; one-liner [TTS.*] prints stay on).
DEBUG = os.environ.get("VUI_DEBUG", "1") == "1"


CODEC_HZ = 12.5
VOCODER_CTX = 25  # 2s of codec context for streaming vocoder (CUDA-graphed)
PROMPTS_DIR = Path("prompts")


def _ckpt_id(checkpoint_path: str | None) -> str:
    """Stable short id for a checkpoint, used to key per-model KV caches.

    Parses runs/<run>/<step>.pt → "<run>_<step>"; falls back to a path md5.
    KV/cond_bias/spk_token_emb are model-specific, so prompt files are named
    `{name}.{ckpt_id}.pt` and regenerated when the active checkpoint changes.
    """
    if not checkpoint_path:
        return "unknown"
    p = str(checkpoint_path)
    m = re.search(r"/([A-Za-z0-9]+)/(\d+)\.pt$", p)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return "h" + hashlib.md5(p.encode()).hexdigest()[:9]


def _migrate_legacy_prompts() -> None:
    """One-shot rename of `{name}.pt` → `{name}.{ckpt_id}.pt` for legacy files.

    Old prompts predate the per-checkpoint naming. We rename them so discovery
    treats them as the (possibly stale) cache for whatever checkpoint produced
    them. If `checkpoint` metadata is missing, fall back to `legacyd{d_model}`
    so they don't collide with anything real.
    """
    if not PROMPTS_DIR.exists():
        return
    for legacy in sorted(PROMPTS_DIR.glob("*.pt")):
        if "." in legacy.stem:  # already keyed
            continue
        try:
            saved = torch.load(legacy, map_location="cpu", weights_only=False)
        except Exception:
            continue
        ckpt = saved.get("checkpoint") if isinstance(saved, dict) else None
        d_model = saved.get("d_model", 0) if isinstance(saved, dict) else 0
        old_id = (
            _ckpt_id(ckpt) if ckpt else (f"legacyd{d_model}" if d_model else "legacy")
        )
        target = legacy.with_name(f"{legacy.stem}.{old_id}.pt")
        if target.exists():
            _logf(f"[migrate] {legacy.name} target exists, removing legacy")
            legacy.unlink()
        else:
            _logf(f"[migrate] {legacy.name} -> {target.name}")
            legacy.rename(target)


# --- Timestamped logging to both stdout and debug_dump/tts_worker.log ---
# The TTS worker is a child process so stdout is only visible if the parent
# pipes it; everything we print here is also appended to a file for post-hoc
# diagnosis of bad generations. Timestamps are elapsed ms since worker start
# so they line up with `[main.*]` times in server.log via process start order.
_T0 = time.monotonic()
_LOG_DIR = Path("debug_dump")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / "tts_worker.log"
_LOG_F = None


def _logf(msg: str) -> None:
    global _LOG_F
    stamp = f"[t={int((time.monotonic() - _T0) * 1000):>7d}ms] "
    line = stamp + msg
    print(line, flush=True)
    if _LOG_F is None:
        _LOG_F = open(_LOG_PATH, "a", buffering=1)
        _LOG_F.write(f"\n--- worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    _LOG_F.write(line + "\n")


class TTSEngine:
    """Server-side TTS adapter over a single-row `vui.engine.Engine`.

    Owns one persistent `Row` for the lifetime of the worker process. All
    decode compute flows through `self.engine` / `self.row`; this class
    layers on queue I/O, prompt building, VAD trimming, KV save/load, and
    conversation logging.
    """

    def __init__(self, model: Vui, codec_dec, codec_enc, spk_enc=None):
        self.model = model
        self.tok = model.text_tokenizer
        self.codec_dec = codec_dec
        self.codec_enc = codec_enc
        self.spk_enc = spk_enc
        self.device = model.device
        self.n_q = model.config.model.n_quantizers
        self.max_seqlen = model.decoder.max_seqlen
        self.sc_id = self.tok.special_to_id["[SC]"]

        # Mutable prompt state (for KV save/load + conversation log).
        # `self.T` is a property delegating to row.offset so the server
        # status commands keep working.
        self.prompt_segments: list[tuple[str, torch.Tensor]] | None = None
        self.prompt_codes: torch.Tensor | None = None
        self.prompt_text: str | None = None
        self.prompt_T = 0
        self.spk_token_emb: torch.Tensor | None = None
        # Cached projected speaker token for the user. Computed once from
        # the first user turn that has audio, then prepended before every
        # subsequent user turn's text — matches `streamed_tts` training
        # which emits `[spk] text [SC] audio` per turn. Cleared on reset().
        self._user_spk_token: torch.Tensor | None = None
        self._prompt_rms: float = 0.0

        # Conversation debug dir — each turn saved immediately as its own file
        self._conv_dir: Path | None = None
        self._conv_turn_idx = 0

        # KV sequence log — human-readable record of what's in the cache
        self._seq_parts: list[str] = []

        # Set up by setup()
        self.engine: Engine | None = None
        self.row = None  # vui.engine.Row
        self.stream_enc: StreamingCodecEncoder | None = None

    def _log_seq(self, op: str):
        if not DEBUG:
            return
        # Each entry is (text, T_after). Display just the text.
        if not self._seq_parts:
            seq = "(empty)"
        else:
            seq = " ".join(p[0] if isinstance(p, tuple) else p for p in self._seq_parts)
        _logf(f"[TTS.seq] {op} T={self.T} | {seq}")

    def _seq_reset(self):
        self._seq_parts.clear()

    def _seq_add(self, part: str):
        if not DEBUG:
            return
        # Tag with current T so cancel-rewind can trim entries that no
        # longer correspond to actual KV content.
        self._seq_parts.append((part, self.T))

    def _seq_trim_to(self, T: int):
        """Drop seq entries logically past T. Called when KV is rewound by
        cancel — without this the seq log shows historical _add_user calls
        as if they were still in the cache (very misleading debugging)."""
        kept = []
        for p in self._seq_parts:
            if isinstance(p, tuple):
                _, t_after = p
                if t_after <= T:
                    kept.append(p)
            else:
                kept.append(p)  # untagged (legacy) — keep
        self._seq_parts = kept

    @property
    def T(self) -> int:
        return self.row.offset if self.row is not None else 0

    @T.setter
    def T(self, value: int) -> None:
        # Setting T is used by legacy paths — forward to row rewind.
        if self.row is not None:
            self.engine._rewind_row(self.row, value)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        codec_dtype: torch.dtype = torch.float32,
    ) -> "TTSEngine":
        print(f"[TTS] Loading model (codec_dtype={codec_dtype})...")
        model = Vui.from_pretrained_inf(checkpoint_path).cuda()

        print("[TTS] Loading codec decoder + encoder...")
        codec_dec = QwenCodecDecoder.from_pretrained().cuda().to(codec_dtype).eval()
        codec_enc = QwenCodecEncoder.from_pretrained().cuda().to(codec_dtype).eval()

        spk_enc = None
        if model.spk_proj is not None:
            print("[TTS] Loading speaker encoder...")
            spk_enc = QwenSpeakerEncoder.from_pretrained()

        engine = cls(model, codec_dec, codec_enc, spk_enc)
        engine.checkpoint_path = checkpoint_path
        engine.codec_dtype = codec_dtype
        engine.setup()
        return engine

    def setup(self):
        print("[TTS] Building Engine (backbone + RQ + vocoder CUDA graphs)...")
        self.engine = Engine(
            model=self.model, codec=self.codec_dec, max_rows=1, vocoder_ctx=VOCODER_CTX
        )
        self.row = self.engine.new_row()

        print("[TTS] Setting up streaming codec encoder...")
        self.stream_enc = StreamingCodecEncoder(
            self.codec_enc,
            n_quantizers=self.n_q,
            max_secs=10,
            min_chunk_frames=12,
        )
        self.stream_enc.setup_graph()

        print(
            f"[TTS] Context: {self.max_seqlen} codes "
            f"({self.max_seqlen / CODEC_HZ:.1f}s @ {CODEC_HZ}Hz)"
        )
        print("[TTS] Warming up...")
        cfg = GenConfig(temperature=0.8, top_k=300, max_secs=3)
        for text in ["Hello warmup.", "Testing one two three."]:
            self.row.render(text, cfg)
            self.row.reset()
        print("[TTS] Ready!")

    # --- KV + conditioning ---

    def reset(self):
        self.row.reset()
        self._seq_reset()
        self._user_spk_token = None

    def rewind(self) -> int:
        if self.prompt_T > 0:
            self.row.rewind()
        else:
            self.row.reset()

        # AR KV is back at prompt_offset — bring the codec streaming state
        # back in step. Without this, _codec_ctx._stack/_buf carry every
        # prior turn's audio while the AR thinks it's right after the
        # prompt; the next agent reply's first frame decodes against stale
        # state. set_prompt resets _buf to prompt codes and closes _stack
        # so the next stream() auto-reprefills from the prompt cleanly.
        if self.prompt_codes is not None:
            self.row._codec_ctx.set_prompt(
                self.prompt_codes.to(self.device).T.unsqueeze(0)
            )
        else:
            self.row._codec_ctx.reset()

        # Trim seq log back to just prompt parts
        def _is_prompt(p):
            text = p[0] if isinstance(p, tuple) else p
            return text.startswith("[prompt]") or text == "[SC]"

        self._seq_parts = [p for p in self._seq_parts if _is_prompt(p)]
        self._log_seq("rewind")
        return self.row.offset

    def set_conditioning(self, wps: float = 0.0, sq: list | None = None):
        self.engine.set_conditioning(
            sq_scores=tuple(sq) if sq else None, wps_score=wps
        )

    # --- Prefill ---

    def prefill_prompt(
        self,
        segments: list[tuple[str, torch.Tensor]],
        settings: dict,
        prompt_audio_24k=None,
        cond_bias_override=None,
    ):
        """Prefill KV cache with multi-segment prompt.

        segments: list of (text, codes) where codes is (T_seg, Q).
        Writes [spk] text_i codes_i per segment via Row.prefill.
        """
        if cond_bias_override is not None:
            # Direct override for KV re-load paths; bypass set_conditioning
            self.engine.model._cond_bias.copy_(cond_bias_override)
        else:
            self.set_conditioning(
                wps=settings.get("wps_score", 0.0),
                sq=settings.get("sq_scores"),
            )

        if prompt_audio_24k is not None and prompt_audio_24k.numel() > 0:
            self._prompt_rms = prompt_audio_24k.float().pow(2).mean().sqrt().item()

        spk_emb = None
        if (
            self.spk_enc is not None
            and self.model.spk_proj is not None
            and prompt_audio_24k is not None
        ):
            max_samples = 30 * 24000
            spk_audio = prompt_audio_24k[:max_samples]
            spk_emb = self.spk_enc.embed(spk_audio, sr=24000)

        seg_list = [Segment(text=t, codes=c) for t, c in segments]
        self.row.reset()
        self.row.prefill(seg_list, spk_emb=spk_emb)

        # Critical: agent generation prepends row._spk_token before every chunk
        # (matching streamed_tts training format). On a fresh upload spk_emb was
        # computed and projected into row._spk_token by Row.prefill. On context
        # reset no audio is passed, so row._spk_token is None — restore it from
        # the cached projected token, otherwise the rest of the conversation
        # generates without [spk] and quality drops massively (~8x WER in A/B).
        if self.row._spk_token is None and self.spk_token_emb is not None:
            self.row._spk_token = self.spk_token_emb.to(self.device, self.engine.dtype)

        self.prompt_segments = segments
        self.prompt_codes = torch.cat([c for _, c in segments], dim=0)
        self.prompt_text = " ".join(t for t, _ in segments if t)
        self.prompt_T = self.row.offset
        ctx_buf = self.row._codec_ctx._buf
        ctx_frames = ctx_buf.shape[2] if ctx_buf is not None else 0
        ctx_q = ctx_buf.shape[1] if ctx_buf is not None else 0
        seg_shapes = [
            (t[:30], tuple(c.shape) if c is not None else ())
            for t, c in segments
        ]
        _logf(
            f"[TTS.prompt] segments={len(segments)} {seg_shapes} "
            f"T=0->{self.row.offset} codec_ctx={ctx_frames}f/Q={ctx_q} "
            f"prompt_wps={self.row.prompt_wps:.2f} "
            f"spk_token={self.row._spk_token is not None} "
            f"spk_emb_used={spk_emb is not None}"
        )
        if spk_emb is not None and self.row._spk_token is not None:
            self.spk_token_emb = self.row._spk_token

        self._seq_reset()
        for t, c in segments:
            nf = c.shape[0] if c is not None else 0
            nq = c.shape[1] if c is not None and c.dim() > 1 else 0
            self._seq_add(f'[prompt] [spk] "{t}" [{nf}f/Q={nq}]')
        self._seq_add("[SC]")
        self._log_seq("prefill_prompt")

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self._prompt_rms < 1e-6:
            return audio
        rms = audio.float().pow(2).mean().sqrt().item()
        if rms < 1e-6:
            return audio
        gain = max(0.25, min(4.0, self._prompt_rms / rms))
        return audio * gain

    def build_prompt_from_audio(
        self, audio_16k_np, text: str, audio_24k_np=None
    ) -> list[tuple[str, torch.Tensor]]:
        """Run codec encode + ASR-aligned segmentation.

        text: full transcript (caller is responsible for ASR; alignment uses
            wav2vec2 here against this text).
        audio_24k_np: if provided, used for codec encoding (avoids 16k->24k resample).
        """
        from vui.prompt_utils import build_prompt_segments

        audio_16k = torch.from_numpy(audio_16k_np).float()
        if audio_24k_np is not None:
            audio_24k = torch.from_numpy(audio_24k_np).float()
        else:
            audio_24k = None

        def _encode(_a16: torch.Tensor) -> torch.Tensor:
            if audio_24k is not None:
                a24 = audio_24k
            else:
                from julius.resample import resample_frac

                a24 = resample_frac(_a16, 16000, 24000)
            audio_in = a24.to(self.codec_dtype).cuda().reshape(1, 1, -1)
            codes = self.codec_enc.encode(audio_in)
            return codes[0, : self.n_q].T.long().cpu()

        return build_prompt_segments(
            audio_16k,
            encode_codes=_encode,
            transcribe=lambda _a: text,
            align_device="cpu",
        )

    @staticmethod
    def vad_trim_codes(codes: torch.Tensor, audio_16k=None) -> torch.Tensor:
        """Trim silence from codec codes using VAD on the source audio.
        codes: (T, Q), audio_16k: numpy float32 16kHz mono or None.
        Returns trimmed codes.

        audio_16k may include a pre-buffer (captured before VAD triggered
        recording) that the codec encoder never saw. We estimate the offset
        from the duration mismatch: codec_duration = codes.shape[0] / 12.5,
        audio_duration = len(audio_16k) / 16000. The front of audio_16k that
        exceeds codec_duration is pre-buffer.
        """
        if audio_16k is None or len(audio_16k) < 1600:
            return codes
        from vui.serving.stream.vad import SileroVAD

        audio_dur = len(audio_16k) / 16000
        codec_dur = codes.shape[0] / CODEC_HZ
        pre_buffer_secs = max(0.0, audio_dur - codec_dur)

        vad = SileroVAD(
            sample_rate=16000, confidence=0.5, start_secs=0.1, stop_secs=0.3
        )
        chunk_size = 512
        speech_start = None
        speech_end = 0
        for i in range(0, len(audio_16k) - chunk_size + 1, chunk_size):
            chunk = audio_16k[i : i + chunk_size]
            result = vad.process(chunk)
            t_sec = i / 16000
            if result is not None:
                from vui.serving.stream.vad import VADState

                if result == VADState.SPEAKING and speech_start is None:
                    speech_start = max(0, t_sec - 0.1)
                elif result == VADState.QUIET and speech_start is not None:
                    speech_end = t_sec + 0.1
        if speech_start is None:
            return codes
        if speech_end <= speech_start:
            speech_end = len(audio_16k) / 16000

        # Shift VAD timestamps into codec time (remove pre-buffer offset)
        speech_start = max(0.0, speech_start - pre_buffer_secs)
        speech_end = max(0.0, speech_end - pre_buffer_secs)

        f_start = max(0, int(speech_start * CODEC_HZ))
        f_end = min(codes.shape[0], int(speech_end * CODEC_HZ) + 1)
        if f_end <= f_start:
            return codes
        trimmed = codes[f_start:f_end]
        if trimmed.shape[0] != codes.shape[0]:
            print(
                f"[TTS] VAD trimmed: {codes.shape[0]} -> {trimmed.shape[0]} frames "
                f"({speech_start:.2f}s - {speech_end:.2f}s, pre_buf={pre_buffer_secs:.2f}s)"
            )
        return trimmed

    def _ensure_user_spk_token(self, audio_16k) -> None:
        """Compute + cache the user's projected [spk] token from their audio.
        First call only; idempotent thereafter. Skips if no audio, no spk
        encoder, no spk projection, or audio is too short to be useful.

        Training computes spk embeddings on 24kHz audio
        (`scripts/compute_spk_emb.py`), so we resample 16k -> 24k first
        (matching the prompt-upload path in server.py).
        """
        if (
            self._user_spk_token is not None
            or audio_16k is None
            or self.spk_enc is None
            or self.model.spk_proj is None
        ):
            return
        a16 = (
            audio_16k
            if isinstance(audio_16k, torch.Tensor)
            else torch.from_numpy(audio_16k).float()
        )
        if a16.numel() < 16000:  # < 1s, embedding would be unreliable
            return
        from julius.resample import resample_frac

        t0 = time.perf_counter()
        a24 = resample_frac(a16.unsqueeze(0), 16000, 24000).squeeze(0)
        # Cap at 30s — encoder pools over time, more doesn't help (matches
        # prompt path).
        a24 = a24[: 30 * 24000]
        with torch.inference_mode():
            spk_emb = self.spk_enc.embed(a24, sr=24000)
            self._user_spk_token = self.model.embed_speaker(spk_emb).to(
                self.engine.dtype
            )
        print(
            f"[TTS] Cached user spk_token from {a16.numel()/16000:.2f}s audio "
            f"(24k resampled, {(time.perf_counter()-t0)*1000:.0f}ms)"
        )

    def prefill_user_turn(
        self, text: str = "", codes=None, audio_16k=None, settings: dict | None = None
    ):
        T0 = self.row.offset
        # Use locally accumulated codes from stream_feed/stream_stop if none passed
        if codes is None and self._user_codes_parts:
            codes = torch.cat(self._user_codes_parts, dim=0)
        n_in = codes.shape[0] if codes is not None else 0
        n_words = len(text.split()) if text else 0
        if codes is not None and audio_16k is not None:
            codes = self.vad_trim_codes(codes, audio_16k)
        n_codes = codes.shape[0] if codes is not None else 0

        # Cache the user's [spk] token on first turn; prepend to every turn
        # thereafter so the format matches `streamed_tts` training.
        self._ensure_user_spk_token(audio_16k)
        had_spk = self._user_spk_token is not None
        t0 = time.perf_counter()
        if had_spk:
            with torch.inference_mode():
                self.engine._prefill_emb(self.row, self._user_spk_token)
        self.row.add_user(text=text or "", codes=codes)
        dt = (time.perf_counter() - t0) * 1000
        ctx_buf = self.row._codec_ctx._buf
        ctx_frames = ctx_buf.shape[2] if ctx_buf is not None else 0
        ctx_q = ctx_buf.shape[1] if ctx_buf is not None else 0
        _logf(
            f"[TTS.user] spk={had_spk}, text={n_words}w '{text[:50]}', "
            f"codes={n_in}->{n_codes}f ({n_codes/CODEC_HZ:.2f}s), "
            f"audio_16k={'yes' if audio_16k is not None else 'no'}, "
            f"T={T0}->{self.row.offset} (+{self.row.offset - T0}), "
            f"codec_ctx={ctx_frames}f/Q={ctx_q}, {dt:.1f}ms"
        )
        self._log_turn("user", text, codes)
        spk = "[user_spk] " if had_spk else ""
        self._seq_add(f'{spk}[user] "{text}" [SC] [{n_codes}f]')
        self._log_seq("prefill_user_turn")

    def prefill_text(self, text: str):
        """Write free-form text into the row with the current cond_bias."""
        with torch.inference_mode():
            ids = self.tok.encode(simple_clean(text)).to("cuda")
            emb = self.model.token_emb(ids[None]).to(self.engine.dtype)
            self.engine._prefill_emb(self.row, emb + self.engine._cond_bias)
        self._seq_add(f'[text] "{text}"')
        self._log_seq("prefill_text")

    def prefill_text_sc(self, text: str) -> float:
        """Feed text tokens and return P([SC]) from text logits.

        Used for learned endpointing: high P([SC]) means the model thinks
        the current speaker is done.
        """
        with torch.inference_mode():
            ids = self.tok.encode(simple_clean(text)).to("cuda")
            emb = self.model.token_emb(ids[None]).to(self.engine.dtype)
            out = self.engine._prefill_emb(self.row, emb)
            hidden = out[:, -1]
            text_logits = F.linear(hidden.float(), self.model.token_emb.weight.float())
            sc_prob = torch.softmax(text_logits[0], dim=-1)[self.sc_id].item()
        return sc_prob

    # --- Generation ---

    # --- Generation ---

    def generate(
        self,
        text: str,
        settings: dict,
        cancel_event,
        audio_queue: Queue | None = None,
        *,
        is_new_turn: bool = True,
        is_final_chunk: bool = False,
        context: str = "",
    ) -> dict:
        """High-level generate via row.stream().

        Emits timing / audio / chunk_done events to audio_queue (if set).
        Returns {"total_secs", "total_frames", "total_gen_time", "cancelled", "codes"}.
        """
        max_seq = self.max_seqlen - 10

        # Context reset if near limit
        if self.row.offset >= max_seq - 200:
            self.row.reset()
            if self.prompt_segments is not None:
                self.prefill_prompt(self.prompt_segments, settings)
            if context:
                self.prefill_text(context)
            if audio_queue:
                audio_queue.put({"type": "context_reset", "T": self.row.offset})

        max_duration = settings.get("max_duration", 120)
        wps = self.row.prompt_wps if self.row.prompt_wps > 0 else DEFAULT_WPS
        n_words = len(text.split())
        max_duration = min(max_duration, n_words / wps * 1.5 + 1.0)
        cfg = GenConfig(
            temperature=settings.get("temperature", 0.9),
            top_k=int(settings.get("top_k", 300)),
            top_p=settings.get("top_p"),
            rep_penalty=settings.get("rep_penalty", 1.1),
            rep_window=int(settings.get("rep_window", 0)),
            eos_threshold=settings.get("eos_threshold", 0.4),
            max_secs=max_duration,
            chunk_words=int(settings.get("chunk_words", 20)),
            first_chunk_words=(
                int(settings.get("first_chunk_words", 0)) if is_new_turn else 0
            ),
            n_codebooks=int(settings.get("n_codebooks", 0)),
            gate_frames=int(settings.get("gate_frames", 0)),
            gate_entropy_max=float(settings.get("gate_entropy_max", 1.9)),
            gate_retries=int(settings.get("gate_retries", 2)),
        )

        t_turn_start = time.perf_counter()
        gen_codes: list[torch.Tensor] = []
        total_frames = 0
        total_secs = 0.0
        cancelled = False
        first_frame_done = False
        t_prefill = t_turn_start

        # Pull pre-chunk log for parity with old server logs
        chunks_preview = chunk_text(text, min_words=cfg.chunk_words)
        ctx_buf = self.row._codec_ctx._buf
        codec_ctx_frames = ctx_buf.shape[2] if ctx_buf is not None else 0
        codec_ctx_q = ctx_buf.shape[1] if ctx_buf is not None else 0
        spk_token_set = self.row._spk_token is not None
        user_spk_token_set = self._user_spk_token is not None
        # Rep-penalty state sanity: with rep_window=0 + penalty>1, counts
        # accumulate forever inside a turn (no decay), so counts_max growing
        # over a long multi-chunk turn → penalty factor = penalty**counts_max
        # which can easily exceed 100x and drive logits off distribution.
        rep_counts = self.engine._rep_counts[0] if self.engine is not None else None
        if rep_counts is not None:
            nz = (rep_counts > 0).sum().item()
            cmax = float(rep_counts.max().item())
            csum = float(rep_counts.sum().item())
        else:
            nz = cmax = csum = 0
        rep_hist_fill = self.engine._rep_filled if self.engine is not None else 0
        rep_hist_win = self.engine._rep_window_cur if self.engine is not None else 0
        # Conditioning bias + spk fingerprints (catches set_conditioning being
        # called unexpectedly mid-conversation)
        bias_rms = float(
            self.engine.model._cond_bias.float().pow(2).mean().sqrt().item()
        )
        spk_n = (
            float(self.row._spk_token.float().norm().item())
            if self.row._spk_token is not None
            else 0.0
        )
        user_spk_n = (
            float(self._user_spk_token.float().norm().item())
            if self._user_spk_token is not None
            else 0.0
        )
        _logf(
            f"[TTS.gen] T={self.row.offset} (prompt_T={self.prompt_T}) "
            f"codec_ctx={codec_ctx_frames}f/Q={codec_ctx_q} "
            f"spk={spk_token_set}(norm={spk_n:.2f}) "
            f"user_spk={user_spk_token_set}(norm={user_spk_n:.2f}) "
            f"bias_rms={bias_rms:.4f} "
            f"new_turn={is_new_turn} final={is_final_chunk} "
            f"cfg(temp={cfg.temperature} top_k={cfg.top_k} "
            f"rep={cfg.rep_penalty}/w{cfg.rep_window} eos={cfg.eos_threshold} "
            f"nq={cfg.n_codebooks} chunk_w={cfg.chunk_words} first_cw={cfg.first_chunk_words} "
            f"max_s={cfg.max_secs:.1f} wps={wps:.2f}) "
            f"text='{text[:60]}'"
        )
        _logf(
            f"[TTS.rep] nonzero={nz} sum={csum:.0f} max={cmax:.0f} "
            f"hist_fill={rep_hist_fill}/{rep_hist_win} "
            f"(penalty_at_max={cfg.rep_penalty**cmax:.2f}x)"
        )
        _logf(
            f"[TTS.gen]   -> {len(chunks_preview)} sub-chunks, "
            f"sc={[c['sc'] for c in chunks_preview]}"
        )

        self._seq_add(f'[agent] [spk] "{text}" [generating...]')
        self._log_seq("generate")

        try:
            # reset_rep=False on continuation chunks so rep-penalty history
            # spans the whole assistant turn (matches demo.py's render()).
            # final_turn=True on the last LLM chunk so the last sub-chunk
            # gets [SC] appended to its text (matches data.py streamed_tts
            # where [SC] marks "next speaker differs").
            stream_iter = self.row.stream(
                text,
                cfg,
                cancel=cancel_event,
                reset_rep=is_new_turn,
                final_turn=is_final_chunk,
            )
            # Look-ahead so the final frame(s) can be tapered to avoid a click
            # at end-of-utterance. Final chunks hold up to 3 frames (~240ms)
            # so we can fade the last 200ms; non-final chunks keep the cheap
            # 1-frame hold so TTFB only takes the 80ms hit on the final chunk.
            from collections import deque as _deque

            HOLD_N = 3 if is_final_chunk else 1
            FADE_SAMPLES = 4800  # 200ms at 24kHz
            held: "_deque[tuple[torch.Tensor, int]]" = _deque(maxlen=HOLD_N)
            for audio_chunk in stream_iter:
                if not first_frame_done:
                    torch.cuda.synchronize()
                    first_frame_ms = (time.perf_counter() - t_prefill) * 1000
                    ttfb_ms = first_frame_ms  # row.stream includes prefill
                    if audio_queue:
                        audio_queue.put(
                            {
                                "type": "timing",
                                "prefill_ms": 0.0,
                                "prefill_tokens": 0,
                                "first_frame_ms": first_frame_ms,
                                "ttfb_ms": ttfb_ms,
                            }
                        )
                    first_frame_done = True
                # Single per-frame sync: audio to CPU. No codes collection in the
                # hot loop — conv log records just text + timings for now.
                wav = audio_chunk[0, 0].detach().float().cpu()
                total_frames += 1
                total_secs += 1 / CODEC_HZ
                # If the hold buffer is full, the oldest frame is committed
                # (no fade — the tail of the chunk is what gets the ramp).
                if len(held) == HOLD_N and audio_queue:
                    old_wav, old_T = held.popleft()
                    audio_queue.put(
                        {
                            "type": "audio",
                            "data": old_wav,
                            "T": old_T,
                            "secs": 1 / CODEC_HZ,
                        }
                    )
                held.append((wav, self.row.offset))
                if cancel_event.is_set():
                    cancelled = True
                    break

            if held and audio_queue:
                if is_final_chunk or cancelled:
                    # Concat the buffered tail (up to 240ms) and apply a
                    # linear ramp across the trailing FADE_SAMPLES so we end
                    # the utterance on silence — kills any tail click from
                    # the model's last frame.
                    combined = torch.cat([w for w, _ in held]).clone()
                    fade_n = min(FADE_SAMPLES, combined.numel())
                    ramp = torch.linspace(1.0, 0.0, fade_n, dtype=combined.dtype)
                    combined[-fade_n:] *= ramp
                    audio_queue.put(
                        {
                            "type": "audio",
                            "data": combined,
                            "T": held[-1][1],
                            "secs": combined.numel() / 24000,
                        }
                    )
                else:
                    for hwav, hT in held:
                        audio_queue.put(
                            {
                                "type": "audio",
                                "data": hwav,
                                "T": hT,
                                "secs": 1 / CODEC_HZ,
                            }
                        )
        except Exception:
            traceback.print_exc()
            raise

        total_gen_time = time.perf_counter() - t_turn_start
        # Trailing N frames of the rolling codec buffer ARE the codes just
        # generated — `_stream_decode_frame` appends each agent frame via
        # `CodecCtx._append`. One `.cpu()` at end, no per-frame sync.
        ctx_buf = self.row._codec_ctx._buf
        if ctx_buf is not None and total_frames > 0:
            all_codes = ctx_buf[0, :, -total_frames:].T.detach().cpu()
        else:
            all_codes = None

        if audio_queue and not cancelled:
            audio_queue.put(
                {
                    "type": "chunk_done",
                    "chunk_idx": 1,
                    "text": text,
                    "secs": total_secs,
                    "frames": total_frames,
                    "gen_time": total_gen_time,
                    # Position after this chunk: server uses this to rewind
                    # KV to the boundary of the last fully-heard chunk on
                    # barge-in (so the user keeps the assistant text they
                    # actually heard in TTS context).
                    "T": self.row.offset,
                }
            )

        self._log_turn("assistant", text, all_codes, secs=total_secs)
        if self._seq_parts:
            last = self._seq_parts[-1]
            last_text = last[0] if isinstance(last, tuple) else last
            if last_text.startswith("[agent]"):
                new_text = f'[agent] [spk] "{text}" [{total_frames}f]'
                if cancelled:
                    new_text += " CANCELLED"
                if isinstance(last, tuple):
                    self._seq_parts[-1] = (new_text, last[1])
                else:
                    self._seq_parts[-1] = new_text
                self._log_seq("generate done")
        return {
            "total_secs": total_secs,
            "total_frames": total_frames,
            "total_gen_time": total_gen_time,
            "cancelled": cancelled,
            "codes": all_codes,
        }

    # --- Conversation debug log ---
    # Each turn saved as its own file the moment it happens — no flush needed.
    # debug_dump/2026-03-20_143052/
    #   prompt.pt          — saved once on prefill_prompt
    #   00_user.pt         — user text + codes
    #   01_assistant.pt    — reply text + codes
    #   02_user.pt
    #   ...

    def start_conversation(self, base_dir: Path):
        from datetime import datetime

        folder = base_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        self._conv_dir = folder
        self._conv_turn_idx = 0
        if self.prompt_text or self.prompt_codes is not None:
            torch.save(
                {
                    "prompt_text": self.prompt_text,
                    "prompt_codes": self.prompt_codes,
                },
                folder / "prompt.pt",
            )
        print(f"[TTS] New conversation: {folder}")

    def end_conversation(self):
        if self._conv_dir and self._conv_turn_idx > 0:
            print(
                f"[TTS] Ended conversation: {self._conv_dir} ({self._conv_turn_idx} turns)"
            )
        self._conv_dir = None
        self._conv_turn_idx = 0

    def _log_turn(self, role: str, text: str, codes=None, secs: float | None = None):
        if self._conv_dir is None:
            return
        idx = self._conv_turn_idx
        self._conv_turn_idx += 1
        data = {"role": role, "text": text, "T": self.T, "t": time.time()}
        if codes is not None:
            data["codes"] = codes.cpu() if codes.is_cuda else codes
        if secs is not None:
            data["secs"] = secs
        torch.save(data, self._conv_dir / f"{idx:02d}_{role}.pt")
        print(f"[TTS] Logged turn {idx}: {role} '{text[:50]}'")

    # --- KV save/load ---

    def save_kv(self, path: Path, name: str):
        T = self.row.offset
        kv_data = []
        for kv in self.model.decoder.flash_kv_caches:
            kv_data.append(
                {
                    "k": kv.k_cache[:, :T].cpu().clone(),
                    "v": kv.v_cache[:, :T].cpu().clone(),
                }
            )
        torch.save(
            {
                "name": name,
                "codes": self.prompt_codes,
                "text": self.prompt_text,
                "T": T,
                "n_q": self.n_q,
                "d_model": self.model.config.model.d_model,
                "checkpoint": getattr(self, "checkpoint_path", None),
                "kv": kv_data,
                "cond_bias": self.engine.model._cond_bias.cpu().clone(),
                "spk_token_emb": (
                    self.spk_token_emb.cpu().clone()
                    if self.spk_token_emb is not None
                    else None
                ),
            },
            path,
        )

    def _match_n_q(self, codes: torch.Tensor) -> torch.Tensor:
        """Align saved codes' last dim to the current model's n_q.
        Trim if larger; zero-pad if smaller (older prompt with fewer Q)."""
        q = codes.shape[-1]
        if q == self.n_q:
            return codes
        if q > self.n_q:
            print(f"[TTS] Trimming saved prompt codes: Q {q} -> {self.n_q}")
            return codes[..., : self.n_q].contiguous()
        import torch.nn.functional as F

        print(f"[TTS] Padding saved prompt codes: Q {q} -> {self.n_q} (zero-fill)")
        return F.pad(codes, (0, self.n_q - q)).contiguous()

    def _reprefill_from_saved(self, saved):
        """Re-prefill prompt from saved codes+text (when KV cache is stale)."""
        codes = self._match_n_q(saved["codes"])
        text = saved.get("text", "")
        saved_bias = saved["cond_bias"].to(self.engine.model._cond_bias.device)
        # Set spk_token_emb BEFORE prefill so it's included in the KV cache
        spk = saved.get("spk_token_emb")
        self.spk_token_emb = spk.to(self.device) if spk is not None else None
        self.row._spk_token = self.spk_token_emb
        # Pass saved cond_bias directly to skip set_conditioning
        self.prefill_prompt([(text, codes)], {}, cond_bias_override=saved_bias)

    def load_kv(self, path: Path):
        if path.suffix == ".safetensors":
            return self._load_safetensors_prompt(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)

        # d_model must match
        saved_d = saved.get("d_model", 0)
        if saved_d and saved_d != self.model.config.model.d_model:
            raise ValueError(
                f"KV cache d_model mismatch: saved={saved_d}, model={self.model.config.model.d_model}"
            )

        # If checkpoint changed or n_q/codes-Q changed, KV is stale — re-prefill
        # from codes. Older prompts (e.g. Abraham.pt) omit `n_q`; fall back to
        # inferring from `codes.shape[-1]` so a Q-mismatched prompt triggers a
        # full reprefill (which also seeds codec_ctx + spk_token correctly).
        saved_ckpt = saved.get("checkpoint")
        cur_ckpt = getattr(self, "checkpoint_path", None)
        saved_nq = saved.get("n_q") or int(saved["codes"].shape[-1])
        needs_reprefill = (saved_nq and saved_nq != self.n_q) or (
            saved_ckpt and cur_ckpt and saved_ckpt != cur_ckpt
        )
        if needs_reprefill:
            reason = []
            if saved_nq and saved_nq != self.n_q:
                reason.append(f"n_q {saved_nq}->{self.n_q}")
            if saved_ckpt and cur_ckpt and saved_ckpt != cur_ckpt:
                reason.append("checkpoint changed")
            print(f"[TTS] KV stale ({', '.join(reason)}), re-prefilling from codes")
            self._reprefill_from_saved(saved)
            return saved.get("name", path.stem)

        # KV shape sanity check
        saved_kv_dim = saved["kv"][0]["k"].shape[-1]
        model_kv_dim = self.model.decoder.flash_kv_caches[0].k_cache.shape[-1]
        if saved_kv_dim != model_kv_dim:
            raise ValueError(
                f"KV cache head_dim mismatch: saved={saved_kv_dim}, model={model_kv_dim}"
            )

        self.row.reset()
        self.engine.model._cond_bias.copy_(
            saved["cond_bias"].to(self.engine.model._cond_bias.device)
        )

        T = saved["T"]
        for kv, kv_saved in zip(self.model.decoder.flash_kv_caches, saved["kv"]):
            k, v = kv_saved["k"].to(self.device), kv_saved["v"].to(self.device)
            n_kv_heads = kv.k_cache.shape[2]
            if k.dim() == 4 and k.shape[1] == n_kv_heads and k.shape[2] == T:
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
            kv.k_cache[:, :T].copy_(k)
            kv.v_cache[:, :T].copy_(v)
        with torch.inference_mode():
            self.model.decoder.flash_kv_caches[0].seq_lens.fill_(T)

        self.row._prompt_offset = T
        self.prompt_T = T
        self.prompt_codes = self._match_n_q(saved["codes"])
        self.prompt_text = saved.get("text", "")
        # Keep prompt_segments in sync so a later `reprefill` (on /reset) uses
        # the prompt we just loaded, not the one from a prior upload. Codes
        # are matched to current model's n_q so reprefill doesn't fail when
        # set_prompt seeds _codec_ctx with a Q-mismatched buffer.
        self.prompt_segments = [(self.prompt_text, self.prompt_codes)]
        spk = saved.get("spk_token_emb")
        self.row._spk_token = spk.to(self.device) if spk is not None else None
        self.spk_token_emb = self.row._spk_token
        # CRITICAL: seed the codec rolling context with prompt codes, matching
        # what _prefill_row does. Without this, the streaming vocoder's
        # decode_cached_frame runs with no prompt context → garbled audio.
        # Was the root cause of "server outputs rubbish after load_kv" — the
        # direct-KV-copy fast path above restores seq_lens but the per-row
        # CodecCtx buffer is still empty from construction.
        self.row._codec_ctx.set_prompt(self.prompt_codes.to(self.device).T.unsqueeze(0))
        _logf(
            f"[TTS.load_kv] direct-KV path: T={T} "
            f"codes_Q={self.prompt_codes.shape[-1]} codec_ctx_seeded={self.prompt_codes.shape[0]}f "
            f"spk_token={'from_saved' if spk is not None else 'NONE(saved had no spk_token_emb)'}"
        )
        self._seq_reset()
        nf = self.prompt_codes.shape[0]
        nq = self.prompt_codes.shape[1] if self.prompt_codes.dim() > 1 else 0
        self._seq_add(f'[prompt] [spk] "{self.prompt_text}" [{nf}f/Q={nq}]')
        self._seq_add("[SC]")
        self._log_seq("load_kv")
        return saved.get("name", path.stem)

    def _load_safetensors_prompt(self, path: Path) -> str:
        import json as _json

        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
            cfg = _json.loads(metadata.get("config", "{}"))
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        saved_d = int(cfg.get("d_model", 0))
        if saved_d and saved_d != self.model.config.model.d_model:
            raise ValueError(
                f"Prompt d_model mismatch: saved={saved_d}, "
                f"model={self.model.config.model.d_model}"
            )

        saved = {
            "codes": tensors["codes"],
            "text": cfg.get("text", ""),
            "cond_bias": tensors["cond_bias"],
            "spk_token_emb": tensors.get("spk_token_emb"),
        }
        print(f"[TTS] Loading portable prompt {path.name} (re-prefill from codes)")
        self._reprefill_from_saved(saved)
        self._log_seq("load_kv")
        return cfg.get("name", path.stem)

    # --- Per-checkpoint KV cache (load by name, regenerate if stale) ---

    def _kv_path_for(self, name: str) -> Path:
        return PROMPTS_DIR / f"{name}.{_ckpt_id(self.checkpoint_path)}.pt"

    def _find_reprefill_source(self, name: str) -> Path | None:
        """Find any file containing codes+text for `name` (any ckpt)."""
        candidates = sorted(PROMPTS_DIR.glob(f"{name}.*.pt"))
        if candidates:
            return candidates[0]
        legacy = PROMPTS_DIR / f"{name}.pt"
        if legacy.exists():
            return legacy
        st = PROMPTS_DIR / f"{name}.safetensors"
        if st.exists():
            return st
        return None

    def _read_codes_text(self, src: Path) -> tuple[torch.Tensor, str]:
        if src.suffix == ".safetensors":
            import json as _json

            from safetensors import safe_open

            with safe_open(src, framework="pt") as f:
                cfg = _json.loads((f.metadata() or {}).get("config", "{}"))
                codes = f.get_tensor("codes")
            return codes, cfg.get("text", "")
        saved = torch.load(src, map_location="cpu", weights_only=False)
        return saved["codes"], saved.get("text", "")

    def _reprefill_from_wav(self, wav: Path, txt: Path) -> None:
        """Re-run upload pipeline: decode wav → aligned segments → prefill.
        Produces a fresh spk_token + cond_bias against the current model."""
        from julius.resample import resample_frac
        from torchcodec.decoders import AudioDecoder

        QWEN_SR = 24000
        decoder_24k = AudioDecoder(
            wav.read_bytes(), sample_rate=QWEN_SR, num_channels=1
        )
        audio_24k = decoder_24k.get_all_samples().data.squeeze(0)
        max_samples = 180 * QWEN_SR
        if len(audio_24k) > max_samples:
            audio_24k = audio_24k[:max_samples]
        if audio_24k.abs().max() > 0:
            audio_24k = audio_24k / audio_24k.abs().max()
        audio_16k = resample_frac(audio_24k.unsqueeze(0), QWEN_SR, 16000).squeeze(0)
        text = txt.read_text().strip()
        audio_16k_np = audio_16k.numpy()
        audio_24k_np = audio_24k.numpy()
        segments = self.build_prompt_from_audio(audio_16k_np, text, audio_24k_np)
        self.prefill_prompt(
            segments, {}, prompt_audio_24k=torch.from_numpy(audio_24k_np).float()
        )

    def load_kv_by_name(self, name: str) -> str:
        """Load prompt by name, keyed to the active checkpoint.

        Fast path: `{name}.{ckpt_id}.pt` exists → direct KV restore.
        Else: regenerate from a model-agnostic source ({name}.wav+.txt
        preferred for fresh spk_token; otherwise codes+text from any
        sibling .pt/.safetensors), then persist `{name}.{ckpt_id}.pt`
        so future loads hit the fast path.
        """
        # Switching prompt = new session: drop the prior caller's projected
        # [user_spk] token (would otherwise be prepended on the next user turn,
        # priming the new voice's KV with the wrong speaker) and close any
        # open turn-log dir.
        self._user_spk_token = None
        self.end_conversation()

        keyed = self._kv_path_for(name)
        if keyed.exists():
            return self.load_kv(keyed)

        wav = PROMPTS_DIR / f"{name}.wav"
        txt = PROMPTS_DIR / f"{name}.txt"
        cur_id = _ckpt_id(self.checkpoint_path)
        if wav.exists() and txt.exists():
            _logf(f"[TTS.regen] '{name}' from wav+txt for ckpt={cur_id}")
            self._reprefill_from_wav(wav, txt)
        else:
            src = self._find_reprefill_source(name)
            if src is None:
                raise FileNotFoundError(
                    f"no prompt source for '{name}' in {PROMPTS_DIR}"
                )
            _logf(
                f"[TTS.regen] '{name}' from {src.name} (codes-only, no spk audio) "
                f"for ckpt={cur_id}"
            )
            codes, text = self._read_codes_text(src)
            codes = self._match_n_q(codes)
            self.prefill_prompt([(text, codes)], {})

        PROMPTS_DIR.mkdir(exist_ok=True)
        self.save_kv(keyed, name)
        _logf(f"[TTS.regen] saved {keyed.name}")
        return name

    # --- Incremental user-turn prefill ---

    def _reset_user_prefill(self):
        """Reset incremental prefill state for a new user turn."""
        self._user_codes_parts: list[torch.Tensor] = []
        self._user_codes_consumed: int = 0  # frames already prefilled
        self._user_text_prefilled: list[str] = []  # texts already in KV
        self._segment_start_idx: int = 0  # parts index where current segment began

    def _get_aligner(self):
        """Lazy-load wav2vec2 aligner on CPU."""
        if not hasattr(self, "_aligner_model"):
            from vui.align import _load

            self._aligner_model, self._aligner_dict = _load("cpu")
            print("[TTS] wav2vec2 aligner loaded (CPU)")
        return self._aligner_model, self._aligner_dict

    def prefill_user_chunk(
        self,
        text: str,
        stable_end_time: float = 0.0,
        audio_16k=None,
        *,
        final: bool = False,
    ) -> int:
        """Incremental user-turn KV prefill using fwhisper timestamps.

        Each chunk writes a complete text [SC] codes pair to KV,
        matching training format. Codes sliced via stable_end_time * 12.5Hz.
        """
        n_parts = len(self._user_codes_parts)
        self._ensure_user_spk_token(audio_16k)
        all_codes = (
            torch.cat(self._user_codes_parts, dim=0) if self._user_codes_parts else None
        )
        total_frames = all_codes.shape[0] if all_codes is not None else 0
        _logf(
            f"[TTS.chunk] >> {'FINAL' if final else 'incr'} "
            f"text='{text[:60]}' end_t={stable_end_time:.3f}s "
            f"codec_parts={n_parts} total_frames={total_frames} "
            f"consumed={self._user_codes_consumed} "
            f"prefilled_texts={self._user_text_prefilled} "
            f"spk_token={'yes' if self._user_spk_token is not None else 'no'} "
            f"T={self.row.offset}"
        )

        # Compute new text not yet prefilled
        if self._user_text_prefilled:
            already = " ".join(self._user_text_prefilled)
            if text.lower().startswith(already.lower()):
                chunk_text = text[len(already) :].strip()
            else:
                chunk_text = text
                _logf(
                    f"[TTS.chunk] WARN: text doesn't start with prefilled "
                    f"already='{already[:40]}' text='{text[:40]}'"
                )
        else:
            chunk_text = text

        if not final:
            end_frame = min(int(stable_end_time * CODEC_HZ), total_frames)
            start_frame = self._user_codes_consumed

            if not chunk_text or end_frame <= start_frame:
                _logf(
                    f"[TTS.chunk] skip: chunk_text='{chunk_text[:30]}' "
                    f"frames={start_frame}->{end_frame}/{total_frames}"
                )
                return 0

            is_first = self._user_codes_consumed == 0
            if is_first and self._user_spk_token is not None:
                _logf(f"[TTS.chunk] writing [user_spk] token T={self.row.offset}")
                with torch.inference_mode():
                    self.engine._prefill_emb(self.row, self._user_spk_token)

            chunk_codes = all_codes[start_frame:end_frame]
            self._user_codes_consumed = end_frame
            self._user_text_prefilled.append(chunk_text)
            self.row.add_user(text=chunk_text, codes=chunk_codes, final=False)

            n = chunk_codes.shape[0]
            spk = (
                "[user_spk] " if (is_first and self._user_spk_token is not None) else ""
            )
            self._seq_add(f'{spk}[user] "{chunk_text}" [{n}f]')
            _logf(
                f"[TTS.chunk] KV written: '{chunk_text[:50]}' "
                f"frames={start_frame}->{end_frame} ({n}f) T={self.row.offset}"
            )
            return n

        # FINAL: remaining text + codes
        is_first = self._user_codes_consumed == 0
        if is_first and self._user_spk_token is not None:
            _logf(f"[TTS.chunk] FINAL: writing [user_spk] token T={self.row.offset}")
            with torch.inference_mode():
                self.engine._prefill_emb(self.row, self._user_spk_token)

        remaining_codes = (
            all_codes[self._user_codes_consumed :] if all_codes is not None else None
        )
        if remaining_codes is not None and remaining_codes.shape[0] == 0:
            remaining_codes = None
        n_remaining = remaining_codes.shape[0] if remaining_codes is not None else 0

        _logf(
            f"[TTS.chunk] FINAL: chunk_text='{(chunk_text or '')[:50]}' "
            f"remaining_codes={n_remaining}f "
            f"consumed={self._user_codes_consumed}/{total_frames}"
        )

        if chunk_text:
            # New text not yet prefilled — write full text [SC] codes pair
            self.row.add_user(text=chunk_text, codes=remaining_codes, final=True)
            spk = (
                "[user_spk] " if (is_first and self._user_spk_token is not None) else ""
            )
            self._seq_add(f'{spk}[user] "{chunk_text}" [SC] [{n_remaining}f]')
        else:
            # Text already prefilled by incrementals — write [SC] then trailing audio
            with torch.inference_mode():
                self.engine._prefill_emb(self.row, self.engine._sc_emb)
                if remaining_codes is not None:
                    self.engine._prefill_emb(
                        self.row, self.engine._audio_emb(remaining_codes)
                    )
                    self.row._codec_ctx.add(
                        remaining_codes.T.unsqueeze(0).to(self.engine.device)
                    )
            self._seq_add(f"[SC] [+{n_remaining}f trailing codes]")

        _logf(f"[TTS.chunk] FINAL done T={self.row.offset}")
        self._log_seq("prefill_user_chunk FINAL")
        self._log_turn("user", text or "", all_codes)
        return n_remaining

    def align_prefill_chunk(
        self,
        text: str,
        audio_16k,
        *,
        final: bool = False,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> int:
        """Align text to audio, slice accumulated codec codes, prefill into KV.

        Called incrementally as ASR lines finalize during user speech.
        - final=False: intermediate chunk — align text to audio, slice codes, prefill without [SC]
        - final=True: closing chunk — text is the FULL turn text; we compute the
          remaining tail (not yet prefilled) and use all remaining codes + [SC].

        start_time/end_time from Moonshine are used to slice the audio for
        wav2vec2 (avoids aligning the full recording every time).
        """
        import numpy as np

        from vui.align import align_words

        # For the final call, compute remaining text not yet prefilled
        if final and self._user_text_prefilled:
            already = " ".join(self._user_text_prefilled)
            if text.startswith(already):
                remaining = text[len(already) :].strip()
            else:
                remaining = text
            chunk_text = remaining
        else:
            chunk_text = text

        # Concatenate all accumulated codes
        if not self._user_codes_parts:
            if final:
                self.row.add_user(text=chunk_text, codes=None, final=True)
            return 0

        all_codes = torch.cat(self._user_codes_parts, dim=0)  # (T_total, Q)
        total_frames = all_codes.shape[0]

        if not final:
            self._get_aligner()
            audio_np = (
                audio_16k if isinstance(audio_16k, np.ndarray) else audio_16k.numpy()
            )

            # Slice audio around the Moonshine timing (with 0.3s margin)
            margin = 0.3
            sr = 16000
            s0 = max(0, int((start_time - margin) * sr))
            s1 = min(len(audio_np), int((end_time + margin) * sr))
            audio_slice = audio_np[s0:s1]
            slice_offset = s0 / sr  # seconds offset of slice start

            t0 = time.perf_counter()
            words = align_words(
                torch.from_numpy(audio_slice).float(), chunk_text, device="cpu"
            )
            align_ms = (time.perf_counter() - t0) * 1000

            if not words:
                print(
                    f"[TTS] Alignment failed for '{chunk_text[:40]}' ({align_ms:.0f}ms), skipping"
                )
                return 0

            # Word timings are relative to slice — convert to absolute
            abs_end = slice_offset + words[-1]["end"]
            end_frame = min(int(abs_end * CODEC_HZ), total_frames)
            print(
                f"[TTS] Aligned '{chunk_text[:40]}' -> end={abs_end:.3f}s frame={end_frame} "
                f"({align_ms:.0f}ms, audio_slice={s0/sr:.1f}-{s1/sr:.1f}s)"
            )
        else:
            end_frame = total_frames

        # Slice codes for this chunk
        start_frame = self._user_codes_consumed
        if end_frame <= start_frame:
            if final:
                self.row.add_user(text=chunk_text, codes=None, final=True)
            return 0

        chunk_codes = all_codes[start_frame:end_frame]
        self._user_codes_consumed = end_frame
        if not final:
            self._user_text_prefilled.append(text)

        self.row.add_user(text=chunk_text, codes=chunk_codes, final=final)
        n = chunk_codes.shape[0]
        ctx_buf = self.row._codec_ctx._buf
        ctx_frames = ctx_buf.shape[2] if ctx_buf is not None else 0
        ctx_q = ctx_buf.shape[1] if ctx_buf is not None else 0
        _logf(
            f"[TTS.align] chunk='{chunk_text[:40]}' "
            f"frames={start_frame}-{end_frame} ({n}f), "
            f"final={final}, spk_prefixed=NO, "
            f"T={self.row.offset}, codec_ctx={ctx_frames}f/Q={ctx_q}"
        )
        if final:
            self._log_turn("user", text, all_codes)
        return n

    # --- Codec encode ---

    def encode_full(self, audio_np) -> torch.Tensor | None:
        audio_t = torch.from_numpy(audio_np).float().cuda().to(self.codec_dtype).reshape(1, 1, -1)
        codes = self.codec_enc.encode(audio_t)
        return codes[0, : self.n_q].T.long().cpu()

    def stream_start(self, *, continuation: bool = False):
        """Begin a fresh codec-encode session for the user's voice.

        With `continuation=True` we keep the accumulated incremental
        prefill state (`_user_codes_parts`, `_user_codes_consumed`,
        `_user_text_prefilled`) so a brief mid-utterance pause-resume
        doesn't throw away prior chunks already written to KV.
        """
        self.stream_enc.reset()
        if not continuation:
            self._reset_user_prefill()
        # Mark where this segment starts in `_user_codes_parts`. On
        # stream_stop we splice the re-encoded segment in place of its
        # streamed partials (re-encoding the whole audio buffer is more
        # accurate than per-frame streaming output) without clobbering
        # earlier-segment codes that incremental prefills already
        # depend on.
        self._segment_start_idx = len(self._user_codes_parts)
        self._stream_samples_fed = 0
        self._stream_frames_emitted = 0
        self._stream_start_t = time.monotonic()
        _logf(
            f"[TTS.stream_start] buf reset"
            f"{' (continuation, prior_parts=' + str(self._segment_start_idx) + ')' if continuation else ''}"
        )

    def stream_flush(self) -> torch.Tensor | None:
        """Flush encoder buffer without resetting — captures pending frames."""
        codes = self.stream_enc.flush()
        if codes is not None:
            codes_cpu = codes.cpu()
            self._user_codes_parts.append(codes_cpu)
            self._stream_frames_emitted += codes_cpu.shape[0]
            _logf(
                f"[TTS.stream_flush] +{codes_cpu.shape[0]}f (total={self._stream_frames_emitted})"
            )
            return codes_cpu
        _logf("[TTS.stream_flush] nothing to flush")
        return None

    def stream_feed(self, audio_np) -> torch.Tensor | None:
        audio_t = self._normalize_audio(torch.from_numpy(audio_np).float())
        self._stream_samples_fed += audio_t.numel()
        codes = self.stream_enc.feed(audio_t)
        if codes is not None:
            codes_cpu = codes.cpu()
            self._user_codes_parts.append(codes_cpu)
            self._stream_frames_emitted += codes_cpu.shape[0]
            fed_s = self._stream_samples_fed / 24000
            _logf(
                f"[TTS.stream_feed] +{codes_cpu.shape[0]}f "
                f"total={self._stream_frames_emitted}f ({fed_s:.1f}s audio)"
            )
            return codes_cpu
        return None

    def stream_stop(self) -> torch.Tensor | None:
        self.stream_enc.flush()
        torch.cuda.synchronize()
        total_frames = self.stream_enc._emitted
        from vui.qwen_codec import DOWNSAMPLE_RATE as _DS

        expected = self._stream_samples_fed // _DS
        fed_s = self._stream_samples_fed / 24000
        dur_s = time.monotonic() - self._stream_start_t
        _logf(
            f"[TTS.stream_stop] fed_samples={self._stream_samples_fed} "
            f"({fed_s:.2f}s audio in {dur_s:.2f}s wall) "
            f"expected~{expected}f, encoder_emitted={total_frames}f, "
            f"parts_emitted_sum={self._stream_frames_emitted}f"
            f"{'  MISMATCH' if abs(total_frames - expected) > 3 else ''}"
        )
        if total_frames == 0:
            return None
        if self.stream_enc._graph is not None:
            self.stream_enc._graph.replay()
            return self.stream_enc._codes_out[0, :, :total_frames].T.long().cpu()
        else:
            c = self.stream_enc.encoder(self.stream_enc._audio_buf)
            return c[0, :, :total_frames].T.long().cpu()


# ---------------------------------------------------------------------------
# Process entry point — thin dispatch over TTSEngine
# ---------------------------------------------------------------------------


def tts_process(
    cmd_queue: Queue, audio_queue: Queue, checkpoint_path: str, cancel_event=None
):
    import multiprocessing
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if cancel_event is None:
        cancel_event = multiprocessing.Event()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    _inf_ctx = torch.inference_mode()
    _inf_ctx.__enter__()

    _migrate_legacy_prompts()

    engine = TTSEngine.from_checkpoint(checkpoint_path)
    audio_queue.put({"type": "ready"})

    debug_dir = Path("debug_dump")
    debug_dir.mkdir(exist_ok=True)

    while True:
        try:
            msg = cmd_queue.get()
        except Exception:
            break

        cmd = msg.get("cmd")
        t_cmd = time.perf_counter()
        if cmd != "stream_feed":
            _logf(f"[TTS.cmd] >> {cmd} T={engine.T}")

        if cmd == "shutdown":
            engine.end_conversation()
            break

        elif cmd == "reset":
            _logf("=" * 60)
            _logf("RESET — ending conversation, resetting engine")
            _logf("=" * 60)
            engine.end_conversation()
            engine.reset()
            audio_queue.put({"type": "done", "T": 0, "total_secs": 0})

        elif cmd == "rewind":
            T = engine.rewind()
            print(f"[TTS] Rewind -> T={T}")
            audio_queue.put({"type": "done", "T": T, "total_secs": 0})

        elif cmd == "reprefill":
            # Re-prefill prompt from stored segments with new settings (e.g. n_codebooks changed)
            try:
                settings = msg.get("settings", {})
                if engine.prompt_segments is not None:
                    n_cb = int(settings.get("n_codebooks", 0))
                    print(f"[TTS] Re-prefilling prompt with n_codebooks={n_cb}")
                    engine.prefill_prompt(engine.prompt_segments, settings)
                    audio_queue.put(
                        {
                            "type": "prompt_loaded",
                            "T": engine.T,
                            "text": engine.prompt_text or "",
                        }
                    )
                else:
                    engine.reset()
                    audio_queue.put({"type": "done", "T": 0, "total_secs": 0})
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "set_cond":
            engine.set_conditioning(
                wps=msg.get("wps", 0),
                sq=msg.get("sq"),
            )

        elif cmd == "allocate_and_prefill":
            try:
                audio_24k_np = msg.get("audio_24k")
                prompt_audio_24k = (
                    torch.from_numpy(audio_24k_np).float()
                    if audio_24k_np is not None
                    else None
                )
                audio_16k_np = msg["audio_16k"]
                full_text = msg.get("text") or ""
                print(
                    f"[TTS] Building prompt: {len(audio_16k_np)/16000:.1f}s audio, "
                    f"text='{full_text[:60]}'"
                )
                segments = engine.build_prompt_from_audio(
                    audio_16k_np, full_text, audio_24k_np
                )
                engine.prefill_prompt(
                    segments,
                    msg.get("settings", {}),
                    prompt_audio_24k=prompt_audio_24k,
                )
                torch.save(
                    {
                        "segments": segments,
                        "text": engine.prompt_text,
                        "T": engine.T,
                        "settings": msg.get("settings", {}),
                    },
                    debug_dir / "prompt.pt",
                )
                _logf(
                    f"[TTS.cmd] << allocate_and_prefill T={engine.T} {(time.perf_counter()-t_cmd)*1000:.0f}ms"
                )
                prompt_rms = 0.0
                if prompt_audio_24k is not None and prompt_audio_24k.numel() > 0:
                    prompt_rms = prompt_audio_24k.pow(2).mean().sqrt().item()
                audio_queue.put(
                    {
                        "type": "prompt_loaded",
                        "T": engine.T,
                        "text": engine.prompt_text or "",
                        "prompt_rms": prompt_rms,
                    }
                )
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "generate":
            if cancel_event.is_set():
                _logf(f"[TTS.cmd] << generate SKIPPED (cancelled) T={engine.T}")
                continue
            try:
                tts_text = msg["text"]
                settings = msg.get("settings", {})

                result = engine.generate(
                    tts_text,
                    settings,
                    cancel_event,
                    audio_queue=audio_queue,
                    is_new_turn=msg.get("new_turn", True),
                    is_final_chunk=msg.get("is_final", False),
                    context=msg.get("context", ""),
                )
                status = "CANCELLED" if result["cancelled"] else "done"
                _logf(
                    f"[TTS.cmd] << generate {status} T={engine.T} "
                    f"{result['total_frames']}f/{result['total_secs']:.1f}s "
                    f"in {(time.perf_counter()-t_cmd)*1000:.0f}ms"
                )

                if engine._conv_dir is not None:
                    engine._conv_turn_idx += 1
                    torch.save(
                        {
                            "role": "gen",
                            "text": tts_text,
                            "codes": result["codes"],
                            "T_start": engine.T - result["total_frames"],
                            "T_end": engine.T,
                            "frames": result["total_frames"],
                            "secs": result["total_secs"],
                            "settings": settings,
                        },
                        engine._conv_dir / f"{engine._conv_turn_idx:02d}_gen.pt",
                    )

                if result["cancelled"]:
                    audio_queue.put({"type": "cancelled", "T": engine.T})
                else:
                    audio_queue.put(
                        {
                            "type": "done",
                            "T": engine.T,
                            "total_secs": result["total_secs"],
                            "total_frames": result["total_frames"],
                            "total_gen_time": result["total_gen_time"],
                        }
                    )
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "prefill_user_turn":
            try:
                if DEBUG and engine._conv_dir is None:
                    engine.start_conversation(debug_dir)
                _text = msg.get("text", "")
                _codes = msg.get("codes")
                _n_codes = _codes.shape[0] if _codes is not None else 0
                _n_words = len(_text.split()) if _text else 0
                _logf(
                    f"[TTS.cmd] >> prefill_user_turn: text={_n_words}w '{_text[:60]}', "
                    f"codes={_n_codes}f ({_n_codes/12.5:.2f}s), "
                    f"audio_16k={'yes' if msg.get('audio_16k') is not None else 'no'}"
                )
                engine.prefill_user_turn(
                    text=_text,
                    codes=_codes,
                    audio_16k=msg.get("audio_16k"),
                    settings=msg.get("settings"),
                )
                _logf(
                    f"[TTS.cmd] << prefill_user_turn T={engine.T} {(time.perf_counter()-t_cmd)*1000:.0f}ms"
                )
                audio_queue.put({"type": "user_prefilled", "T": engine.T})
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "prefill_text":
            try:
                engine.prefill_text(msg["text"])
                _logf(
                    f"[TTS.cmd] << prefill_text T={engine.T} {(time.perf_counter()-t_cmd)*1000:.0f}ms"
                )
                audio_queue.put({"type": "text_prefilled", "T": engine.T})
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "prefill_text_sc":
            try:
                sc_prob = engine.prefill_text_sc(msg["text"])
                _logf(
                    f"[TTS.cmd] << prefill_text_sc T={engine.T} sc={sc_prob:.3f} {(time.perf_counter()-t_cmd)*1000:.0f}ms"
                )
                audio_queue.put(
                    {
                        "type": "sc_prob",
                        "prob": sc_prob,
                        "text": msg["text"],
                        "T": engine.T,
                    }
                )
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "save_kv":
            try:
                name = msg["name"]
                PROMPTS_DIR.mkdir(exist_ok=True)
                safe_name = re.sub(r"[^\w\-]", "_", name)
                engine.save_kv(engine._kv_path_for(safe_name), name)
                audio_queue.put({"type": "kv_saved", "name": name, "ok": True})
            except Exception as e:
                traceback.print_exc()
                audio_queue.put(
                    {
                        "type": "kv_saved",
                        "name": msg.get("name", ""),
                        "ok": False,
                        "msg": str(e),
                    }
                )

        elif cmd == "load_kv":
            try:
                name = msg.get("name") or msg["file"]
                loaded_name = engine.load_kv_by_name(name)
                audio_queue.put(
                    {
                        "type": "kv_loaded",
                        "name": loaded_name,
                        "ok": True,
                        "T": engine.T,
                        "text": engine.prompt_text,
                    }
                )
            except Exception:
                traceback.print_exc()
                audio_queue.put(
                    {
                        "type": "kv_loaded",
                        "name": msg.get("file", ""),
                        "ok": False,
                        "T": 0,
                        "text": "",
                    }
                )

        elif cmd == "cancel":
            rewind_to = msg.get("rewind_to")
            if rewind_to is not None and rewind_to < engine.T:
                _logf(f"[TTS.cmd] cancel: rewinding T={engine.T} -> {rewind_to}")
                engine.T = rewind_to
                # Trim seq log so debug output reflects actual KV state.
                engine._seq_trim_to(rewind_to)
            else:
                _logf(f"[TTS.cmd] cancel: T={engine.T}")
            cancel_event.clear()

        elif cmd == "get_state":
            audio_queue.put(
                {
                    "type": "state",
                    "T": engine.T,
                    "max_T": engine.max_seqlen,
                    "has_prompt": engine.prompt_codes is not None,
                    "prompt_text": engine.prompt_text or "",
                }
            )

        elif cmd == "encode_full":
            try:
                t0 = time.perf_counter()
                codes = engine.encode_full(msg["audio"])
                t_enc = (time.perf_counter() - t0) * 1000
                print(f"[TTS] Full encode: {t_enc:.1f}ms, {codes.shape[0]} frames")
                audio_queue.put({"type": "encoded", "codes": codes})
            except Exception:
                traceback.print_exc()
                audio_queue.put({"type": "encoded", "codes": None})

        elif cmd == "decode_prompt":
            try:
                if engine.prompt_codes is not None:
                    codes = engine.prompt_codes.long().to(engine.device).T.unsqueeze(0)
                    with torch.autocast("cuda", enabled=False):
                        audio = engine.codec_dec.decode_chunked(codes, chunk_size=200)
                    wav = audio[0, 0].cpu().float()
                    audio_queue.put(
                        {
                            "type": "prompt_audio",
                            "audio": wav,
                            "sample_rate": 24000,
                            "codes_shape": list(engine.prompt_codes.shape),
                        }
                    )
                else:
                    audio_queue.put({"type": "prompt_audio", "audio": None})
            except Exception:
                traceback.print_exc()
                audio_queue.put({"type": "prompt_audio", "audio": None})

        elif cmd == "stream_start":
            engine.stream_start(continuation=bool(msg.get("continuation", False)))

        elif cmd == "stream_flush":
            try:
                codes = engine.stream_flush()
                if codes is not None:
                    audio_queue.put({"type": "codes", "codes": codes})
            except Exception:
                traceback.print_exc()

        elif cmd == "stream_feed":
            try:
                codes = engine.stream_feed(msg["audio"])
                if codes is not None:
                    audio_queue.put({"type": "codes", "codes": codes})
            except Exception:
                traceback.print_exc()

        elif cmd == "stream_stop":
            try:
                t0 = time.perf_counter()
                old_parts = len(engine._user_codes_parts)
                old_frames = sum(p.shape[0] for p in engine._user_codes_parts)
                codes = engine.stream_stop()
                t_flush = (time.perf_counter() - t0) * 1000
                n = codes.shape[0] if codes is not None else 0
                print(
                    f"[TTS] Stream stop: {t_flush:.1f}ms "
                    f"parts={old_parts}({old_frames}f) -> re-encoded={n}f "
                    f"consumed={engine._user_codes_consumed}f"
                )
                # Splice the re-encoded segment into _user_codes_parts in
                # place of its streamed partials. With continuation across
                # pause-resume we keep prior segments' codes intact —
                # those are referenced by prior incremental prefills.
                if codes is not None:
                    prior = engine._user_codes_parts[: engine._segment_start_idx]
                    engine._user_codes_parts = prior + [codes]
                audio_queue.put({"type": "codes_final", "n_frames": n})
            except Exception:
                traceback.print_exc()
                audio_queue.put({"type": "codes_final", "n_frames": 0})

        elif cmd == "prefill_user_chunk":
            final = msg.get("final", False)
            reply_type = "user_prefilled" if final else "user_chunk_prefilled"
            try:
                n = engine.prefill_user_chunk(
                    text=msg.get("text", ""),
                    stable_end_time=msg.get("stable_end_time", 0.0),
                    audio_16k=msg.get("audio_16k"),
                    final=final,
                )
                audio_queue.put(
                    {"type": reply_type, "frames": n, "final": final, "T": engine.T}
                )
            except Exception:
                traceback.print_exc()
                audio_queue.put(
                    {"type": reply_type, "frames": 0, "final": final, "T": engine.T}
                )

        elif cmd == "align_prefill_chunk":
            final = msg.get("final", False)
            reply_type = "user_prefilled" if final else "user_chunk_prefilled"
            try:
                n = engine.align_prefill_chunk(
                    text=msg.get("text", ""),
                    audio_16k=msg.get("audio_16k"),
                    final=final,
                    start_time=msg.get("start_time", 0.0),
                    end_time=msg.get("end_time", 0.0),
                )
                audio_queue.put(
                    {"type": reply_type, "frames": n, "final": final, "T": engine.T}
                )
            except Exception:
                traceback.print_exc()
                audio_queue.put(
                    {"type": reply_type, "frames": 0, "final": final, "T": engine.T}
                )

    print("[TTS] Shutting down")
