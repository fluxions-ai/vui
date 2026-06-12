"""TTS worker process for Apple Silicon using MLX.

Same queue interface as tts_worker.py but uses MLX for the main model
and MPS for codec encode/decode. No CUDA graphs needed.
"""

from __future__ import annotations

import os
import time
import traceback
from multiprocessing import Queue
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from vui.inference import simple_clean
from vui.mlx.tts.generate import (
    CODEC_HZ,
    RepPenalty,
    _sample_top_k,
    compute_cond_bias,
)
from vui.mlx.tts.model import VuiMLX
from vui.mlx.tts.weights import load_quantized
from vui.serving.stream.tts_worker import chunk_text

PROMPTS_DIR = Path("prompts")
SQ_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]


class MLXTTSEngine:
    def __init__(self, model: VuiMLX, codec_dec, codec_enc):
        self.model = model
        self.tok = model.text_tokenizer
        self.codec_dec = codec_dec
        self.codec_enc = codec_enc
        self.n_q = model.audio_emb.n_quantizers
        self.sc_id = model.sc_id

        self.Q = model.rq_transformer.n_quantizers
        self.CS = model.rq_transformer.codebook_size

        # Mutable state
        self.T = 0
        self.prompt_T = 0
        self.prompt_codes = None  # mx.array
        self.prompt_text = None
        self.cond_bias = mx.zeros((1, 1, model.d_model))
        self.codec_ctx: list[torch.Tensor] = []

        self._conv_dir: Path | None = None
        self._conv_turn_idx = 0

        # Compile RQ blocks for faster decode
        model.rq_transformer.compile_forward()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "MLXTTSEngine":
        precision = os.environ.get("VUI_MLX_PRECISION", "int8")
        print(f"[TTS-MLX] Loading model ({precision})...")
        model, config = load_quantized(checkpoint_path, precision)

        print("[TTS-MLX] Loading MLX codec decoder...")
        from vui.mlx.tts.codec import load_codec_decoder_mlx

        mlx_codec = load_codec_decoder_mlx()

        print("[TTS-MLX] Loading codec encoder + speaker encoder...")
        from vui.qwen_codec import QwenCodecEncoder
        codec_enc = QwenCodecEncoder.from_pretrained().cpu().float().eval()
        from vui.qwen_spk_enc import QwenSpeakerEncoder
        spk_enc = QwenSpeakerEncoder.from_pretrained()

        engine = cls(model, None, codec_enc)
        engine._mlx_codec = mlx_codec
        engine._spk_enc = spk_enc
        engine.checkpoint_path = checkpoint_path

        engine.set_conditioning(sq=SQ_DEFAULTS)
        print("[TTS-MLX] Ready!")
        return engine

    def _warmup(self):
        model = self.model
        model.decoder.reset_cache()
        model.decoder.make_cache()
        ids = self.tok.encode("Warmup test.")
        ids_mx = mx.array(ids.numpy().astype(np.int32))
        text_emb = model.token_emb(ids_mx[None]) + self.cond_bias
        out = model.decoder(text_emb)
        hidden = out[:, -1:]
        code0 = _sample_top_k(model.codec_head(hidden[:, 0]) / 0.8, 300)
        model.rq_transformer.generate(hidden[:, 0], code0, 0.5, 300)
        mx.eval(model.decoder.kv_caches[0].state)
        model.decoder.reset_cache()

    # --- State management ---

    def reset(self):
        self.model.decoder.reset_cache()
        self.T = 0
        self.codec_ctx = []

    def rewind(self) -> int:
        if (
            self.prompt_T > 0
            and self.prompt_codes is not None
            and self.prompt_text is not None
        ):
            self.reset()
            self._do_prefill_prompt(self.prompt_codes, self.prompt_text)
        else:
            self.reset()
        return self.T

    # --- Conditioning ---

    def set_conditioning(self, wps: float = 0.0, sq: list | None = None):
        self.cond_bias = compute_cond_bias(self.model, sq=sq, wps=wps)

    # --- Prefill ---

    def _do_prefill_prompt(self, codes_mx: mx.array, text: str):
        from vui.mlx.tts.generate import prefill_prompt

        prefill_prompt(self.model, text, codes_mx)
        # Count tokens for T tracking
        ids = self.tok.encode(simple_clean(text))
        n_text = len(ids)
        pc = codes_mx[0].T if codes_mx.ndim == 3 else codes_mx
        n_audio = pc.shape[0]
        self.T = n_text + n_audio

    def prefill_prompt(self, codes, text: str, settings: dict, prompt_audio_24k=None):
        self.set_conditioning(
            wps=settings.get("wps_score", 1.0),
            sq=settings.get("sq_scores"),
        )

        self.model.decoder.reset_cache()
        self.model.decoder.make_cache()

        # Encode text (no cond_bias on prompt)
        if text:
            ids = self.tok.encode(simple_clean(text))
            ids_mx = mx.array(ids.numpy().astype(np.int32))
            # Append [SC] to prompt text
            ids_mx = mx.concatenate([ids_mx, mx.array([self.sc_id])])
            text_emb = self.model.token_emb(ids_mx[None])
            self.model.decoder(text_emb)
            n_text = ids_mx.shape[0]
        else:
            n_text = 0

        # Convert codes to MLX if needed
        if isinstance(codes, torch.Tensor):
            codes_mx = mx.array(codes.numpy().astype(np.int32))
        else:
            codes_mx = codes

        # codes_mx: (1, Q, T) or (T, Q)
        if codes_mx.ndim == 3:
            pc = codes_mx[0].T  # (T, Q)
        else:
            pc = codes_mx

        # Truncate codebooks if requested
        n_cb = int(settings.get("n_codebooks", 0))
        if n_cb > 0 and pc.shape[-1] > n_cb:
            padded = mx.zeros((pc.shape[0], self.n_q), dtype=pc.dtype)
            padded = padded.at[:, :n_cb].add(pc[:, :n_cb])
            pc = padded

        audio_emb = self.model.audio_emb(pc)[None]
        self.model.decoder(audio_emb)
        n_audio = pc.shape[0]

        mx.eval([c.state for c in self.model.decoder.kv_caches])

        self.T = n_text + n_audio
        self.prompt_T = self.T
        self.prompt_codes = codes_mx
        self.prompt_text = text

        # Seed codec context
        if isinstance(codes, torch.Tensor):
            pc_cpu = codes.cpu() if codes.is_cuda else codes
        else:
            pc_cpu = torch.from_numpy(np.array(codes_mx))
        if pc_cpu.dim() == 3:
            pc_cpu = pc_cpu.squeeze(0)
        self.codec_ctx = [pc_cpu[i] for i in range(pc_cpu.shape[0])]

    def prefill_user_turn(
        self, text: str = "", codes=None, audio_16k=None, settings: dict | None = None
    ):
        if codes is not None and audio_16k is not None:
            from vui.serving.stream.tts_worker import TTSEngine

            codes = TTSEngine.vad_trim_codes(codes, audio_16k)

        if not self.model.decoder.kv_caches:
            self.model.decoder.make_cache()

        if text:
            ids = self.tok.encode(simple_clean(text))
            ids_mx = mx.array(ids.numpy().astype(np.int32))
            ids_mx = mx.concatenate([ids_mx, mx.array([self.sc_id])])
            t_emb = self.model.token_emb(ids_mx[None])
            self.model.decoder(t_emb)
            mx.eval([c.state for c in self.model.decoder.kv_caches])
            self.T += ids_mx.shape[0]

        if codes is not None:
            if isinstance(codes, torch.Tensor):
                codes_mx = mx.array(codes.numpy().astype(np.int32))
            else:
                codes_mx = codes
            if codes_mx.ndim == 3:
                codes_mx = (
                    codes_mx[0].T
                    if codes_mx.shape[0] == 1
                    else codes_mx.reshape(-1, codes_mx.shape[-1])
                )
            elif codes_mx.ndim == 2:
                pass  # (T, Q) already

            n_cb = int((settings or {}).get("n_codebooks", 0))
            if n_cb > 0 and codes_mx.shape[-1] > n_cb:
                padded = mx.zeros((codes_mx.shape[0], self.n_q), dtype=codes_mx.dtype)
                padded = padded.at[:, :n_cb].add(codes_mx[:, :n_cb])
                codes_mx = padded

            a_emb = self.model.audio_emb(codes_mx)[None]
            self.model.decoder(a_emb)
            mx.eval([c.state for c in self.model.decoder.kv_caches])
            self.T += codes_mx.shape[0]

        if codes is not None:
            if isinstance(codes, torch.Tensor):
                uc = codes.cpu() if codes.is_cuda else codes
            else:
                uc = torch.from_numpy(np.array(codes))
            if uc.dim() == 3:
                uc = uc.squeeze(0)
            for i in range(uc.shape[0]):
                self.codec_ctx.append(uc[i])

        n_codes = codes.shape[0] if codes is not None else 0
        print(
            f"[TTS-MLX] Prefilled user turn: text='{text[:60]}', codes={n_codes} frames, T={self.T}"
        )
        self._log_turn("user", text, codes)

    def prefill_text(self, text: str):
        if not self.model.decoder.kv_caches:
            self.model.decoder.make_cache()
        ids = self.tok.encode(simple_clean(text))
        ids_mx = mx.array(ids.numpy().astype(np.int32))
        emb = self.model.token_emb(ids_mx[None]) + self.cond_bias
        self.model.decoder(emb)
        mx.eval([c.state for c in self.model.decoder.kv_caches])
        self.T += ids_mx.shape[0]

    def prefill_text_sc(self, text: str) -> float:
        if not self.model.decoder.kv_caches:
            self.model.decoder.make_cache()
        ids = self.tok.encode(simple_clean(text))
        ids_mx = mx.array(ids.numpy().astype(np.int32))
        emb = self.model.token_emb(ids_mx[None])
        out = self.model.decoder(emb)
        mx.eval([c.state for c in self.model.decoder.kv_caches])
        self.T += ids_mx.shape[0]

        hidden = out[:, -1]
        text_logits = hidden @ self.model.token_emb.weight.T
        sc_prob = float(mx.softmax(text_logits[0], axis=-1)[self.sc_id].item())
        return sc_prob

    # --- Generation ---

    def generate_turn(
        self,
        chunk: dict,
        cancel_event,
        audio_queue: Queue | None = None,
        *,
        temperature: float = 0.8,
        top_k: int = 300,
        rep_penalty: float = 1.4,
        rep_window: int = 24,
        n_codebooks: int = 0,
        remaining_frames: int = 0,
    ):
        t_turn_start = time.perf_counter()
        model = self.model
        Q, CS = self.Q, self.CS
        rep = RepPenalty(Q, CS, rep_penalty, rep_window)
        rq_temp = temperature

        min_frames = int(0.5 * CODEC_HZ)
        n_words = len(chunk["text"].split())
        word_limit = max(3.0, min(15.0, n_words / 3.0 * 2.0))
        max_per_turn = int(word_limit * CODEC_HZ)
        if remaining_frames > 0:
            max_per_turn = min(max_per_turn, remaining_frames)

        rq_Q = n_codebooks if n_codebooks > 0 else Q

        if not model.decoder.kv_caches:
            model.decoder.make_cache()

        def emit(msg):
            if audio_queue is not None:
                audio_queue.put(msg)

        def decode_audio_frame(codes_mx):
            # codes_mx: (Q,) MLX array -> (1, Q, 1) for codec
            frame = codes_mx[None, :, None]
            if n_codebooks > 0:
                frame = frame[:, :n_codebooks]
            audio = self._mlx_codec.decode_frame(frame)
            mx.eval(audio)
            return torch.from_numpy(np.array(audio.flatten())).float()

        total_secs = 0
        total_frames = 0
        cancelled = False

        # Text prefill
        t_prefill = time.perf_counter()
        ids = self.tok.encode(chunk["text"])
        ids_mx = mx.array(ids.numpy().astype(np.int32))
        text_emb = model.token_emb(ids_mx[None]) + self.cond_bias
        out = model.decoder(text_emb)
        self.T += ids_mx.shape[0]
        mx.eval([c.state for c in model.decoder.kv_caches])
        prefill_ms = (time.perf_counter() - t_prefill) * 1000

        # First frame
        t_first = time.perf_counter()
        hidden = out[:, -1]
        code0 = _sample_top_k(model.codec_head(hidden) / temperature, top_k)

        logit_bias = rep.rq_logit_bias()
        first_codes = model.rq_transformer.generate(
            hidden, code0, rq_temp, top_k, logit_bias, max_q=rq_Q
        )
        mx.eval(first_codes)
        rep.update(first_codes[0])

        all_codes = [first_codes[0]]

        # Prefill MLX codec with context
        self._mlx_codec.reset_state()
        if self.codec_ctx:
            ctx_codes_pt = torch.stack(self.codec_ctx).T.unsqueeze(0)
            if n_codebooks > 0:
                ctx_codes_pt = ctx_codes_pt[:, :n_codebooks]
            ctx_mx = mx.array(ctx_codes_pt.numpy().astype(np.int32))
            self._mlx_codec.prefill(ctx_mx)
            mx.eval(self._mlx_codec.parameters())

        audio = decode_audio_frame(first_codes[0])
        first_frame_ms = (time.perf_counter() - t_first) * 1000
        ttfb_ms = (time.perf_counter() - t_turn_start) * 1000

        emit(
            {
                "type": "timing",
                "prefill_ms": prefill_ms,
                "prefill_tokens": int(ids_mx.shape[0]),
                "first_frame_ms": first_frame_ms,
                "ttfb_ms": ttfb_ms,
            }
        )

        total_secs += 1 / CODEC_HZ
        total_frames += 1
        emit({"type": "audio", "data": audio, "T": self.T, "secs": 1 / CODEC_HZ})

        codes_in = first_codes[0]

        for step in range(1, max_per_turn):
            if cancel_event.is_set():
                cancelled = True
                break

            # Decode step
            emb = model.audio_emb(codes_in[None])[None]
            h = model.decoder(emb)
            hidden = h[:, 0]
            self.T += 1

            cb0_logits = model.codec_head(hidden)
            eos_logit = model.eos_head(hidden)

            penalised = rep.apply_cb0(cb0_logits)
            code0 = _sample_top_k(penalised / temperature, top_k)

            logit_bias = rep.rq_logit_bias()
            codes_frame = model.rq_transformer.generate(
                hidden, code0, rq_temp, top_k, logit_bias, max_q=rq_Q
            )

            mx.eval(codes_frame, eos_logit)

            if step >= min_frames and float(mx.sigmoid(eos_logit).item()) > 0.5:
                break

            rep.update(codes_frame[0])
            all_codes.append(codes_frame[0])
            codes_in = codes_frame[0]

            audio = decode_audio_frame(codes_frame[0])
            total_secs += 1 / CODEC_HZ
            total_frames += 1
            emit({"type": "audio", "data": audio, "T": self.T, "secs": 1 / CODEC_HZ})

            if step % 50 == 0:
                mx.clear_cache()

        # Add generated codes to codec context
        for c in all_codes:
            c_np = np.array(c)
            c_pt = torch.from_numpy(c_np).long()
            if c_pt.shape[0] < self.Q:
                padded = torch.zeros(self.Q, dtype=c_pt.dtype)
                padded[: c_pt.shape[0]] = c_pt
                c_pt = padded
            self.codec_ctx.append(c_pt)

        raw_codes = (
            torch.stack([torch.from_numpy(np.array(c)).long() for c in all_codes])
            if all_codes
            else None
        )
        return total_secs, total_frames, cancelled, raw_codes

    def generate(
        self,
        text: str,
        settings: dict,
        cancel_event,
        audio_queue: Queue | None = None,
        *,
        is_new_turn: bool = True,
        context: str = "",
    ) -> dict:
        max_duration = settings.get("max_duration", 120)
        max_frames = int(max_duration * CODEC_HZ)
        total_secs = 0
        total_frames = 0
        total_gen_time = 0
        cancelled = False

        chunks = chunk_text(text, min_words=settings.get("chunk_words", 20))
        if chunks:
            chunks[0]["sc"] = is_new_turn
        print(f"[TTS-MLX]   -> {len(chunks)} sub-chunks")

        gen_all_codes = []
        for chunk_idx, chunk in enumerate(chunks):
            if cancelled or total_frames >= max_frames:
                break
            if chunk_idx > 0:
                chunk["sc"] = False

            print(f"[TTS-MLX]   chunk {chunk_idx+1}: T={self.T} '{chunk['text'][:60]}'")
            t_chunk = time.perf_counter()
            secs, frames, was_cancelled, raw_codes = self.generate_turn(
                chunk,
                cancel_event,
                audio_queue,
                temperature=settings.get("temperature", 0.8),
                top_k=settings.get("top_k", 300),
                rep_penalty=settings.get("rep_penalty", 1.4),
                rep_window=settings.get("rep_window", 24),
                n_codebooks=int(settings.get("n_codebooks", 0)),
                remaining_frames=max_frames - total_frames,
            )
            if raw_codes is not None:
                gen_all_codes.append(raw_codes)
            chunk_gen_time = time.perf_counter() - t_chunk
            total_secs += secs
            total_frames += frames
            total_gen_time += chunk_gen_time
            print(
                f"[TTS-MLX]   chunk {chunk_idx+1} done: {frames} frames, {secs:.1f}s, T={self.T}"
            )

            if was_cancelled:
                cancelled = True
                break

            if audio_queue:
                audio_queue.put(
                    {
                        "type": "chunk_done",
                        "chunk_idx": chunk_idx + 1,
                        "text": chunk["text"][:80],
                        "secs": secs,
                        "frames": frames,
                        "gen_time": chunk_gen_time,
                    }
                )

        all_codes = torch.cat(gen_all_codes) if gen_all_codes else None
        self._log_turn("assistant", text, all_codes, secs=total_secs)
        return {
            "total_secs": total_secs,
            "total_frames": total_frames,
            "total_gen_time": total_gen_time,
            "cancelled": cancelled,
            "codes": all_codes,
        }

    # --- Prompt loading ---

    def _load_prompt_by_name(self, name: str, settings: dict | None = None):
        """Load a voice prompt by name (wav+txt or .pt), encode, and prefill."""
        settings = settings or {}
        wav_path = PROMPTS_DIR / f"{name}.wav"
        txt_path = PROMPTS_DIR / f"{name}.txt"

        if not wav_path.exists():
            raise FileNotFoundError(f"No prompt wav at {wav_path}")

        # Read audio
        from torchcodec.decoders import AudioDecoder
        from julius.resample import resample_frac

        dec = AudioDecoder(str(wav_path), num_channels=1).get_all_samples()
        audio_16k = resample_frac(dec.data, int(dec.sample_rate), 16000).squeeze(0)
        audio_24k = resample_frac(audio_16k.unsqueeze(0), 16000, 24000)

        # Transcribe if no .txt
        if txt_path.exists():
            text = txt_path.read_text().strip()
        else:
            import mlx_whisper
            text = mlx_whisper.transcribe(
                audio_16k.numpy(),
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language="en",
                verbose=False,
            )["text"].strip()
            txt_path.write_text(text)
            print(f"[TTS-MLX] Saved ASR transcript to {txt_path}")

        # Encode audio to codes
        enc = self._get_codec_enc()
        device = next(enc.parameters()).device
        audio_in = audio_24k.float().to(device).unsqueeze(0)
        with torch.inference_mode():
            codes = enc.encode(audio_in)
        codes_tq = codes[0, :self.n_q].T.long().cpu()  # (T, Q)

        # Prefill
        codes_mx = mx.array(codes_tq.numpy().astype(np.int32))
        self.reset()
        self.model.decoder.make_cache()

        # Speaker embedding
        spk_emb_mx = None
        if self.model.spk_proj is not None:
            from vui.qwen_spk_enc import QwenSpeakerEncoder
            if not hasattr(self, '_spk_enc'):
                self._spk_enc = QwenSpeakerEncoder.from_pretrained()
            spk_emb = self._spk_enc.embed(audio_24k.squeeze(0), sr=24000)
            spk_emb_mx = mx.array(spk_emb.numpy())
            spk_token = self.model.spk_proj(spk_emb_mx).reshape(1, 1, -1)
            self.model.decoder(spk_token)

        # Text + codes
        ids = self.tok.encode(simple_clean(text))
        ids_mx = mx.array(ids.numpy().astype(np.int32))
        self.model.decoder(self.model.token_emb(ids_mx[None]))
        audio_emb = self.model.audio_emb(codes_mx)[None]
        self.model.decoder(audio_emb)
        mx.eval([c.state for c in self.model.decoder.kv_caches])

        n_text = ids_mx.shape[0]
        n_audio = codes_mx.shape[0]
        self.T = (1 if spk_emb_mx is not None else 0) + n_text + n_audio
        self.prompt_T = self.T
        self.prompt_codes = codes_mx
        self.prompt_text = text

        # Seed codec context
        self.codec_ctx = [codes_tq[i] for i in range(codes_tq.shape[0])]

        print(f"[TTS-MLX] Loaded prompt '{name}': {text[:60]}... ({n_audio} frames, T={self.T})")

    # --- Codec encode ---

    def _get_codec_enc(self):
        if self.codec_enc is None:
            print("[TTS-MLX] Loading codec encoder on CPU (lazy)...")
            from vui.qwen_codec import QwenCodecEncoder
            self.codec_enc = QwenCodecEncoder.from_pretrained().cpu().float().eval()
        return self.codec_enc

    def encode_full(self, audio_np) -> torch.Tensor | None:
        enc = self._get_codec_enc()
        device = next(enc.parameters()).device
        audio_t = torch.from_numpy(audio_np).float().to(device).reshape(1, 1, -1)
        with torch.inference_mode():
            codes = enc.encode(audio_t)
        return codes[0, : self.n_q].T.long().cpu()

    # --- Conversation debug log ---

    def start_conversation(self, base_dir: Path):
        from datetime import datetime

        folder = base_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        self._conv_dir = folder
        self._conv_turn_idx = 0
        if self.prompt_text or self.prompt_codes is not None:
            torch.save(
                {"prompt_text": self.prompt_text, "prompt_codes": self.prompt_codes},
                folder / "prompt.pt",
            )
        print(f"[TTS-MLX] New conversation: {folder}")

    def end_conversation(self):
        if self._conv_dir and self._conv_turn_idx > 0:
            print(
                f"[TTS-MLX] Ended conversation: {self._conv_dir} ({self._conv_turn_idx} turns)"
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
            data["codes"] = (
                codes.cpu()
                if isinstance(codes, torch.Tensor) and codes.is_cuda
                else codes
            )
        if secs is not None:
            data["secs"] = secs
        torch.save(data, self._conv_dir / f"{idx:02d}_{role}.pt")


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------


def tts_process_mlx(
    cmd_queue: Queue, audio_queue: Queue, checkpoint_path: str, cancel_event=None
):
    import multiprocessing
    import signal
    import threading

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if cancel_event is None:
        cancel_event = multiprocessing.Event()

    try:
        engine = MLXTTSEngine.from_checkpoint(checkpoint_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        audio_queue.put({"type": "error", "msg": str(e)})
        return
    audio_queue.put({"type": "ready"})

    debug_dir = Path("debug_dump")
    debug_dir.mkdir(exist_ok=True)
    debug_gen_idx = 0

    while True:
        try:
            msg = cmd_queue.get()
        except Exception:
            break

        cmd = msg.get("cmd")

        if cmd == "shutdown":
            engine.end_conversation()
            break

        elif cmd == "reset":
            engine.end_conversation()
            engine.reset()
            audio_queue.put({"type": "done", "T": 0, "total_secs": 0})

        elif cmd == "rewind":
            T = engine.rewind()
            print(f"[TTS-MLX] Rewind -> T={T}")
            audio_queue.put({"type": "done", "T": T, "total_secs": 0})

        elif cmd == "reprefill":
            try:
                settings = msg.get("settings", {})
                if engine.prompt_codes is not None:
                    n_cb = int(settings.get("n_codebooks", 0))
                    print(f"[TTS-MLX] Re-prefilling prompt with n_codebooks={n_cb}")
                    engine.prefill_prompt(
                        engine.prompt_codes,
                        engine.prompt_text,
                        settings,
                    )
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
                prompt_audio_24k = msg.get("audio_24k")
                if prompt_audio_24k is not None:
                    prompt_audio_24k = torch.from_numpy(prompt_audio_24k).float()
                codes_in = msg["codes"]
                print(
                    f"[TTS-MLX] Prompt codes shape: {codes_in.shape}, dtype: {codes_in.dtype}"
                )
                engine.prefill_prompt(
                    codes_in,
                    msg.get("text"),
                    msg.get("settings", {}),
                    prompt_audio_24k=prompt_audio_24k,
                )
                debug_gen_idx = 0
                audio_queue.put(
                    {
                        "type": "prompt_loaded",
                        "T": engine.T,
                        "text": engine.prompt_text or "",
                    }
                )
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "generate":
            cancel_event.clear()
            try:
                tts_text = msg["text"]
                settings = msg.get("settings", {})
                print(
                    f"[TTS-MLX] Generate: new_turn={msg.get('new_turn', True)} T={engine.T} '{tts_text[:80]}'"
                )

                result = engine.generate(
                    tts_text,
                    settings,
                    cancel_event,
                    audio_queue=audio_queue,
                    is_new_turn=msg.get("new_turn", True),
                    context=msg.get("context", ""),
                )

                debug_gen_idx += 1
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
                if engine._conv_dir is None:
                    engine.start_conversation(debug_dir)
                engine.prefill_user_turn(
                    text=msg.get("text", ""),
                    codes=msg.get("codes"),
                    audio_16k=msg.get("audio_16k"),
                    settings=msg.get("settings"),
                )
                audio_queue.put({"type": "user_prefilled", "T": engine.T})
            except Exception as e:
                traceback.print_exc()
                audio_queue.put({"type": "error", "msg": str(e)})

        elif cmd == "prefill_user_chunk":
            final = msg.get("final", False)
            reply_type = "user_prefilled" if final else "user_chunk_prefilled"
            try:
                text = msg.get("text", "")
                if text:
                    engine.prefill_user_turn(text=text, settings=msg.get("settings"))
                audio_queue.put(
                    {"type": reply_type, "frames": 0, "final": final, "T": engine.T}
                )
            except Exception:
                traceback.print_exc()
                audio_queue.put(
                    {"type": reply_type, "frames": 0, "final": final, "T": engine.T}
                )

        elif cmd == "prefill_text_sc":
            try:
                sc_prob = engine.prefill_text_sc(msg["text"])
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
            audio_queue.put(
                {
                    "type": "kv_saved",
                    "name": msg.get("name", ""),
                    "ok": False,
                    "msg": "KV save not supported in MLX mode",
                }
            )

        elif cmd == "load_kv":
            try:
                name = msg.get("name") or msg.get("file", "")
                engine._load_prompt_by_name(name, msg.get("settings", {}))
                audio_queue.put(
                    {
                        "type": "kv_loaded",
                        "name": name,
                        "ok": True,
                        "T": engine.T,
                        "text": engine.prompt_text or "",
                    }
                )
            except Exception as e:
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
            print(f"[TTS-MLX] Cancel: T={engine.T}")
            cancel_event.set()

        elif cmd == "get_state":
            audio_queue.put(
                {
                    "type": "state",
                    "T": engine.T,
                    "max_T": 8192,
                    "has_prompt": engine.prompt_codes is not None,
                    "prompt_text": engine.prompt_text or "",
                }
            )

        elif cmd == "encode_full":
            try:
                t0 = time.perf_counter()
                codes = engine.encode_full(msg["audio"])
                t_enc = (time.perf_counter() - t0) * 1000
                print(f"[TTS-MLX] Full encode: {t_enc:.1f}ms, {codes.shape[0]} frames")
                audio_queue.put({"type": "encoded", "codes": codes})
            except Exception:
                traceback.print_exc()
                audio_queue.put({"type": "encoded", "codes": None})

        elif cmd == "decode_prompt":
            try:
                if engine.prompt_codes is not None:
                    pc = engine.prompt_codes
                    if isinstance(pc, torch.Tensor):
                        pc = mx.array(pc.cpu().numpy().astype(np.int32))
                    # pc: (T, Q) MLX array — decode with MLX codec
                    engine._mlx_codec.reset_state()
                    audio_parts = []
                    for i in range(pc.shape[0]):
                        frame = pc[i][None, :, None]  # (1, Q, 1)
                        a = engine._mlx_codec.decode_frame(frame)
                        mx.eval(a)
                        audio_parts.append(np.array(a.flatten()))
                    wav = torch.from_numpy(np.concatenate(audio_parts)).float()
                    audio_queue.put(
                        {
                            "type": "prompt_audio",
                            "audio": wav,
                            "sample_rate": 24000,
                            "codes_shape": list(pc.shape),
                        }
                    )
                else:
                    audio_queue.put({"type": "prompt_audio", "audio": None})
            except Exception:
                import traceback as _tb
                _tb.print_exc()
                audio_queue.put({"type": "prompt_audio", "audio": None})

        elif cmd == "stream_start":
            pass  # User audio encoding skipped on MLX — text-only prefill

        elif cmd == "stream_feed":
            pass

        elif cmd == "stream_stop":
            audio_queue.put({"type": "codes_final", "codes": None})

    print("[TTS-MLX] Shutting down")
