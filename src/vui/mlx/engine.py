"""Engine-API-compatible MLX backend (Apple Silicon).

`vui.engine.Engine()` dispatches here automatically when CUDA is unavailable,
so the README Python API works unchanged on M-series:

    from vui.engine import Engine, GenConfig
    engine = Engine()
    with engine.new_row() as row:
        codes, audio = row.render("Hello there!", GenConfig(temperature=0.7))

Single-row only: the Row surface (prefill / add_user / stream / render /
rewind / reset) matches the CUDA engine; continuous batching (max_rows > 1),
two-speaker prefill, and the entropy/probe gates stay CUDA-only. Repetition
penalty state is per-chunk here rather than per-render-call.
"""

import os

import mlx.core as mx
import numpy as np
import torch

from vui.inference import chunk_text
from vui.mlx.tts.generate import CODEC_HZ, compute_cond_bias
from vui.mlx.tts.stream import TTSStream
from vui.mlx.tts.weights import load_quantized


def load_official_prompt(
    voice: str, prompt_dir: str | None = None
) -> tuple[str, mx.array, mx.array, mx.array | None]:
    """Returns (transcript, codes (T,Q) int32, spk_token (1,1,d), cond_bias|None).

    `prompt_dir` reads local <voice>.safetensors/.txt; default downloads the
    pre-baked set from the HF repo (codes + pre-projected speaker token +
    cond_bias, so no torch codec encoder is needed for the official voices).
    """
    if prompt_dir:
        st = mx.load(f"{prompt_dir}/{voice}.safetensors")
        txt_path = f"{prompt_dir}/{voice}.txt"
    else:
        from huggingface_hub import hf_hub_download

        st = mx.load(hf_hub_download("fluxions/vui", f"prompts/{voice}.safetensors"))
        txt_path = hf_hub_download("fluxions/vui", f"prompts/{voice}.txt")
    with open(txt_path) as f:
        text = f.read().strip()
    codes = st["codes"].astype(mx.int32)  # (T, Q)
    spk_token = st["spk_token_emb"].astype(mx.float32)  # (1, 1, d) pre-projected
    cond_bias = st.get("cond_bias")
    if cond_bias is not None:
        cond_bias = cond_bias.astype(mx.float32)
    return text, codes, spk_token, cond_bias


def _to_mx_codes(codes) -> mx.array:
    """(T, Q) torch long / numpy / mx -> mx int32."""
    if isinstance(codes, mx.array):
        return codes.astype(mx.int32)
    if isinstance(codes, torch.Tensor):
        codes = codes.detach().cpu().numpy()
    return mx.array(np.asarray(codes).astype(np.int32))


def _audio_to_torch(audio_mx: mx.array) -> torch.Tensor:
    """(S,) mx float -> (1, 1, S) torch float32."""
    return torch.from_numpy(np.array(audio_mx)).float().reshape(1, 1, -1)


class MLXRow:
    """Single conversation slot on the MLX engine. Mirrors vui.engine.Row."""

    def __init__(self, engine: "MLXEngine"):
        self._engine = engine
        self._prompt_offset = 0
        self._prompt_codes: mx.array | None = None  # for codec re-warm on rewind
        self._spk_token: mx.array | None = None
        self._closed = False
        self.prompt_wps: float = 0.0

    @property
    def idx(self) -> int:
        return 0

    @property
    def offset(self) -> int:
        return self._engine.model.decoder.cache_T

    def prefill(self, segments, spk_emb=None, segments_2=None, spk_emb_2=None) -> int:
        if segments_2 is not None or spk_emb_2 is not None:
            raise NotImplementedError("two-speaker prefill is CUDA-only")
        return self._engine._prefill_row(self, segments, spk_emb)

    def add_user(self, text: str = "", codes=None, *, final: bool = True) -> int:
        return self._engine._add_user(self, text, codes, final=final)

    def stream(self, text, cfg=None, cancel=None, *, reset_rep=True, final_turn=False):
        """Yield (1, 1, 1920) float32 audio tensors per frame (CPU)."""
        from vui.engine import GenConfig

        cfg = cfg or GenConfig()
        for _codes, audio in self._engine._stream_row(self, text, cfg, cancel):
            yield _audio_to_torch(audio)
        if final_turn:
            self._engine._append_sc()

    def render(self, text, cfg=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a full turn. Returns (codes (T, Q) long, audio (1, 1, S))."""
        from vui.engine import GenConfig

        cfg = cfg or GenConfig()
        all_codes, all_audio = [], []
        for codes, audio in self._engine._stream_row(self, text, cfg, None):
            all_codes.append(torch.from_numpy(np.array(codes)).long())
            all_audio.append(np.array(audio))
        if not all_codes:
            return torch.zeros(0, self._engine.Q, dtype=torch.long), torch.zeros(1, 1, 0)
        codes_t = torch.stack(all_codes)
        audio_t = torch.from_numpy(np.concatenate(all_audio)).float().reshape(1, 1, -1)
        return codes_t, audio_t

    def rewind(self) -> int:
        """Rewind KV to end-of-prompt."""
        return self._engine._rewind_row(self, self._prompt_offset)

    def reset(self) -> int:
        """Rewind KV to 0."""
        self._prompt_codes = None
        return self._engine._rewind_row(self, 0)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._engine._row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MLXEngine:
    """Engine-API adapter over TTSStream. One row; no CUDA anywhere."""

    def __init__(
        self,
        name: str = "vui-190k",
        *,
        model=None,
        codec=None,
        max_rows: int = 1,
        max_seq: int | None = None,
        codec_dtype=None,
        vocoder_ctx: int = 25,
        precision: str | None = None,
    ):
        if max_rows != 1:
            raise NotImplementedError(
                "MLX Engine supports max_rows=1; continuous batching is CUDA-only"
            )
        if model is not None or codec is not None:
            raise TypeError(
                "MLX Engine loads its own MLX model/codec — the model=/codec= "
                "overrides take torch modules and are CUDA-only"
            )
        from vui.engine import Engine as _CudaEngine

        path = _CudaEngine.NAMES.get(name, name)
        precision = precision or os.environ.get("VUI_MLX_PRECISION", "int8")
        print(f"[Engine-MLX] Loading model {path} ({precision}) ...")
        self.model, self.config = load_quantized(path, precision)
        from vui.mlx.tts.codec import load_codec_decoder_mlx

        print("[Engine-MLX] Loading MLX codec decoder ...")
        self.codec = load_codec_decoder_mlx()
        self.model.rq_transformer.compile_forward()

        self.max_rows = 1
        self.tok = self.model.text_tokenizer
        self.Q = self.model.rq_transformer.n_quantizers
        self.CS = self.model.rq_transformer.codebook_size
        self.D = self.model.d_model
        self._ctx = TTSStream(self.model, self.codec, mx.zeros((1, 1, self.D)))
        self._row: MLXRow | None = None

    # -- conditioning -------------------------------------------------------

    @property
    def cond_bias(self) -> mx.array:
        return self._ctx.cond_bias

    @cond_bias.setter
    def cond_bias(self, bias: mx.array) -> None:
        self._ctx.cond_bias = bias

    def set_conditioning(self, *, sq_scores=None, wps_score: float = 0.0) -> None:
        self._ctx.cond_bias = compute_cond_bias(
            self.model, sq=list(sq_scores) if sq_scores is not None else None,
            wps=wps_score,
        )

    # -- rows ---------------------------------------------------------------

    def new_row(self) -> MLXRow:
        if self._row is not None and not self._row._closed:
            raise RuntimeError("MLX Engine supports one open row at a time")
        self._ctx.reset()
        self._ctx._ensure_cache()
        row = MLXRow(self)
        self._row = row
        return row

    def reset(self) -> None:
        if self._row is not None:
            self._row.close()
        self._ctx.reset()

    # -- internals ----------------------------------------------------------

    def _append_sc(self) -> None:
        sc = self.model.token_emb(mx.array([[self.model.sc_id]]))
        self.model.decoder(sc)
        mx.eval([c.state for c in self.model.decoder.kv_caches])

    def _prefill_row(self, row: MLXRow, segments, spk_emb) -> int:
        """[spk] text_i codes_i per segment, [SC] on the last text — matches
        the CUDA engine's _prefill_speaker_segments(final=True)."""
        ctx = self._ctx
        ctx._ensure_cache()

        spk_token = None
        if spk_emb is not None:
            emb = spk_emb if isinstance(spk_emb, mx.array) else mx.array(
                spk_emb.detach().float().cpu().numpy()
            )
            if emb.ndim == 3 and emb.shape[-1] == self.D:
                spk_token = emb  # pre-projected token (official prompts)
            elif self.model.spk_proj is not None:
                spk_token = self.model.spk_proj(emb).reshape(1, 1, -1)
        row._spk_token = spk_token
        ctx._spk_token = spk_token  # re-injected before each generated chunk

        all_codes = []
        last = len(segments) - 1
        for i, seg in enumerate(segments):
            if spk_token is not None:
                self.model.decoder(spk_token)
            if seg.text:
                ctx.add_text(seg.text, sc=(i == last))
            if seg.codes is not None:
                codes_mx = _to_mx_codes(seg.codes)
                self.model.decoder(self.model.audio_emb(codes_mx)[None])
                all_codes.append(codes_mx)
        mx.eval([c.state for c in self.model.decoder.kv_caches])

        if all_codes:
            row._prompt_codes = mx.concatenate(all_codes, axis=0)
            self._warm_codec(row._prompt_codes)

        row._prompt_offset = self.model.decoder.cache_T

        total_words = sum(len(s.text.split()) for s in segments if s.text)
        total_frames = sum(s.codes.shape[0] for s in segments if s.codes is not None)
        if total_frames > 0 and total_words > 0:
            row.prompt_wps = total_words / (total_frames / CODEC_HZ)
        return row._prompt_offset

    def _warm_codec(self, codes_mx: mx.array) -> None:
        self.codec.reset_state()
        self.codec.prefill(codes_mx.T[None])
        mx.eval(self.codec.parameters())
        self._ctx._codec_ready = True

    def _add_user(self, row: MLXRow, text: str, codes, *, final: bool = True) -> int:
        """text [SC] codes — same layout as the CUDA engine's _add_user."""
        ctx = self._ctx
        ctx._ensure_cache()
        if text:
            ctx.add_text(text, sc=final)
        elif final:
            self._append_sc()
        if codes is not None:
            codes_mx = _to_mx_codes(codes)
            self.model.decoder(self.model.audio_emb(codes_mx)[None])
            self._warm_codec(codes_mx)
        mx.eval([c.state for c in self.model.decoder.kv_caches])
        return self.model.decoder.cache_T

    def _rewind_row(self, row: MLXRow, offset: int) -> int:
        for c in self.model.decoder.kv_caches:
            c.offset = min(c.offset, offset)
        if offset > 0 and row._prompt_codes is not None:
            self._warm_codec(row._prompt_codes)
        else:
            self.codec.reset_state()
            self._ctx._codec_ready = False
        return offset

    def _stream_row(self, row: MLXRow, text: str, cfg, cancel):
        """Chunk the text like the CUDA engine and yield (codes, audio) frames."""
        chunks = chunk_text(
            text, min_words=cfg.chunk_words, sentence_only=cfg.sentence_only,
            single_speaker=True,
        )
        min_frames = max(1, int(cfg.min_secs * CODEC_HZ))
        max_total = int(cfg.max_secs * CODEC_HZ)
        total = 0
        for ch in chunks:
            n_words = len(ch["text"].split())
            max_frames = min(cfg.max_turn_frames(n_words, row.prompt_wps),
                             max_total - total)
            if max_frames <= 0:
                break
            for codes, audio in self._ctx.generate(
                ch["text"],
                temperature=cfg.temperature,
                top_k=cfg.top_k or 300,
                max_frames=max_frames,
                eos_threshold=cfg.eos_threshold,
                min_frames=min_frames,
                rep_penalty=cfg.rep_penalty,
                rep_window=cfg.rep_window,
                n_codebooks=cfg.n_codebooks,
            ):
                if cancel is not None and cancel.is_set():
                    return
                total += 1
                yield codes, audio
