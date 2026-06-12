"""Unified TTS engine: streaming + batched inference over one Vui model.

One class, one API, one set of CUDA graphs. `max_rows=1` default for the
common streaming / single-conversation case; `max_rows=N` for batched
multi-conversation inference.

Usage (streaming, B=1):

    engine = Engine()  # loads "vui-nano" from HuggingFace by default
    with engine.new_row() as row:
        row.prefill([Segment(prompt_text, prompt_codes)], spk_emb=emb)
        for audio in row.stream("Hello!", GenConfig(temperature=0.9)):
            play(audio)

Usage (batched, B=N):

    engine = Engine(max_rows=4)
    rows = [engine.new_row() for _ in range(4)]
    for r in rows:
        r.prefill([Segment(prompt_text, prompt_codes)])
    audios = engine.render_all(rows, TEXTS, GenConfig())
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from vui.inference import chunk_text, simple_clean
from vui.model import Vui
from vui.qwen_codec import FRAME_RATE
from vui.qwen_codec import SAMPLE_RATE as QWEN_SR
from vui.qwen_codec import CodecCtx, QwenCodecDecoder

DEFAULT_WPS = 3.0  # fallback words-per-second when prompt_wps unavailable

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One voice-reference unit: a transcript + its codec codes.

    Most commonly both are present (text + codes for a ~10s reference clip).
    Either can be empty/None for text-only or codes-only contexts.
    """

    text: str = ""
    codes: Tensor | None = None  # (T, Q) long


@dataclass(frozen=True)
class GenConfig:
    """Sampling + generation settings shared by stream() and render*()."""

    temperature: float = 0.9
    top_k: int | None = 100
    top_p: float | None = None
    rep_penalty: float = 1.1
    rep_window: int = 24  # ~2s of cb0 history — enough to break long silence/filler loops without blocking natural repetition
    eos_threshold: float = 0.45
    min_secs: float = 0.5
    max_turn_secs: float = 15.0
    max_secs: float = 30.0
    chunk_words: int = 5
    first_chunk_words: int = 0  # >0: first sub-chunk uses fewer words for lower TTFB
    n_codebooks: int = 0  # 0 = use all Q quantizers
    max_turn_wps: float = 0.0  # >0: override WPS for chunk duration cap
    sentence_only: bool = False  # split on .!? only (matches llm_stream_chunks)
    # Hallucination gate: collect first `gate_frames` codes from chunk 0 without
    # vocoder decode, check mean cb0 entropy, rewind+retry at lower temp if
    # above `gate_entropy_max`. Costs `gate_frames * frame_step` extra TTFB.
    # 0 = disabled (default). Threshold is in nats over CS codes (uniform
    # 1024 ≈ 6.93). Training cb0 CE ~1.2-1.5, so a healthy decode tracks
    # ~1.5-2.0 at temp 0.9; >3 means the model is confused.
    gate_frames: int = 0
    gate_entropy_max: float = 1.9
    gate_retries: int = 2

    def max_turn_frames(self, n_words: int, prompt_wps: float = 0.0) -> int:
        """Max frames for a chunk, scaled by word count and WPS.

        The *1.6 multiplier and +0.6s offset are leeway over the prompt's WPS
        so the model can take its time on emphatic / slow phrases without
        getting cut off — gets clipped less often at the cost of letting some
        long-tail generations run a bit further.
        """
        ceil = int(self.max_turn_secs * FRAME_RATE)
        wps = self.max_turn_wps if self.max_turn_wps > 0 else prompt_wps
        if wps <= 0:
            wps = DEFAULT_WPS
        if n_words <= 0:
            return ceil
        secs = n_words / wps * 1.6 + 0.6
        return min(int(secs * FRAME_RATE), ceil)


@dataclass(frozen=True)
class RenderRequest:
    """One TTS request for continuous batching."""

    segments: list[Segment]
    spk_emb: Tensor | None = None
    text: str = ""
    user_text: str | None = None
    user_codes: Tensor | None = None
    segments_2: list[Segment] | None = None
    spk_emb_2: Tensor | None = None


@dataclass
class _ActiveSlot:
    """Tracks one in-flight request in the continuous batching loop."""

    row: Row
    request_idx: int
    chunks: list[dict]
    chunk_idx: int = 0
    frame_count: int = 0
    turn_count: int = 0
    max_turn_frames: int = 0
    last_codes: Tensor | None = None
    seq_offset: int = 0


def _make_buckets(max_bs: int) -> list[int]:
    buckets = []
    b = 1
    while b < max_bs:
        buckets.append(b)
        b *= 2
    if not buckets or buckets[-1] != max_bs:
        buckets.append(max_bs)
    return buckets


def _get_bucket(batch_size: int, buckets: list[int]) -> int:
    for b in buckets:
        if batch_size <= b:
            return b
    return buckets[-1]


# ---------------------------------------------------------------------------
# Row: one conversation slot on an Engine
# ---------------------------------------------------------------------------


class Row:
    """A single conversation slot.

    Holds: slot index into the engine's shared flash KV cache, the
    end-of-prompt offset for rewind, a per-row CodecCtx for streaming
    decode, and an optional speaker token embedding from prefill.
    """

    def __init__(self, engine: "Engine", idx: int):
        self._engine = engine
        self._idx = idx
        self._prompt_offset = 0
        self._spk_token: Tensor | None = None
        self._spk_token_2: Tensor | None = None  # optional 2nd speaker
        self._active_speaker = 0  # 0 or 1 — flipped on [SC] chunks
        self._codec_ctx = CodecCtx(engine.codec)
        self._closed = False
        self.prompt_wps: float = 0.0

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def offset(self) -> int:
        return int(
            self._engine.model.decoder.flash_kv_caches[0].seq_lens[self._idx].item()
        )

    def prefill(
        self,
        segments: list[Segment],
        spk_emb: Tensor | None = None,
        segments_2: list[Segment] | None = None,
        spk_emb_2: Tensor | None = None,
    ) -> int:
        """Prefill this row with `[spk] text_i codes_i` for each segment.

        spk_emb is projected once via model.embed_speaker and re-injected
        before every segment's text (matching training chunk format).

        For two-speaker conversations, pass `segments_2` + `spk_emb_2`; both
        speakers are prefilled in order, and the stream/render loop alternates
        between them on each `[SC]` chunk. Sets self._prompt_offset to the
        new offset (used by rewind()).
        """
        return self._engine._prefill_row(self, segments, spk_emb, segments_2, spk_emb_2)

    def add_user(
        self, text: str = "", codes: Tensor | None = None, *, final: bool = True
    ) -> int:
        """Write user turn chunk. final=True appends [SC] to close the turn."""
        return self._engine._add_user(self, text, codes, final=final)

    def stream(
        self,
        text: str,
        cfg: GenConfig = GenConfig(),
        cancel=None,
        *,
        reset_rep: bool = True,
        final_turn: bool = False,
    ) -> Iterator[Tensor]:
        """Generate audio frame-by-frame using the vocoder CUDA graph.

        Yields (1, 1, DOWNSAMPLE_RATE) float audio tensors on GPU. Caller
        concatenates or moves to CPU as needed. B=1 only.

        Each text sub-chunk gets `[spk] text audio` written to the KV,
        matching the streamed_tts training format which produces `[spk]`
        per segment.

        `cancel`: optional object with `.is_set()` method (threading.Event or
        multiprocessing.Event). Checked per frame; stream exits cleanly if set.

        `reset_rep`: zero the repetition-penalty history at the start of this
        call. Defaults True (matches `render()`'s one-shot semantics). Set
        False when continuing a multi-call assistant turn — e.g. when the
        streaming server feeds LLM chunks one at a time and you want the
        rep history to span the whole turn (matches demo.py's render() path).
        """
        if self._engine.max_rows != 1:
            raise RuntimeError(
                f"stream() requires Engine(max_rows=1), got {self._engine.max_rows}"
            )
        yield from self._engine._stream_row(
            self, text, cfg, cancel, reset_rep=reset_rep, final_turn=final_turn
        )

    def render(self, text: str, cfg: GenConfig = GenConfig()) -> tuple[Tensor, Tensor]:
        """Generate a full turn non-streaming. Returns (codes (T,Q), audio (1,1,S))."""
        codes = self._engine._render_row(self, text, cfg)
        audio = self._engine._decode_full(codes)
        return codes, audio

    def rewind(self) -> int:
        """Rewind KV to end-of-prompt."""
        return self._engine._rewind_row(self, self._prompt_offset)

    def reset(self) -> int:
        """Rewind KV to 0."""
        return self._engine._rewind_row(self, 0)

    def close(self) -> None:
        """Release the slot back to the engine's free pool."""
        if not self._closed:
            self._engine._release_row(self)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Unified TTS engine — streaming (B=1) and batched (B=N) inference.

    One flash KV cache sized at `max_rows`, one batched backbone decode graph,
    one per-row RQ decode (B=1 looped for correctness), vocoder CUDA graph
    for per-frame streaming decode.

    `Engine()` with no args loads `vui-nano` from HuggingFace. Pass a name
    (resolved via `Engine.NAMES`), a HF filename, or a local path. Advanced
    callers can inject `model=` / `codec=` directly to bypass loading.
    """

    NAMES = {
        "vui-nano": "vui-nano.safetensors",
    }

    def __init__(
        self,
        name: str = "vui-nano",
        *,
        model: Vui | None = None,
        codec: QwenCodecDecoder | None = None,
        max_rows: int = 1,
        max_seq: int | None = None,
        codec_dtype: torch.dtype = torch.float32,
        vocoder_ctx: int = 25,
    ):
        if model is None:
            path = self.NAMES.get(name, name)
            print(f"[Engine] Loading model {path} ...")
            model = Vui.from_pretrained_inf(path).cuda()
        if codec is None:
            print(f"[Engine] Loading codec (dtype={codec_dtype}) ...")
            codec = QwenCodecDecoder.from_pretrained().cuda().to(codec_dtype).eval()
        self.model = model
        self.codec = codec
        self.max_rows = max_rows
        self.vocoder_ctx = vocoder_ctx
        self.device = model.device
        self.dtype = model.dtype
        self.Q = model.config.model.n_quantizers
        self.CS = model.config.model.codebook_size
        self.D = model.config.model.d_model
        self.tok = model.text_tokenizer
        sc_id = self.tok.special_to_id["[SC]"]
        with torch.inference_mode():
            self._sc_emb = self.model.token_emb(
                torch.tensor([[sc_id]], device=self.device)
            ).to(self.dtype)

        trained = model.decoder.max_seqlen
        self.max_seq = max_seq or trained
        if self.max_seq > trained:
            raise ValueError(
                f"max_seq={self.max_seq} exceeds trained ceiling {trained}"
            )

        # Model-owned cond_bias buffer (sq/wps biases applied to text emb)
        if not hasattr(model, "_cond_bias") or model._cond_bias is None:
            model._cond_bias = torch.zeros(
                1, 1, self.D, device=self.device, dtype=self.dtype
            )
        self._cond_bias = model._cond_bias

        self._setup_graphs()
        self._setup_rep_state()

        self._free: set[int] = set(range(max_rows))
        self._rows: dict[int, Row] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        max_rows: int = 1,
        max_seq: int | None = None,
        codec_dtype: torch.dtype = torch.float32,
        vocoder_ctx: int = 25,
    ) -> "Engine":
        """Backward-compat shim. Prefer `Engine(name)` / `Engine()`."""
        return cls(
            checkpoint_path,
            max_rows=max_rows,
            max_seq=max_seq,
            codec_dtype=codec_dtype,
            vocoder_ctx=vocoder_ctx,
        )

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_graphs(self) -> None:
        """Capture backbone batched decode graph, RQ B=1 graph, vocoder graph."""
        model = self.model
        device, dtype = self.device, self.dtype

        # Flash KV cache sized at B=max_rows+1 (extra slot for padding target)
        model.decoder.allocate_flash_kv_cache(
            batch_size=self.max_rows + 1,
            device=device,
            dtype=dtype,
            max_seqlen=self.max_seq,
        )
        self._pad_slot = self.max_rows

        # Buckets for continuous batching
        self._buckets = _make_buckets(self.max_rows)

        # RQ CUDA graph: capture at bucket sizes + B=1 for advance_turn
        rq_batch_sizes = tuple(sorted(set(self._buckets) | {1, self.max_rows}))
        with sdpa_kernel([SDPBackend.MATH]):
            model.rq_transformer.setup_cuda_graph_kv(
                device, dtype, top_k=100, batch_sizes=rq_batch_sizes
            )
        self._zero_rq_bias = torch.zeros(
            self.Q - 1, self.CS, device=device, dtype=torch.float32
        )

        # Static buffers for the OLD batched backbone decode graph (B=max_rows)
        N = self.max_rows
        self._g_codes_in = torch.zeros(N, self.Q, device=device, dtype=torch.long)
        self._g_hidden_out = torch.zeros(N, self.D, device=device, dtype=dtype)
        self._g_cb0_out = torch.zeros(N, self.CS, device=device, dtype=dtype)
        self._g_eos_out = torch.zeros(N, 1, device=device, dtype=dtype)

        self._decode_graph = self._capture_decode_graph()

        # Static buffers for BUCKETED continuous batching graphs
        B_max = self._buckets[-1]
        self._gc_codes_in = torch.zeros(B_max, self.Q, device=device, dtype=torch.long)
        self._gc_hidden_out = torch.zeros(B_max, self.D, device=device, dtype=dtype)
        self._gc_cb0_out = torch.zeros(B_max, self.CS, device=device, dtype=dtype)
        self._gc_eos_out = torch.zeros(B_max, 1, device=device, dtype=dtype)
        self._g_slot_idx = torch.full(
            (B_max,), self._pad_slot, device=device, dtype=torch.int32
        )
        self._g_seq_inc = torch.zeros(
            self.max_rows + 1, device=device, dtype=torch.int32
        )

        print(f"[Engine] Capturing bucketed graphs at {self._buckets} ...")
        self._bucketed_graphs = self._capture_bucketed_decode_graphs()

        # Streaming codec graph (post-pre_transformer chain at chunk size 1).
        # Conv state lives on persistent module buffers, so the graph survives
        # across streaming-context exits/re-entries done by CodecCtx.prefill.
        if self.max_rows == 1:
            print("[Engine] Capturing streaming codec graph ...")
            self.codec.setup_streaming_graph(batch_size=1)

    def _backbone_decode_body(self) -> None:
        """One autoregressive decode step across all max_rows slots."""
        model = self.model
        decoder = model.decoder
        B = self.max_rows
        emb = model.audio_emb(self._g_codes_in).unsqueeze(1).to(self.dtype)
        pos = decoder.flash_kv_caches[0].seq_lens[:B]
        x = emb
        for i, (block, kv) in enumerate(zip(decoder.blocks, decoder.flash_kv_caches)):
            freqs_cis = decoder._get_freqs(pos, i)
            x = block.forward_flash(x, kv, freqs_cis, per_sample_freqs=True)
        hidden = decoder.norm(x)[:, 0]  # (B, d)
        self._g_hidden_out.copy_(hidden)
        self._g_cb0_out.copy_(model.codec_head(hidden))
        self._g_eos_out.copy_(model.eos_head(hidden))
        decoder.flash_kv_caches[0].seq_lens[:B] += 1

    def _capture_decode_graph(self) -> torch.cuda.CUDAGraph:
        """Warm then capture the batched backbone decode step."""
        self._g_codes_in.zero_()
        saved = self.model.decoder.flash_kv_caches[0].seq_lens[: self.max_rows].clone()
        with (
            torch.inference_mode(),
            sdpa_kernel([SDPBackend.MATH]),
        ):
            for _ in range(3):
                self._backbone_decode_body()
                self.model.decoder.flash_kv_caches[0].seq_lens[: self.max_rows].copy_(
                    saved
                )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                self._backbone_decode_body()
        # Capture advanced seq_lens by 1; rewind for caller
        self.model.decoder.flash_kv_caches[0].seq_lens[: self.max_rows].copy_(saved)
        return graph

    def _set_codes_in(self, row_idx: int, codes: Tensor) -> None:
        Q_out = codes.shape[0]
        if Q_out < self.Q:
            self._g_codes_in[row_idx].zero_()
            self._g_codes_in[row_idx, :Q_out] = codes
        else:
            self._g_codes_in[row_idx] = codes

    # ------------------------------------------------------------------
    # Bucketed decode graphs (continuous batching)
    # ------------------------------------------------------------------

    def _backbone_decode_body_b(self, B: int) -> None:
        """Decode step for B active slots with slot indirection via cache_batch_idx."""
        model = self.model
        decoder = model.decoder
        slot_idx = self._g_slot_idx[:B]
        emb = model.audio_emb(self._gc_codes_in[:B]).unsqueeze(1).to(self.dtype)
        pos = decoder.flash_kv_caches[0].seq_lens[slot_idx.long()]
        x = emb
        for i, (block, kv) in enumerate(zip(decoder.blocks, decoder.flash_kv_caches)):
            freqs_cis = decoder._get_freqs(pos, i)
            x = block.forward_flash(
                x, kv, freqs_cis, per_sample_freqs=True, cache_batch_idx=slot_idx
            )
        hidden = decoder.norm(x)[:, 0]
        self._gc_hidden_out[:B].copy_(hidden)
        self._gc_cb0_out[:B].copy_(model.codec_head(hidden))
        self._gc_eos_out[:B].copy_(model.eos_head(hidden))
        decoder.flash_kv_caches[0].seq_lens.add_(self._g_seq_inc)

    def _capture_bucketed_decode_graphs(self) -> dict[int, torch.cuda.CUDAGraph]:
        graphs: dict[int, torch.cuda.CUDAGraph] = {}
        seq_lens = self.model.decoder.flash_kv_caches[0].seq_lens
        for B in self._buckets:
            self._g_slot_idx[:B].fill_(self._pad_slot)
            self._g_seq_inc.zero_()
            self._gc_codes_in[:B].zero_()
            saved = seq_lens.clone()
            with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
                for _ in range(3):
                    self._backbone_decode_body_b(B)
                    seq_lens.copy_(saved)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    self._backbone_decode_body_b(B)
            seq_lens.copy_(saved)
            graphs[B] = graph
        return graphs

    def _bucketed_decode_step(self, B_active: int) -> tuple[Tensor, Tensor, Tensor]:
        """Replay the bucketed backbone graph. Returns (hidden, cb0, eos) for active slots."""
        B_bucket = _get_bucket(B_active, self._buckets)
        self._g_slot_idx[B_active:B_bucket].fill_(self._pad_slot)
        self._bucketed_graphs[B_bucket].replay()
        return (
            self._gc_hidden_out[:B_active],
            self._gc_cb0_out[:B_active],
            self._gc_eos_out[:B_active],
        )

    # ------------------------------------------------------------------
    # Rep-penalty state (GPU-vectorized, no Python loops)
    # ------------------------------------------------------------------

    def _setup_rep_state(self) -> None:
        """Initialize the rep-penalty state tensors once per engine."""
        N = self.max_rows
        self._rep_counts = torch.zeros(
            N, self.CS, device=self.device, dtype=torch.float32
        )
        self._rep_history: Tensor | None = None  # lazy-init on first use
        self._rep_head = 0
        self._rep_filled = 0
        self._rep_unit = torch.ones(N, 1, device=self.device, dtype=torch.float32)
        self._rep_neg_unit = -self._rep_unit
        self._rep_penalty_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._rep_ones_1d = torch.ones(N, device=self.device, dtype=torch.float32)
        self._rep_window_cur = 0

    def _rep_reset(self, penalty: float, window: int) -> None:
        """Reset the rep-penalty buffers for a fresh multi-row decode run."""
        self._rep_counts.zero_()
        self._rep_head = 0
        self._rep_filled = 0
        self._rep_penalty_t.fill_(penalty)
        if penalty <= 1.0:
            return
        if window > 0 and (self._rep_history is None or self._rep_window_cur != window):
            self._rep_history = torch.zeros(
                self.max_rows, window, device=self.device, dtype=torch.long
            )
            self._rep_window_cur = window
        elif window > 0 and self._rep_history is not None:
            self._rep_history.zero_()

    def _rep_apply(self, cb0_logits: Tensor, penalty: float) -> Tensor:
        """Apply cb0 rep-penalty across all rows. (N, CS) -> (N, CS)."""
        if penalty <= 1.0:
            return cb0_logits
        factor = torch.pow(self._rep_penalty_t, self._rep_counts)
        return torch.where(cb0_logits > 0, cb0_logits / factor, cb0_logits * factor)

    def _rep_push(self, codes: Tensor, penalty: float, window: int) -> None:
        """Push sampled cb0 codes (N,) into the rolling count tensors."""
        if penalty <= 1.0:
            return
        if window > 0 and self._rep_history is not None:
            if self._rep_filled == window:
                old = self._rep_history[:, self._rep_head].unsqueeze(1)
                self._rep_counts.scatter_add_(1, old, self._rep_neg_unit)
            self._rep_history[:, self._rep_head] = codes
            self._rep_head = (self._rep_head + 1) % window
            if self._rep_filled < window:
                self._rep_filled += 1
        self._rep_counts.scatter_add_(1, codes.unsqueeze(1), self._rep_unit)

    # Slot-indexed rep methods for continuous batching (operate on arbitrary slot subsets)

    def _rep_reset_slot(self, slot: int) -> None:
        self._rep_counts[slot].zero_()

    def _rep_apply_slots(
        self, cb0_logits: Tensor, slot_indices: Tensor, penalty: float
    ) -> Tensor:
        if penalty <= 1.0:
            return cb0_logits
        counts = self._rep_counts[slot_indices]
        factor = torch.pow(self._rep_penalty_t, counts)
        return torch.where(cb0_logits > 0, cb0_logits / factor, cb0_logits * factor)

    def _rep_push_slots(
        self, codes: Tensor, slot_indices: Tensor, penalty: float
    ) -> None:
        if penalty <= 1.0:
            return
        B = slot_indices.shape[0]
        self._rep_counts.index_put_(
            (slot_indices, codes.long()),
            self._rep_ones_1d[:B],
            accumulate=True,
        )

    # ------------------------------------------------------------------
    # Row lifecycle
    # ------------------------------------------------------------------

    def new_row(self) -> Row:
        """Claim a free conversation slot."""
        if not self._free:
            raise RuntimeError(f"No free rows (max_rows={self.max_rows} all in use)")
        idx = min(self._free)
        self._free.remove(idx)
        row = Row(self, idx)
        self._rows[idx] = row
        with torch.inference_mode():
            self.model.decoder.flash_kv_caches[0].seq_lens[idx] = 0
        return row

    def _release_row(self, row: Row) -> None:
        self._rows.pop(row.idx, None)
        self._free.add(row.idx)
        row._codec_ctx.reset()

    def reset(self) -> None:
        """Release all rows and zero the flash KV seq_lens.

        Use when switching speakers: per-conversation state (KV contents,
        codec rolling context, speaker tokens) becomes invalid, but the
        engine's graphs, model weights, and codec stay valid.
        """
        for row in list(self._rows.values()):
            row.close()
        with torch.inference_mode():
            self.model.decoder.flash_kv_caches[0].seq_lens.zero_()

    def _rewind_row(self, row: Row, offset: int) -> int:
        with torch.inference_mode():
            self.model.decoder.flash_kv_caches[0].seq_lens[row.idx] = offset
        return offset

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def set_conditioning(
        self,
        *,
        sq_scores: tuple[float, ...] | None = None,
        wps_score: float = 0.0,
    ) -> None:
        """Set the model's cond_bias (applied additively to text embeddings)."""
        self.model.set_cond_bias(sq_scores=sq_scores, wps_score=wps_score)
        self._cond_bias = self.model._cond_bias

    # ------------------------------------------------------------------
    # Prefill (shared between prompt, user, and agent)
    # ------------------------------------------------------------------

    def _prefill_emb(self, row: Row, emb: Tensor) -> Tensor:
        """Write (1, T, d) emb into row's slot at current offset.

        Uses per-row seq_lens slicing via the shared flash KV cache. Returns
        the decoder's norm output (1, T, d) for first-frame hidden.
        """
        decoder = self.model.decoder
        T = emb.shape[1]
        offset = row.offset
        positions = torch.arange(offset, offset + T, device=self.device)
        x = emb
        for i, (block, kv) in enumerate(zip(decoder.blocks, decoder.flash_kv_caches)):
            freqs_cis = decoder._get_freqs(positions, i)
            sliced = _SlicedKV(
                k_cache=kv.k_cache[row.idx : row.idx + 1],
                v_cache=kv.v_cache[row.idx : row.idx + 1],
                seq_lens=kv.seq_lens[row.idx : row.idx + 1],
            )
            x = block.forward_flash(x, sliced, freqs_cis)
        decoder.flash_kv_caches[0].seq_lens[row.idx] += T
        return decoder.norm(x)

    def _text_emb(self, text: str, with_cond_bias: bool, noisy: bool = False) -> Tensor:
        ids = self.tok.encode(simple_clean(text)).to(self.device)
        emb = self.model.token_emb(ids[None]).to(self.dtype)
        if with_cond_bias:
            emb = emb + self._cond_bias
        if noisy and hasattr(self.model, "noisy_emb"):
            emb = emb + self.model.noisy_emb.to(self.dtype)
        return emb

    def _audio_emb(self, codes: Tensor) -> Tensor:
        pc = codes.to(self.device)
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        return self.model.embed_audio(pc).to(self.dtype)

    def _embed_speaker(self, spk_emb: Tensor | None) -> Tensor | None:
        if spk_emb is None or self.model.spk_proj is None:
            return None
        return self.model.embed_speaker(spk_emb).to(self.dtype)

    def _prefill_speaker_segments(
        self,
        row: Row,
        segments: list[Segment],
        spk_token: Tensor | None,
        *,
        final: bool = False,
    ) -> None:
        last = len(segments) - 1
        for i, seg in enumerate(segments):
            if spk_token is not None:
                self._prefill_emb(row, spk_token)
            if seg.text:
                self._prefill_emb(row, self._text_emb(seg.text, with_cond_bias=False))
            if final and i == last:
                self._prefill_emb(row, self._sc_emb)
            if seg.codes is not None:
                self._prefill_emb(row, self._audio_emb(seg.codes))

    def _prefill_row(
        self,
        row: Row,
        segments: list[Segment],
        spk_emb: Tensor | None,
        segments_2: list[Segment] | None = None,
        spk_emb_2: Tensor | None = None,
    ) -> int:
        """Prefill [spk] text codes ... into a row, starting from its current offset.

        For two-speaker conversations, pass `segments_2` + `spk_emb_2`: both
        are prefilled in order, and the row's `_active_speaker` is initialized
        so the first generation chunk will be speaker 0.
        """
        spk_token = self._embed_speaker(spk_emb)
        spk_token_2 = self._embed_speaker(spk_emb_2)
        row._spk_token = spk_token
        row._spk_token_2 = spk_token_2
        row._active_speaker = 0

        # Seed the row's codec rolling buffer with all prompt codes (both speakers)
        all_codes: list[Tensor] = []
        for seg in segments:
            if seg.codes is not None:
                all_codes.append(seg.codes)
        if segments_2 is not None:
            for seg in segments_2:
                if seg.codes is not None:
                    all_codes.append(seg.codes)
        if all_codes:
            row._codec_ctx.set_prompt(
                torch.cat([c.to(self.device) for c in all_codes], dim=0).T.unsqueeze(0)
            )

        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            # Always end each speaker's block with [SC] before its last audio:
            # matches training (streamed_tts) where [SC] is appended to a turn's
            # text whenever the next turn is a different speaker. For speaker 1
            # in two-speaker mode the next turn is speaker 2, so [SC] is needed;
            # for the trailing speaker the next "turn" is the assistant reply.
            self._prefill_speaker_segments(row, segments, spk_token, final=True)
            if segments_2 is not None:
                self._prefill_speaker_segments(row, segments_2, spk_token_2, final=True)
        row._prompt_offset = row.offset

        # Estimate WPS from prompt segments for max_turn_frames fallback
        all_segs = list(segments) + (list(segments_2) if segments_2 else [])
        total_words = sum(len(s.text.split()) for s in all_segs if s.text)
        total_frames = sum(s.codes.shape[0] for s in all_segs if s.codes is not None)
        if total_frames > 0 and total_words > 0:
            row.prompt_wps = total_words / (total_frames / FRAME_RATE)

        return row._prompt_offset

    def _add_user(
        self, row: Row, text: str, codes: Tensor | None, *, final: bool = True
    ) -> int:
        """Write a user turn chunk into the KV cache.

        final=True (default):  text [SC] codes  — closes the user turn.
        final=False:           text codes        — intermediate chunk, no [SC].
        """
        T0 = row.offset
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            n_text = 0
            if text:
                text_emb = self._text_emb(text, with_cond_bias=False, noisy=True)
                n_text = text_emb.shape[1]
                self._prefill_emb(row, text_emb)
            if final:
                self._prefill_emb(row, self._sc_emb)
            n_codes = 0
            if codes is not None:
                n_codes = codes.shape[0]
                self._prefill_emb(row, self._audio_emb(codes))
                row._codec_ctx.add(codes.T.unsqueeze(0).to(self.device))
        sc = "+[SC]" if final else ""
        print(
            f"[Engine._add_user] T={T0}->{row.offset} "
            f"text={n_text}tok{sc} codes={n_codes}f "
            f"'{text[:40]}'"
        )
        return row.offset

    def _active_spk_token(self, row: Row) -> Tensor | None:
        """Return the speaker token for the row's currently-active speaker."""
        if row._active_speaker == 1 and row._spk_token_2 is not None:
            return row._spk_token_2
        return row._spk_token

    def _add_agent_text(self, row: Row, text: str, *, final: bool = False) -> Tensor:
        """Write `[spk] reply_text` for one chunk. Returns final hidden (d,).

        Matches training (`data.py:streamed_tts`): `[spk]` is written
        per-segment, where each segment is `[spk] text audio`. Multiple
        same-speaker segments stack as `[spk] t1 a1 [spk] t2 a2 ...` with
        `[SC]` only appended to text when the next speaker differs.

        `final=True` appends `[SC]` to the text (before audio is generated)
        — used on the last sub-chunk of a turn so the model sees the same
        speaker-change signal it was trained with.
        """
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            spk = self._active_spk_token(row)
            if spk is not None:
                self._prefill_emb(row, spk)
            out = self._prefill_emb(row, self._text_emb(text, with_cond_bias=True))
            if final:
                out = self._prefill_emb(row, self._sc_emb)
        return out[0, -1]  # (d,)

    # ------------------------------------------------------------------
    # Decode loop — used by stream(), render(), render_all()
    # ------------------------------------------------------------------

    def _decode_step(self) -> tuple[Tensor, Tensor, Tensor]:
        """Replay the batched backbone graph. Returns (hidden, cb0, eos)."""
        self._decode_graph.replay()
        return self._g_hidden_out, self._g_cb0_out, self._g_eos_out

    def _sample_cb0(self, logits: Tensor, cfg: GenConfig) -> Tensor:
        """Sample cb0 across all rows. (N, CS) -> (N,)."""
        probs = F.softmax(logits.float() / cfg.temperature, dim=-1)
        if cfg.top_k is not None and cfg.top_k > 0:
            topv, topi = probs.topk(cfg.top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter_(1, topi, topv)
            mask = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            probs = mask
        return torch.multinomial(probs, 1).squeeze(-1)

    def _render_rows(
        self,
        rows: list[Row],
        texts: list[str],
        cfg: GenConfig,
    ) -> dict[int, Tensor]:
        """Core batched decode loop. Returns {row.idx: (T, Q) codes}.

        Pre-condition: each row has been prefilled (prompt, optional user)
        via row.prefill(). This method adds the agent text per-chunk and
        runs the batched decode.
        """
        model = self.model
        rq = model.rq_transformer
        N = self.max_rows
        max_frames = int(cfg.max_secs * FRAME_RATE)
        min_frames = int(cfg.min_secs * FRAME_RATE)
        max_seq = self.max_seq - 10

        # Pre-chunk each row's text
        row_chunks: dict[int, list[dict]] = {
            r.idx: chunk_text(
                t,
                min_words=cfg.chunk_words,
                sentence_only=cfg.sentence_only,
                single_speaker=(r._spk_token_2 is None),
            )
            for r, t in zip(rows, texts)
        }
        row_chunk_idx: dict[int, int] = {r.idx: 0 for r in rows}
        row_frame_count: dict[int, int] = {r.idx: 0 for r in rows}
        row_turn_count: dict[int, int] = {
            r.idx: 0 for r in rows
        }  # frames in current turn
        row_max_turn: dict[int, int] = {
            r.idx: cfg.max_turn_frames(0, r.prompt_wps) for r in rows
        }

        frame_buf = torch.zeros(
            N, max_frames, self.Q, device=self.device, dtype=torch.long
        )
        active = {r.idx: True for r in rows}
        self._rep_reset(cfg.rep_penalty, cfg.rep_window)

        _MIN_FPW = 2.0
        _RETRY_TEMPS = (0.5, 0.3)
        row_chunk_offset: dict[int, int] = {}
        row_retry_idx: dict[int, int] = {r.idx: 0 for r in rows}

        def _advance_turn(row: Row, temp: float | None = None) -> Tensor | None:
            """Prefill the next chunk's text and sample the first frame.
            Returns (Q,) first_codes on GPU, or None if no more chunks."""
            ci = row_chunk_idx[row.idx]
            chunks = row_chunks[row.idx]
            if ci >= len(chunks):
                return None
            if chunks[ci]["sc"] and row._spk_token_2 is not None:
                row._active_speaker = 1 - row._active_speaker
            chunk_text_str = chunks[ci]["text"]
            n_words = len(chunk_text_str.split())
            row_max_turn[row.idx] = cfg.max_turn_frames(n_words, row.prompt_wps)
            row_chunk_offset[row.idx] = row.offset
            t = temp or cfg.temperature
            hidden = self._add_agent_text(row, chunk_text_str)  # (d,)
            cb0 = model.codec_head(hidden.unsqueeze(0))  # (1, CS)
            # Use the same windowed rep bookkeeping as `_stream`. The previous
            # inline scatter_add incremented `_rep_counts` but skipped
            # `_rep_history`, so chunk-start codes never aged out of the
            # window — by ~frame 57 the count drift was enough to flip the
            # cb0 sample vs stream() on the same RNG.
            cb0 = self._rep_apply(cb0, cfg.rep_penalty)
            probs = F.softmax(cb0.float() / t, dim=-1)
            code0 = torch.multinomial(probs, 1).squeeze(-1)
            self._rep_push(code0, cfg.rep_penalty, cfg.rep_window)
            frame = rq.generate_cuda_graph_kv(
                hidden.unsqueeze(0),
                code0,
                t,
                self._zero_rq_bias,
                n_quantizers=cfg.n_codebooks,
            )  # (1, Q')
            self._set_codes_in(row.idx, frame[0])
            _ridx = row_frame_count[row.idx]
            frame_buf[row.idx, _ridx, : frame.shape[1]] = frame[0]
            row_frame_count[row.idx] += 1
            row_turn_count[row.idx] = 1
            return frame[0]

        def _retry_chunk(row: Row) -> bool:
            """Rewind KV and retry the current chunk at lower temperature.
            Returns True if a retry was attempted, False if fpw is fine or
            all retry temps exhausted."""
            ci = row_chunk_idx[row.idx]
            n_words = len(row_chunks[row.idx][ci]["text"].split())
            turn_n = row_turn_count[row.idx]
            fpw = turn_n / n_words if n_words > 0 else 999
            if fpw >= _MIN_FPW:
                row_retry_idx[row.idx] = 0
                return False
            ri = row_retry_idx[row.idx]
            if ri >= len(_RETRY_TEMPS):
                chunk_text_str = row_chunks[row.idx][ci]["text"]
                print(
                    f"[retry] row {row.idx} chunk {ci} gave up after {len(_RETRY_TEMPS)} retries (fpw={fpw:.1f}): {chunk_text_str[:60]}"
                )
                row_retry_idx[row.idx] = 0
                return False
            retry_temp = _RETRY_TEMPS[ri]
            row_retry_idx[row.idx] = ri + 1
            chunk_text_str = row_chunks[row.idx][ci]["text"]
            print(
                f"[retry] row {row.idx} chunk {ci} fpw={fpw:.1f} -> retry {ri+1}/{len(_RETRY_TEMPS)} at temp={retry_temp}: {chunk_text_str[:60]}"
            )
            saved = row_chunk_offset[row.idx]
            model.decoder.flash_kv_caches[0].seq_lens[row.idx : row.idx + 1] = saved
            row_frame_count[row.idx] -= turn_n
            self._rep_reset(cfg.rep_penalty, cfg.rep_window)
            _advance_turn(row, temp=retry_temp)
            return True

        # Everything runs under inference_mode: model is bf16 (no autocast
        # needed), and the backbone/RQ graphs mutate flash_kv_caches.seq_lens
        # which become inference tensors after capture.
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            for r in rows:
                first = _advance_turn(r)
                if first is None:
                    active[r.idx] = False

            # Per-step scratch bias sized at B=N (RQ graph captured at N)
            zero_rq_bias_bN = torch.zeros(
                self.Q - 1, self.CS, device=self.device, dtype=torch.float32
            )

            total_steps = 0
            while any(active.values()) and total_steps < max_frames:
                # Pre-iteration max_turn check: advance any row that hit
                # max_turn from the previous iteration's writes BEFORE running
                # _decode_step. Stream's `for step in range(1, max_per_turn)`
                # exits via range exhaustion without an extra _decode_step or
                # multinomial; matching that here keeps render bit-equal with
                # stream across chunk boundaries (otherwise render advances
                # seq_lens by 1 and consumes ~16 multinomials more per
                # max_turn boundary, diverging the model state).
                for r in rows:
                    if not active[r.idx]:
                        continue
                    if row_turn_count[r.idx] >= row_max_turn[r.idx]:
                        if _retry_chunk(r):
                            continue
                        row_retry_idx[r.idx] = 0
                        row_chunk_idx[r.idx] += 1
                        nxt = _advance_turn(r)
                        if nxt is None:
                            active[r.idx] = False
                if not any(active.values()):
                    break

                hidden_all, cb0_all, eos_all = self._decode_step()
                total_steps += 1

                cb0_penalised = self._rep_apply(cb0_all, cfg.rep_penalty)
                code0s = self._sample_cb0(cb0_penalised, cfg)  # (N,)
                self._rep_push(code0s, cfg.rep_penalty, cfg.rep_window)

                eos_cpu = torch.sigmoid(eos_all).float().cpu().numpy().flatten()

                # One batched RQ call for all N rows — the batched-RQ bug in
                # _generate_graph_body_kv was fixed (see model.py), so this is
                # correct and ~1.6x faster than the old per-row serial loop.
                if self.max_rows > 1:
                    frames_all = rq.generate_cuda_graph_kv(
                        hidden_all,
                        code0s,
                        cfg.temperature,
                        zero_rq_bias_bN,
                        n_quantizers=cfg.n_codebooks,
                    )  # (N, Q)
                else:
                    frames_all = None  # fall through to the B=1 path below

                for r in rows:
                    if not active[r.idx]:
                        continue
                    if r.offset >= max_seq or row_frame_count[r.idx] >= max_frames:
                        active[r.idx] = False
                        continue
                    turn_n = row_turn_count[r.idx]
                    hit_eos = (
                        turn_n >= min_frames and eos_cpu[r.idx] > cfg.eos_threshold
                    )
                    if hit_eos:
                        if _retry_chunk(r):
                            continue
                        row_retry_idx[r.idx] = 0
                        row_chunk_idx[r.idx] += 1
                        nxt = _advance_turn(r)
                        if nxt is None:
                            active[r.idx] = False
                        continue
                    if frames_all is not None:
                        frame = frames_all[r.idx]
                    else:
                        frame = rq.generate_cuda_graph_kv(
                            hidden_all[r.idx : r.idx + 1],
                            code0s[r.idx : r.idx + 1],
                            cfg.temperature,
                            self._zero_rq_bias,
                            n_quantizers=cfg.n_codebooks,
                        )[0]
                    self._set_codes_in(r.idx, frame)
                    _pidx = row_frame_count[r.idx]
                    frame_buf[r.idx, _pidx, : frame.shape[0]] = frame
                    row_frame_count[r.idx] += 1
                    row_turn_count[r.idx] += 1

        # Transfer valid frames to CPU
        result: dict[int, Tensor] = {}
        for r in rows:
            n = row_frame_count[r.idx]
            if n > 0:
                result[r.idx] = frame_buf[r.idx, :n].cpu()
        return result

    def _render_row(self, row: Row, text: str, cfg: GenConfig) -> Tensor:
        """Non-streaming single-row render. Returns (T, Q) codes."""
        results = self._render_rows([row], [text], cfg)
        return results.get(row.idx, torch.zeros(0, self.Q, dtype=torch.long))

    def render_all(
        self,
        rows: list[Row],
        texts: list[str],
        cfg: GenConfig = GenConfig(),
    ) -> list[Tensor]:
        """Batched non-streaming render. Returns one audio tensor per row."""
        results = self._render_rows(rows, texts, cfg)
        out: list[Tensor] = []
        for r in rows:
            codes = results.get(r.idx, torch.zeros(0, self.Q, dtype=torch.long))
            out.append(self._decode_full(codes))
        return out

    # ------------------------------------------------------------------
    # Continuous batching
    # ------------------------------------------------------------------

    def render_continuous(
        self,
        requests: list[RenderRequest],
        cfg: GenConfig = GenConfig(),
    ) -> list[Tensor]:
        """Process requests via continuous batching with bucketed CUDA graphs.

        Accepts more requests than max_rows. Slots are freed as requests finish
        and immediately backfilled from the pending queue. Returns one audio
        tensor per request, in input order.
        """
        if not requests:
            return []

        model = self.model
        rq = model.rq_transformer
        max_frames = int(cfg.max_secs * FRAME_RATE)
        min_frames = int(cfg.min_secs * FRAME_RATE)
        max_seq = self.max_seq - 10

        frame_bufs: list[list[Tensor]] = [[] for _ in requests]
        pending: deque[int] = deque(range(len(requests)))
        active: dict[int, _ActiveSlot] = {}  # physical slot -> info
        free_slots: set[int] = set(range(self.max_rows))

        self._rep_penalty_t.fill_(cfg.rep_penalty)

        def _advance_slot(info: _ActiveSlot) -> bool:
            """Prefill next chunk text, sample first frame. False if no more chunks."""
            row = info.row
            ci = info.chunk_idx
            if ci >= len(info.chunks):
                return False
            chunk = info.chunks[ci]
            if chunk["sc"] and row._spk_token_2 is not None:
                row._active_speaker = 1 - row._active_speaker
            chunk_text_str = chunk["text"]
            n_words = len(chunk_text_str.split())
            info.max_turn_frames = cfg.max_turn_frames(n_words, row.prompt_wps)
            hidden = self._add_agent_text(row, chunk_text_str)
            cb0 = model.codec_head(hidden.unsqueeze(0))
            rep_slice = self._rep_counts[row.idx : row.idx + 1]
            if cfg.rep_penalty > 1.0:
                f = torch.pow(self._rep_penalty_t, rep_slice)
                cb0 = torch.where(cb0 > 0, cb0 / f, cb0 * f)
            probs = F.softmax(cb0.float() / cfg.temperature, dim=-1)
            code0 = torch.multinomial(probs, 1).squeeze(-1)
            if cfg.rep_penalty > 1.0:
                self._rep_counts[row.idx, code0.item()] += 1.0
            frame = rq.generate_cuda_graph_kv(
                hidden.unsqueeze(0),
                code0,
                cfg.temperature,
                self._zero_rq_bias,
                n_quantizers=cfg.n_codebooks,
            )
            info.last_codes = frame[0].clone()
            frame_bufs[info.request_idx].append(info.last_codes.unsqueeze(0))
            info.frame_count += 1
            info.turn_count = 1
            info.seq_offset = row.offset
            return True

        def _backfill() -> None:
            while pending and free_slots:
                req_idx = pending.popleft()
                req = requests[req_idx]
                slot = min(free_slots)
                free_slots.remove(slot)
                with torch.inference_mode():
                    self.model.decoder.flash_kv_caches[0].seq_lens[slot] = 0
                self._rep_reset_slot(slot)
                row = Row(self, slot)
                self._rows[slot] = row
                row.prefill(
                    req.segments,
                    spk_emb=req.spk_emb,
                    segments_2=req.segments_2,
                    spk_emb_2=req.spk_emb_2,
                )
                if req.user_text or req.user_codes is not None:
                    row.add_user(req.user_text or "", req.user_codes)
                chunks = chunk_text(
                    req.text,
                    min_words=cfg.chunk_words,
                    sentence_only=cfg.sentence_only,
                    single_speaker=(row._spk_token_2 is None),
                )
                if not chunks:
                    row.close()
                    free_slots.add(slot)
                    frame_bufs[req_idx] = []
                    continue
                info = _ActiveSlot(row=row, request_idx=req_idx, chunks=chunks)
                if not _advance_slot(info):
                    row.close()
                    free_slots.add(slot)
                    continue
                active[slot] = info

        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            _backfill()
            B_max = self._buckets[-1]
            rq_code0s_buf = torch.zeros(B_max, device=self.device, dtype=torch.long)

            while active:
                slot_list = sorted(active.keys())
                B_active = len(slot_list)
                B_bucket = _get_bucket(B_active, self._buckets)

                # Pack inputs
                self._g_seq_inc.zero_()
                for qi, phys_slot in enumerate(slot_list):
                    info = active[phys_slot]
                    Q_out = info.last_codes.shape[0]
                    if Q_out < self.Q:
                        self._gc_codes_in[qi].zero_()
                        self._gc_codes_in[qi, :Q_out] = info.last_codes
                    else:
                        self._gc_codes_in[qi] = info.last_codes
                    self._g_slot_idx[qi] = phys_slot
                    self._g_seq_inc[phys_slot] = 1

                hidden_all, cb0_all, eos_all = self._bucketed_decode_step(B_active)
                for phys_slot in slot_list:
                    active[phys_slot].seq_offset += 1

                slot_tensor = self._g_slot_idx[:B_active].long()
                cb0_penalised = self._rep_apply_slots(
                    cb0_all, slot_tensor, cfg.rep_penalty
                )
                code0s = self._sample_cb0(cb0_penalised, cfg)
                self._rep_push_slots(code0s, slot_tensor, cfg.rep_penalty)

                eos_flags = (torch.sigmoid(eos_all).flatten() > cfg.eos_threshold).cpu()

                # Batched RQ at bucket size
                rq_hidden = self._gc_hidden_out[:B_bucket]
                rq_code0s_buf[:B_active] = code0s
                frames_all = rq.generate_cuda_graph_kv(
                    rq_hidden,
                    rq_code0s_buf[:B_bucket],
                    cfg.temperature,
                    self._zero_rq_bias,
                    n_quantizers=cfg.n_codebooks,
                )

                frames_cloned = frames_all[:B_active].clone()

                to_remove: list[int] = []
                for qi, phys_slot in enumerate(slot_list):
                    info = active[phys_slot]
                    info.row

                    if info.frame_count >= max_frames or info.seq_offset >= max_seq:
                        to_remove.append(phys_slot)
                        continue

                    turn_n = info.turn_count
                    hit_eos = turn_n >= min_frames and eos_flags[qi]
                    hit_max_turn = turn_n >= info.max_turn_frames

                    if hit_eos or hit_max_turn:
                        info.chunk_idx += 1
                        if not _advance_slot(info):
                            to_remove.append(phys_slot)
                        continue

                    frame = frames_cloned[qi]
                    info.last_codes = frame
                    frame_bufs[info.request_idx].append(frame.unsqueeze(0))
                    info.frame_count += 1
                    info.turn_count += 1

                for phys_slot in to_remove:
                    info = active.pop(phys_slot)
                    info.row.close()
                    free_slots.add(phys_slot)

                _backfill()

        # Decode all accumulated codes to audio
        audios: list[Tensor] = []
        for req_idx in range(len(requests)):
            bufs = frame_bufs[req_idx]
            if bufs:
                codes = torch.cat(bufs, dim=0)
                audios.append(self._decode_full(codes))
            else:
                audios.append(torch.zeros(1, 1, 0, device=self.device))
        return audios

    # ------------------------------------------------------------------
    # Streaming single-row path (vocoder graph per frame)
    # ------------------------------------------------------------------

    def _stream_row(
        self,
        row: Row,
        text: str,
        cfg: GenConfig,
        cancel=None,
        *,
        reset_rep: bool = True,
        final_turn: bool = False,
    ) -> Iterator[Tensor]:
        """Per-frame streaming decode for B=1 rows. Yields audio tensors."""
        model = self.model
        rq = model.rq_transformer
        max_frames = int(cfg.max_secs * FRAME_RATE)
        min_frames = int(cfg.min_secs * FRAME_RATE)
        max_seq = self.max_seq - 10

        single_speaker = row._spk_token_2 is None
        chunks = chunk_text(
            text,
            min_words=cfg.chunk_words,
            sentence_only=cfg.sentence_only,
            single_speaker=single_speaker,
        )
        if not chunks:
            return
        if cfg.first_chunk_words > 0 and cfg.first_chunk_words < cfg.chunk_words:
            first = chunks[0]["text"]
            words = first.split()
            if len(words) > cfg.first_chunk_words:
                head = " ".join(words[: cfg.first_chunk_words])
                tail = " ".join(words[cfg.first_chunk_words :])
                rest_chunks = chunk_text(
                    tail,
                    min_words=cfg.chunk_words,
                    sentence_only=cfg.sentence_only,
                    single_speaker=single_speaker,
                )
                sc = chunks[0]["sc"]
                chunks = [{"text": head, "sc": sc}] + rest_chunks + chunks[1:]

        # Seed the codec ONCE per session — first stream call (or after a
        # set_prompt that closed the stack). Subsequent calls keep accumulating
        # streaming state across chunks and hard-reset at the 10s boundary
        # inside decode_frame (matches the encoder _slide behaviour and
        # training's independent 10s chunks).
        with torch.inference_mode():
            if row._codec_ctx._stack is None:
                row._codec_ctx.prefill(n_codebooks=cfg.n_codebooks)

        if reset_rep:
            self._rep_reset(cfg.rep_penalty, cfg.rep_window)
        total_frames = 0

        # Initialize active speaker: if both speakers set, generation starts
        # on speaker 1 (index 0) unless the first chunk is marked sc=True.
        def _cancelled() -> bool:
            return cancel is not None and cancel.is_set()

        _GATE_RETRY_TEMPS = (0.5, 0.3)

        def _entropy_top1(probs: Tensor) -> tuple[float, float]:
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean().item()
            p1 = probs.max(-1).values.mean().item()
            return ent, p1

        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            last_idx = len(chunks) - 1
            for ci, chunk in enumerate(chunks):
                if total_frames >= max_frames or row.offset >= max_seq:
                    return
                if _cancelled():
                    return
                if chunk["sc"] and row._spk_token_2 is not None:
                    row._active_speaker = 1 - row._active_speaker
                is_last_chunk = final_turn and ci == last_idx

                # Gate only the first chunk (where rep state was just reset
                # and hallucinations are most common). Mid-turn gating would
                # need rep-state snapshot/restore — skipped for now.
                use_gate = cfg.gate_frames > 0 and ci == 0 and reset_rep
                chunk_start = row.offset
                gate_target = cfg.gate_frames

                n_words = len(chunk["text"].split())
                max_per_turn = cfg.max_turn_frames(n_words, row.prompt_wps)

                retry_idx = 0
                while True:  # retry loop (single pass when gate disabled / passes)
                    cur_temp = cfg.temperature
                    if retry_idx > 0:
                        cur_temp = _GATE_RETRY_TEMPS[
                            min(retry_idx - 1, len(_GATE_RETRY_TEMPS) - 1)
                        ]
                        # Rewind: seq_lens to chunk start + rep state.
                        # Codec ctx is untouched because the gate buffers codes
                        # without calling _stream_decode_frame.
                        model.decoder.flash_kv_caches[0].seq_lens[
                            row.idx : row.idx + 1
                        ] = chunk_start
                        self._rep_reset(cfg.rep_penalty, cfg.rep_window)

                    hidden = self._add_agent_text(
                        row, chunk["text"], final=is_last_chunk
                    )  # (d,)

                    cb0 = model.codec_head(hidden.unsqueeze(0))
                    cb0 = self._rep_apply(cb0, cfg.rep_penalty)
                    probs = F.softmax(cb0.float() / cur_temp, dim=-1)
                    held: list[Tensor] = []
                    entropies: list[float] = []
                    top1s: list[float] = []
                    if use_gate:
                        e, p1 = _entropy_top1(probs)
                        entropies.append(e)
                        top1s.append(p1)
                    code0 = torch.multinomial(probs, 1).squeeze(-1)
                    self._rep_push(code0, cfg.rep_penalty, cfg.rep_window)
                    frame = rq.generate_cuda_graph_kv(
                        hidden.unsqueeze(0),
                        code0,
                        cur_temp,
                        self._zero_rq_bias,
                        n_quantizers=cfg.n_codebooks,
                    )  # (1, Q')
                    self._set_codes_in(0, frame[0])

                    if use_gate:
                        held.append(frame[0])
                    else:
                        yield self._stream_decode_frame(frame[0], cfg.n_codebooks)
                        total_frames += 1

                    retry_triggered = False
                    for step in range(1, max_per_turn):
                        if total_frames >= max_frames or row.offset >= max_seq:
                            for hc in held:
                                yield self._stream_decode_frame(hc, cfg.n_codebooks)
                                total_frames += 1
                            return
                        if _cancelled():
                            return
                        hidden_all, cb0_all, eos_all = self._decode_step()
                        cb0_pen = self._rep_apply(cb0_all, cfg.rep_penalty)

                        gathering = use_gate and len(held) < gate_target
                        # Sample at cur_temp so retries actually use the lower
                        # temp (cfg.temperature is frozen).
                        probs_s = F.softmax(cb0_pen.float() / cur_temp, dim=-1)
                        if gathering:
                            e, p1 = _entropy_top1(probs_s)
                            entropies.append(e)
                            top1s.append(p1)
                        if cfg.top_k is not None and cfg.top_k > 0:
                            topv, topi = probs_s.topk(cfg.top_k, dim=-1)
                            mask = torch.zeros_like(probs_s).scatter_(1, topi, topv)
                            probs_s = mask / mask.sum(-1, keepdim=True).clamp_min(1e-9)
                        code0s = torch.multinomial(probs_s, 1).squeeze(-1)
                        self._rep_push(code0s, cfg.rep_penalty, cfg.rep_window)

                        if (
                            step >= min_frames
                            and torch.sigmoid(eos_all[0, 0]).item() > cfg.eos_threshold
                        ):
                            for hc in held:
                                yield self._stream_decode_frame(hc, cfg.n_codebooks)
                                total_frames += 1
                            held = []
                            break

                        frame = rq.generate_cuda_graph_kv(
                            hidden_all[0:1],
                            code0s[0:1],
                            cur_temp,
                            self._zero_rq_bias,
                            n_quantizers=cfg.n_codebooks,
                        )
                        self._set_codes_in(0, frame[0])

                        if gathering:
                            held.append(frame[0])
                            if len(held) == gate_target:
                                avg_e = sum(entropies) / len(entropies)
                                avg_p1 = sum(top1s) / len(top1s)
                                if (
                                    avg_e > cfg.gate_entropy_max
                                    and retry_idx < cfg.gate_retries
                                ):
                                    nxt_t = _GATE_RETRY_TEMPS[
                                        min(retry_idx, len(_GATE_RETRY_TEMPS) - 1)
                                    ]
                                    print(
                                        f"[stream-gate] H={avg_e:.2f} p1={avg_p1:.3f} "
                                        f"> thr={cfg.gate_entropy_max:.2f} -> retry "
                                        f"{retry_idx + 1}/{cfg.gate_retries} @ temp={nxt_t}"
                                    )
                                    retry_triggered = True
                                    break
                                print(
                                    f"[stream-gate] H={avg_e:.2f} p1={avg_p1:.3f} pass "
                                    f"({len(held)}f, retry={retry_idx})"
                                )
                                for hc in held:
                                    yield self._stream_decode_frame(
                                        hc, cfg.n_codebooks
                                    )
                                    total_frames += 1
                                held = []
                                use_gate = False
                        else:
                            yield self._stream_decode_frame(
                                frame[0], cfg.n_codebooks
                            )
                            total_frames += 1

                    if retry_triggered:
                        retry_idx += 1
                        continue
                    if held:
                        avg_e = sum(entropies) / len(entropies) if entropies else 0
                        print(
                            f"[stream-gate] short chunk ({len(held)}f < {gate_target}), "
                            f"H={avg_e:.2f} flush w/o gate decision"
                        )
                        for hc in held:
                            yield self._stream_decode_frame(hc, cfg.n_codebooks)
                            total_frames += 1
                    break  # exit retry loop, advance to next chunk

    def _stream_decode_frame(self, frame_codes: Tensor, n_codebooks: int) -> Tensor:
        """Decode one audio frame from (Q,) codes via vocoder graph."""
        # Pad to full Q for the codec rolling context
        if frame_codes.shape[0] < self.Q:
            padded = torch.zeros(self.Q, device=self.device, dtype=frame_codes.dtype)
            padded[: frame_codes.shape[0]] = frame_codes
            frame_codes = padded
        codes_t = frame_codes.unsqueeze(0).unsqueeze(-1)  # (1, Q, 1)
        decode_codes = codes_t[:, :n_codebooks] if n_codebooks > 0 else codes_t
        return self._row_codec_ctx.decode_frame(decode_codes, store_codes=codes_t)

    # Kept public so stream callers can reach the active row's codec
    @property
    def _row_codec_ctx(self) -> CodecCtx:
        # At max_rows=1 there's only row 0; return its CodecCtx
        if 0 in self._rows:
            return self._rows[0]._codec_ctx
        raise RuntimeError("no active row for streaming codec decode")

    # ------------------------------------------------------------------
    # Full-codec decode (non-streaming)
    # ------------------------------------------------------------------

    def _decode_full(self, codes: Tensor) -> Tensor:
        """Full codec forward on accumulated codes. codes: (T, Q) -> (1, 1, S)."""
        if codes.shape[0] == 0:
            return torch.zeros(1, 1, 0, device=self.device)
        codes_bqt = codes.T.unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            return self.codec.decode(codes_bqt)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def sample_rate(self) -> int:
        return int(QWEN_SR)


# ---------------------------------------------------------------------------
# Private: per-row KV slice (used by _prefill_emb)
# ---------------------------------------------------------------------------


@dataclass
class _SlicedKV:
    k_cache: Tensor
    v_cache: Tensor
    seq_lens: Tensor
