import re

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from vui.model import Vui
from vui.qwen_codec import DOWNSAMPLE_RATE, FRAME_RATE, QwenCodecDecoder
from vui.sampling import multinomial, sample_top_k, sample_top_p, sample_top_p_top_k
from vui.tokenizer import SPECIAL_TOKENS


def ensure_spaces_around_tags(text: str):
    text = re.sub(
        r"(?<![<\[\s])(\[)",
        lambda m: (
            f"\n{m.group(1)}"
            if m.start() > 0 and text[m.start() - 1] == "\n"
            else f" {m.group(1)}"
        ),
        text,
    )
    text = re.sub(
        r"(?<!\d\])(\])(?![>\]\s])",
        lambda m: (
            f"{m.group(1)}\n"
            if m.end() < len(text) and text[m.end()] == "\n"
            else f"{m.group(1)} "
        ),
        text,
    )
    return text.strip()

REPLACE = [
    ("—", ","),
    ("'", "'"),
    (":", ","),
    (";", ","),
]

# engine = None
wm = None


@torch.inference_mode()
def asr(chunk, model=None, prefix=None, prompt=None):
    import whisper

    global wm
    if model is not None:
        wm = model
    elif wm is None:
        wm = whisper.load_model("turbo", "cuda")

    chunk = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(chunk, n_mels=wm.dims.n_mels).to(wm.device)
    options = whisper.DecodingOptions(
        language="en", without_timestamps=True, prefix=prefix, prompt=prompt
    )
    if len(mel.shape) != 3:
        mel = mel[None]
    result = whisper.decode(wm, mel, options)
    return result[0].text


_moonshine_transcriber = None


def asr_moonshine(audio: Tensor, sr_in: int = 24000, model_arch: int = 4) -> str:
    """Transcribe a (T,) or (1, T) audio tensor with moonshine_voice (CPU, fast).

    Expects audio sampled at sr_in; will resample to 16 kHz internally.
    """
    import ctypes

    import numpy as np
    from julius.resample import resample_frac
    from moonshine_voice import Transcriber, get_model_for_language
    from moonshine_voice.errors import check_error
    from moonshine_voice.moonshine_api import TranscriptC

    global _moonshine_transcriber
    if _moonshine_transcriber is None:
        model_path, arch = get_model_for_language("en", model_arch)
        _moonshine_transcriber = Transcriber(model_path=model_path, model_arch=arch)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.detach().float().cpu()
    if sr_in != 16000:
        audio = resample_frac(audio, sr_in, 16000)
    arr = np.ascontiguousarray(audio.squeeze(0).numpy(), dtype=np.float32)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out = ctypes.POINTER(TranscriptC)()
    err = _moonshine_transcriber._lib.moonshine_transcribe_without_streaming(
        _moonshine_transcriber._handle, ptr, len(arr), 16000, 0, ctypes.byref(out)
    )
    check_error(err)
    transcript = _moonshine_transcriber._parse_transcript(out)
    return " ".join(line.text.strip() for line in transcript.lines if line.text.strip())


# def replace_numbers_with_words(text):
#     global engine
#     if engine is None:
#         engine = inflect.engine()
#     def number_to_words(match):
#         number = match.group()
#         return engine.number_to_words(number) + " "
#     return re.sub(r"\d+", number_to_words, text)


valid_non_speech = [t for t in SPECIAL_TOKENS if t.startswith("[") and t.endswith("]")]


def remove_all_invalid_non_speech(txt):
    """
    Remove all non-speech markers that are not in the valid_non_speech list.
    Only keeps valid non-speech markers like [breath], [sigh], etc.
    """
    # Find all text within square brackets
    bracket_pattern = r"\[([^\]]+)\]"
    brackets = re.findall(bracket_pattern, txt)

    # For each bracketed text, check if it's in our valid list
    for bracket in brackets:
        bracket_with_brackets = f"[{bracket}]"
        if bracket_with_brackets not in valid_non_speech:
            txt = txt.replace(bracket_with_brackets, "")

    return txt


def simple_clean(text):
    text = re.sub(r"(\d+)am", r"\1 AM", text)
    text = re.sub(r"(\d+)pm", r"\1 PM", text)
    # text = replace_numbers_with_words(text)
    text = ensure_spaces_around_tags(text)
    text = remove_all_invalid_non_speech(text)

    # Normalize curly quotes first.
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    # Single quotes that aren't sandwiched between two letters are treated
    # as opening/closing wraps (or stray punctuation) and converted to double
    # quotes so the double-quote stripping below removes them. Apostrophes
    # inside a word (don't, isn't, can't) are preserved because both sides
    # are letters.
    text = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", '"', text)
    text = text.replace('"', "")
    text = text.replace("%", " percent")
    text = text.replace("*", "")
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("(", "").replace(")", "")
    text = text.replace(";", "")
    text = text.replace("–", " ")
    text = text.replace("—", "")
    # Remove ASCII hyphens that aren't inside a compound word (e.g. "-word"
    # or a stray "-"), but keep intra-word hyphens like "well-known".
    text = re.sub(r"(?<!\w)-|-(?!\w)", "", text)
    text = text.replace(":", "")
    text = text.replace("…", "...")
    text = text.replace("s...", "s")
    text = re.sub(r"\bvs\.?\b", "versus", text, flags=re.IGNORECASE)

    # replace repeating \n with just one \n
    text = re.sub(r"\n+", "\n", text)

    # Final spacing pass: collapse repeated spaces/tabs and trim whitespace
    # around newlines so we never emit doubled spaces.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    ntxt = text.strip()
    if not (ntxt.endswith(".") or ntxt.endswith("?")):
        ntxt += "."
    return ntxt


# ---------------------------------------------------------------------------
# RQ Transformer + Qwen Codec inference (CUDA-graphed, flash-attn)
# ---------------------------------------------------------------------------


class InferenceState:
    """Holds CUDA graphs + codec for fast streaming inference."""

    def __init__(
        self,
        model: Vui,
        codec: QwenCodecDecoder,
        chunk_frames: int = 7,
        ctx_frames: int = 3,
        sq_scores: tuple[float, ...] | None = None,
        wps_score: float = 1.0,
        codec_graphs: bool = True,
        compile_rq: bool = False,
    ):
        """compile_rq: use torch.compile fullgraph for the RQ transformer
        (~2.8x faster than the manual CUDA graph). Only safe in single-threaded
        contexts — DO NOT use from gradio worker threads (will hang)."""
        self.model = model
        self.codec = codec
        self.chunk_frames = chunk_frames
        self.ctx_frames = ctx_frames
        self.device = model.device
        self.dtype = model.dtype
        self._codec_graphs = codec_graphs
        self._compile_rq = compile_rq
        self._setup_graphs()
        self.model.set_cond_bias(sq_scores=sq_scores, wps_score=wps_score)

    def _setup_graphs(self):
        device, dtype = self.device, self.dtype
        # Shared memory pool so all graphs can coexist without conflicts
        self._graph_pool = torch.cuda.graph_pool_handle()
        # Backbone decode graph (flash-attn KV cache)
        self.model.setup_decode_graph(pool=self._graph_pool)
        if self._compile_rq:
            # torch.compile fullgraph path: ~2.14ms vs ~5.2ms manual graph.
            # Wrap to match the (hidden, code0, temp_float, bias) signature.
            # mode="reduce-overhead": cudagraph trees on, no kernel autotune.
            # Needed for the ~2ms/call speed (default mode is 40ms/call from
            # uncaptured Python dispatch). max-autotune would add ~15% more
            # speed but re-benchmarks every process (~80s warmup) — not worth it.
            compiled = torch.compile(
                self.model.rq_transformer.generate_kv_compilable,
                mode="reduce-overhead",
                fullgraph=True,
            )
            _temp_buf = torch.zeros(1, device=device, dtype=torch.float32)

            def _rq_compiled(hidden, code0, temperature, bias):
                _temp_buf.fill_(temperature)
                return compiled(hidden, code0, _temp_buf, bias)

            self._rq_generate = _rq_compiled
        else:
            self.model.rq_transformer.setup_cuda_graph_kv(
                device, dtype, pool=self._graph_pool
            )
            self._rq_generate = self.model.rq_transformer.generate_cuda_graph_kv
        # Codec graphs: first chunk (no context) and subsequent (with context)
        self._codec_graph_first = None
        self._codec_graph_ctx = None
        if self._codec_graphs:
            with torch.autocast("cuda", torch.bfloat16):
                self.codec.setup_decode_graph(
                    self.chunk_frames, device, dtype, pool=self._graph_pool
                )
                self._codec_graph_first = self.codec._decode_graph
                self._codec_buf_first = self.codec._graph_codes_in
                self._codec_audio_first = self.codec._graph_audio_out

                total = self.ctx_frames + self.chunk_frames
                self.codec.setup_decode_graph(
                    total, device, dtype, pool=self._graph_pool
                )
                self._codec_graph_ctx = self.codec._decode_graph
                self._codec_buf_ctx = self.codec._graph_codes_in
                self._codec_audio_ctx = self.codec._graph_audio_out

    def reset(self):
        """Move the position pointer back to 0. Equivalent to `rewind(0)`.

        IMPORTANT: this does NOT touch the K/V tensor data. Only the position
        counter (`flash_kv_caches[0].seq_lens`) and `self.offset` are updated.
        Subsequent `forward_flash` writes will overwrite the K/V tensors in
        place at the new positions; nothing is allocated or zeroed.
        """
        self.rewind(0)

    def teardown(self):
        """Fully deallocate KV caches (invalidates backbone graph)."""
        self.model.decoder.deallocate_kv_cache()

    def prefill(
        self, segments: list[tuple[str, str | Tensor]], cond_last: bool = True
    ) -> int:
        """Prefill KV cache with interleaved text/audio segments.

        segments: list of ("text", "hello world") or ("audio", codes_tensor)
            codes_tensor: (T, Q) int tensor of audio codes
        cond_last: if True, only apply cond_bias to the last segment (generation text).
            Prompt text/audio should NOT have cond_bias (matching demo.py behavior).

        Returns total offset (number of tokens in KV cache).
        Stores last hidden state in self._prefill_hidden for first-frame decode.
        """
        cond_bias = self.model._cond_bias
        out = None
        for i, (seg_type, data) in enumerate(segments):
            is_last = i == len(segments) - 1
            bias = cond_bias if (not cond_last or is_last) else 0
            if seg_type == "text":
                encoded = self.model.text_tokenizer(
                    [data], padding="longest", return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(self.device)
                emb = self.model.token_emb(input_ids) + bias
                n = input_ids.shape[1]
            else:
                emb = self.model.embed_audio(data.to(self.device)).unsqueeze(0) + bias
                n = emb.shape[1]
            positions = torch.arange(self.offset, self.offset + n, device=self.device)
            out = self.model.decoder.forward_flash(emb, positions)
            self.offset += n
        if out is not None:
            hidden = out[:, -1]
            self._prefill_hidden = hidden
            self._prefill_logits = self.model.codec_head(hidden)
            self._prefill_eos = self.model.eos_head(hidden)
        return self.offset

    def _prefill(self, text: str) -> int:
        """Legacy text-only prefill. Resets cache first."""
        return self.prefill([("text", text)])

    _ckpt_offset: int = 0
    _ckpt_spk_token: Tensor | None = None
    _ckpt_spk_token_2: Tensor | None = None

    @torch.inference_mode()
    def prefill_prompt(
        self,
        prompt_segments: list[tuple[str, Tensor]] | None = None,
        prompt_segments_2: list[tuple[str, Tensor]] | None = None,
        spk_emb: Tensor | None = None,
        spk_emb_2: Tensor | None = None,
    ) -> int:
        """Prefill prompt segments (text + audio + spk tokens) into the
        backbone flash KV cache. Resets state first. Returns end offset.

        Use with `checkpoint()` + `rewind()` for low-latency multi-turn
        streaming where the prompt is reused across turns.
        """
        self.reset()
        model = self.model
        tok = model.text_tokenizer
        device = self.device
        spk_token = (
            model.embed_speaker(spk_emb)
            if spk_emb is not None and model.spk_proj is not None
            else None
        )
        spk_token_2 = (
            model.embed_speaker(spk_emb_2)
            if spk_emb_2 is not None and model.spk_proj is not None
            else None
        )
        # Cache spk tokens for resume-mode stream_frames calls
        self._ckpt_spk_token = spk_token
        self._ckpt_spk_token_2 = spk_token_2
        with (
            torch.autocast("cuda", torch.bfloat16, True),
            sdpa_kernel([SDPBackend.MATH]),
        ):
            for segs, seg_spk in (
                (prompt_segments, spk_token),
                (prompt_segments_2, spk_token_2),
            ):
                if segs is None:
                    continue
                for seg_text, seg_codes in segs:
                    if seg_spk is not None:
                        _forward_flash_emb(self, seg_spk)
                    if seg_text:
                        ids = tok.encode(simple_clean(seg_text)).to(device)
                        _forward_flash_emb(self, model.token_emb(ids[None]))
                    pc = seg_codes.to(device)
                    if pc.dim() == 2:
                        pc = pc.unsqueeze(0)
                    _forward_flash_emb(self, model.embed_audio(pc))
        return self.offset

    def checkpoint(self) -> int:
        """Save current KV-cache offset as a restore point. Returns it."""
        self._ckpt_offset = self.offset
        return self.offset

    def rewind(self, offset: int | None = None) -> int:
        """Move the position pointer to `offset` (default: last checkpoint).

        Only two values change: the int `self.offset` and the int counter
        `flash_kv_caches[0].seq_lens[0]`. The K/V tensor data is NEVER
        zeroed or copied — positions [0:offset] in the cache stay exactly
        as they were last written, and any "stale" K/V at positions
        [offset:] is logically discarded (flash_attn won't read past
        seq_lens, and the next `forward_flash` call will overwrite it
        in place).

        This is the sole way to reposition the cache. `reset()` is just
        `rewind(0)`.
        """
        target = offset if offset is not None else getattr(self, "_ckpt_offset", 0)
        # Defensive: cache may have been deallocated externally.
        if self.model.decoder.flash_kv_caches is None:
            self._setup_graphs()
        self.model.decoder.flash_kv_caches[0].seq_lens[:1].fill_(target)
        self.offset = target
        return target

    def _decode_codec_chunk(self, codes_tensor: Tensor, has_ctx: bool) -> Tensor:
        Q = codes_tensor.shape[1]
        n = codes_tensor.shape[2]
        expected_first = self.chunk_frames
        expected_ctx = self.ctx_frames + self.chunk_frames
        if not has_ctx and n == expected_first and self._codec_graph_first is not None:
            self._codec_buf_first[:, :Q].copy_(codes_tensor)
            self._codec_buf_first[:, Q:].zero_()
            self._codec_graph_first.replay()
            return self._codec_audio_first.clone()
        elif has_ctx and n == expected_ctx and self._codec_graph_ctx is not None:
            self._codec_buf_ctx[:, :Q].copy_(codes_tensor)
            self._codec_buf_ctx[:, Q:].zero_()
            self._codec_graph_ctx.replay()
            audio = self._codec_audio_ctx.clone()
            return audio[..., self.ctx_frames * DOWNSAMPLE_RATE :]
        else:
            audio = self.codec.decode(codes_tensor)
            if has_ctx:
                audio = audio[..., self.ctx_frames * DOWNSAMPLE_RATE :]
            return audio


CODEC_HZ = FRAME_RATE  # 12.5 frames/sec


# Sentence-end matcher shared with `vui.serving.stream.llm.llm_stream_chunks`.
# Terminal punctuation, optional close-quote/paren, followed by space, `[`,
# or end-of-string. Duplicated here to avoid an inference.py ↔ stream.llm
# import cycle; keep the two definitions in sync.
_SENT_END_RE = re.compile(r'[.!?]+(?:["\'\)\]]+)?(?=\s|\[|$)')


def _sentence_split(text: str, max_words: int) -> list[str]:
    """Sentence-bounded splitter matching `llm_stream_chunks`: break on
    [.!?]; hard word-count fallback at max_words for runaway clauses
    (never breaks on commas)."""
    out: list[str] = []
    buf = text
    while buf:
        m = _SENT_END_RE.search(buf)
        if m is not None:
            piece = buf[: m.end()].strip()
            if piece:
                out.append(piece)
            buf = buf[m.end() :].lstrip()
            continue
        words = buf.split()
        if len(words) > max_words:
            piece = " ".join(words[:max_words]).rstrip(",;:")
            if piece:
                out.append(piece)
            buf = " ".join(words[max_words:])
        else:
            piece = buf.strip()
            if piece:
                out.append(piece)
            buf = ""
    return out


def chunk_text(
    text: str,
    min_words: int = 5,
    sentence_only: bool = False,
    single_speaker: bool = False,
) -> list[dict]:
    """Split text into turns, matching training's streamed_tts mode.

    Newlines and [SC] markers = speaker changes. Within a speaker:
    - default: split at sentence boundaries (.!?) and before commas, then
      merge so each chunk is ~min_words. Chunks never end on a comma or a
      [tag].
    - sentence_only=True: split only at sentence boundaries (.!?), with a
      hard word-count fallback at min_words for runaway sentences.
      Mirrors `llm_stream_chunks` so demo / batched paths chunk the same
      way the streaming server does at LLM-output time.
    - single_speaker=True: never mark a chunk as a speaker change. Newlines
      stay as paragraph breaks but don't flip speaker; explicit `[SC]`
      markers in the input are stripped of their speaker-change semantics.
      Use when the row only has one speaker prompt — the [SC] marker has
      no target to flip to and just clutters previews/logs.
    """
    raw_lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    lines = []
    for line in raw_lines:
        parts = re.split(r"\[SC\]", line)
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                lines.append({"text": part, "sc_explicit": i > 0})

    chunks: list[dict] = []
    for line_idx, line_info in enumerate(lines):
        line = simple_clean(line_info["text"])
        is_new_speaker = (line_idx > 0) or line_info["sc_explicit"]
        if single_speaker:
            is_new_speaker = False
        if sentence_only:
            fixed = _sentence_split(line, max_words=min_words)
        else:
            # Split after .!? or before comma-space (comma stays with next clause)
            clauses = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+|\s*,\s+", line) if s.strip()
            ]
            merged: list[str] = []
            for clause in clauses:
                if merged and len(merged[-1].split()) < min_words:
                    merged[-1] += " " + clause
                else:
                    merged.append(clause)
            if len(merged) > 1 and len(merged[-1].split()) < min_words:
                merged[-2] += " " + merged[-1]
                merged.pop()
            # Don't end a chunk on a [tag] or trailing comma
            fixed = []
            for m in merged:
                m = m.rstrip(",").rstrip()
                if fixed and re.search(r"\[.*?\]\s*$", fixed[-1]):
                    fixed[-1] += " " + m
                else:
                    fixed.append(m)
        for i, sent in enumerate(fixed):
            if sent:
                chunks.append({"text": sent, "sc": is_new_speaker and i == 0})
    return chunks


def _apply_rep_penalty_cb0_fast(
    logits: Tensor, counts: dict[int, int], penalty: float
) -> Tensor:
    """Apply repetition penalty using a pre-built {token: count} dict.
    Caller maintains the rolling counts incrementally to avoid O(window) work
    per frame. Skips the .clone() unless we actually have penalties to apply."""
    if penalty <= 1.0 or not counts:
        return logits
    logits = logits.clone()
    for tok_id, count in counts.items():
        p = penalty**count
        v = logits[0, tok_id]
        logits[0, tok_id] = v / p if v > 0 else v * p
    return logits


# Backward-compat shim still used by callers that prefer the simple API.
def _apply_rep_penalty_cb0(
    logits: Tensor, past: list[int], penalty: float, window: int
) -> Tensor:
    if penalty <= 1.0 or not past:
        return logits
    history = past[-window:] if window > 0 else past
    from collections import Counter as _C

    return _apply_rep_penalty_cb0_fast(logits, dict(_C(history)), penalty)


def _compute_rq_logit_bias(
    past_codes_per_q: list[list[int]],
    penalty: float,
    window: int,
    n_quantizers: int,
    codebook_size: int,
    device,
) -> Tensor | None:
    if penalty <= 1.0:
        return None
    import math as _math
    from collections import Counter as _C

    bias = torch.zeros(
        n_quantizers - 1, codebook_size, device=device, dtype=torch.float32
    )
    log_p = _math.log(penalty)
    for q_idx in range(n_quantizers - 1):
        cb = q_idx + 1
        if cb >= len(past_codes_per_q):
            continue
        history = past_codes_per_q[cb][-window:] if window > 0 else past_codes_per_q[cb]
        if not history:
            continue
        counts = _C(history)
        for tok_id, count in counts.items():
            bias[q_idx, tok_id] = -count * log_p
    return bias


def _sample_codes(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Tensor:
    probs = F.softmax(logits / temperature, dim=-1)
    if top_p is not None and top_k is not None:
        return sample_top_p_top_k(probs, top_p, top_k)
    if top_p is not None and top_p > 0:
        return sample_top_p(probs, top_p)
    if top_k is not None and top_k > 0:
        return sample_top_k(probs, top_k)
    return multinomial(probs, num_samples=1)


@torch.inference_mode()
def stream_frames(
    state: InferenceState,
    text: str,
    *,
    prompt_segments: list[tuple[str, Tensor]] | None = None,
    prompt_segments_2: list[tuple[str, Tensor]] | None = None,
    spk_emb: Tensor | None = None,
    spk_emb_2: Tensor | None = None,
    temperature: float = 0.9,
    top_k: int | None = 100,
    top_p: float | None = None,
    rep_penalty: float = 1.0,
    rep_window: int = 0,
    sq_scores: tuple[float, ...] | None = None,
    wps_score: float = 0.0,
    chunk_words: int = 5,
    max_secs: float = 120.0,
    max_turn_secs: float = 15.0,
    eos_threshold: float = 0.45,
    min_turn_secs: float = 0.5,
    resume: bool = False,
):
    """Low-level per-frame code generator. Yields `(code_frame, turn_end)`.

    resume: if True, skip `state.reset()` and skip prompt-segment prefill —
        assume `state.prefill_prompt(...)` was already called and `state.offset`
        has been rewound (via `state.rewind()`) to the post-prompt checkpoint.
        Use this for low-latency multi-turn streaming where the prompt's KV
        cache is reused across turns.

    code_frame: (Q,) long tensor of audio codes for one frame, ON GPU
                (caller moves to CPU if needed — GPU tensors are faster to
                re-embed for the codec chunk decode).
    turn_end: `None` for intra-turn frames, or the turn's chunk_info dict
              (with 'text', 'sc', 'frames', 'secs' populated) when the final
              frame of that turn is emitted.

    This is the canonical decode loop. `render_codes_stream`,
    `render_audio_stream` and tts_worker all consume it.

    Handles: text chunking, prompt prefill, speaker embs, [SC] alternation,
    cond_bias, rep penalty, top_k/top_p, EOS/max limits.
    """
    model = state.model
    device = state.device
    tok = model.text_tokenizer
    sc_id = tok.special_to_id["[SC]"]

    # wps_score intentionally NOT passed — disabled at inference per user request.
    model.set_cond_bias(sq_scores=sq_scores, wps_score=0.0)
    cond_bias = model._cond_bias  # (1, 1, d)

    spk_token = (
        model.embed_speaker(spk_emb)
        if spk_emb is not None and model.spk_proj is not None
        else None
    )
    spk_token_2 = (
        model.embed_speaker(spk_emb_2)
        if spk_emb_2 is not None and model.spk_proj is not None
        else None
    )
    # Resume mode: fall back to spk tokens cached during prefill_prompt
    if resume:
        if spk_token is None:
            spk_token = state._ckpt_spk_token
        if spk_token_2 is None:
            spk_token_2 = state._ckpt_spk_token_2

    chunks = chunk_text(text, min_words=chunk_words)
    if not chunks:
        return

    has_rq = model.rq_transformer is not None
    Q = model.rq_transformer.n_quantizers if has_rq else 0
    CS = model.rq_transformer.codebook_size if has_rq else 0
    zero_rq_bias = (
        torch.zeros(max(Q - 1, 0), CS, device=device, dtype=torch.float32)
        if has_rq
        else None
    )

    # Incremental rolling counter for cb0 rep-penalty (O(1) per frame).
    from collections import deque as _deque

    cb0_window: _deque[int] = _deque(maxlen=rep_window if rep_window > 0 else None)
    cb0_counts: dict[int, int] = {}

    def rq_sample(cb0_logits: Tensor, hidden: Tensor) -> Tensor:
        penalised = _apply_rep_penalty_cb0_fast(cb0_logits, cb0_counts, rep_penalty)
        code0 = _sample_codes(penalised, temperature, top_k, top_p).squeeze(-1)
        c = int(code0.item())
        # Slide the rolling window: drop the oldest if at capacity, then push.
        if cb0_window.maxlen is not None and len(cb0_window) == cb0_window.maxlen:
            old = cb0_window[0]  # about to be evicted by append
            cb0_counts[old] -= 1
            if cb0_counts[old] == 0:
                del cb0_counts[old]
        cb0_window.append(c)
        cb0_counts[c] = cb0_counts.get(c, 0) + 1
        return state._rq_generate(hidden, code0, temperature, zero_rq_bias).clone()

    max_frames = int(max_secs * CODEC_HZ)
    max_per_turn = int(max_turn_secs * CODEC_HZ)
    min_frames = int(min_turn_secs * CODEC_HZ)
    max_seq = model.decoder.max_seqlen - 10

    if not resume:
        state.reset()  # zero seq_lens, offset=0
    total_frames = 0

    with (
        torch.autocast("cuda", torch.bfloat16, True),
        sdpa_kernel([SDPBackend.MATH]),
    ):
        # --- Prompt prefill (interleaved text/audio segments) ---
        if not resume and (
            prompt_segments is not None or prompt_segments_2 is not None
        ):
            first_sc = chunks[0]["sc"]

            def _prefill_speaker(segs, is_last_speaker):
                if segs is None:
                    return
                seg_spk = spk_token_2 if is_last_speaker else spk_token
                for seg_idx, (seg_text, seg_codes) in enumerate(segs):
                    # Per training format: [spk] [text] [audio] before EVERY segment
                    if seg_spk is not None:
                        _forward_flash_emb(state, seg_spk)
                    if seg_text:
                        ids = tok.encode(simple_clean(seg_text)).to(device)
                        append_sc = (
                            seg_idx == len(segs) - 1
                            and first_sc
                            and (is_last_speaker or prompt_segments_2 is None)
                        )
                        if append_sc:
                            ids = torch.cat([ids, torch.tensor([sc_id], device=device)])
                        _forward_flash_emb(state, model.token_emb(ids[None]))
                    pc = seg_codes.to(device)
                    if pc.dim() == 2:
                        pc = pc.unsqueeze(0)
                    _forward_flash_emb(state, model.embed_audio(pc))

            _prefill_speaker(prompt_segments, is_last_speaker=False)
            _prefill_speaker(prompt_segments_2, is_last_speaker=True)

        current_speaker = 1 if prompt_segments_2 is not None else 0

        for turn_idx, chunk in enumerate(chunks):
            if state.offset >= max_seq or total_frames >= max_frames:
                break

            if chunk["sc"]:
                current_speaker = 1 - current_speaker

            turn_spk = (
                spk_token_2
                if (current_speaker == 1 and spk_token_2 is not None)
                else spk_token
            )
            if turn_spk is not None:
                _forward_flash_emb(state, turn_spk)

            ids = tok.encode(chunk["text"]).to(device)
            next_is_sc = turn_idx + 1 < len(chunks) and chunks[turn_idx + 1]["sc"]
            if next_is_sc:
                ids = torch.cat([ids, torch.tensor([sc_id], device=device)])
            text_emb = model.token_emb(ids[None]) + cond_bias
            n = text_emb.size(1)
            if state.offset + n >= max_seq:
                break
            out = _forward_flash_emb(state, text_emb)
            hidden = out[:, -1]

            if has_rq:
                first_codes = rq_sample(model.codec_head(hidden), hidden)
            else:
                logits = model.audio_head(hidden)
                first_codes = _sample_codes(logits, temperature, top_k, top_p).squeeze(
                    -1
                )

            # Buffer pending frames so we can tag the last one with chunk_info.
            # (We need to know if the next frame is EOS before emitting the
            # current frame as "turn_end".)
            pending: Tensor = first_codes[0]  # (Q,) on GPU
            turn_frame_count = 1
            total_frames += 1
            codes_in = first_codes[0]
            turn_over = False

            for step in range(1, max_per_turn):
                if state.offset >= max_seq or total_frames >= max_frames:
                    turn_over = True
                    break
                hidden_next, cb0_logits, eos_logit = model.decode_step(
                    codes_in, state.offset
                )
                state.offset += 1
                if has_rq:
                    next_codes = rq_sample(cb0_logits.clone(), hidden_next.clone())
                else:
                    lg = model.audio_head(hidden_next)
                    next_codes = _sample_codes(lg, temperature, top_k, top_p).squeeze(
                        -1
                    )
                if (
                    step >= min_frames
                    and torch.sigmoid(eos_logit).item() > eos_threshold
                ):
                    turn_over = True
                    break
                # Previous pending frame is not the last → emit with turn_end=None
                yield pending, None
                pending = next_codes[0]
                turn_frame_count += 1
                total_frames += 1
                codes_in = next_codes[0]

            # Emit the final pending frame tagged as turn_end
            chunk["frames"] = turn_frame_count
            chunk["secs"] = turn_frame_count / CODEC_HZ
            yield pending, chunk
            if not turn_over:
                # max_per_turn reached without EOS
                continue


def _forward_flash_emb(state: "InferenceState", emb: Tensor) -> Tensor:
    """Prefill embedding into the flash KV cache, advance state.offset."""
    model = state.model
    n = emb.shape[1]
    positions = torch.arange(state.offset, state.offset + n, device=state.device)
    out = model.decoder.forward_flash(emb, positions)
    state.offset += n
    return out


def render_codes_stream(state: InferenceState, text: str, **kwargs):
    """Per-turn generator: yields `(turn_codes, chunk_info)`.

    turn_codes: (T, Q) long tensor on CPU.
    Wraps `stream_frames` — accumulates frames until each turn ends.
    """
    pending: list[Tensor] = []
    for frame, turn_end in stream_frames(state, text, **kwargs):
        pending.append(frame.cpu())
        if turn_end is not None:
            yield torch.stack(pending), turn_end
            pending = []


def render_codes(
    state: InferenceState, text: str, **kwargs
) -> tuple[Tensor | None, list[dict]]:
    """Batched variant: returns (all_codes, chunks_meta)."""
    pieces: list[Tensor] = []
    metas: list[dict] = []
    for turn_codes, chunk in render_codes_stream(state, text, **kwargs):
        pieces.append(turn_codes)
        metas.append(chunk)
    if not pieces:
        return None, metas
    return torch.cat(pieces, dim=0), metas


def render_audio_stream(
    state: InferenceState,
    text: str,
    *,
    chunk_frames: int | None = None,
    ctx_frames: int | None = None,
    first_chunk_frames: int | None = None,
    **kwargs,
):
    """Low-latency streaming audio.

    Consumes `stream_frames` and emits audio every `chunk_frames` frames
    using `state._decode_codec_chunk` (which has first/ctx codec CUDA graphs).

    chunk_frames: frames per codec emission (default state.chunk_frames, ~7).
    ctx_frames: codec context window (default state.ctx_frames, ~3).
    first_chunk_frames: emit the first chunk after only this many frames for
        lower TTFB (default == chunk_frames). Requires a codec graph for that
        size — if not available, falls back to non-graph codec decode.
    """
    chunk_size = chunk_frames if chunk_frames is not None else state.chunk_frames
    ctx_size = ctx_frames if ctx_frames is not None else state.ctx_frames
    first_size = first_chunk_frames if first_chunk_frames is not None else chunk_size
    device = state.device

    buf: list[Tensor] = []  # pending GPU frames for this turn
    emitted = 0  # count of frames already sent to codec this turn
    have_emitted_first = False  # whether this turn has emitted at least once

    def _emit(target_size: int, force_flush: bool) -> Tensor | None:
        nonlocal emitted, have_emitted_first
        n_ready = len(buf) - emitted
        if n_ready == 0:
            return None
        if not force_flush and n_ready < target_size:
            return None
        take = n_ready if force_flush else target_size
        has_ctx = emitted > 0
        ctx_used = min(ctx_size, emitted) if has_ctx else 0
        slab = torch.stack(buf[emitted - ctx_used : emitted + take])  # (T, Q)
        codes_bqt = slab.T.unsqueeze(0).to(device)
        audio = state._decode_codec_chunk(codes_bqt, has_ctx=has_ctx)
        emitted += take
        have_emitted_first = True
        return audio

    for frame, turn_end in stream_frames(state, text, **kwargs):
        buf.append(frame)
        if turn_end is None:
            # Intra-turn: try to emit a chunk. First chunk uses first_size.
            target = first_size if not have_emitted_first else chunk_size
            audio = _emit(target, force_flush=False)
            if audio is not None:
                yield audio
        else:
            # Last frame of a turn: flush remainder for this turn.
            tail = _emit(chunk_size, force_flush=True)
            if tail is not None:
                yield tail
            # Carry the tail of this turn's frames into the next turn so
            # its first emission decodes with codec context. Without this
            # the codec cold-starts at every text-chunk boundary (zero
            # conv state, no preceding samples) → audible stutter.
            keep = min(ctx_size, len(buf))
            buf = buf[-keep:] if keep > 0 else []
            emitted = keep


def render_audio(state: InferenceState, text: str, **kwargs) -> Tensor:
    """Batched audio: returns full (1, 1, samples) waveform."""
    pieces = list(render_audio_stream(state, text, **kwargs))
    if not pieces:
        return torch.zeros(1, 1, 0, device=state.device)
    return torch.cat(pieces, dim=-1)
