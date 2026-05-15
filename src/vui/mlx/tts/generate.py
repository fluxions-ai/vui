"""MLX generation / inference engine for Vui TTS."""

import math
import time

import mlx.core as mx
import numpy as np

from vui.inference import simple_clean
from vui.mlx.tts.model import VuiMLX

CODEC_HZ = 12.5


def _sample_top_k(logits: mx.array, top_k: int) -> mx.array:
    top_vals = mx.topk(logits, top_k, axis=-1)
    return mx.random.categorical(
        mx.where(logits < top_vals[:, -1:], mx.array(-1e9), logits)
    )


class RepPenalty:
    """Tracks code frequencies as MLX tensors. No Python-level sync."""

    def __init__(
        self, n_quantizers: int, codebook_size: int, penalty: float, window: int
    ):
        self.Q = n_quantizers
        self.CS = codebook_size
        self.penalty = penalty
        self.window = window
        self.log_p = math.log(max(penalty, 1.001))
        # Ring buffer of past codes: (window, Q). window=0 means unbounded.
        buf_size = window if window > 0 else 512
        self.history = mx.zeros((buf_size, n_quantizers), dtype=mx.int32)
        self.pos = 0
        self.count = 0

    def update(self, codes: mx.array):
        # codes: (Q,) — already eval'd
        idx = self.pos % self.history.shape[0]
        self.history[idx] = codes[: self.Q]
        self.pos += 1
        self.count = min(self.count + 1, self.history.shape[0])

    def apply_cb0(self, logits: mx.array) -> mx.array:
        if self.penalty <= 1.0 or self.count == 0:
            return logits
        # Count frequencies for codebook 0 from history
        active = self.history[: self.count, 0]  # (count,)
        freq = mx.zeros((self.CS,))
        freq = freq.at[active].add(mx.ones((active.shape[0],)))
        # Multiplicative penalty matching PyTorch: divide positive logits,
        # multiply negative logits by penalty^count
        factor = mx.power(self.penalty, freq)
        return mx.where(logits > 0, logits / factor, logits * factor)

    def rq_logit_bias(self) -> mx.array | None:
        if self.penalty <= 1.0 or self.count == 0:
            return None
        active = self.history[: self.count]  # (count, Q)
        bias = mx.zeros((self.Q - 1, self.CS))
        for q in range(1, self.Q):
            codes_q = active[:, q]
            freq = mx.zeros((self.CS,))
            freq = freq.at[codes_q].add(mx.ones((codes_q.shape[0],)))
            bias = bias.at[q - 1].add(-freq * self.log_p)
        return bias


def compute_cond_bias(
    model: VuiMLX, sq: list[float] | None = None, wps: float = 0.0
) -> mx.array:
    bias = mx.zeros((1, 1, model.d_model))
    if sq is not None and model.sq_proj is not None:
        bias = bias + model.sq_proj(mx.array([sq])).reshape(1, 1, -1)
    if wps > 0 and model.wps_proj is not None:
        bias = bias + model.wps_proj(mx.array([[wps]])).reshape(1, 1, -1)
    return bias


def prefill_prompt(
    model: VuiMLX,
    prompt_text: str,
    prompt_codes: mx.array,
    spk_emb: mx.array | None = None,
    speaker_change: bool = False,
):
    model.decoder.reset_cache()
    model.decoder.make_cache()

    # Speaker embedding token (if model supports it)
    if spk_emb is not None and model.spk_proj is not None:
        spk_token = model.spk_proj(spk_emb).reshape(1, 1, -1)
        model.decoder(spk_token)

    # Prompt text (+ [SC] only when next speaker differs)
    ids = model.text_tokenizer.encode(simple_clean(prompt_text))
    ids_mx = mx.array(np.array(ids, dtype=np.int32))
    if speaker_change:
        ids_mx = mx.concatenate([ids_mx, mx.array([model.sc_id])])
    text_emb = model.token_emb(ids_mx[None])
    model.decoder(text_emb)

    # Prompt audio codes
    pc = prompt_codes[0].T if prompt_codes.ndim == 3 else prompt_codes
    audio_emb = model.audio_emb(pc)[None]
    model.decoder(audio_emb)

    mx.eval([c.state for c in model.decoder.kv_caches])


def generate(
    model: VuiMLX,
    text: str,
    cond_bias: mx.array,
    temperature: float = 0.7,
    top_k: int = 50,
    rep_penalty: float = 1.4,
    rep_window: int = 24,
    max_frames: int = 187,
    eos_threshold: float = 0.45,
    min_frames: int = 6,
    compile_rq: bool = False,
    spk_emb: mx.array | None = None,
) -> tuple[list[mx.array], int, float]:
    """Generate audio codes from text. Returns (codes_list, n_frames, gen_time)."""
    Q = model.rq_transformer.n_quantizers
    CS = model.rq_transformer.codebook_size
    rq_temp = temperature
    rep = RepPenalty(Q, CS, rep_penalty, rep_window)

    if compile_rq:
        model.rq_transformer.compile_forward()

    if not model.decoder.kv_caches:
        model.decoder.make_cache()

    # Re-inject speaker token before this chunk's text (matches training format)
    if spk_emb is not None and model.spk_proj is not None:
        spk_token = model.spk_proj(spk_emb).reshape(1, 1, -1)
        model.decoder(spk_token)

    # Text prefill (with cond_bias)
    ids = model.text_tokenizer.encode(text)
    ids_mx = mx.array(np.array(ids, dtype=np.int32))
    text_emb = model.token_emb(ids_mx[None]) + cond_bias
    out = model.decoder(text_emb)

    # First frame from text hidden state
    hidden = out[:, -1:]
    code0 = _sample_top_k(model.codec_head(hidden[:, 0]) / temperature, top_k)
    first_codes = model.rq_transformer.generate(hidden[:, 0], code0, rq_temp, top_k)

    all_codes = [first_codes[0]]
    codes_in = first_codes[0]
    mx.eval(codes_in)
    rep.update(codes_in)

    t0 = time.perf_counter()

    for step in range(1, max_frames):
        emb = model.audio_emb(codes_in[None])[None]
        h = model.decoder(emb)
        hidden = h[:, 0]
        cb0_logits = model.codec_head(hidden)
        eos_logit = model.eos_head(hidden)

        penalised = rep.apply_cb0(cb0_logits)
        code0 = _sample_top_k(penalised / temperature, top_k)

        logit_bias = rep.rq_logit_bias()
        codes_frame = model.rq_transformer.generate(
            hidden, code0, rq_temp, top_k, logit_bias
        )

        mx.eval(codes_frame, eos_logit)

        if step >= min_frames and float(mx.sigmoid(eos_logit).item()) > eos_threshold:
            break

        rep.update(codes_frame[0])
        all_codes.append(codes_frame[0])
        codes_in = codes_frame[0]

        if step % 50 == 0:
            mx.clear_cache()

    elapsed = time.perf_counter() - t0
    return all_codes, len(all_codes), elapsed
