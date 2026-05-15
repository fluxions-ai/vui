"""Simple streaming TTS interface for MLX.

Usage:
    ctx = TTSStream(model, codec, cond_bias)
    ctx.add_speaker(spk_emb)
    ctx.add_text(prompt_text, sc=True)
    ctx.add_audio(prompt_codes)
    for codes, audio in ctx.generate(text):
        # codes: (Q,) per frame, audio: (1920,) float32
        play(audio)
    ctx.reset()
"""

import mlx.core as mx
import numpy as np

from vui.inference import simple_clean
from vui.mlx.tts.generate import RepPenalty
from vui.mlx.tts.model import VuiMLX


def _sample_top_k(logits: mx.array, top_k: int, temperature: float) -> mx.array:
    logits = logits / temperature
    top_vals = mx.topk(logits, top_k, axis=-1)
    logits = mx.where(logits < top_vals[:, -1:], mx.array(-1e9), logits)
    return mx.random.categorical(logits)


class TTSStream:
    def __init__(self, model: VuiMLX, codec=None, cond_bias: mx.array | None = None):
        self.model = model
        self.codec = codec
        self.cond_bias = (
            cond_bias if cond_bias is not None else mx.zeros((1, 1, model.d_model))
        )
        self._codec_ready = False
        self._spk_token: mx.array | None = None

    def reset(self):
        self.model.decoder.reset_cache()
        if self.codec is not None:
            self.codec.reset_state()
        self._codec_ready = False
        self._spk_token = None

    def add_speaker(self, spk_emb: mx.array):
        self._ensure_cache()
        self._spk_token = self.model.spk_proj(spk_emb).reshape(1, 1, -1)
        self.model.decoder(self._spk_token)

    def add_text(self, text: str, sc: bool = False):
        self._ensure_cache()
        ids = self.model.text_tokenizer.encode(simple_clean(text))
        ids_mx = mx.array(np.array(ids, dtype=np.int32))
        if sc:
            ids_mx = mx.concatenate([ids_mx, mx.array([self.model.sc_id])])
        emb = self.model.token_emb(ids_mx[None])
        self.model.decoder(emb)

    def add_audio(self, codes: mx.array):
        self._ensure_cache()
        # codes: (1, Q, T) or (T, Q)
        if codes.ndim == 3:
            pc = codes[0].T  # (T, Q)
        else:
            pc = codes
        emb = self.model.audio_emb(pc)[None]  # (1, T, d)
        self.model.decoder(emb)
        # Warm codec conv state with these codes
        if self.codec is not None:
            codec_codes = codes if codes.ndim == 3 else codes.T[None]
            self.codec.reset_state()
            self.codec.prefill(codec_codes)
            mx.eval(self.codec.parameters())
            self._codec_ready = True

    def generate(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 300,
        max_frames: int = 187,
        eos_threshold: float = 0.45,
        min_frames: int = 6,
        rep_penalty: float = 1.4,
        rep_window: int = 24,
    ):
        """Yield (codes, audio) per frame. audio is None if no codec."""
        self._ensure_cache()
        m = self.model
        Q = m.rq_transformer.n_quantizers
        CS = m.rq_transformer.codebook_size
        rq_temp = temperature
        rep = RepPenalty(Q, CS, rep_penalty, rep_window)

        # Re-inject speaker token before each new turn (matches training format)
        if self._spk_token is not None:
            m.decoder(self._spk_token)

        # Text prefill with cond_bias
        ids = m.text_tokenizer.encode(simple_clean(text))
        ids_mx = mx.array(ids.numpy().astype(np.int32))
        text_emb = m.token_emb(ids_mx[None]) + self.cond_bias
        out = m.decoder(text_emb)

        # First frame from last text hidden
        hidden = out[:, -1:]
        code0 = _sample_top_k(m.codec_head(hidden[:, 0]), top_k, temperature)
        codes = m.rq_transformer.generate(hidden[:, 0], code0, rq_temp, top_k)
        codes_in = codes[0]
        mx.eval(codes_in)
        rep.update(codes_in)

        audio = self._decode_frame(codes_in) if self.codec is not None else None
        yield codes_in, audio

        for step in range(1, max_frames):
            emb = m.audio_emb(codes_in[None])[None]
            h = m.decoder(emb)
            hidden = h[:, 0]

            eos_logit = m.eos_head(hidden)
            cb0_logits = rep.apply_cb0(m.codec_head(hidden))
            code0 = _sample_top_k(cb0_logits, top_k, temperature)
            logit_bias = rep.rq_logit_bias()
            codes = m.rq_transformer.generate(hidden, code0, rq_temp, top_k, logit_bias)

            mx.eval(codes, eos_logit)

            if (
                step >= min_frames
                and float(mx.sigmoid(eos_logit).item()) > eos_threshold
            ):
                break

            codes_in = codes[0]
            rep.update(codes_in)
            audio = self._decode_frame(codes_in) if self.codec is not None else None
            yield codes_in, audio

    def _decode_frame(self, codes: mx.array) -> mx.array:
        # codes: (Q,) -> (1, Q, 1) for codec
        frame = codes[None, :, None]
        if not self._codec_ready:
            self.codec.reset_state()
            self.codec.prefill(frame)
            mx.eval(self.codec.parameters())
            self._codec_ready = True
            return mx.zeros((1920,))
        audio = self.codec.decode_frame(frame)
        mx.eval(audio)
        return audio.flatten()

    def _ensure_cache(self):
        if not self.model.decoder.kv_caches:
            self.model.decoder.make_cache()
