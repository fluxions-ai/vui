"""Minimal Qwen3-TTS 12Hz codec encoder + decoder.

Clean PyTorch reimplementation. No HuggingFace runtime deps, fully torch.compile compatible.

Encoder: Audio (B,1,T) @ 24kHz -> Codes (B, 16, T/1920) @ 12.5 Hz
Decoder: Codes (B, 16, T) -> Audio (B, 1, T*1920) @ 24kHz

Usage:
    encoder = QwenCodecEncoder.from_pretrained()
    decoder = QwenCodecDecoder.from_pretrained()
    codes = encoder.encode(audio)       # (B, 1, T) -> (B, 16, T//1920)
    audio = decoder.decode(codes)       # (B, 16, T) -> (B, 1, T*1920)
"""

import glob
import os
from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vui.streaming import State, StreamingContainer, StreamingModule

SAMPLE_RATE = 24000
DOWNSAMPLE_RATE = 1920
FRAME_RATE = SAMPLE_RATE / DOWNSAMPLE_RATE  # 12.5
N_CODEBOOKS = 16

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


@dataclass
class _ConvState(State):
    previous: Tensor  # references the module's persistent _prev_buf

    def reset(self) -> None:
        self.previous.zero_()


class CausalConv1d(StreamingModule):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        effective_kernel = (kernel - 1) * dilation + 1
        self.pad = effective_kernel - stride
        # Persistent state buffer (lazy alloc on first streaming entry).
        # Keeping it on the module means the same tensor is reused across
        # streaming context entries — required so a CUDA graph captured over
        # the streaming forward keeps working after exit/re-enter.
        self._prev_buf: Tensor | None = None

    def _init_streaming_state(self, batch_size: int) -> _ConvState | None:
        if self.pad == 0:
            return None
        param = next(self.parameters())
        target_shape = (batch_size, self.conv.in_channels, self.pad)
        if (
            self._prev_buf is None
            or self._prev_buf.shape != target_shape
            or self._prev_buf.dtype != param.dtype
            or self._prev_buf.device != param.device
        ):
            self._prev_buf = torch.zeros(
                *target_shape, dtype=param.dtype, device=param.device
            )
        else:
            self._prev_buf.zero_()
        return _ConvState(batch_size, self._prev_buf.device, self._prev_buf)

    def forward(self, x: Tensor) -> Tensor:
        state = self._streaming_state
        if state is None:
            if self.pad > 0:
                x = F.pad(x, (self.pad, 0))
            return self.conv(x)
        if self.pad > 0:
            x = torch.cat([state.previous, x], dim=-1)
            state.previous.copy_(x[..., -self.pad :])
        return self.conv(x)


@dataclass
class _TrConvState(State):
    partial: Tensor  # references the module's persistent _partial_buf

    def reset(self) -> None:
        self.partial.zero_()


class CausalTransConv1d(StreamingModule):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, bias: bool = True
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, bias=bias)
        self.trim_right = kernel - stride
        # Persistent state buffer for graph stability (see CausalConv1d._prev_buf).
        self._partial_buf: Tensor | None = None

    def _init_streaming_state(self, batch_size: int) -> _TrConvState | None:
        if self.trim_right == 0:
            return None
        param = next(self.parameters())
        target_shape = (batch_size, self.conv.out_channels, self.trim_right)
        if (
            self._partial_buf is None
            or self._partial_buf.shape != target_shape
            or self._partial_buf.dtype != param.dtype
            or self._partial_buf.device != param.device
        ):
            self._partial_buf = torch.zeros(
                *target_shape, dtype=param.dtype, device=param.device
            )
        else:
            self._partial_buf.zero_()
        return _TrConvState(batch_size, self._partial_buf.device, self._partial_buf)

    def forward(self, x: Tensor) -> Tensor:
        state = self._streaming_state
        y = self.conv(x)
        PT = self.trim_right
        if state is None:
            if PT > 0:
                y = y[..., :-PT]
            return y
        if PT > 0:
            # Add stored partial to leading PT samples (overlap from prior call).
            y = y.clone()  # don't write into upstream-owned tensor
            y[..., :PT] = y[..., :PT] + state.partial
            # Save trailing PT as next-call partial (subtract bias to avoid double-count).
            for_partial = y[..., -PT:].clone()
            if self.conv.bias is not None:
                for_partial = for_partial - self.conv.bias[:, None]
            state.partial.copy_(for_partial)
            y = y[..., :-PT]
        return y


class RoPE(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class EuclideanCodebook(nn.Module):
    def __init__(self, codebook_size: int = 2048, dim: int = 256):
        super().__init__()
        self.register_buffer("embed", torch.zeros(codebook_size, dim))

    def encode(self, x: Tensor) -> Tensor:
        dists = torch.cdist(
            x.unsqueeze(0), self.embed.to(x.dtype).unsqueeze(0)
        ).squeeze(0)
        return dists.argmin(dim=-1)

    def decode(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.embed)


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_dim: int = 256,
        codebook_size: int = 2048,
        n_quantizers: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, codebook_dim, 1, bias=False)
        self.output_proj = nn.Conv1d(codebook_dim, input_dim, 1, bias=False)
        self.codebooks = nn.ModuleList(
            [
                EuclideanCodebook(codebook_size, codebook_dim)
                for _ in range(n_quantizers)
            ]
        )

    def encode(self, x: Tensor, n_q: int | None = None) -> Tensor:
        h = self.input_proj(x).float()
        B, C, T = h.shape
        residual = h.permute(0, 2, 1).reshape(B * T, C)
        n_q = n_q or len(self.codebooks)
        all_codes = []
        for cb in self.codebooks[:n_q]:
            indices = cb.encode(residual)
            quantized = cb.embed.float()[indices]
            residual = residual - quantized
            all_codes.append(indices.view(B, T))
        return torch.stack(all_codes, dim=0)

    def decode(self, codes: Tensor) -> Tensor:
        quantized = torch.zeros(
            1, device=codes.device, dtype=self.output_proj.weight.dtype
        )
        for i, cb in enumerate(self.codebooks[: codes.shape[0]]):
            q = cb.decode(codes[i])  # (B, T, codebook_dim)
            quantized = quantized + q.transpose(1, 2)  # (B, codebook_dim, T)
        return self.output_proj(quantized)


class SplitResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_dim: int = 256,
        codebook_size: int = 2048,
        n_semantic: int = 1,
        n_acoustic: int = 31,
    ):
        super().__init__()
        self.n_semantic = n_semantic
        self.semantic = ResidualVQ(input_dim, codebook_dim, codebook_size, n_semantic)
        self.acoustic = ResidualVQ(input_dim, codebook_dim, codebook_size, n_acoustic)

    def encode(self, x: Tensor, n_q: int = 16) -> Tensor:
        sem_codes = self.semantic.encode(x, self.n_semantic)
        n_acq = n_q - self.n_semantic
        if n_acq > 0:
            acq_codes = self.acoustic.encode(x, n_acq)
            return torch.cat([sem_codes, acq_codes], dim=0)
        return sem_codes

    def decode(self, codes: Tensor) -> Tensor:
        # codes: (B, n_q, T)
        codes_t = codes.transpose(0, 1)  # (n_q, B, T)
        sem = self.semantic.decode(codes_t[: self.n_semantic])
        if codes_t.shape[0] > self.n_semantic:
            acq = self.acoustic.decode(codes_t[self.n_semantic :])
            return sem + acq
        return sem


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim: int, compress: int = 2):
        super().__init__()
        hidden = dim // compress
        self.block = nn.Sequential(
            nn.ELU(),
            CausalConv1d(dim, hidden, kernel=3),
            nn.ELU(),
            CausalConv1d(hidden, dim, kernel=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ConvEncoder(nn.Module):
    def __init__(
        self, ratios: list[int] = [4, 5, 6, 8], dim: int = 64, hidden: int = 512
    ):
        super().__init__()
        layers: list[nn.Module] = [CausalConv1d(1, dim, kernel=7)]
        for ratio in ratios:
            layers.append(ResBlock(dim))
            layers.append(nn.ELU())
            layers.append(CausalConv1d(dim, dim * 2, kernel=ratio * 2, stride=ratio))
            dim *= 2
        layers.append(nn.ELU())
        layers.append(CausalConv1d(dim, hidden, kernel=3))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class EncTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, mlp_dim: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_scale = nn.Parameter(torch.zeros(d_model))

        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, d_model, bias=False)
        self.mlp_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, D = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_scale * self.o_proj(attn_out)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp_scale * self.fc2(F.gelu(self.fc1(h)))
        return x


class EncTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_dim: int = 2048,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncTransformerBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)]
        )
        self.rope = RoPE(d_model // n_heads)

    def forward(self, x: Tensor) -> Tensor:
        cos, sin = self.rope(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class QwenCodecEncoder(StreamingContainer):
    def __init__(self):
        super().__init__()
        self.conv_encoder = ConvEncoder(ratios=[4, 5, 6, 8], dim=64, hidden=512)
        self.transformer = EncTransformer(
            d_model=512, n_layers=8, n_heads=8, mlp_dim=2048
        )
        self.downsample = CausalConv1d(512, 512, kernel=4, stride=2, bias=False)
        self.quantizer = SplitResidualVQ(
            input_dim=512,
            codebook_dim=256,
            codebook_size=2048,
            n_semantic=1,
            n_acoustic=31,
        )
        self.n_q = N_CODEBOOKS

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_encoder(x)
        h = self.transformer(h.transpose(1, 2)).transpose(1, 2)
        h = self.downsample(h)
        codes = self.quantizer.encode(h, self.n_q)
        return codes.permute(1, 0, 2)

    def encode(self, audio: Tensor) -> Tensor:
        return self.forward(audio)

    @classmethod
    def from_pretrained(
        cls, repo_id: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    ) -> "QwenCodecEncoder":
        hf_state = _load_safetensors(repo_id, prefix="encoder.")
        model = cls()
        sd = model.state_dict()
        for our_key, hf_key in _encoder_key_map().items():
            if our_key in sd and hf_key in hf_state:
                sd[our_key] = hf_state[hf_key]
        model.load_state_dict(sd)
        _load_encoder_codebooks(model, hf_state)
        return model


class CodecCtx:
    """Managed codec decode with automatic 10s context window.

    Two modes:

    - **Full decode (demo, batch)**: `__call__(codes)` runs `decoder(ctx + codes)`
      over a rolling 10s window. Stateless on the decoder side.
    - **Streaming (server)**: `prefill()` enters the decoder's `streaming(B=1)`
      context (held in `self._stack`), seeds state by feeding any prior codes,
      and then each `decode_frame(codes_1)` calls `decoder(codes_1)` which
      returns audio for that frame. State (per-conv `previous`/`partial`,
      pre_transformer KV) lives in the StreamingModule tree. `reset()` exits
      the streaming context.

    Buffer management (`set_prompt`, `add`, `_buf`) is shared by both modes:
    the buffer is the source for full-decode rolling context AND the seed for
    streaming prefill.
    """

    def __init__(self, decoder: "QwenCodecDecoder", max_ctx_secs: float = 10.0):
        self.decoder = decoder
        self.max_ctx = int(max_ctx_secs * FRAME_RATE)  # 125 frames
        self._prompt_len = 0
        self._buf: Tensor | None = None  # (B, Q, T) rolling buffer
        self._stack: ExitStack | None = None  # holds decoder.streaming() ctx
        self._frames_since_prefill = 0  # bumped per decode_frame; resets on prefill
        self._prefill_n_codebooks = 0

    def set_prompt(self, codes: Tensor):
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        self._prompt_len = codes.shape[2]
        self._buf = codes
        # Buffer changed → drop any prior streaming state so the next
        # decode_frame seeds from this new prompt (or starts cold).
        self._close_stack()

    def add(self, codes: Tensor):
        """Add codes to buffer without decoding. For KV-cached streaming path."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        self._append(codes)

    @property
    def n_frames(self) -> int:
        return self._buf.shape[2] if self._buf is not None else 0

    def get_context(self, max_frames: int | None = None) -> Tensor | None:
        """Get context codes windowed to max_ctx, preserving prompt."""
        if self._buf is None:
            return None
        limit = max_frames or self.max_ctx
        T = self._buf.shape[2]
        if T <= limit:
            return self._buf
        if self._prompt_len > 0 and self._prompt_len < limit:
            avail = limit - self._prompt_len
            return torch.cat(
                [
                    self._buf[:, :, : self._prompt_len],
                    self._buf[:, :, -avail:],
                ],
                dim=2,
            )
        return self._buf[:, :, -limit:]

    def _append(self, codes: Tensor):
        if self._buf is None:
            self._buf = codes
        else:
            # Defensive: align Q (codebook dim) if a stale prompt context
            # had a different n_q than the codes we're appending. Truncate
            # both to the smaller count so cat doesn't crash.
            if codes.shape[1] != self._buf.shape[1]:
                min_q = min(codes.shape[1], self._buf.shape[1])
                if self._buf.shape[1] != min_q:
                    self._buf = self._buf[:, :min_q].contiguous()
                if codes.shape[1] != min_q:
                    codes = codes[:, :min_q].contiguous()
            self._buf = torch.cat([self._buf, codes], dim=2)
        # Trim to 2x max_ctx to bound memory
        limit = self.max_ctx * 2
        if self._buf.shape[2] > limit:
            if self._prompt_len > 0 and self._prompt_len < limit:
                self._buf = torch.cat(
                    [
                        self._buf[:, :, : self._prompt_len],
                        self._buf[:, :, -(limit - self._prompt_len) :],
                    ],
                    dim=2,
                )
            else:
                self._buf = self._buf[:, :, -limit:]

    # --- Full decode path (demo/batch) ---

    @torch.inference_mode()
    def __call__(self, codes: Tensor) -> Tensor:
        """Decode codes to audio. Any length. Returns audio for new codes only."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        T_new = codes.shape[2]
        ctx = self.get_context()
        ctx_t = ctx.shape[2] if ctx is not None else 0

        if ctx_t + T_new <= self.max_ctx:
            if ctx is not None:
                inp = torch.cat([ctx, codes], dim=2)
                audio = self.decoder(inp)
                self._append(codes)
                return audio[..., ctx_t * DOWNSAMPLE_RATE :]
            else:
                audio = self.decoder(codes)
                self._append(codes)
                return audio

        return self._decode_chunked(codes)

    def _decode_chunked(self, codes: Tensor) -> Tensor:
        ctx = self.get_context(self.max_ctx // 2)
        T = codes.shape[2]
        wavs = []
        pos = 0

        while pos < T:
            ctx_t = ctx.shape[2] if ctx is not None else 0
            chunk_size = self.max_ctx - ctx_t
            end = min(pos + chunk_size, T)
            chunk = codes[:, :, pos:end]

            if ctx is not None:
                inp = torch.cat([ctx, chunk], dim=2)
                audio = self.decoder(inp)
                wavs.append(audio[..., ctx_t * DOWNSAMPLE_RATE :])
            else:
                audio = self.decoder(chunk)
                wavs.append(audio)

            ctx_limit = min(self.max_ctx // 2, end)
            ctx = codes[:, :, end - ctx_limit : end]
            pos = end

        self._append(codes)
        return torch.cat(wavs, dim=-1)

    # --- Streaming path (server) ---

    def prefill(self, n_codebooks: int = 0, device: str | torch.device = "cuda"):
        """Enter decoder.streaming(1) and seed state from the buffered codes.

        Closes any prior streaming context, allocates fresh state on every
        StreamingModule, then runs the decoder over the partial-window tail
        of `_buf` so the codec's internal position aligns with the absolute
        audio frame index modulo `max_ctx`.

        The alignment matters because training (`scripts/qwen-encode.py`)
        encoded audio in independent 10s chunks (`CHUNK_SECS = 10`, no
        overlap). Codes at absolute frame positions `k * max_ctx` are
        "fresh-start" codes from a cold encoder run; the codec decoder
        learned to decode them against cold state. Resetting the codec
        anywhere else — or carrying continuation context past `k * max_ctx`
        — is out of distribution.

        Concretely: `_frames_since_prefill` is set to `buf_size % max_ctx`
        after this call, so the boundary check `_fsp >= max_ctx` next fires
        at absolute frame `(buf_size // max_ctx + 1) * max_ctx` — the next
        encoder boundary, where the AR will emit fresh-start codes.

        Wrapped in inference_mode so persistent state buffers (which are
        inference tensors after setup_streaming_graph) can be mutated.
        """
        self._prefill_n_codebooks = n_codebooks
        self._close_stack()
        self._stack = ExitStack()
        self._stack.enter_context(torch.inference_mode())
        self._stack.enter_context(self.decoder.streaming(1))
        self._reseed(device)

    def _reseed(self, device: str | torch.device = "cuda") -> None:
        """Seed state with the partial-window tail of `_buf`.

        Feeds `buf_size % max_ctx` codes through the decoder so the codec
        position lands at `buf_size % max_ctx` — aligned with the absolute
        audio frame index modulo `max_ctx`. See `prefill` docstring for why.
        """
        if self._buf is None or self._buf.shape[2] == 0:
            self._frames_since_prefill = 0
            return
        n = self._buf.shape[2] % self.max_ctx
        if n == 0:
            self._frames_since_prefill = 0
            return
        seed = self._buf[:, :, -n:].to(device)
        if self._prefill_n_codebooks > 0:
            seed = seed[:, : self._prefill_n_codebooks]
        self.decoder(seed)
        self._frames_since_prefill = n

    def _hard_reset(self, device: str | torch.device = "cuda") -> None:
        """Clear streaming state at the 10s boundary — no seeding.

        Mirrors `StreamingCodecEncoder._slide`: training encodes audio in
        independent 10s chunks, so the decoder is also asked to start cold at
        the boundary. The next decode_frame goes through with empty conv
        state and an empty pre_transformer KV — same as the first frame of a
        training clip. Buffers (`_buf`, `_prompt_len`) are unchanged so the
        rolling-buffer accounting (e.g. `n_frames`) keeps working.
        """
        self._close_stack()
        self._stack = ExitStack()
        self._stack.enter_context(torch.inference_mode())
        self._stack.enter_context(self.decoder.streaming(1))
        self._frames_since_prefill = 0

    def decode_frame(
        self,
        codes: Tensor,
        vocoder_ctx: int | None = None,  # accepted for API compat; unused
        store_codes: Tensor | None = None,
    ) -> Tensor:
        """Decode a single frame inside the active streaming context.

        Re-prefills every `max_ctx` frames so the pre_transformer KV stays
        within the 10s training context window.

        store_codes: if provided, store these in the rolling buffer instead of
            `codes` (useful when decoding with fewer codebooks but storing all Q).
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        dev = codes.device if codes.is_cuda else "cuda"
        if self._stack is None:
            # No prior prefill — auto-enter streaming context, start cold.
            self._hard_reset(device=dev)
            audio = self.decoder(codes)
        elif self._frames_since_prefill >= self.max_ctx:
            # 10s boundary: training encoded in independent 10s chunks, so
            # decoder state at boundaries is OOD. Hard-reset and emit the
            # cold-state output — matches the training distribution, which
            # makes streaming bit-exact with the 10s-batch reference.
            self._hard_reset(device=dev)
            audio = self.decoder(codes)
        else:
            audio = self.decoder(codes)
        self._append(store_codes if store_codes is not None else codes)
        self._frames_since_prefill += 1
        return audio

    def _close_stack(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def reset(self):
        self._prompt_len = 0
        self._buf = None
        self._frames_since_prefill = 0
        self._close_stack()


class StreamingCodecEncoder:
    """Streaming codec encoder with CUDA graph. Bit-exact match with full encode.

    Uses a fixed-size buffer and re-encodes from scratch each call via CUDA graph.
    Since all convolutions are causal and attention uses is_causal=True,
    right-padded zeros don't affect output at earlier positions.

    Usage:
        stream_enc = StreamingCodecEncoder(encoder)
        stream_enc.setup_graph()  # once, after model is on GPU
        # In audio callback:
        codes = stream_enc.feed(audio_chunk)  # (T,) float 24kHz
        if codes is not None:
            # codes: (n_new_frames, n_quantizers) long
            all_codes.append(codes)
        # When done:
        codes = stream_enc.flush()
        stream_enc.reset()
    """

    def __init__(
        self,
        encoder: QwenCodecEncoder,
        n_quantizers: int = N_CODEBOOKS,
        max_secs: float = 10,
        min_chunk_frames: int = 12,
    ):
        self.encoder = encoder
        self.encoder.n_q = n_quantizers
        self.device = next(encoder.parameters()).device
        self.n_q = n_quantizers
        self.max_samples = int(max_secs * SAMPLE_RATE)
        self.max_samples = (self.max_samples // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        self.max_frames = self.max_samples // DOWNSAMPLE_RATE
        self.min_chunk_samples = min_chunk_frames * DOWNSAMPLE_RATE
        self.dtype = next(encoder.parameters()).dtype
        self._audio_buf = torch.zeros(
            1, 1, self.max_samples, device=self.device, dtype=self.dtype
        )
        self._pos = 0
        self._emitted = 0
        self._graph = None

    def _slide(self):
        """Reset buffer at 10s boundary to match training encoding.

        Training encodes audio in independent 10s chunks with no overlap.
        Resetting here reproduces the same boundary discontinuities so that
        codes match the training distribution exactly.
        """
        self._audio_buf.zero_()
        self._pos = 0
        self._emitted = 0

    def setup_graph(self):
        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                self.encoder(self._audio_buf)
            torch.cuda.synchronize()

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._codes_out = self.encoder(self._audio_buf)
        # _codes_out: (1, n_q, max_frames)

    @torch.inference_mode()
    def feed(self, audio: Tensor) -> Tensor | None:
        """Feed audio (float, 24kHz mono). Returns (n_new_frames, n_q) or None."""
        audio = audio.detach().flatten()
        n = len(audio)
        if n == 0:
            return None

        space = self.max_samples - self._pos
        if n > space:
            self._slide()
            space = self.max_samples - self._pos
            n = min(n, space)
        if n == 0:
            return None

        self._audio_buf[0, 0, self._pos : self._pos + n].copy_(
            audio.to(device=self.device, dtype=self.dtype)
        )
        self._pos += n

        n_frames = self._pos // DOWNSAMPLE_RATE
        if n_frames <= self._emitted:
            return None
        if (self._pos - self._emitted * DOWNSAMPLE_RATE) < self.min_chunk_samples:
            return None

        if self._graph is not None:
            self._graph.replay()
            codes = self._codes_out
        else:
            codes = self.encoder(self._audio_buf)

        new_codes = codes[0, :, self._emitted : n_frames].T.long().cpu()
        self._emitted = n_frames
        return new_codes

    @torch.inference_mode()
    def flush(self) -> Tensor | None:
        """Encode remaining audio (zero-pads partial frame)."""
        remainder = self._pos % DOWNSAMPLE_RATE
        if remainder > 0:
            self._pos += DOWNSAMPLE_RATE - remainder
            # Buffer already zero-initialized beyond _pos

        n_frames = self._pos // DOWNSAMPLE_RATE
        if n_frames <= self._emitted:
            return None

        if self._graph is not None:
            self._graph.replay()
            codes = self._codes_out
        else:
            codes = self.encoder(self._audio_buf)

        new_codes = codes[0, :, self._emitted : n_frames].T.long().cpu()
        self._emitted = n_frames
        return new_codes

    def reset(self):
        self._audio_buf.zero_()
        self._pos = 0
        self._emitted = 0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


class SnakeBeta(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        a = self.alpha.unsqueeze(0).unsqueeze(-1).exp()
        b = self.beta.unsqueeze(0).unsqueeze(-1).exp()
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a).pow(2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = F.gelu(self.pwconv1(x))
        x = self.gamma * self.pwconv2(x)
        return residual + x.transpose(1, 2)


class DecoderResUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.block = nn.ModuleList(
            [
                SnakeBeta(in_dim),
                CausalTransConv1d(in_dim, out_dim, kernel=stride * 2, stride=stride),
                DecoderResUnit(out_dim, dilation=1),
                DecoderResUnit(out_dim, dilation=3),
                DecoderResUnit(out_dim, dilation=9),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class DecTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 1024,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        attn_dim = n_heads * head_dim  # 16 * 64 = 1024

        self.input_layernorm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.v_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, d_model, bias=False)
        self.attn_scale = nn.Parameter(torch.zeros(d_model))

        self.post_attention_layernorm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, mlp_dim, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, d_model, bias=False)
        self.mlp_scale = nn.Parameter(torch.zeros(d_model))

        # KV cache. Two modes:
        # - **Streaming** (graph-friendly): fixed-size buffers `(B, n_heads, max_seq, head_dim)`,
        #   allocated by `alloc_streaming_cache(B, max_seq)`. New k/v written via
        #   `index_copy_` at a tensor `position_idx`. Reused across streaming-context
        #   exits/re-entries so a captured CUDA graph still references valid memory.
        # - **Parallel/legacy incremental**: cat-based, allocated on the fly.
        self._k_cache: Tensor | None = None
        self._v_cache: Tensor | None = None
        self._streaming_cache: bool = (
            False  # True when caches are fixed-size streaming buffers
        )

    def alloc_streaming_cache(self, batch_size: int, max_seq: int) -> None:
        """(Re-)allocate fixed-size KV buffers for streaming. Idempotent.

        Does NOT zero the cache — positions past `position_idx` are masked
        out by the attention mask, so stale values never affect the SDPA
        output. Skipping the zero saves bandwidth on every streaming entry.
        """
        param = next(self.parameters())
        target = (batch_size, self.n_heads, max_seq, self.head_dim)
        if (
            self._k_cache is None
            or not self._streaming_cache
            or self._k_cache.shape != target
            or self._k_cache.dtype != param.dtype
            or self._k_cache.device != param.device
        ):
            self._k_cache = torch.zeros(*target, dtype=param.dtype, device=param.device)
            self._v_cache = torch.zeros(*target, dtype=param.dtype, device=param.device)
            self._streaming_cache = True

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_idx: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        B, T, D = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        if position_idx is not None and self._streaming_cache:
            # Streaming path with fixed-size cache. Write new k/v at position_idx,
            # then SDPA over the full cache with a position-bounded mask.
            self._k_cache.index_copy_(2, position_idx, k)
            self._v_cache.index_copy_(2, position_idx, v)
            attn_out = F.scaled_dot_product_attention(
                q, self._k_cache, self._v_cache, attn_mask=attn_mask
            )
        else:
            # Parallel/offline: one-shot causal SDPA over q. Does NOT touch
            # the persistent _k_cache (which the streaming graph references) —
            # zero-cost path in/out of streaming.
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        x = x + self.attn_scale * self.o_proj(attn_out)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp_scale * self.down_proj(
            F.silu(self.gate_proj(h)) * self.up_proj(h)
        )
        return x


class DecTransformer(StreamingModule):
    def __init__(
        self,
        latent_dim: int = 1024,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 1024,
        max_seq: int = 250,  # 2 × 10s training context — fits seed + next cycle before re-prefill
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.layers = nn.ModuleList(
            [
                DecTransformerBlock(d_model, n_heads, head_dim, mlp_dim)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        self.rope = RoPE(head_dim)
        self.max_seq = max_seq
        self._cached_len = 0
        # Streaming-only persistent buffers (lazy alloc on first streaming entry).
        # Held on the module so they survive streaming-context exits/re-entries —
        # required for the captured CUDA graph to keep referring to live memory.
        self._position_idx: Tensor | None = (
            None  # (1,) long: write index along cache dim
        )
        self._cos_table: Tensor | None = None  # (max_seq, head_dim)
        self._sin_table: Tensor | None = None  # (max_seq, head_dim)
        self._arange_max_seq: Tensor | None = None  # (max_seq,) long: for mask compare

    def _init_streaming_state(self, batch_size: int) -> State:
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        if (
            self._position_idx is None
            or self._cos_table is None
            or self._cos_table.dtype != dtype
            or self._cos_table.device != device
        ):
            self._position_idx = torch.zeros(1, dtype=torch.long, device=device)
            cos, sin = self.rope(self.max_seq, device, dtype)
            self._cos_table = cos.contiguous()
            self._sin_table = sin.contiguous()
            self._arange_max_seq = torch.arange(self.max_seq, device=device)
        else:
            self._position_idx.zero_()
        # Allocate fixed-size KV per layer (idempotent — reuses if shape/dtype match)
        for layer in self.layers:
            layer.alloc_streaming_cache(batch_size, self.max_seq)
        self._cached_len = 0
        return State(batch_size, device)

    def forward(self, x: Tensor) -> Tensor:
        if self._streaming_state is not None:
            return self._forward_streaming(x)
        x = self.input_proj(x)
        cos, sin = self.rope(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.output_proj(self.norm(x))

    def reset_cache(self) -> None:
        """Back-compat: reset streaming KV state. No-op if not in streaming mode."""
        if self._position_idx is not None:
            self._position_idx.zero_()

    def _forward_streaming(self, x: Tensor) -> Tensor:
        """Streaming forward at fixed buffer size — graph-friendly.

        Reads/writes via tensor indices so all kernel launches have static
        shapes. Position counter and mask are computed via tensor ops
        (`index_select`, `arange < pos`) which the graph captures.

        Caller's pre-condition: the rolling code buffer has been seeded into
        the cache by entering streaming and feeding prior context once.
        """
        T = x.shape[1]
        x = self.input_proj(x)
        # Indices [pos, pos+T) — positions to write the new K/V to and
        # to look up RoPE (cos, sin) for.
        idx = self._position_idx + self._arange_max_seq[:T]
        cos = self._cos_table.index_select(0, idx)  # (T, head_dim)
        sin = self._sin_table.index_select(0, idx)
        # Causal mask with prefix: query i (at absolute position idx[i]) attends
        # to all cache slots j with arange[j] <= idx[i]. With a (1,1,1,max_seq)
        # mask the seed pass (T>1) was non-causal across the new queries, which
        # poisons the K/V cache vs the offline `is_causal=True` forward.
        mask = (self._arange_max_seq.unsqueeze(0) <= idx.unsqueeze(-1)).view(
            1, 1, T, self.max_seq
        )
        for layer in self.layers:
            x = layer(x, cos, sin, position_idx=idx, attn_mask=mask)
        # Advance position. Tensor add is graph-captured — replays advance by T each call.
        self._position_idx.add_(T)
        return self.output_proj(self.norm(x))


class QwenCodecDecoder(StreamingContainer):
    def __init__(self):
        super().__init__()
        latent_dim = 1024
        decoder_dim = 1536
        upsample_rates = [8, 5, 4, 3]
        upsampling_ratios = [2, 2]

        self.quantizer = SplitResidualVQ(
            input_dim=512,
            codebook_dim=256,
            codebook_size=2048,
            n_semantic=1,
            n_acoustic=15,
        )
        self.pre_conv = CausalConv1d(512, latent_dim, kernel=3)
        self.pre_transformer = DecTransformer(
            latent_dim=latent_dim,
            d_model=512,
            n_layers=8,
            n_heads=16,
            mlp_dim=1024,
        )

        # Upsample: 2x, 2x = 4x total
        self.upsample = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CausalTransConv1d(latent_dim, latent_dim, kernel=r, stride=r),
                        ConvNeXtBlock(latent_dim),
                    ]
                )
                for r in upsampling_ratios
            ]
        )

        # BigVGAN-style decoder: 8x5x4x3 = 480x
        dim = decoder_dim
        dec_layers: list[nn.Module] = [CausalConv1d(latent_dim, dim, kernel=7)]
        for rate in upsample_rates:
            out_dim = dim // 2
            dec_layers.append(DecoderBlock(dim, out_dim, rate))
            dim = out_dim
        dec_layers.append(SnakeBeta(dim))
        dec_layers.append(CausalConv1d(dim, 1, kernel=7))
        self.decoder = nn.ModuleList(dec_layers)

        # Total upsample: 4 * 480 = 1920

        # CUDA graph over the entire single-frame streaming forward
        # (quantizer → pre_conv → pre_transformer → upsample → BigVGAN → clamp).
        # Captured by `setup_streaming_graph` at fixed input shape (B, n_q, 1).
        # All state buffers are persistent on the module tree so the graph
        # survives streaming-context exits/re-entries.
        self._stream_graph: torch.cuda.CUDAGraph | None = None
        self._stream_codes_in: Tensor | None = None
        self._stream_out: Tensor | None = None

    def setup_streaming_graph(
        self, batch_size: int = 1, device=None, dtype=None, pool=None
    ) -> None:
        """Capture a CUDA graph over the entire single-frame streaming forward.

        Captures `forward(codes_1frame)` end-to-end:
            quantizer → pre_conv → pre_transformer (fixed-size KV) → upsample
            → BigVGAN decoder → clamp

        All state lives in persistent module buffers (conv `_prev_buf`/
        `_partial_buf`, transformer `_k_cache`/`_v_cache`/`_position_idx`,
        plus `_cos_table`/`_sin_table`/`_arange_max_seq`), so the graph
        replay correctly mutates them in place across calls — and survives
        streaming-context exits/re-entries done by `CodecCtx.prefill`.

        After capture, `forward(codes)` with chunk size 1 in streaming
        context replays the graph instead of running eagerly.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        with torch.inference_mode():
            opened_here = False
            stack = ExitStack()
            if self._streaming_state is None:
                stack.enter_context(self.streaming(batch_size))
                opened_here = True
            try:
                self._stream_codes_in = torch.zeros(
                    batch_size, N_CODEBOOKS, 1, device=device, dtype=torch.long
                )
                # Warm up so cuDNN picks an algorithm before capture.
                for _ in range(3):
                    self._streaming_forward_body(self._stream_codes_in)
                # Reset state so capture starts clean (and so position_idx
                # snapshot at capture time matches a fresh streaming entry).
                self.reset_streaming()
                self.pre_transformer._position_idx.zero_()
                torch.cuda.synchronize()
                self._stream_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._stream_graph, pool=pool):
                    self._stream_out = self._streaming_forward_body(
                        self._stream_codes_in
                    )
                # Capture advanced state by 1; reset so live decode starts clean.
                self.reset_streaming()
                self.pre_transformer._position_idx.zero_()
            finally:
                if opened_here:
                    stack.close()

    def _streaming_forward_body(self, codes: Tensor) -> Tensor:
        """Body of streaming forward — quantizer through clamp. No graph fast-path."""
        h = self.quantizer.decode(codes)
        h = self.pre_conv(h).transpose(1, 2)
        h = self.pre_transformer(h)
        h = h.transpose(1, 2)
        return self._vocoder_chain(h)

    def forward(self, codes: Tensor) -> Tensor:
        # codes: (B, 16, T) long -> audio (B, 1, T*1920)
        # Fast path: streaming + chunk size 1 + pre-captured graph
        if (
            self._streaming_state is not None
            and self._stream_graph is not None
            and self._stream_codes_in is not None
            and codes.shape == self._stream_codes_in.shape
        ):
            self._stream_codes_in.copy_(codes)
            self._stream_graph.replay()
            return self._stream_out
        h = self.quantizer.decode(codes)  # (B, 512, T)
        h = self.pre_conv(h).transpose(1, 2)  # (B, T, 1024)
        h = self.pre_transformer(h)  # (B, T, 1024)
        h = h.transpose(1, 2)  # (B, 1024, T)
        return self._vocoder_chain(h)

    def _vocoder_chain(self, h: Tensor) -> Tensor:
        """Stateful upsample + BigVGAN decoder + clamp on (B, 1024, T) latent."""
        for blocks in self.upsample:
            for block in blocks:
                h = block(h)
        for layer in self.decoder:
            h = layer(h)
        return h.clamp(-1, 1)

    def decode(self, codes: Tensor) -> Tensor:
        return self.forward(codes)

    def decode_chunked(
        self, codes: Tensor, chunk_size: int = 300, ctx: int = 25
    ) -> Tensor:
        # Batch-decode path. If a row is mid-stream the codec's StreamingModules
        # carry live `_streaming_state`; routing chunk_size>1 through that path
        # consumes the active row's pre_transformer KV position and (once
        # position_idx + chunk_size exceeds max_seq=250) overflows the RoPE
        # table → out-of-bounds gather assert. Save & null streaming state for
        # the duration; persistent buffers (_prev_buf/_partial_buf/_k_cache/
        # _position_idx) are not mutated by the eager forwards.
        saved: list[tuple[StreamingModule, State]] = []

        def _save(m: StreamingModule) -> None:
            if m._streaming_state is not None:
                saved.append((m, m._streaming_state))
                m._streaming_state = None

        self._walk_streaming(_save)
        try:
            T = codes.shape[2]
            wavs = []
            start = 0
            while start < T:
                end = min(start + chunk_size, T)
                ctx_size = min(ctx, start)
                chunk = codes[:, :, start - ctx_size : end]
                wav = self.forward(chunk)
                wavs.append(wav[..., ctx_size * DOWNSAMPLE_RATE :])
                start = end
            return torch.cat(wavs, dim=-1)
        finally:
            for m, s in saved:
                m._streaming_state = s

    def setup_decode_graph(self, n_frames: int, device, dtype=torch.float16, pool=None):
        """Capture CUDA graph for fixed-size codec decode."""
        self._graph_codes_in = torch.zeros(
            1, N_CODEBOOKS, n_frames, device=device, dtype=torch.long
        )

        for _ in range(3):
            self.forward(self._graph_codes_in)
        torch.cuda.synchronize()

        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph, pool=pool):
            self._graph_audio_out = self.forward(self._graph_codes_in)

    @classmethod
    def from_pretrained(
        cls, repo_id: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    ) -> "QwenCodecDecoder":
        hf_state = _load_safetensors(repo_id, prefix="decoder.")
        model = cls()
        sd = model.state_dict()
        for our_key, hf_key in _decoder_key_map().items():
            if our_key in sd and hf_key in hf_state:
                if sd[our_key].shape != hf_state[hf_key].shape:
                    raise ValueError(
                        f"Shape mismatch: {our_key} {sd[our_key].shape} vs {hf_key} {hf_state[hf_key].shape}"
                    )
                sd[our_key] = hf_state[hf_key]
        model.load_state_dict(sd)
        _load_decoder_codebooks(model, hf_state)
        return model


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _load_safetensors(repo_id: str, prefix: str) -> dict[str, Tensor]:
    import filelock
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    lock_path = os.path.join(cache_dir, "qwen-codec-download.lock")
    with filelock.FileLock(lock_path):
        model_dir = snapshot_download(repo_id)
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    state = {}
    for f in st_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                if k.startswith(prefix):
                    state[k[len(prefix) :]] = st.get_tensor(k)
    return state


def _load_ema_codebook(cb: EuclideanCodebook, usage: Tensor, embed_sum: Tensor):
    cb.embed.copy_(embed_sum / usage.clamp(min=1e-5).unsqueeze(1))


def _encoder_key_map() -> dict[str, str]:
    m = {}
    conv_layers = [0, 3, 6, 9, 12, 14]
    for i in conv_layers:
        m[f"conv_encoder.layers.{i}.conv.weight"] = f"encoder.layers.{i}.conv.weight"
        m[f"conv_encoder.layers.{i}.conv.bias"] = f"encoder.layers.{i}.conv.bias"

    for i in [1, 4, 7, 10]:
        m[f"conv_encoder.layers.{i}.block.1.conv.weight"] = (
            f"encoder.layers.{i}.block.1.conv.weight"
        )
        m[f"conv_encoder.layers.{i}.block.1.conv.bias"] = (
            f"encoder.layers.{i}.block.1.conv.bias"
        )
        m[f"conv_encoder.layers.{i}.block.3.conv.weight"] = (
            f"encoder.layers.{i}.block.3.conv.weight"
        )
        m[f"conv_encoder.layers.{i}.block.3.conv.bias"] = (
            f"encoder.layers.{i}.block.3.conv.bias"
        )

    for i in range(8):
        p = f"transformer.layers.{i}"
        h = f"encoder_transformer.layers.{i}"
        m[f"{p}.input_layernorm.weight"] = f"{h}.input_layernorm.weight"
        m[f"{p}.input_layernorm.bias"] = f"{h}.input_layernorm.bias"
        m[f"{p}.q_proj.weight"] = f"{h}.self_attn.q_proj.weight"
        m[f"{p}.k_proj.weight"] = f"{h}.self_attn.k_proj.weight"
        m[f"{p}.v_proj.weight"] = f"{h}.self_attn.v_proj.weight"
        m[f"{p}.o_proj.weight"] = f"{h}.self_attn.o_proj.weight"
        m[f"{p}.attn_scale"] = f"{h}.self_attn_layer_scale.scale"
        m[f"{p}.post_attention_layernorm.weight"] = (
            f"{h}.post_attention_layernorm.weight"
        )
        m[f"{p}.post_attention_layernorm.bias"] = f"{h}.post_attention_layernorm.bias"
        m[f"{p}.fc1.weight"] = f"{h}.mlp.fc1.weight"
        m[f"{p}.fc2.weight"] = f"{h}.mlp.fc2.weight"
        m[f"{p}.mlp_scale"] = f"{h}.mlp_layer_scale.scale"

    m["downsample.conv.weight"] = "downsample.conv.weight"

    sem = "quantizer.semantic_residual_vector_quantizer"
    m["quantizer.semantic.input_proj.weight"] = f"{sem}.input_proj.weight"
    m["quantizer.semantic.output_proj.weight"] = f"{sem}.output_proj.weight"
    acq = "quantizer.acoustic_residual_vector_quantizer"
    m["quantizer.acoustic.input_proj.weight"] = f"{acq}.input_proj.weight"
    m["quantizer.acoustic.output_proj.weight"] = f"{acq}.output_proj.weight"
    return m


def _load_encoder_codebooks(model: QwenCodecEncoder, hf_state: dict[str, Tensor]):
    sem = "quantizer.semantic_residual_vector_quantizer"
    _load_ema_codebook(
        model.quantizer.semantic.codebooks[0],
        hf_state[f"{sem}.layers.0.codebook.cluster_usage"],
        hf_state[f"{sem}.layers.0.codebook.embed_sum"],
    )
    acq = "quantizer.acoustic_residual_vector_quantizer"
    for i in range(31):
        _load_ema_codebook(
            model.quantizer.acoustic.codebooks[i],
            hf_state[f"{acq}.layers.{i}.codebook.cluster_usage"],
            hf_state[f"{acq}.layers.{i}.codebook.embed_sum"],
        )


def _decoder_key_map() -> dict[str, str]:
    m = {}

    # pre_conv
    m["pre_conv.conv.weight"] = "pre_conv.conv.weight"
    m["pre_conv.conv.bias"] = "pre_conv.conv.bias"

    # pre_transformer
    m["pre_transformer.input_proj.weight"] = "pre_transformer.input_proj.weight"
    m["pre_transformer.input_proj.bias"] = "pre_transformer.input_proj.bias"
    m["pre_transformer.output_proj.weight"] = "pre_transformer.output_proj.weight"
    m["pre_transformer.output_proj.bias"] = "pre_transformer.output_proj.bias"
    m["pre_transformer.norm.weight"] = "pre_transformer.norm.weight"

    for i in range(8):
        p = f"pre_transformer.layers.{i}"
        h = f"pre_transformer.layers.{i}"
        m[f"{p}.input_layernorm.weight"] = f"{h}.input_layernorm.weight"
        m[f"{p}.q_proj.weight"] = f"{h}.self_attn.q_proj.weight"
        m[f"{p}.k_proj.weight"] = f"{h}.self_attn.k_proj.weight"
        m[f"{p}.v_proj.weight"] = f"{h}.self_attn.v_proj.weight"
        m[f"{p}.o_proj.weight"] = f"{h}.self_attn.o_proj.weight"
        m[f"{p}.attn_scale"] = f"{h}.self_attn_layer_scale.scale"
        m[f"{p}.post_attention_layernorm.weight"] = (
            f"{h}.post_attention_layernorm.weight"
        )
        m[f"{p}.gate_proj.weight"] = f"{h}.mlp.gate_proj.weight"
        m[f"{p}.up_proj.weight"] = f"{h}.mlp.up_proj.weight"
        m[f"{p}.down_proj.weight"] = f"{h}.mlp.down_proj.weight"
        m[f"{p}.mlp_scale"] = f"{h}.mlp_layer_scale.scale"

    # upsample stages
    for i in range(2):
        m[f"upsample.{i}.0.conv.weight"] = f"upsample.{i}.0.conv.weight"
        m[f"upsample.{i}.0.conv.bias"] = f"upsample.{i}.0.conv.bias"
        m[f"upsample.{i}.1.dwconv.conv.weight"] = f"upsample.{i}.1.dwconv.conv.weight"
        m[f"upsample.{i}.1.dwconv.conv.bias"] = f"upsample.{i}.1.dwconv.conv.bias"
        m[f"upsample.{i}.1.norm.weight"] = f"upsample.{i}.1.norm.weight"
        m[f"upsample.{i}.1.norm.bias"] = f"upsample.{i}.1.norm.bias"
        m[f"upsample.{i}.1.pwconv1.weight"] = f"upsample.{i}.1.pwconv1.weight"
        m[f"upsample.{i}.1.pwconv1.bias"] = f"upsample.{i}.1.pwconv1.bias"
        m[f"upsample.{i}.1.pwconv2.weight"] = f"upsample.{i}.1.pwconv2.weight"
        m[f"upsample.{i}.1.pwconv2.bias"] = f"upsample.{i}.1.pwconv2.bias"
        m[f"upsample.{i}.1.gamma"] = f"upsample.{i}.1.gamma"

    # BigVGAN decoder: layers 0-6
    # 0: initial conv
    m["decoder.0.conv.weight"] = "decoder.0.conv.weight"
    m["decoder.0.conv.bias"] = "decoder.0.conv.bias"

    # 1-4: DecoderBlocks
    for i in range(1, 5):
        pfx = f"decoder.{i}"
        # block.0: SnakeBeta
        m[f"{pfx}.block.0.alpha"] = f"{pfx}.block.0.alpha"
        m[f"{pfx}.block.0.beta"] = f"{pfx}.block.0.beta"
        # block.1: TransConv
        m[f"{pfx}.block.1.conv.weight"] = f"{pfx}.block.1.conv.weight"
        m[f"{pfx}.block.1.conv.bias"] = f"{pfx}.block.1.conv.bias"
        # block.2-4: ResidualUnits
        for j in range(2, 5):
            m[f"{pfx}.block.{j}.act1.alpha"] = f"{pfx}.block.{j}.act1.alpha"
            m[f"{pfx}.block.{j}.act1.beta"] = f"{pfx}.block.{j}.act1.beta"
            m[f"{pfx}.block.{j}.conv1.conv.weight"] = (
                f"{pfx}.block.{j}.conv1.conv.weight"
            )
            m[f"{pfx}.block.{j}.conv1.conv.bias"] = f"{pfx}.block.{j}.conv1.conv.bias"
            m[f"{pfx}.block.{j}.act2.alpha"] = f"{pfx}.block.{j}.act2.alpha"
            m[f"{pfx}.block.{j}.act2.beta"] = f"{pfx}.block.{j}.act2.beta"
            m[f"{pfx}.block.{j}.conv2.conv.weight"] = (
                f"{pfx}.block.{j}.conv2.conv.weight"
            )
            m[f"{pfx}.block.{j}.conv2.conv.bias"] = f"{pfx}.block.{j}.conv2.conv.bias"

    # 5: final SnakeBeta
    m["decoder.5.alpha"] = "decoder.5.alpha"
    m["decoder.5.beta"] = "decoder.5.beta"
    # 6: final conv
    m["decoder.6.conv.weight"] = "decoder.6.conv.weight"
    m["decoder.6.conv.bias"] = "decoder.6.conv.bias"

    # quantizer projections
    m["quantizer.semantic.input_proj.weight"] = "quantizer.rvq_first.input_proj.weight"
    m["quantizer.semantic.output_proj.weight"] = (
        "quantizer.rvq_first.output_proj.weight"
    )
    m["quantizer.acoustic.input_proj.weight"] = "quantizer.rvq_rest.input_proj.weight"
    m["quantizer.acoustic.output_proj.weight"] = "quantizer.rvq_rest.output_proj.weight"

    return m


def _load_decoder_codebooks(model: QwenCodecDecoder, hf_state: dict[str, Tensor]):
    # Semantic (1 codebook)
    _load_ema_codebook(
        model.quantizer.semantic.codebooks[0],
        hf_state["quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"],
        hf_state["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"],
    )
    # Acoustic (15 codebooks)
    for i in range(15):
        _load_ema_codebook(
            model.quantizer.acoustic.codebooks[i],
            hf_state[f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"],
            hf_state[f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"],
        )
