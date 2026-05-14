"""Moonshine Streaming ASR model for MLX."""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EncoderConfig:
    hidden_size: int = 320
    intermediate_size: int = 1280
    hidden_act: str = "gelu"
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    attention_bias: bool = False
    sample_rate: int = 16000
    frame_ms: float = 5.0
    sliding_windows: list = field(default_factory=lambda: [[16, 4]] * 6)
    head_dim: int | None = None

    def __post_init__(self):
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads


@dataclass
class StreamingConfig:
    encoder_config: dict | EncoderConfig | None = None
    vocab_size: int = 32768
    hidden_size: int = 320
    intermediate_size: int = 1280
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.8
    bos_token_id: int = 1
    eos_token_id: int = 2
    decoder_start_token_id: int = 1
    tie_word_embeddings: bool = False
    head_dim: int | None = None
    encoder_hidden_size: int | None = None

    def __post_init__(self):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig(
                **{
                    k: v
                    for k, v in self.encoder_config.items()
                    if k in EncoderConfig.__dataclass_fields__
                }
            )
        elif self.encoder_config is None:
            self.encoder_config = EncoderConfig()
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        if self.encoder_hidden_size is None:
            self.encoder_hidden_size = self.encoder_config.hidden_size

    @classmethod
    def from_dict(cls, d: dict) -> "StreamingConfig":
        rope = d.get("rope_parameters", {})
        return cls(
            **{
                k: v
                for k, v in {
                    **d,
                    "rope_theta": rope.get("rope_theta", 10000.0),
                    "partial_rotary_factor": rope.get("partial_rotary_factor", 0.8),
                }.items()
                if k in cls.__dataclass_fields__
            }
        )


# --- Encoder ---


class FrameCMVN(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        rms = mx.sqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + self.eps)
        return centered / rms


class AsinhCompression(nn.Module):
    def __init__(self, k_init: float = 0.75):
        super().__init__()
        self.log_k = mx.log(mx.array(k_init))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.arcsinh(mx.exp(self.log_k) * x)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1

    def __call__(self, x: mx.array) -> mx.array:
        pad = mx.zeros((x.shape[0], self.left_pad, x.shape[2]))
        x = mx.concatenate([pad, x], axis=1)
        x = mx.conv1d(x, self.weight, stride=self.stride)
        if self.bias is not None:
            x = x + self.bias
        return x


class UnitOffsetLayerNorm(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.ln = nn.LayerNorm(dims, affine=False)
        self.gamma = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.ln(x) * (self.gamma + 1.0)


class EncoderMLP(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class EncoderAttention(nn.Module):
    def __init__(self, config: EncoderConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=bias
        )
        sw = config.sliding_windows[layer_idx]
        self.left_window = sw[0]
        self.right_window = sw[1]

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        q_idx = mx.arange(T)[:, None]
        kv_idx = mx.arange(T)[None, :]
        dist = q_idx - kv_idx
        valid = ((dist >= 0) & (dist < self.left_window)) | (
            (dist < 0) & ((-dist) < self.right_window)
        )
        mask = mx.where(valid, mx.array(0.0), mx.array(-1e9))

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(o)


class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig, layer_idx: int):
        super().__init__()
        self.self_attn = EncoderAttention(config, layer_idx)
        self.mlp = EncoderMLP(config)
        self.input_layernorm = UnitOffsetLayerNorm(config.hidden_size)
        self.post_attention_layernorm = UnitOffsetLayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class EncoderEmbedder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.frame_len = int(round(config.sample_rate * config.frame_ms / 1000.0))
        self.cmvn = FrameCMVN()
        self.comp = AsinhCompression()
        self.linear = nn.Linear(self.frame_len, config.hidden_size, bias=False)
        self.conv1 = CausalConv1d(
            config.hidden_size, config.hidden_size * 2, kernel_size=5, stride=2
        )
        self.conv2 = CausalConv1d(
            config.hidden_size * 2, config.hidden_size, kernel_size=5, stride=2
        )

    def __call__(self, audio: mx.array) -> mx.array:
        B = audio.shape[0]
        T_frames = audio.shape[1] // self.frame_len
        x = audio[:, : T_frames * self.frame_len].reshape(B, T_frames, self.frame_len)
        x = self.cmvn(x)
        x = self.comp(x)
        x = nn.silu(self.linear(x))
        x = nn.silu(self.conv1(x))
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embedder = EncoderEmbedder(config)
        self.layers = [EncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.final_norm = UnitOffsetLayerNorm(config.hidden_size)

    def __call__(self, audio: mx.array) -> mx.array:
        x = self.embedder(audio)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# --- Decoder ---


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.stack([-x2, x1], axis=-1).reshape(x.shape)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    half = cos.shape[-1] // 2
    cos = mx.repeat(cos[..., :half], 2, axis=-1)
    sin = mx.repeat(sin[..., :half], 2, axis=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    return mx.concatenate([q_embed, q_pass], axis=-1), mx.concatenate(
        [k_embed, k_pass], axis=-1
    )


class DecoderAttention(nn.Module):
    def __init__(self, config: StreamingConfig, is_causal: bool):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.is_causal = is_causal
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        cos_sin: tuple[mx.array, mx.array] | None = None,
        key_value_states: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        B, T, _ = x.shape
        is_cross = key_value_states is not None

        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        if is_cross and cache is not None:
            k, v = cache
        else:
            src = key_value_states if is_cross else x
            S = src.shape[1]
            k = (
                self.k_proj(src)
                .reshape(B, S, self.num_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(src)
                .reshape(B, S, self.num_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

        if not is_cross and cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, *cos_sin)

        if not is_cross and cache is not None:
            prev_k, prev_v = cache
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)

        new_cache = (k, v)

        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        mask = None
        if self.is_causal and T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            if k.shape[2] > T:
                prefix = mx.zeros((T, k.shape[2] - T))
                mask = mx.concatenate([prefix, mask], axis=1)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(o), new_cache


class DecoderMLP(nn.Module):
    def __init__(self, config: StreamingConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x, gate = mx.split(x, 2, axis=-1)
        return self.fc2(nn.silu(gate) * x)


class DecoderLayer(nn.Module):
    def __init__(self, config: StreamingConfig):
        super().__init__()
        self.self_attn = DecoderAttention(config, is_causal=True)
        self.encoder_attn = DecoderAttention(config, is_causal=False)
        self.mlp = DecoderMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, affine=True, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, affine=True, bias=False
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, affine=True, bias=False)

    def __call__(
        self,
        x: mx.array,
        encoder_out: mx.array,
        cos_sin: tuple[mx.array, mx.array],
        self_cache: tuple | None = None,
        cross_cache: tuple | None = None,
    ) -> tuple[mx.array, tuple, tuple]:
        residual = x
        x = self.input_layernorm(x)
        x, new_self_cache = self.self_attn(x, cos_sin=cos_sin, cache=self_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x, new_cross_cache = self.encoder_attn(
            x, key_value_states=encoder_out, cache=cross_cache
        )
        x = residual + x

        residual = x
        x = self.final_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_self_cache, new_cross_cache


class Decoder(nn.Module):
    def __init__(self, config: StreamingConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, affine=True, bias=False)
        self.pos_emb = nn.Embedding(
            config.max_position_embeddings, config.encoder_hidden_size
        )
        if config.encoder_hidden_size != config.hidden_size:
            self.proj = nn.Linear(
                config.encoder_hidden_size, config.hidden_size, bias=False
            )
        else:
            self.proj = None
        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        rope_dim = rope_dim - (rope_dim % 2)
        self._inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, rope_dim, 2, dtype=mx.float32) / rope_dim)
        )

    def _rope(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        freqs = (
            position_ids[:, :, None].astype(mx.float32) * self._inv_freq[None, None, :]
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    def __call__(
        self,
        tokens: mx.array,
        encoder_out: mx.array,
        cache: list[dict] | None = None,
    ) -> tuple[mx.array, list[dict]]:
        enc_pos = self.pos_emb(mx.arange(encoder_out.shape[1]))
        enc = encoder_out + enc_pos
        if self.proj is not None:
            enc = self.proj(enc)

        x = self.embed_tokens(tokens)

        if cache is None:
            cache = [{"self": None, "cross": None} for _ in range(len(self.layers))]

        offset = cache[0]["self"][0].shape[2] if cache[0]["self"] is not None else 0
        position_ids = mx.arange(offset, offset + tokens.shape[1])[None, :]
        cos_sin = self._rope(position_ids)

        new_cache = []
        for i, layer in enumerate(self.layers):
            x, new_self, new_cross = layer(
                x,
                enc,
                cos_sin,
                self_cache=cache[i]["self"],
                cross_cache=cache[i]["cross"],
            )
            new_cache.append({"self": new_self, "cross": new_cross})

        return self.norm(x), new_cache


# --- Full model ---


class MoonshineStreaming(nn.Module):
    def __init__(self, config: StreamingConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder_config)
        self.decoder = Decoder(config)
        if not config.tie_word_embeddings:
            self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tokenizer = None

    def encode(self, audio: mx.array) -> mx.array:
        if audio.ndim == 1:
            audio = audio[None, :]
        return self.encoder(audio)

    def decode_step(
        self,
        tokens: mx.array,
        encoder_out: mx.array,
        cache: list[dict] | None = None,
    ) -> tuple[mx.array, list[dict]]:
        hidden, cache = self.decoder(tokens, encoder_out, cache=cache)
        if self.config.tie_word_embeddings:
            logits = self.decoder.embed_tokens.as_linear(hidden)
        else:
            logits = self.proj_out(hidden)
        return logits, cache

    def transcribe(
        self,
        audio: mx.array,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        if audio.ndim == 1:
            audio = audio[None, :]

        encoder_out = self.encoder(audio)
        mx.eval(encoder_out)

        tokens = [self.config.decoder_start_token_id]
        cache = None

        for _ in range(max_tokens):
            tok = mx.array([[tokens[-1]]], dtype=mx.int32)
            logits, cache = self.decode_step(tok, encoder_out, cache)
            mx.eval(logits)
            logits = logits[:, -1, :]
            if temperature > 0:
                next_token = int(mx.random.categorical(logits / temperature))
            else:
                next_token = int(logits.argmax())
            if next_token == self.config.eos_token_id:
                break
            tokens.append(next_token)

        generated = tokens[1:]
        if self._tokenizer is not None:
            return self._tokenizer.decode(generated, skip_special_tokens=True)
        return "".join(chr(t) if t < 128 else f"<{t}>" for t in generated)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            if key.startswith("proj_out."):
                new_key = key
            if "rotary_emb" in new_key:
                continue
            if "conv" in new_key and "weight" in new_key and value.ndim == 3:
                value = mx.transpose(value, (0, 2, 1))
            sanitized[new_key] = value
        return sanitized
