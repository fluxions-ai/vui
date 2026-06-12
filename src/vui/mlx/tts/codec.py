"""MLX codec decoder with streaming conv state (Mimi-style).

Each CausalConv1d/CausalTransConv1d caches its padding between step() calls.
No vocoder_ctx reprocessing — only the new frame flows through.
Fused SnakeBeta Metal kernel for activation.
"""

import os

import mlx.core as mx
import mlx.nn as nn

DOWNSAMPLE_RATE = 1920

# --- Fused SnakeBeta Metal kernel ---

_snake_beta_kernel = mx.fast.metal_kernel(
    name="snake_beta",
    input_names=["x", "alpha", "beta"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        uint C = alpha_shape[0];
        uint c = elem % C;
        T a = metal::exp(alpha[c]);
        T b = metal::exp(beta[c]);
        T val = x[elem];
        T s = metal::sin(a * val);
        out[elem] = val + (s * s) / (b + T(1e-9));
    """,
)


class SnakeBeta(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = mx.zeros((dim,))
        self.beta = mx.zeros((dim,))

    def __call__(self, x):
        out = _snake_beta_kernel(
            inputs=[x, self.alpha, self.beta],
            template=[("T", x.dtype)],
            grid=(x.size, 1, 1),
            threadgroup=(min(256, x.size), 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )
        return out[0]


# --- Streaming conv layers ---


class CausalConv1d(nn.Module):
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
        self.pad = (kernel - 1) * dilation + 1 - stride
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = mx.zeros((out_ch, kernel, in_ch // groups))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self._prev = None

    def __call__(self, x):
        if self.pad > 0:
            x = mx.pad(x, [(0, 0), (self.pad, 0), (0, 0)])
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    def step(self, x):
        """Streaming: prepend cached padding instead of zero-pad."""
        if self._prev is None:
            x = mx.pad(x, [(0, 0), (self.pad, 0), (0, 0)])
        else:
            x = mx.concatenate([self._prev, x], axis=1)
        if self.pad > 0:
            self._prev = x[:, -self.pad :]
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    def reset_state(self):
        self._prev = None


class CausalTransConv1d(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, bias: bool = True
    ):
        super().__init__()
        self.trim_right = kernel - stride
        self.stride = stride
        self.weight = mx.zeros((out_ch, kernel, in_ch))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self._partial = None

    def __call__(self, x):
        y = mx.conv_transpose1d(x, self.weight, stride=self.stride)
        if self.bias is not None:
            y = y + self.bias
        if self.trim_right > 0:
            y = y[:, : -self.trim_right]
        return y

    def step(self, x):
        """Streaming: accumulate overlap from previous call."""
        y = mx.conv_transpose1d(x, self.weight, stride=self.stride)
        if self.bias is not None:
            y = y + self.bias
        if self.trim_right > 0:
            if self._partial is not None:
                y = y.at[:, : self.trim_right].add(self._partial)
            partial = y[:, -self.trim_right :]
            if self.bias is not None:
                partial = partial - self.bias
            self._partial = partial
            y = y[:, : -self.trim_right]
        return y

    def reset_state(self):
        self._partial = None


# --- Vocoder blocks ---


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1_weight = mx.zeros((4 * dim, dim))
        self.pwconv1_bias = mx.zeros((4 * dim,))
        self.pwconv2_weight = mx.zeros((dim, 4 * dim))
        self.pwconv2_bias = mx.zeros((dim,))
        self.gamma = mx.full((dim,), 1e-6)

    def __call__(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = nn.gelu(x @ self.pwconv1_weight.T + self.pwconv1_bias)
        x = self.gamma * (x @ self.pwconv2_weight.T + self.pwconv2_bias)
        return residual + x

    def step(self, x):
        residual = x
        x = self.dwconv.step(x)
        x = self.norm(x)
        x = nn.gelu(x @ self.pwconv1_weight.T + self.pwconv1_bias)
        x = self.gamma * (x @ self.pwconv2_weight.T + self.pwconv2_bias)
        return residual + x

    def reset_state(self):
        self.dwconv.reset_state()


class DecoderResUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel=1)

    def __call__(self, x):
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))

    def step(self, x):
        return x + self.conv2.step(self.act2(self.conv1.step(self.act1(x))))

    def reset_state(self):
        self.conv1.reset_state()
        self.conv2.reset_state()


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.act = SnakeBeta(in_dim)
        self.tconv = CausalTransConv1d(
            in_dim, out_dim, kernel=stride * 2, stride=stride
        )
        self.res1 = DecoderResUnit(out_dim, dilation=1)
        self.res2 = DecoderResUnit(out_dim, dilation=3)
        self.res3 = DecoderResUnit(out_dim, dilation=9)

    def __call__(self, x):
        x = self.tconv(self.act(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x

    def step(self, x):
        x = self.tconv.step(self.act(x))
        x = self.res1.step(x)
        x = self.res2.step(x)
        x = self.res3.step(x)
        return x

    def reset_state(self):
        self.tconv.reset_state()
        self.res1.reset_state()
        self.res2.reset_state()
        self.res3.reset_state()


# --- Transformer ---


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


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
        attn_dim = n_heads * head_dim

        self.input_layernorm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.v_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, d_model, bias=False)
        self.attn_scale = mx.zeros((d_model,))

        self.post_attention_layernorm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, mlp_dim, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, d_model, bias=False)
        self.mlp_scale = mx.zeros((d_model,))

        self._k_cache = None
        self._v_cache = None

    def __call__(self, x, offset: int = 0):
        B, T, _ = x.shape
        h = self.input_layernorm(x)
        q = (
            self.q_proj(h)
            .reshape(B, T, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(h)
            .reshape(B, T, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(h)
            .reshape(B, T, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        q = mx.fast.rope(
            q, self.head_dim, traditional=True, base=10000.0, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, self.head_dim, traditional=True, base=10000.0, scale=1.0, offset=offset
        )

        is_prefill = self._k_cache is None
        if not is_prefill:
            k = mx.concatenate([self._k_cache, k], axis=2)
            v = mx.concatenate([self._v_cache, v], axis=2)
        self._k_cache = k
        self._v_cache = v

        scale = self.head_dim**-0.5
        S = k.shape[2]
        if is_prefill and T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(S)
            mask = mask[-T:]
        else:
            mask = None
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        x = x + self.attn_scale * self.o_proj(attn_out)

        h = self.post_attention_layernorm(x)
        x = x + self.mlp_scale * self.down_proj(
            nn.silu(self.gate_proj(h)) * self.up_proj(h)
        )
        return x

    def reset_state(self):
        self._k_cache = None
        self._v_cache = None


class DecTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.layers = [
            DecTransformerBlock(d_model, n_heads, head_dim, mlp_dim)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        self._cached_len = 0

    def __call__(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, offset=0)
        return self.output_proj(self.norm(x))

    def forward_incremental(self, x, n_new: int):
        x = self.input_proj(x)
        offset = self._cached_len
        for layer in self.layers:
            x = layer(x, offset=offset)
        self._cached_len += x.shape[1]
        return self.output_proj(self.norm(x))

    def reset_state(self):
        self._cached_len = 0
        for layer in self.layers:
            layer.reset_state()


# --- Quantizer (decode only) ---


class EuclideanCodebook(nn.Module):
    def __init__(self, codebook_size: int = 2048, dim: int = 256):
        super().__init__()
        self.embed = mx.zeros((codebook_size, dim))

    def decode(self, indices):
        return self.embed[indices]


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_dim: int = 256,
        codebook_size: int = 2048,
        n_quantizers: int = 1,
    ):
        super().__init__()
        self.output_proj_weight = mx.zeros((input_dim, 1, codebook_dim))
        self.codebooks = [
            EuclideanCodebook(codebook_size, codebook_dim) for _ in range(n_quantizers)
        ]

    def decode(self, codes_per_q):
        quantized = None
        for i, cb in enumerate(self.codebooks[: len(codes_per_q)]):
            q = cb.decode(codes_per_q[i])
            quantized = q if quantized is None else quantized + q
        return mx.conv1d(quantized, self.output_proj_weight)


class SplitResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_dim: int = 256,
        codebook_size: int = 2048,
        n_semantic: int = 1,
        n_acoustic: int = 15,
    ):
        super().__init__()
        self.n_semantic = n_semantic
        self.semantic = ResidualVQ(input_dim, codebook_dim, codebook_size, n_semantic)
        self.acoustic = ResidualVQ(input_dim, codebook_dim, codebook_size, n_acoustic)

    def decode(self, codes):
        codes_t = [codes[:, i] for i in range(codes.shape[1])]
        sem = self.semantic.decode(codes_t[: self.n_semantic])
        if len(codes_t) > self.n_semantic:
            acq = self.acoustic.decode(codes_t[self.n_semantic :])
            return sem + acq
        return sem


# --- Main decoder ---


class QwenCodecDecoderMLX(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 1024
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
            head_dim=64,
            mlp_dim=1024,
        )
        self.upsample = [
            [
                CausalTransConv1d(latent_dim, latent_dim, kernel=2, stride=2),
                ConvNeXtBlock(latent_dim),
            ],
            [
                CausalTransConv1d(latent_dim, latent_dim, kernel=2, stride=2),
                ConvNeXtBlock(latent_dim),
            ],
        ]
        decoder_dim = 1536
        self.dec_conv_in = CausalConv1d(latent_dim, decoder_dim, kernel=7)
        self.dec_blocks = [
            DecoderBlock(decoder_dim, decoder_dim // 2, 8),
            DecoderBlock(decoder_dim // 2, decoder_dim // 4, 5),
            DecoderBlock(decoder_dim // 4, decoder_dim // 8, 4),
            DecoderBlock(decoder_dim // 8, decoder_dim // 16, 3),
        ]
        self.dec_act_out = SnakeBeta(decoder_dim // 16)
        self.dec_conv_out = CausalConv1d(decoder_dim // 16, 1, kernel=7)

    def _vocoder(self, h):
        for stage in self.upsample:
            for block in stage:
                h = block(h)
        h = self.dec_conv_in(h)
        for block in self.dec_blocks:
            h = block(h)
        h = self.dec_act_out(h)
        h = self.dec_conv_out(h)
        return mx.clip(h, -1, 1)

    def _vocoder_step(self, h):
        """Streaming vocoder: each layer uses cached conv state."""
        for stage in self.upsample:
            for block in stage:
                h = block.step(h) if hasattr(block, "step") else block(h)
        h = self.dec_conv_in.step(h)
        for block in self.dec_blocks:
            h = block.step(h)
        h = self.dec_act_out(h)
        h = self.dec_conv_out.step(h)
        return mx.clip(h, -1, 1)

    def forward(self, codes):
        h = self.quantizer.decode(codes)
        h = self.pre_conv(h)
        h = self.pre_transformer(h)
        return self._vocoder(h)

    def prefill(self, codes):
        """Prefill transformer KV cache + warm all conv states with context."""
        self.reset_state()
        h = self.quantizer.decode(codes)
        h = self.pre_conv.step(h)
        h = self.pre_transformer.forward_incremental(h, h.shape[1])
        self._vocoder_step(h)

    def decode_frame(self, new_codes):
        """Decode 1 new frame using cached state. No context reprocessing."""
        h_new = self.quantizer.decode(new_codes)
        h_conv = self.pre_conv.step(h_new)
        h_conv = self.pre_transformer.forward_incremental(h_conv, h_conv.shape[1])
        return self._vocoder_step(h_conv)

    def reset_state(self):
        self.pre_transformer.reset_state()
        self.pre_conv.reset_state()
        for stage in self.upsample:
            for block in stage:
                if hasattr(block, "reset_state"):
                    block.reset_state()
        self.dec_conv_in.reset_state()
        for block in self.dec_blocks:
            block.reset_state()
        self.dec_conv_out.reset_state()


# --- Weight loading ---

_MLX_CACHE = os.path.expanduser("~/.cache/vui/mlx_weights/codec_decoder.safetensors")


def load_codec_decoder_mlx() -> QwenCodecDecoderMLX:
    """Load codec decoder. Uses cached MLX safetensors if available."""
    if os.path.exists(_MLX_CACHE):
        model = QwenCodecDecoderMLX()
        model.load_weights(_MLX_CACHE)
        mx.eval(model.parameters())
        return model
    return _convert_codec_from_pytorch()


def _convert_codec_from_pytorch() -> QwenCodecDecoderMLX:
    from vui.qwen_codec import QwenCodecDecoder

    def _torch_to_mlx(t):
        return mx.array(t.float().numpy())

    pt_model = QwenCodecDecoder.from_pretrained().cpu().float().eval()
    mlx_model = QwenCodecDecoderMLX()
    sd = pt_model.state_dict()
    t2m = _torch_to_mlx

    for i, cb in enumerate(pt_model.quantizer.semantic.codebooks):
        mlx_model.quantizer.semantic.codebooks[i].embed = t2m(cb.embed)
    for i, cb in enumerate(pt_model.quantizer.acoustic.codebooks):
        mlx_model.quantizer.acoustic.codebooks[i].embed = t2m(cb.embed)
    mlx_model.quantizer.semantic.output_proj_weight = t2m(
        sd["quantizer.semantic.output_proj.weight"].permute(0, 2, 1)
    )
    mlx_model.quantizer.acoustic.output_proj_weight = t2m(
        sd["quantizer.acoustic.output_proj.weight"].permute(0, 2, 1)
    )

    def load_conv(mlx_conv, pt_prefix):
        mlx_conv.weight = t2m(sd[f"{pt_prefix}.weight"].permute(0, 2, 1))
        if f"{pt_prefix}.bias" in sd:
            mlx_conv.bias = t2m(sd[f"{pt_prefix}.bias"])

    def load_tconv(mlx_tconv, pt_prefix):
        mlx_tconv.weight = t2m(sd[f"{pt_prefix}.weight"].permute(1, 2, 0))
        if f"{pt_prefix}.bias" in sd:
            mlx_tconv.bias = t2m(sd[f"{pt_prefix}.bias"])

    load_conv(mlx_model.pre_conv, "pre_conv.conv")

    pt = mlx_model.pre_transformer
    pt.input_proj.weight = t2m(sd["pre_transformer.input_proj.weight"])
    pt.input_proj.bias = t2m(sd["pre_transformer.input_proj.bias"])
    pt.output_proj.weight = t2m(sd["pre_transformer.output_proj.weight"])
    pt.output_proj.bias = t2m(sd["pre_transformer.output_proj.bias"])
    pt.norm.weight = t2m(sd["pre_transformer.norm.weight"])
    for i, layer in enumerate(pt.layers):
        p = f"pre_transformer.layers.{i}"
        layer.input_layernorm.weight = t2m(sd[f"{p}.input_layernorm.weight"])
        layer.q_proj.weight = t2m(sd[f"{p}.q_proj.weight"])
        layer.k_proj.weight = t2m(sd[f"{p}.k_proj.weight"])
        layer.v_proj.weight = t2m(sd[f"{p}.v_proj.weight"])
        layer.o_proj.weight = t2m(sd[f"{p}.o_proj.weight"])
        layer.attn_scale = t2m(sd[f"{p}.attn_scale"])
        layer.post_attention_layernorm.weight = t2m(
            sd[f"{p}.post_attention_layernorm.weight"]
        )
        layer.gate_proj.weight = t2m(sd[f"{p}.gate_proj.weight"])
        layer.up_proj.weight = t2m(sd[f"{p}.up_proj.weight"])
        layer.down_proj.weight = t2m(sd[f"{p}.down_proj.weight"])
        layer.mlp_scale = t2m(sd[f"{p}.mlp_scale"])

    for i in range(2):
        load_tconv(mlx_model.upsample[i][0], f"upsample.{i}.0.conv")
        cnb = mlx_model.upsample[i][1]
        load_conv(cnb.dwconv, f"upsample.{i}.1.dwconv.conv")
        cnb.norm.weight = t2m(sd[f"upsample.{i}.1.norm.weight"])
        cnb.norm.bias = t2m(sd[f"upsample.{i}.1.norm.bias"])
        cnb.pwconv1_weight = t2m(sd[f"upsample.{i}.1.pwconv1.weight"])
        cnb.pwconv1_bias = t2m(sd[f"upsample.{i}.1.pwconv1.bias"])
        cnb.pwconv2_weight = t2m(sd[f"upsample.{i}.1.pwconv2.weight"])
        cnb.pwconv2_bias = t2m(sd[f"upsample.{i}.1.pwconv2.bias"])
        cnb.gamma = t2m(sd[f"upsample.{i}.1.gamma"])

    load_conv(mlx_model.dec_conv_in, "decoder.0.conv")
    for i, block in enumerate(mlx_model.dec_blocks):
        p = f"decoder.{i + 1}"
        block.act.alpha = t2m(sd[f"{p}.block.0.alpha"])
        block.act.beta = t2m(sd[f"{p}.block.0.beta"])
        load_tconv(block.tconv, f"{p}.block.1.conv")
        for j, res in enumerate([block.res1, block.res2, block.res3]):
            r = f"{p}.block.{j + 2}"
            res.act1.alpha = t2m(sd[f"{r}.act1.alpha"])
            res.act1.beta = t2m(sd[f"{r}.act1.beta"])
            load_conv(res.conv1, f"{r}.conv1.conv")
            res.act2.alpha = t2m(sd[f"{r}.act2.alpha"])
            res.act2.beta = t2m(sd[f"{r}.act2.beta"])
            load_conv(res.conv2, f"{r}.conv2.conv")

    mlx_model.dec_act_out.alpha = t2m(sd["decoder.5.alpha"])
    mlx_model.dec_act_out.beta = t2m(sd["decoder.5.beta"])
    load_conv(mlx_model.dec_conv_out, "decoder.6.conv")
    mx.eval(mlx_model.parameters())

    # Cache for next time
    os.makedirs(os.path.dirname(_MLX_CACHE), exist_ok=True)
    mx.save_safetensors(_MLX_CACHE, dict(nn.utils.tree_flatten(mlx_model.parameters())))
    return mlx_model
