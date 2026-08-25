import math
from contextlib import nullcontext
from functools import partial, wraps
from os import path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from einops.layers.torch import Rearrange
from pydantic import BaseModel
from torch import Tensor, int32
from torch.amp import autocast
from torch.nn import Module
from torch.nn.utils.parametrizations import weight_norm

def decompile_state_dict(state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: bool | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = True,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32: bool = True,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer(
                "implicit_codebook", implicit_codebook, persistent=False
            )

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: Tensor):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        device_type = z.device.type

        with torch.autocast(device_type=device_type, enabled=False):
            if self.channel_first:
                z = rearrange(z, "b d ... -> b ... d")
                z, ps = pack_one(z, "b * d")

            assert (
                z.shape[-1] == self.dim
            ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

            z = self.project_in(z)

            z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

            # whether to force quantization step to be full precision or not

            force_f32 = self.force_quantization_f32
            quantization_context = (
                partial(autocast, device_type=device_type, enabled=False)
                if force_f32
                else nullcontext
            )

            with quantization_context():
                orig_dtype = z.dtype

                if force_f32 and orig_dtype not in self.allowed_dtypes:
                    z = z.float()

                codes = self.quantize(z)

                # returning indices could be optional

                indices = None

                if self.return_indices:
                    indices = self.codes_to_indices(codes)

                codes = rearrange(codes, "b n c d -> b n (c d)")

                codes = codes.type(orig_dtype)

            # project out

            out = self.project_out(codes)

            # reconstitute image or video dimensions

            if self.channel_first:
                out = unpack_one(out, ps, "b * d")
                out = rearrange(out, "b ... d -> b d ...")

                indices = maybe(unpack_one)(indices, ps, "b * c")

            if not self.keep_num_codebooks_dim and self.return_indices:
                indices = maybe(rearrange)(indices, "... 1 -> ...")

            # return quantized output and indices

            return out, indices


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    # @torch.compile(dynamic=True)
    def forward(self, z: Tensor):
        return self.model(z)


class FiniteScalarQuantize(nn.Module):
    def __init__(
        self, latent_dim: int, levels: list[int], *, stride: int = 1, mlp: bool = False
    ):
        super().__init__()

        self.stride = stride

        codebook_dim = len(levels)

        self.in_proj = WNConv1d(latent_dim, codebook_dim, kernel_size=1)
        self.quantize = FSQ(levels=levels, channel_first=True)
        self.out_proj = WNConv1d(codebook_dim, latent_dim, kernel_size=1)

        if mlp:
            self.mlp = nn.Sequential(
                Rearrange("B C T -> B T C"),
                nn.Linear(latent_dim, 4 * latent_dim),
                nn.GELU(),
                nn.Linear(4 * latent_dim, latent_dim),
                Rearrange("B T C -> B C T"),
            )
        else:
            self.mlp = None

    def from_indices(self, indices: Tensor):
        B, T = indices.size()
        z_q = self.quantize.indices_to_codes(indices)
        z_q = self.out_proj(z_q)
        return z_q

    def forward(self, z: Tensor, *args):
        if self.stride > 1:
            z = F.avg_pool1d(z, self.stride, stride=self.stride)

        z_e = self.in_proj(z)  # z_e : (B x D x T)

        # we're channels first
        # scale = scale.unsqueeze(-1)

        # z_e = z_e / scale
        z_q, indices = self.quantize(z_e)
        # z_q = z_q * scale

        z_q = self.out_proj(z_q)

        if self.stride > 1:
            z_e = z_e.repeat_interleave(self.stride, dim=-1)
            z_q = z_q.repeat_interleave(self.stride, dim=-1)
            indices = indices.repeat_interleave(self.stride, dim=-1)

        if self.mlp is not None:
            z_q = self.mlp(z_q)

        return z_q, indices, z_e


class ResidualFiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        n_quantizers: int,
        levels: list[int],
        strides: list[int] | None = None,
        quantizer_dropout: float = 0.0,
        mlp: bool = False,
    ):
        super().__init__()

        self.n_quantizers = n_quantizers
        self.quantizer_dropout = quantizer_dropout

        strides = [1] * n_quantizers if strides is None else strides

        assert (
            len(strides) == n_quantizers
        ), "Strides must be provided for each codebook"

        scales = []
        quantizers = []
        levels_tensor = torch.tensor(levels, dtype=torch.float32)

        for i in range(n_quantizers):
            scales.append((levels_tensor - 1) ** -i)
            quantizers.append(
                FiniteScalarQuantize(
                    latent_dim=latent_dim, levels=levels, stride=strides[i], mlp=mlp
                )
            )

        self.quantizers = nn.ModuleList(quantizers)

        self.register_buffer("scales", torch.stack(scales), persistent=False)

        codebooks = [
            quantizer.quantize.implicit_codebook for quantizer in self.quantizers
        ]
        self.codebooks = torch.stack(codebooks, dim=0)

    def from_indices(self, indices: Tensor):
        B, Q, T = indices.size()

        z_q = 0.0

        for i, quantizer in enumerate(self.quantizers):
            z_q_i = quantizer.from_indices(indices[:, i])
            z_q = z_q + z_q_i

        return z_q

    def forward(self, z: Tensor, n_quantizers: int | None = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
        """
        B = z.shape[0]
        z_q = 0
        residual = z

        indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_quantizers

        if self.training:
            n_quantizers = torch.ones((B,)) * self.n_quantizers + 1
            dropout = torch.randint(1, self.n_quantizers + 1, (B,))
            n_dropout = int(B * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, indices_i, z_e_i = quantizer(residual)

            residual = residual - z_q_i.detach()

            mask = torch.full((B,), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]

            indices.append(indices_i)
            latents.append(z_e_i)

        indices = torch.stack(indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, indices, latents


class FluacConfig(BaseModel):
    sample_rate: int = 44100

    codebook_size: int | None = None

    encoder_dim: int = 64
    encoder_rates: list[int] = [2, 4, 8, 8]

    quantizer_strides: list[int] | None = None  # SNAC style strides
    n_quantizers: int = 1
    fsq_levels: list[int] | None = [8, 5, 5, 5]  # 1000
    decoder_dim: int = 1536
    decoder_rates: list[int] = [8, 8, 4, 2]

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)

    @property
    def latent_dim(self) -> int:
        return self.encoder_dim * (2 ** len(self.encoder_rates))

    @property
    def effective_codebook_size(self) -> int:
        return math.prod(self.fsq_levels)


class Fluac(nn.Module):
    Q9_22KHZ = "fluac-22hz-22khz.pt"

    def __init__(self, config: FluacConfig):
        super().__init__()

        self.config = config

        self.encoder = Encoder(
            config.encoder_dim, config.encoder_rates, config.latent_dim
        )

        self.quantizer = ResidualFiniteScalarQuantize(
            latent_dim=config.latent_dim,
            n_quantizers=config.n_quantizers,
            levels=config.fsq_levels,
            strides=config.quantizer_strides,
        )

        self.decoder = Decoder(
            config.latent_dim,
            config.decoder_dim,
            config.decoder_rates,
        )

        self.apply(init_weights)

    @staticmethod
    def from_pretrained(name: str = Q9_22KHZ):
        if path.exists(name):
            checkpoint_path = name
        else:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                "fluxions/vui",
                name,
            )

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        config = checkpoint["config"]
        if "model" in config:
            model_config = FluacConfig(**config["model"])
        else:
            model_config = FluacConfig(**config)

        generator = Fluac(model_config).eval()
        ckpt = decompile_state_dict(checkpoint["generator"])
        generator.load_state_dict(ckpt)
        return generator

    def pad(self, waveform: Tensor):
        T = waveform.size(-1)
        right_pad = math.ceil(T / self.config.hop_length) * self.config.hop_length - T
        waveform = F.pad(waveform, (0, right_pad))
        return waveform

    @torch.inference_mode()
    def from_indices(self, indices: Tensor):
        z_q = self.quantizer.from_indices(indices)
        waveform = self.decoder(z_q)
        return waveform

    @torch.inference_mode()
    def encode(self, waveforms: Tensor, n_quantizers: int | None = None):
        # Ensure that waveforms is 3 dima
        waveforms = waveforms.flatten()[None][None]
        waveforms = self.pad(waveforms)
        B, C, T = waveforms.size()
        z = self.encoder(waveforms)
        z_q, codes, latents = self.quantizer(z, n_quantizers=n_quantizers)
        return codes

    def forward(self, waveforms: Tensor, n_quantizers: int | None = None):
        B, C, T = waveforms.size()
        waveforms = self.pad(waveforms)
        z = self.encoder(waveforms)
        z_q, codes, latents = self.quantizer(z, n_quantizers=n_quantizers)

        recons = self.decoder(z_q)
        recons = recons[..., :T]
        return {
            "recons": recons,
            "codes": codes,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def hz(self):
        import numpy as np

        return self.config.sample_rate / np.prod(self.config.encoder_rates).item()


if __name__ == "__main__":
    codec = Fluac.from_pretrained(Fluac.Q9_22KHZ)
    print(codec.config)
    wav = torch.rand(1, 1, 22050)
    wav = codec.pad(wav)
    codes = codec.encode(wav)
    breakpoint()
