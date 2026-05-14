"""Qwen ECAPA-TDNN speaker encoder (8.9M params, 1024-dim output).

Standalone module for computing speaker embeddings from audio.
Weights loaded from Qwen/Qwen3-TTS-12Hz-0.6B-Base safetensors (speaker_encoder.* keys).
"""

import glob
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        w = in_channels // scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(w, out_channels // scale, kernel_size, dilation)
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        outs = []
        for i, part in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                o = part
            elif i == 1:
                o = self.blocks[i - 1](part)
            else:
                o = self.blocks[i - 1](part + o)
            outs.append(o)
        return torch.cat(outs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, 1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, 1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m = x.mean(dim=2, keepdim=True)
        return x * self.sigmoid(self.conv2(self.relu(self.conv1(m))))


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            attention_channels, channels, 1, padding="same", padding_mode="reflect"
        )

    def forward(self, x):
        sl = x.shape[-1]
        m = torch.ones(x.shape[0], 1, sl, device=x.device, dtype=x.dtype) / sl
        mean = (m * x).sum(2)
        std = torch.sqrt((m * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        attn = torch.cat([x, mean.unsqueeze(2).expand_as(x), std.unsqueeze(2).expand_as(x)], 1)
        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = F.softmax(attn, dim=2)
        mean2 = (attn * x).sum(2)
        std2 = torch.sqrt((attn * (x - mean2.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return torch.cat((mean2, std2), 1).unsqueeze(2)


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale=8, se_ch=128, kernel_size=1, dilation=1):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_ch, out_ch, 1, 1)
        self.res2net_block = Res2NetBlock(out_ch, out_ch, scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_ch, out_ch, 1, 1)
        self.se_block = SqueezeExcitationBlock(out_ch, se_ch, out_ch)

    def forward(self, x):
        return self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x)))) + x


class QwenSpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [512, 512, 512, 512, 1536]
        ks = [5, 3, 3, 3, 1]
        dil = [1, 2, 3, 4, 1]
        self.blocks = nn.ModuleList([TimeDelayNetBlock(128, ch[0], ks[0], dil[0])])
        for i in range(1, 4):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(ch[i - 1], ch[i], 8, 128, ks[i], dil[i])
            )
        self.mfa = TimeDelayNetBlock(ch[-1], ch[-1], ks[-1], dil[-1])
        self.asp = AttentiveStatisticsPooling(ch[-1], 128)
        self.fc = nn.Conv1d(ch[-1] * 2, 1024, 1, padding="same", padding_mode="reflect")

    def forward(self, x):
        x = x.transpose(1, 2)
        hs = []
        for layer in self.blocks:
            x = layer(x)
            hs.append(x)
        x = torch.cat(hs[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        return self.fc(x).squeeze(-1)

    @classmethod
    def from_pretrained(cls, repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        model_dir = snapshot_download(repo_id)
        st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        spk_sd = {}
        for f in st_files:
            with safe_open(f, framework="pt", device="cpu") as st:
                for k in st.keys():
                    if k.startswith("speaker_encoder."):
                        spk_sd[k[16:]] = st.get_tensor(k)
        model = cls()
        model.load_state_dict(spk_sd, strict=True)
        model.eval().requires_grad_(False)
        return model

    def embed(self, audio: Tensor, sr: int = 24000) -> Tensor:
        """Compute speaker embedding from raw audio waveform.

        Args:
            audio: (T,) or (1, T) float tensor at `sr` Hz
            sr: sample rate (resampled to 24kHz if different)

        Returns:
            (1024,) float32 embedding
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if sr != 24000:
            from julius.resample import resample_frac

            audio = resample_frac(audio, sr, 24000)
        mel = mel_spectrogram(audio).transpose(1, 2)
        with torch.no_grad():
            return self(mel)[0]


_MEL_BASIS = None
_HANN = None


def _mel_filterbank(sr=24000, n_fft=1024, n_mels=128, fmin=0.0, fmax=12000.0) -> Tensor:
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.arange(n_freqs, dtype=torch.float32) * (sr / n_fft)

    def hz_to_mel(f):
        f = torch.as_tensor(f, dtype=torch.float32)
        return torch.where(
            f >= 1000.0,
            15.0 + 27.0 / torch.log(torch.tensor(6.4)) * torch.log(f / 1000.0),
            f * 3.0 / 200.0,
        )

    def mel_to_hz(m):
        m = torch.as_tensor(m, dtype=torch.float32)
        return torch.where(
            m >= 15.0,
            1000.0 * torch.exp((m - 15.0) * torch.log(torch.tensor(6.4)) / 27.0),
            m * 200.0 / 3.0,
        )

    mels = torch.linspace(
        hz_to_mel(torch.tensor(fmin)).item(),
        hz_to_mel(torch.tensor(fmax)).item(),
        n_mels + 2,
    )
    freqs = mel_to_hz(mels)
    weights = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lower, center, upper = freqs[i], freqs[i + 1], freqs[i + 2]
        up_slope = (fft_freqs - lower) / (center - lower)
        down_slope = (upper - fft_freqs) / (upper - center)
        weights[i] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0) * (
            2.0 / (upper - lower)
        )
    return weights


def mel_spectrogram(y: Tensor) -> Tensor:
    global _MEL_BASIS, _HANN
    if _MEL_BASIS is None:
        _MEL_BASIS = _mel_filterbank()
        _HANN = torch.hann_window(1024)
    mb = _MEL_BASIS.to(y.device)
    hw = _HANN.to(y.device)
    y = F.pad(y.unsqueeze(1), (384, 384), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        1024,
        256,
        1024,
        hw,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(mb, spec), min=1e-5))
