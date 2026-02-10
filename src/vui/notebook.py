import numpy as np
import torch


def play(audio: torch.Tensor | np.ndarray | str, sr=16000, autoplay=True):
    from IPython.display import Audio, display
    from torchcodec.decoders import AudioDecoder

    if isinstance(audio, str):
        samples = AudioDecoder(audio, sample_rate=sr, num_channels=1).get_all_samples()
        audio = samples.data
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    assert audio.numel() > 100, "play() needs a non empty audio array"

    audio = audio.flatten()
    if audio.dim() < 2:
        audio = audio[None]

    # Sum Channels
    if audio.shape[0] > 1:
        audio = audio.sum(dim=0)

    display(Audio(audio.cpu().detach(), rate=sr, autoplay=autoplay, normalize=True))


def plot_mel_spec(mel_spec: torch.Tensor | np.ndarray, title: str = None):
    import matplotlib.pyplot as plt

    mel_spec = mel_spec.squeeze()
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.cpu().numpy()

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(mel_spec, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("frames")
    ax.set_ylabel("channels")

    if title is not None:
        ax.set_title(title)
