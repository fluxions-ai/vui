"""Support for the original-release (pre-1.0) 100M checkpoints.

These use the old architecture — per-quantizer audio embeddings/heads with a
delayed codebook pattern, and the Fluac 22 kHz codec — and do not load into
the current model. Usage:

    from vui.legacy import Vui, render
    model = Vui.from_pretrained("vui-cohost-100m.pt").eval()
    audio = render(model, "Hello there!")  # (1, 1, S) at 22050 Hz

Checkpoints: vui-100m-base.pt, vui-cohost-100m.pt, vui-abraham-100m.pt
(auto-downloaded from https://huggingface.co/fluxions/vui). Runs on CUDA or
CPU — no flash-attn required (MPS is refused: numerically broken there).

On Apple Silicon, run the transformer on MLX (~3.4x faster than CPU):

    from vui.mlx.legacy import load_legacy_mlx
    model, adapter = load_legacy_mlx("vui-cohost-100m.pt")
    audio = render(model, "Hello!", decoder=adapter)
"""

from vui.legacy.inference import generate, render
from vui.legacy.model import Vui

__all__ = ["Vui", "generate", "render"]
