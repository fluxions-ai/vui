import sys

from pydantic import BaseModel


class VuiConfig(BaseModel):
    max_text_tokens: int = 100
    text_size: int = -1
    max_audio_tokens: int = 100

    n_quantizers: int = 9
    codebook_size: int = 1000
    special_token_id: int = 1000
    audio_eos_id: int = 1000 + 1
    audio_pad_id: int = 1000 + 1 + 1
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    bias: bool = False
    dropout: float = 0.0
    use_rotary_emb: bool = True
    rope_dim: int | None = None
    rope_theta: float = 10_000.0
    rope_theta_rescale_factor: float = 1.0


class Config(BaseModel):
    name: str = "base"

    checkpoint: str | dict | None = None

    model: VuiConfig = VuiConfig()


ALL = []
current_module = sys.modules[__name__]
for name in dir(current_module):
    if name.isupper() and isinstance(getattr(current_module, name), Config):
        ALL.append(getattr(current_module, name))

CONFIGS = {v.name: v for v in ALL}
