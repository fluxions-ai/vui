"""Pydantic schema for the model config that travels inside each checkpoint.

Values are read from `ckpt["config"]` at load time (see
`Vui.from_pretrained_inf` and `Config(**ckpt["config"])`). The defaults on
these classes are placeholders only — never relied on at runtime.

`extra="ignore"` lets us load checkpoints that gain new fields without
breaking, and silently drops any leftover training fields.
"""

from pydantic import BaseModel, ConfigDict

from vui.tokenizer import TokenizerConfig


class VuiConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Architecture
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None
    intermediate_size: int | None = None
    bias: bool = False
    dropout: float = 0.0
    max_text_tokens: int = 100
    max_audio_tokens: int = 100
    sinusoidal_cond: bool = False
    spk_emb_dim: int = 1024
    codec_hz: float = 12.5

    # RoPE
    use_rotary_emb: bool = True
    rope_dim: int | None = None
    rope_theta: float = 10_000.0
    rope_theta_rescale_factor: float = 1.0
    global_rope_dim: int | None = None
    global_rope_theta: float | None = None
    window_size: int | None = None
    global_every: int | None = None

    # RQ-Transformer head
    use_rq_transformer: bool = False
    rq_d_model: int = 512
    rq_n_layers: int = 6
    rq_n_heads: int = 6
    n_quantizers: int = 9
    codebook_size: int = 4096

    # Conditioning projection heads (architecture switches)
    has_sq_proj: bool = False
    has_wps_proj: bool = False
    has_spk_proj: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_secs: float = 360
    tokenizer: TokenizerConfig = TokenizerConfig()


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    checkpoint: str | dict | None = None
    model: VuiConfig = VuiConfig()
    data: DataConfig = DataConfig()

    @property
    def max_seq_len(self) -> int:
        audio_tokens = int(self.data.max_secs * self.model.codec_hz)
        text_tokens = int(audio_tokens * 2.5)
        total = audio_tokens + text_tokens
        return 64 * ((total + 63) // 64)


# Sentinel weight keys for each optional projection head — used to infer
# missing `has_*_proj` flags from a checkpoint's state dict.
_PROJ_FLAGS: tuple[tuple[str, str], ...] = (
    ("has_sq_proj", "sq_proj.proj.0.weight"),
    ("has_wps_proj", "wps_proj.proj.0.weight"),
    ("has_spk_proj", "spk_proj.weight"),
)


def infer_optional_modules(config: dict, state_dict_keys) -> dict:
    """Auto-detect `has_*_proj` flags from state-dict keys.

    Older training scripts didn't persist these flags in the saved config
    but the weights are present — so build the model class with the
    matching modules whenever the corresponding sentinel keys exist.
    Any flag already set in the config wins.
    """
    keys = set(state_dict_keys)
    mcfg = config.setdefault("model", {})
    for flag, sentinel in _PROJ_FLAGS:
        if sentinel in keys and flag not in mcfg:
            mcfg[flag] = True
    return config
