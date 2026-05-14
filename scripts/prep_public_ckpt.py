"""Strip a training checkpoint down to what vui-public needs and emit safetensors.

Drops:
  - top-level: optimizer, ema, step, run_id, hours
  - config: every training-time field (lr, beta1, weight_decay, muon_*, ...)
  - config.data: every training-only field (queries, mode_weights, prob_no_align,
    codes_s3_prefix, transcription_model, ...)

Keeps:
  - model state_dict (saved as safetensors body)
  - config.model + minimal config.data (saved as JSON in safetensors metadata)

Usage:
  uv run scripts/prep_public_ckpt.py <input.pt> [<output.safetensors>]
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# What we keep — must match vui-public's `VuiConfig` / `DataConfig` schema.
# Everything else in the source ckpt's config is silently dropped.
KEEP_MODEL_FIELDS = {
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "intermediate_size",
    "bias",
    "dropout",
    "max_text_tokens",
    "max_audio_tokens",
    "sinusoidal_cond",
    "spk_emb_dim",
    "codec_hz",
    "use_rotary_emb",
    "rope_dim",
    "rope_theta",
    "rope_theta_rescale_factor",
    "global_rope_dim",
    "global_rope_theta",
    "window_size",
    "global_every",
    "use_rq_transformer",
    "rq_d_model",
    "rq_n_layers",
    "rq_n_heads",
    "n_quantizers",
    "codebook_size",
}
KEEP_DATA_FIELDS = {"tokenizer", "max_secs"}
PROB_TO_SWITCH = {
    "prob_sq": "has_sq_proj",
    "prob_wps": "has_wps_proj",
    "prob_spk_emb": "has_spk_proj",
}


def strip(in_path: str, out_path: str) -> None:
    print(f"Loading {in_path}...")
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)

    src_cfg = ckpt["config"]
    src_data = src_cfg.get("data", {})
    src_model = src_cfg["model"]

    new_model = {k: src_model[k] for k in KEEP_MODEL_FIELDS if k in src_model}
    for prob_field, switch_field in PROB_TO_SWITCH.items():
        new_model[switch_field] = bool(src_data.get(prob_field, 0) > 0)

    new_data = {k: src_data[k] for k in KEEP_DATA_FIELDS if k in src_data}

    new_cfg = {"model": new_model, "data": new_data}

    print(f"Kept model keys ({len(new_model)}): {sorted(new_model)}")
    print(f"Kept data keys ({len(new_data)}): {sorted(new_data)}")
    print(f"Dropped top-level: {sorted(set(ckpt) - {'config', 'model'})}")
    print(f"Dropped config-root keys: {sorted(set(src_cfg) - {'model', 'data'})}")
    print(f"Dropped model keys: {sorted(set(src_model) - KEEP_MODEL_FIELDS)}")
    print(f"Dropped data keys: {sorted(set(src_data) - KEEP_DATA_FIELDS)}")
    print(
        f"Translated switches: "
        f"{ {f: new_model[s] for f, s in PROB_TO_SWITCH.items()} }"
    )

    state_dict = {k: v.contiguous() for k, v in ckpt["model"].items()}
    save_file(state_dict, out_path, metadata={"config": json.dumps(new_cfg)})

    in_mb = Path(in_path).stat().st_size / 1e6
    out_mb = Path(out_path).stat().st_size / 1e6
    print(f"\n{in_mb:.0f}MB -> {out_mb:.0f}MB  {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/prep_public_ckpt.py <input.pt> [<output.safetensors>]"
        )
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vui-nano.safetensors"
    strip(in_path, out_path)
