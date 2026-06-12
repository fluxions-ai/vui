"""Load MLX TTS model weights from safetensors (fast) or PyTorch checkpoint."""

import json
import os

import mlx.core as mx
import mlx.nn as nn

from vui.mlx.tts.model import VuiMLX

MLX_CACHE_DIR = os.path.expanduser("~/.cache/vui/mlx_weights")


def _apply_quant(model: VuiMLX, bits: int, group_size: int = 32):
    """Quantize attn + mlp in both decoder and rq_transformer, in place."""
    for block in model.decoder.blocks:
        nn.quantize(block.attn, bits=bits, group_size=group_size)
        nn.quantize(block.mlp, bits=bits, group_size=group_size)
    for block in model.rq_transformer.blocks:
        nn.quantize(block, bits=bits, group_size=group_size)


def load_quantized(
    checkpoint_path: str, precision: str, group_size: int = 32
) -> tuple[VuiMLX, dict]:
    """Load a VuiMLX model quantized to the given precision.

    Precision-specific safetensors cache is reused across runs. If a cache file
    exists it's loaded directly (bypasses float32 load + quantize). Otherwise
    the float32 model is loaded, quantized in place, and the result is cached.
    """
    cache_key = os.path.basename(checkpoint_path).replace(".pt", "")
    q_path = os.path.join(MLX_CACHE_DIR, f"tts_{cache_key}_{precision}.safetensors")
    cfg_path = os.path.join(MLX_CACHE_DIR, f"tts_{cache_key}_config.json")
    bits = {"int4": 4, "int8": 8}.get(precision, 0)

    if bits == 0:
        # No quantization — regular float32 path.
        return load_model(checkpoint_path)

    if os.path.exists(q_path) and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)
        model = VuiMLX(config)
        _apply_quant(model, bits, group_size)  # replace layers with Quantized*
        model.load_weights(q_path)
        mx.eval(model.parameters())
        return model, config

    # Cold path: load float32, quantize, save
    model, config = load_model(checkpoint_path)
    _apply_quant(model, bits, group_size)
    mx.eval(model.parameters())
    os.makedirs(MLX_CACHE_DIR, exist_ok=True)
    mx.save_safetensors(q_path, dict(nn.utils.tree_flatten(model.parameters())))
    print(f"Saved quantized weights to {q_path}")
    return model, config


def load_model(
    checkpoint_path: str = "vui-nano.safetensors",
) -> tuple[VuiMLX, dict]:
    """Load TTS model. Uses cached MLX safetensors if available, else converts from PyTorch."""
    cache_key = os.path.basename(checkpoint_path).replace(".pt", "")
    st_path = os.path.join(MLX_CACHE_DIR, f"tts_{cache_key}.safetensors")
    cfg_path = os.path.join(MLX_CACHE_DIR, f"tts_{cache_key}_config.json")

    if os.path.exists(st_path) and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)
        model = VuiMLX(config)
        model.load_weights(st_path)
        mx.eval(model.parameters())
        return model, config

    model, config = load_from_pytorch(checkpoint_path)
    os.makedirs(MLX_CACHE_DIR, exist_ok=True)
    mx.save_safetensors(st_path, dict(nn.utils.tree_flatten(model.parameters())))
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    return model, config


def torch_to_mlx(t) -> mx.array:
    return mx.array(t.float().numpy())


def load_from_pytorch(checkpoint_path: str) -> tuple[VuiMLX, dict]:
    """Load a VuiMLX model from a PyTorch or safetensors checkpoint. Requires torch."""
    import torch

    if not os.path.exists(checkpoint_path):
        from vui.hf import download

        checkpoint_path = download(checkpoint_path)

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file, safe_open

        with safe_open(checkpoint_path, framework="pt") as f:
            config = json.loads(f.metadata()["config"])
        raw_state = load_file(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        raw_state = ckpt["model"]
    pt_state = {
        k.replace("module.", "").replace("text_embedding.", "token_emb."): v
        for k, v in raw_state.items()
    }

    model = VuiMLX(config)
    _load_weights(model, pt_state)
    return model, config


def _load_weights(model: VuiMLX, sd: dict):
    def get(key):
        return torch_to_mlx(sd[key]) if key in sd else None

    # Token embedding (handle byte-token migration)
    w = get("token_emb.weight")
    if w is not None:
        if w.shape[0] != model.token_emb.weight.shape[0]:
            expected = model.token_emb.weight.shape[0]
            new_w = mx.zeros((expected, w.shape[1]))
            new_w = new_w.at[: w.shape[0]].add(w)
            w = new_w
        model.token_emb.weight = w

    w = get("audio_emb.embedding.weight")
    if w is not None:
        model.audio_emb.embedding.weight = w

    for key, attr in [
        ("codec_head.weight", "weight"),
        ("eos_head.weight", "weight"),
        ("eos_head.bias", "bias"),
    ]:
        w = get(key)
        if w is not None:
            setattr(model.codec_head if "codec" in key else model.eos_head, attr, w)

    # Conditioning projectors
    for proj, prefix in [(model.sq_proj, "sq_proj"), (model.wps_proj, "wps_proj")]:
        if proj is None:
            continue
        w = get(f"{prefix}.proj.0.weight")
        if w is not None:
            proj.proj0.weight = w
        w = get(f"{prefix}.proj.2.weight")
        if w is not None:
            proj.proj2.weight = w
        w = get(f"{prefix}.freqs")
        if w is not None:
            proj.freqs = w

    # Decoder blocks
    for i, block in enumerate(model.decoder.blocks):
        p = f"decoder.blocks.{i}"
        block.attn_norm.weight = get(f"{p}.attn_norm.weight")
        block.attn.Wqkv.weight = get(f"{p}.attn.Wqkv.weight")
        block.attn.out_proj.weight = get(f"{p}.attn.out_proj.weight")
        block.mlp_norm.weight = get(f"{p}.mlp_norm.weight")
        block.mlp.w1.weight = get(f"{p}.mlp.w1.weight")
        block.mlp.w3.weight = get(f"{p}.mlp.w3.weight")
        block.mlp.w2.weight = get(f"{p}.mlp.w2.weight")
    model.decoder.norm.weight = get("decoder.norm.weight")

    # Speaker projection
    w = get("spk_proj.weight")
    if w is not None and model.spk_proj is not None:
        model.spk_proj.weight = w

    # RQ transformer
    rq = model.rq_transformer
    w = get("rq_transformer.backbone_proj.weight")
    if w is not None and rq.backbone_proj is not None:
        rq.backbone_proj.weight = w
    rq.code_emb.embedding.weight = get("rq_transformer.code_emb.embedding.weight")
    rq.pos_emb.weight = get("rq_transformer.pos_emb.weight")
    rq.head_W = get("rq_transformer.head_W")
    for i, block in enumerate(rq.blocks):
        p = f"rq_transformer.blocks.{i}"
        block.attn_norm.weight = get(f"{p}.attn_norm.weight")
        block.Wqkv.weight = get(f"{p}.Wqkv.weight")
        block.out_proj.weight = get(f"{p}.out_proj.weight")
        block.mlp_norm.weight = get(f"{p}.mlp_norm.weight")
        block.w1.weight = get(f"{p}.w1.weight")
        block.w3.weight = get(f"{p}.w3.weight")
        block.w2.weight = get(f"{p}.w2.weight")
    rq.norm.weight = get("rq_transformer.norm.weight")

    mx.eval(model.parameters())
