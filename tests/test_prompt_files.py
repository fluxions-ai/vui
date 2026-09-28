"""Loading an official voice prompt as torch tensors, from a local folder."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vui.prompt_files import load_official_prompt

D = 8
Q = 16


def _write_prompt(folder, voice, *, text="Hello there.", cond_bias=True):
    tensors = {
        "codes": torch.randint(0, 2048, (30, Q), dtype=torch.int32),
        "spk_token_emb": torch.randn(1, 1, D, dtype=torch.bfloat16),
    }
    if cond_bias:
        tensors["cond_bias"] = torch.randn(1, 1, D, dtype=torch.bfloat16)
    metadata = {"config": json.dumps({"text": text})} if text else None
    save_file(tensors, str(folder / f"{voice}.safetensors"), metadata=metadata)
    return tensors


def test_an_official_prompt_loads_as_torch_tensors(tmp_path):
    saved = _write_prompt(tmp_path, "maeve")

    text, codes, spk_token, cond_bias = load_official_prompt("maeve", prompt_dir=tmp_path)

    assert text == "Hello there."
    assert codes.dtype == torch.long and torch.equal(codes, saved["codes"].long())
    assert spk_token.shape == (1, 1, D) and spk_token.dtype == torch.float32
    assert torch.equal(cond_bias, saved["cond_bias"].float())


def test_a_prompt_without_a_baked_bias_gives_none(tmp_path):
    _write_prompt(tmp_path, "maeve", cond_bias=False)

    *_, cond_bias = load_official_prompt("maeve", prompt_dir=tmp_path)

    assert cond_bias is None


def test_a_prompt_without_a_transcript_is_refused(tmp_path):
    _write_prompt(tmp_path, "maeve", text="")

    with pytest.raises(FileNotFoundError, match="no transcript"):
        load_official_prompt("maeve", prompt_dir=tmp_path)
