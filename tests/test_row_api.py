"""Row / Engine behaviour that needs no GPU: KV truncation, speaker tokens,
cond_bias, user-turn logging.

The engine is faked down to the one row's KV length and its codec context, so
this runs on any box, CPU-only CI included.
"""

from types import SimpleNamespace

import pytest
import torch

from vui.engine import Engine, Row

Q = 16


class _FakeEngine:
    """The KV bookkeeping a Row drives: one row's seq_len, no model."""

    _rewind_row = Engine._rewind_row

    def __init__(self):
        self.codec = None
        kv = SimpleNamespace(seq_lens=torch.zeros(1, dtype=torch.int32))
        self.model = SimpleNamespace(decoder=SimpleNamespace(flash_kv_caches=[kv]))


def _row(written: int, prompt_offset: int = 40) -> Row:
    """A row that has written `written` KV positions, the prompt ending at `prompt_offset`."""
    engine = _FakeEngine()
    row = Row(engine, 0)
    engine.model.decoder.flash_kv_caches[0].seq_lens[0] = written
    row._prompt_offset = prompt_offset
    row._prompt_codes = torch.zeros(1, Q, 25, dtype=torch.long)
    row._codec_ctx._buf = torch.ones(1, Q, 60, dtype=torch.long)
    return row


# ---------------------------------------------------------------- truncate


def test_truncate_moves_the_kv_back_and_leaves_the_codec_buffer():
    row = _row(written=100)
    buf = row._codec_ctx._buf

    assert row.truncate(70) == 70
    assert row.offset == 70
    assert row._codec_ctx._buf is buf


def test_truncate_to_the_end_of_the_prompt_reseeds_the_codec_like_rewind():
    row = _row(written=100, prompt_offset=40)

    assert row.truncate(40) == 40
    assert row.offset == 40
    assert row._codec_ctx._buf.shape == (1, Q, 25)


def test_truncate_to_zero_empties_the_row_like_reset():
    row = _row(written=100)

    assert row.truncate(0) == 0
    assert row.offset == 0
    assert row._codec_ctx._buf is None
    assert row._prompt_codes is None


def test_truncate_to_the_current_offset_is_a_no_op():
    row = _row(written=100)

    assert row.truncate(100) == 100
    assert row.offset == 100


@pytest.mark.parametrize("offset", [-1, 101])
def test_truncate_refuses_an_offset_the_row_has_not_written(offset):
    row = _row(written=100)

    with pytest.raises(ValueError, match="outside the row's KV"):
        row.truncate(offset)
    assert row.offset == 100


# ----------------------------------------------------------- speaker token

D = 8
SPK_DIM = 1024  # QwenSpeakerEncoder.embed output


class _FakeSpeakerEngine:
    """What the prefill path needs to turn a speaker input into a token."""

    _embed_speaker = Engine._embed_speaker

    def __init__(self, with_proj: bool = True):
        self.D = D
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        proj = torch.nn.Linear(SPK_DIM, D) if with_proj else None
        self.model = SimpleNamespace(
            spk_proj=proj,
            embed_speaker=lambda emb: proj(emb).reshape(1, 1, -1),
        )


def test_a_raw_speaker_embedding_is_projected_to_a_token():
    engine = _FakeSpeakerEngine()

    token = engine._embed_speaker(torch.randn(SPK_DIM))

    assert token.shape == (1, 1, D)
    assert token.dtype == torch.bfloat16


def test_a_pre_projected_token_is_used_as_is():
    engine = _FakeSpeakerEngine()
    spk_token_emb = torch.randn(1, 1, D)

    token = engine._embed_speaker(spk_token_emb)

    assert token.dtype == torch.bfloat16
    assert torch.equal(token, spk_token_emb.to(torch.bfloat16))


def test_no_speaker_input_gives_no_token():
    assert _FakeSpeakerEngine()._embed_speaker(None) is None


def test_a_raw_embedding_gives_no_token_without_a_projection():
    engine = _FakeSpeakerEngine(with_proj=False)

    assert engine._embed_speaker(torch.randn(SPK_DIM)) is None


# --------------------------------------------------------------- cond_bias


class _FakeBiasEngine:
    cond_bias = Engine.cond_bias

    def __init__(self, inference_buffer: bool = False):
        if inference_buffer:  # the model's buffer when built under inference_mode
            with torch.inference_mode():
                self._cond_bias = torch.zeros(1, 1, D, dtype=torch.bfloat16)
        else:
            self._cond_bias = torch.zeros(1, 1, D, dtype=torch.bfloat16)


@pytest.mark.parametrize("inference_buffer", [False, True])
def test_setting_cond_bias_copies_into_the_model_buffer(inference_buffer):
    engine = _FakeBiasEngine(inference_buffer)
    buffer = engine._cond_bias
    baked = torch.randn(D)

    engine.cond_bias = baked

    assert engine.cond_bias is buffer
    assert torch.equal(buffer, baked.reshape(1, 1, D).to(torch.bfloat16))


def test_setting_cond_bias_to_none_zeroes_it():
    engine = _FakeBiasEngine()
    engine.cond_bias = torch.ones(1, 1, D)

    engine.cond_bias = None

    assert not engine.cond_bias.any()


# ---------------------------------------------------------------- add_user


class _FakeUserEngine(_FakeEngine):
    """Writes a user turn's embeddings by advancing the row's KV length."""

    _add_user = Engine._add_user

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self._sc_emb = torch.zeros(1, 1, D)

    def _text_emb(self, text, with_cond_bias, noisy=False):
        return torch.zeros(1, len(text.split()), D)

    def _audio_emb(self, codes):
        return torch.zeros(1, codes.shape[0], D)

    def _prefill_emb(self, row, emb):
        self.model.decoder.flash_kv_caches[0].seq_lens[row.idx] += emb.shape[1]


def test_a_user_turn_is_logged_at_debug_level_not_printed(capsys, caplog):
    engine = _FakeUserEngine()
    row = Row(engine, 0)

    with caplog.at_level("DEBUG", logger="vui.engine"):
        row.add_user("my account number is", torch.zeros(5, Q, dtype=torch.long))

    assert row.offset == 4 + 1 + 5
    assert capsys.readouterr().out == ""
    assert "T=0->10" in caplog.text
