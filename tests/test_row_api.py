"""Row / Engine public API that needs no GPU: KV truncation.

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
