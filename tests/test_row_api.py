"""Row / Engine behaviour that needs no GPU: KV truncation, speaker tokens,
cond_bias, user-turn logging.

The engine is faked down to the one row's KV length and its codec context, so
this runs on any box, CPU-only CI included.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vui.engine import Engine, Row, Segment

Q = 16


class _FakeEngine:
    """The KV bookkeeping a Row drives: one row's seq_len, no model."""

    _rewind_row = Engine._rewind_row
    _drop_audio_past = Engine._drop_audio_past

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


def test_truncate_with_no_audio_past_the_offset_leaves_the_codec_buffer():
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


def test_truncate_to_the_current_offset_is_a_no_op():
    row = _row(written=100)

    assert row.truncate(100) == 100
    assert row.offset == 100


@pytest.mark.parametrize("offset", [-1, 0, 39, 101])
def test_truncate_refuses_an_offset_outside_the_prompt_end_to_the_row_offset(offset):
    row = _row(written=100, prompt_offset=40)

    with pytest.raises(ValueError, match="outside 40..100"):
        row.truncate(offset)
    assert row.offset == 100


def test_reset_forgets_the_prompt():
    row = _row(written=100, prompt_offset=40)

    row.reset()

    assert row._prompt_offset == 0
    assert row.rewind() == 0
    assert row.truncate(0) == 0


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


# ---------------------------------------------------------------- add_user


class _FakeUserEngine(_FakeEngine):
    """Writes a user turn's embeddings by advancing the row's KV length."""

    _add_user = Engine._add_user
    _mark_audio_run = Engine._mark_audio_run

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


# ------------------------------------------------------------------ prefill


class _FakePrefillEngine(_FakeUserEngine):
    """The prefill path over the fake KV: speaker token, segments, cond_bias."""

    _prefill_row = Engine._prefill_row
    _prefill_speaker_segments = Engine._prefill_speaker_segments
    _embed_speaker = Engine._embed_speaker

    def __init__(self, inference_buffer: bool = False):
        super().__init__()
        self.D = D
        self.dtype = torch.bfloat16
        self.model.spk_proj = None
        if inference_buffer:  # the model's buffer when built under inference_mode
            with torch.inference_mode():
                self._cond_bias = torch.zeros(1, 1, D, dtype=torch.bfloat16)
        else:
            self._cond_bias = torch.zeros(1, 1, D, dtype=torch.bfloat16)


def _prefill(engine, **kw) -> Row:
    """Two segments of 2 text tokens + 3 frames each."""
    row = Row(engine, 0)
    row.prefill([Segment("one two", torch.zeros(3, Q, dtype=torch.long))] * 2, **kw)
    return row


def test_a_pre_projected_token_is_written_before_each_prompt_segment():
    with_token = _prefill(_FakePrefillEngine(), spk_emb=torch.randn(1, 1, D))
    without = _prefill(_FakePrefillEngine())

    assert with_token.offset - without.offset == 2
    assert with_token._prompt_offset == with_token.offset


@pytest.mark.parametrize("inference_buffer", [False, True])
def test_prefill_copies_the_cond_bias_into_the_engine_buffer(inference_buffer):
    engine = _FakePrefillEngine(inference_buffer)
    buffer = engine._cond_bias
    baked = torch.randn(1, 1, D)

    _prefill(engine, cond_bias=baked)

    assert engine._cond_bias is buffer
    assert torch.equal(buffer, baked.to(torch.bfloat16))


def test_prefill_without_a_cond_bias_leaves_it():
    engine = _FakePrefillEngine()
    engine._cond_bias.fill_(1.0)

    _prefill(engine)

    assert (engine._cond_bias == 1).all()


# ------------------------------------------------- truncate and the codec


class _Decoder:
    """A streaming codec decoder that decodes nothing."""

    def streaming(self, batch_size: int):
        return contextlib.nullcontext()

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 1, codes.shape[2])


class _FakeStreamEngine(_FakePrefillEngine):
    """The prefill and user-turn paths, plus what `_stream_row` does to the KV and the codec."""

    def __init__(self):
        super().__init__()
        self.codec = _Decoder()

    def reply(self, row: Row, n_text: int, n_frames: int) -> list[int]:
        """Write a chunk's text, then one KV position and one codec frame per frame.

        Returns `row.offset` as each frame is yielded, as a barge-in caller notes it.
        """
        kv = self.model.decoder.flash_kv_caches[0].seq_lens
        kv[row.idx] += n_text
        self._mark_audio_run(row, row.offset)
        if row._codec_ctx._stack is None:
            row._codec_ctx.prefill(device="cpu")
        offsets = []
        for _ in range(n_frames):
            offsets.append(row.offset)
            row._codec_ctx.decode_frame(torch.zeros(1, Q, 1, dtype=torch.long))
            kv[row.idx] += 1
        return offsets


def _conversation() -> tuple[_FakeStreamEngine, Row, list[int]]:
    """A 6-frame prompt, a user turn with 5 frames, then a 20-frame reply in two chunks."""
    engine = _FakeStreamEngine()
    row = _prefill(engine)
    row.add_user("hello there", torch.zeros(5, Q, dtype=torch.long))
    offsets = engine.reply(row, n_text=3, n_frames=12) + engine.reply(row, 4, 8)
    assert row._codec_ctx._abs_frames == 6 + 5 + 20
    return engine, row, offsets


def test_truncate_takes_the_frames_past_the_offset_out_of_the_codec():
    _, row, offsets = _conversation()
    heard = 15  # into the second chunk

    row.truncate(min(offsets[heard - 1] + 1, offsets[heard]))

    ctx = row._codec_ctx
    assert ctx._abs_frames == 6 + 5 + heard
    assert ctx.n_frames == 6 + 5 + heard
    assert ctx._stack is None  # the next stream() re-seeds from the frames kept


def test_truncate_at_a_chunk_boundary_keeps_the_whole_first_chunk():
    _, row, offsets = _conversation()

    row.truncate(min(offsets[11] + 1, offsets[12]))

    assert row._codec_ctx._abs_frames == 6 + 5 + 12


def test_truncate_before_the_reply_keeps_the_user_codes():
    _, row, offsets = _conversation()

    row.truncate(offsets[0] - 3)  # the reply's first text, before its audio

    assert row._codec_ctx._abs_frames == 6 + 5


def test_truncate_into_the_user_audio_keeps_the_frames_before_the_offset():
    engine = _FakeStreamEngine()
    row = _prefill(engine)
    row.add_user("hello there", torch.zeros(5, Q, dtype=torch.long))

    row.truncate(row.offset - 2)

    assert row._codec_ctx._abs_frames == 6 + 3


def test_a_chunk_rerolled_before_any_frame_opens_one_run():
    engine, row, _ = _conversation()
    runs = len(row._audio_runs)

    engine._mark_audio_run(row, row.offset)
    engine._mark_audio_run(row, row.offset)

    assert len(row._audio_runs) == runs + 1


def test_rewind_and_prefill_forget_the_audio_runs():
    engine, row, _ = _conversation()

    row.rewind()
    assert row._audio_runs == []
    assert row._codec_ctx._abs_frames == 6

    engine.reply(row, 3, 4)
    row.reset()
    assert row._audio_runs == []
