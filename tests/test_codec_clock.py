"""CodecCtx's 10 s decoder clock, CPU-only.

Training encoded each recording in independent 10 s chunks, so the streaming
decoder must hold exactly the frames since the last absolute 10 s boundary:
every streaming session starts on a boundary and is fed contiguous frames.
The fake decoder's codes carry their absolute frame index, so a session that
skipped the user's frames, or started off a boundary, shows up in the record.
"""

from contextlib import contextmanager

import torch

from vui.qwen_codec import CodecCtx

Q = 16


class _Decoder:
    """Records the frames each streaming session is fed."""

    def __init__(self):
        self.sessions: list[list[int]] = []

    @contextmanager
    def streaming(self, batch_size: int):
        self.sessions.append([])
        yield

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        self.sessions[-1].extend(codes[0, 0].tolist())
        return torch.zeros(1, 1, codes.shape[2])


def _frames(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n).repeat(1, Q, 1)


def _reply(ctx: CodecCtx, start: int, n: int) -> None:
    """What `Engine._stream_row` does: prefill if the state is closed, then decode."""
    if ctx._stack is None:
        ctx.prefill(device="cpu")
    for t in range(start, start + n):
        ctx.decode_frame(_frames(t, 1))


def test_user_codes_count_towards_the_10s_clock():
    dec = _Decoder()
    ctx = CodecCtx(dec)
    # A 12 s prompt: the buffer trim must still keep 10 s of tail after it.
    pos = 150
    ctx.set_prompt(_frames(0, pos))
    for reply, user in [(80, 40), (100, 90), (130, 30), (60, 0)]:
        _reply(ctx, pos, reply)
        pos += reply
        if user:
            ctx.add(_frames(pos, user))
            pos += user

    sessions = [s for s in dec.sessions if s]
    assert sessions
    for s in sessions:
        assert s[0] % ctx.max_ctx == 0, f"session starts off a boundary at {s[0]}"
        assert s == list(range(s[0], s[0] + len(s))), f"session {s[0]} skips frames"
        assert len(s) <= ctx.max_ctx
    assert sessions[-1][-1] == pos - 1


def test_set_prompt_and_reset_restart_the_clock():
    dec = _Decoder()
    ctx = CodecCtx(dec)
    ctx.set_prompt(_frames(0, 100))
    _reply(ctx, 100, 50)
    ctx.add(_frames(150, 20))

    ctx.set_prompt(_frames(0, 100))
    assert ctx._abs_frames == 100
    _reply(ctx, 100, 20)
    assert dec.sessions[-1] == list(range(0, 120))

    ctx.reset()
    assert ctx._abs_frames == 0
    _reply(ctx, 0, 10)
    assert dec.sessions[-1] == list(range(0, 10))
