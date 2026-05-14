"""Reset the conversation when the user asks to start over."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "clear_context",
        "description": "Reset/clear the conversation when the user asks to start over.",
        "parameters": {"type": "object", "properties": {}},
    },
}

RULE = "clear_context: user explicitly asks to reset/clear/start over the conversation."


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    # Let the LLM finish speaking its acknowledgement BEFORE we wipe state.
    # The conversation has already been told (system prompt) to say something
    # short like "Yeah done."; cancelling it cuts that off and substitutes a
    # canned line. Waiting preserves the natural acknowledgement.
    await ctx._wait_generation_done()
    if srv.session.playback_track:
        await srv.session.playback_track.wait_drained()
    await srv._reset_session_state()
    srv.session.ready = not srv._ready_blockers
