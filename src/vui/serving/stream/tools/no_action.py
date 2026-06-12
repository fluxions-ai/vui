"""Default: no tool action — let the conversation stream handle this turn."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "no_action",
        "description": "No action needed. The conversation stream handles this turn.",
        "parameters": {"type": "object", "properties": {}},
    },
}

# RULE intentionally empty — the no_action policy lives in the static
# preamble of `_THOUGHTS_PROMPT` (it's about the overall decision, not a
# per-tool firing rule).
RULE = ""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    return
