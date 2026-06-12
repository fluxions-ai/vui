"""Wipe all stored memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog
from vui.serving.stream.memories import clear_memories

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "clear_memories",
        "description": "Delete all stored memories.",
        "parameters": {"type": "object", "properties": {}},
    },
}

RULE = "clear_memories: ONLY when user explicitly says to clear/wipe/delete all MEMORIES (not tasks)."


async def handle(ctx: "ThoughtsStream", **args) -> None:
    result = clear_memories(ctx.srv)
    _slog(f"[thoughts] memories cleared: {result}")
