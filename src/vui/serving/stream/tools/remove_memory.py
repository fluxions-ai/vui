"""Remove stored memories — by explicit index (preferred) or fuzzy query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog
from vui.serving.stream.memories import remove_memories_by_indices, remove_memory

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "remove_memory",
        "description": (
            "Forget one or more stored memories. Prefer `indices` — pick "
            "the specific memories from the CURRENT MEMORIES list above by "
            "their 1-based index. Use `query` only as a fallback when no "
            "index is obvious."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "1-based indices of memories to remove, as numbered "
                        "in the CURRENT MEMORIES list. For topical bulk "
                        "removals, pick every index that matches the topic."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Fallback: substring of the memory text. Use only "
                        "when indices aren't applicable."
                    ),
                },
            },
        },
    },
}

RULE = """\
remove_memory: user says to forget/remove memories.
- ALWAYS pass `indices` when possible. Read the CURRENT MEMORIES list above (each line is prefixed with [N]) and pick every index that matches the user's request.
- For topical asks ("forget anything about my job", "remove memories about work"): pick ALL indices related to the topic — usually multiple.
- For single-memory removal ("forget I have a dog"): pick the one matching index.
- Use `query` only when there's no visible memory list to pick from.
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    indices = args.get("indices")
    if isinstance(indices, list) and indices:
        result = remove_memories_by_indices(ctx.srv, indices)
    else:
        result = remove_memory(ctx.srv, args.get("query", ""))
    _slog(f"[thoughts] {result}")
