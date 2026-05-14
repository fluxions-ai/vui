"""Store a durable personal fact about the user."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog
from vui.serving.stream.memories import add_memory

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_memory",
        "description": (
            "Store a PERSISTENT fact about who the user IS: name, job, "
            "location, family, pets, allergies, preferences. NOT for what "
            "they did today or transient events. IMPORTANT: if an existing "
            "memory covers the same topic (e.g. daughter's age, job title), "
            "set replaces to the old memory text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Short factual statement, e.g. 'Name is Harry', 'Dog Max passed away'",
                },
                "replaces": {
                    "type": "string",
                    "description": "Old memory text to replace, if updating",
                },
            },
            "required": ["text"],
        },
    },
}

RULE = """\
add_memory:
- ONLY for durable personal facts true weeks from now: name, job title, location, family, pets, allergies, preferences, significant life events (job loss, bereavement, new baby).
- Test: "will this still be true in a month?" If not -> no_action.
- NOT for: activities ("made pasta", "crashed my bike", "finished a book"), plans ("meeting cancelled"), moods ("I'm tired"), thoughts ("thinking about switching jobs").
- Text: short factual statement ("Name is Harry", "Allergic to nuts"). If new info UPDATES or contradicts an existing memory (age changed, job changed, moved city), you MUST set replaces to the old memory text so it gets replaced, not duplicated.
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    result = add_memory(
        ctx.srv, args.get("text", ""), replaces=args.get("replaces")
    )
    _slog(f"[thoughts] memory: {result}")
