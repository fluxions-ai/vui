"""Read out the current task list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream.tasks import handle_list_tasks_voice

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_tasks",
        "description": (
            "List ALL background tasks. Only when user asks about multiple "
            "tasks (\"show my tasks\", \"what tasks are running\")."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

RULE = "list_tasks: user asks to see ALL tasks. \"what tasks do I have\", \"show my tasks\", \"list tasks\". ONLY when asking about multiple tasks."


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    await ctx._wait_generation_done()
    await handle_list_tasks_voice(srv)
    srv.session.ready = not srv._ready_blockers
