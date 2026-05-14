"""Wipe all background tasks in one go."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream.tasks import handle_clear_tasks_voice

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "clear_tasks",
        "description": (
            "Clear/delete ALL background tasks at once. Use only for 'clear "
            "all tasks', 'delete all tasks', 'wipe my tasks'. For a single "
            "task use delete_task."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

RULE = "clear_tasks: user wants to clear/delete ALL tasks at once. \"clear all tasks\", \"delete all tasks\", \"wipe my tasks\". This is about tasks, NOT memories."


async def handle(ctx: "ThoughtsStream", **args) -> None:
    await handle_clear_tasks_voice(ctx.srv)
