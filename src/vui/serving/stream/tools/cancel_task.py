"""Cancel a running background task without removing it from the list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream.tasks import handle_cancel_task_voice

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "cancel_task",
        "description": (
            "Cancel a specific running background task. Use for 'stop X', "
            "'cancel X', 'never mind X'. Leaves the task in the list with "
            "status=cancelled — use delete_task to also remove from the list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Which task to cancel, matched by description",
                },
            },
            "required": ["description"],
        },
    },
}

RULE = "cancel_task: user wants to STOP a specific running task. Triggers: \"cancel that\", \"stop the X task\", \"never mind the search\", \"abort it\". Leaves the entry visible with status=cancelled."


async def handle(ctx: "ThoughtsStream", **args) -> None:
    await handle_cancel_task_voice(ctx.srv, args.get("description", ""))
