"""Cancel and remove a background task from the list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream.tasks import handle_delete_task_voice

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_task",
        "description": (
            "Delete a single task from the list. Use for 'delete X', "
            "'remove that task', 'get rid of the X one'. Cancels first if "
            "running, then removes the entry entirely (different from "
            "cancel_task which leaves the entry visible)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Which task to delete, matched by description",
                },
            },
            "required": ["description"],
        },
    },
}

RULE = "delete_task: user wants to REMOVE a specific task from the list. Triggers: \"delete that one\", \"remove the X task\", \"get rid of it\". Different from cancel: also cancels if running, then removes the entry entirely. Default to delete_task when user says \"delete\" or \"remove\" a single task."


async def handle(ctx: "ThoughtsStream", **args) -> None:
    await handle_delete_task_voice(ctx.srv, args.get("description", ""))
