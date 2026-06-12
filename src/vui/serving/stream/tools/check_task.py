"""Check or replay a single existing task without re-running it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream.tasks import handle_check_task_voice

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_task",
        "description": (
            "Recover or check ONE existing task — read out its current "
            "status or replay its result. Fire when the user wants info "
            "about a task they've already kicked off (status, results, "
            "details). Examples: \"is that done?\", \"is it finished?\", "
            "\"how's it going?\", \"is that ready?\", \"check the results "
            "for that task again\", \"tell me what you found again\", "
            "\"remind me what the task said\", \"what did you find?\", "
            "\"get the results from the task\", \"give me the results\", "
            "\"show me what came back\", \"read me the results\", \"what "
            "were the results?\", \"what's the status?\". Pronouns "
            "(\"that\", \"it\", \"the task\") referring to a task always "
            "mean check_task. Re-speaks the cached result for [done] "
            "tasks (no new external lookup). DO NOT use for genuinely new "
            "requests — only for asking about / replaying an existing task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Which task to check, matched by description. "
                        "If the user says 'the task' / 'that one' without "
                        "naming it, pass an empty string — the handler "
                        "picks the most-recent task."
                    ),
                },
            },
            "required": ["description"],
        },
    },
}

RULE = """\
check_task: user asks about ONE task — its status OR to recover/repeat its result.
- Trigger phrases (any of): "is that done yet?", "how's the email thing going?", "has it finished?", "is that ready?", "check the results for that task again", "tell me what you found again", "remind me what the task said", "get the results from the task", "give me the results", "show me what came back", "read me the results", "what were the results?", "what did you find?", "what's the status?".
- Pronouns "that" / "it" / "the task" referring to a task always mean check_task, NOT list_tasks (list_tasks is for "what tasks are running?", plural / status overview).
- Re-speaks the cached result from `_tasks[id].result` — no new external lookup, just replays.
- Picks the most-recent matching task; pass empty `description` when the user doesn't name a specific one.
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    await ctx._wait_generation_done()
    await handle_check_task_voice(srv, args.get("description", ""))
    srv.session.ready = not srv._ready_blockers
