"""Hand off a request to the Claude task server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Hand off a specific, actionable request to Claude: check "
            "emails, read calendar, send slack message, search the web, "
            "write code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A direct instruction, e.g. 'Check Harry's unread emails' or 'Search the web for X'. Not a research brief.",
                },
            },
            "required": ["task"],
        },
    },
}

RULE = """\
delegate:
- User makes a direct request needing LIVE/EXTERNAL data or actions the assistant cannot answer from training: check emails, read calendar, search the web for current info, send messages, look up live weather/news/scores/prices, write+execute code, read files.
- If the assistant said "let me check", "let me look", "pulling up", "hold on", "give me a sec" about live data -> MUST delegate. This includes hedged answers like "I think it's X but let me check" — the lookup language at the end means delegate.
- If the assistant said "I can't do that" or "I don't have access" for something that requires external access -> MUST delegate.
- NEVER delegate general knowledge (recipes, how-to, science, history, coding explanations).
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    task_desc = args.get("task", "")
    _slog(f"[thoughts] delegating: '{task_desc[:60]}'")

    for tinfo in srv._tasks.values():
        if (
            tinfo.get("status") == "running"
            and tinfo.get("description", "").lower() == task_desc.lower()
        ):
            _slog("[thoughts] duplicate task, skipping")
            return

    task_id = await srv._create_task(task_desc, description=task_desc)
    if task_id:
        ws = srv.session.ws
        if ws and not ws.closed:
            await ws.send_json({"type": "status", "text": f"Task {task_id} running..."})
        srv._pending_task_id = task_id
