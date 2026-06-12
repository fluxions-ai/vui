"""Hand off a request to the Claude task server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


# Filtered out of the evaluator's tool list when the claude-task server is
# unreachable — `_create_task` would fail otherwise.
REQUIRES_TASK_SERVER = True


SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Hand off a specific, actionable request to Claude: check "
            "emails, read calendar, send slack message, write code, "
            "multi-step research. NOT for single-query web lookups — "
            "those go to web_search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A direct instruction, e.g. 'Check Harry's unread emails' or 'Summarise today's Slack DMs'. Not a research brief.",
                },
            },
            "required": ["task"],
        },
    },
}

RULE = """\
delegate:
- User makes a direct request needing AGENTIC tool use beyond a single web search: check emails, read calendar, list/send messages, write+execute code, read files, multi-step research, anything requiring chained tool calls.
- MULTI-STEP RESEARCH briefs go here, NOT web_search. Triggers: plural entities ("top 5 X", "these 3 companies", "all the Y in Z") + verbs like "research", "compare", "summarise", "writeup", "analyse", "investigate", "deep dive". Examples: "research the top 5 EV stocks and summarise their earnings", "compare AWS vs GCP pricing in depth", "find me info on these 3 candidates and tell me which is strongest" -> delegate.
- If the assistant said "let me check", "let me look", "pulling up", "hold on", "give me a sec" about ACCOUNT/PERSONAL data (mail, calendar, files) -> MUST delegate. Plain factual lookups ("what's the weather", "who won X") go to web_search instead.
- If the assistant said "I can't do that" or "I don't have access" for something that requires external account access -> MUST delegate.
- NEVER use delegate for a single factual web lookup — that's web_search. Examples that go to web_search NOT delegate: "search the web for X", "what's the weather/news/score/price", "who is X", "look up X", "current time in X", "latest news".
- NEVER delegate general knowledge (recipes, how-to, science, history, coding explanations).
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    task_desc = args.get("task", "")
    _slog(f"[thoughts] delegating: '{task_desc[:60]}'")

    # Prefix the UI description so users can spot at a glance which rows are
    # claude-task delegations vs. local tools (timers, searches). The prompt
    # sent to the task server is still the bare `task_desc`.
    ui_desc = f"Claude - {task_desc}"
    for tinfo in srv._tasks.values():
        if (
            tinfo.get("status") == "running"
            and tinfo.get("description", "").lower() == ui_desc.lower()
        ):
            _slog("[thoughts] duplicate task, skipping")
            return

    task_id = await srv._create_task(task_desc, description=ui_desc)
    if task_id:
        ws = srv.session.ws
        if ws and not ws.closed:
            await ws.send_json({"type": "status", "text": f"Task {task_id} running..."})
        srv._pending_task_id = task_id
