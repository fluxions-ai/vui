"""Schedule a deferred announcement after N seconds."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog, _spawn

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": "Start a countdown timer. Use when the user asks to set/start a timer for a duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Duration in seconds."},
                "subject": {
                    "type": "string",
                    "description": (
                        "Short subject — what the timer is for, e.g. "
                        "'pasta', 'tea', 'meditation', 'meeting'. If the "
                        "user didn't name a subject explicitly, infer one "
                        "from context or pick a sensible default like "
                        "'kitchen' or 'reminder' — never leave it blank."
                    ),
                },
            },
            "required": ["seconds", "subject"],
        },
    },
}

RULE = """\
set_timer: user says "set a timer for X", "remind me in X minutes", "X minute timer".
- Convert all units to seconds before calling.
- ALWAYS supply a subject — what the timer is for. Pick from the user's words ("pasta timer" -> 'pasta', "remind me to call mum" -> 'call mum'). If they didn't name one, infer from context or pick a sensible default ('kitchen', 'reminder'). Never call set_timer without a subject.
- NOT for calendar reminders (those are delegate).
"""


# Show the timer as a task row so the user can see it counting + cancel it
# via the × button. `auto_done=False` because the row should stay "running"
# until `_fire` completes — the dispatcher would otherwise mark it done
# immediately when `handle()` returns.
TASK = {"surface": True, "auto_done": False}


async def handle(ctx: "ThoughtsStream", **args) -> None:
    seconds = int(args.get("seconds", 0) or 0)
    subject = (args.get("subject") or args.get("label") or "").strip()
    task = ctx.task
    if seconds <= 0:
        if task:
            task.update(status="error", error="invalid duration")
        return
    if not subject:
        subject = "reminder"
    # Avoid "Your timer timer is up." when the LLM falls back to literal
    # 'timer' as the subject for an abstract request like "30 second timer".
    desc = "timer" if subject.lower() == "timer" else f"{subject} timer"
    _slog(f"[set_timer] {desc} for {seconds}s")

    if task:
        task.update(description=f"{desc} ({seconds}s)")

    async def _fire():
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            if task:
                task.update(status="cancelled")
            raise
        if task:
            task.update(status="done", result=f"{desc} fired after {seconds}s")
        await ctx._speak(f"Your {desc} is up.")

    fire_task = _spawn(_fire(), f"timer_{desc}")
    if task:
        task.on_cancel(lambda: fire_task.cancel())
