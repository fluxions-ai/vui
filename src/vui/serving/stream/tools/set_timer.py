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


async def handle(ctx: "ThoughtsStream", **args) -> None:
    seconds = int(args.get("seconds", 0) or 0)
    subject = (args.get("subject") or args.get("label") or "").strip()
    if seconds <= 0:
        return
    if not subject:
        subject = "reminder"
    # Avoid "Your timer timer is up." when the LLM falls back to literal
    # 'timer' as the subject for an abstract request like "30 second timer".
    desc = "timer" if subject.lower() == "timer" else f"{subject} timer"
    _slog(f"[set_timer] {desc} for {seconds}s")

    async def _fire():
        await asyncio.sleep(seconds)
        await ctx._speak(f"Your {desc} is up.")

    _spawn(_fire(), f"timer_{desc}")
