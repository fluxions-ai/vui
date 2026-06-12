"""Adjust the assistant's speaking rate (the `wps_score` TTS setting)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


# Voice-controlled clamp. The slider spans 0-5, but voice nudges stay
# between 2 (slowest comfortable) and 4 (fastest before words mangle).
# 0 = "follow the prompt's natural pace" — this is the default state and
# what "normal pace" resets to. When nudging from 0 we treat 3 as the
# starting point.
WPS_MIN = 2.0
WPS_MAX = 4.0
WPS_NORMAL = 0.0
WPS_NUDGE_BASE = 3.0
WPS_STEP = 0.5


SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_speech_rate",
        "description": (
            "Change the assistant's speaking rate. Use when the user asks "
            "the assistant to talk faster, slower, or at a specific pace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["faster", "slower", "normal", "absolute"],
                    "description": (
                        "'faster' to nudge the rate up, 'slower' to nudge it "
                        "down, 'normal' to reset to the natural prompt pace, "
                        "'absolute' to set a specific wps value."
                    ),
                },
                "wps": {
                    "type": "number",
                    "description": (
                        "Required when direction='absolute'. Words per second "
                        "target (typical conversational range 2–4)."
                    ),
                },
            },
            "required": ["direction"],
        },
    },
}


RULE = """\
set_speech_rate: user asks the assistant to talk faster/slower or at a specific pace.
- Trigger phrases: "speed up", "talk faster", "be quicker", "slow down", "talk slower", "more slowly", "you're talking too fast/slow", "speak at a normal pace", "back to normal", "talk normally".
- For relative requests pass direction='faster' or 'slower' (no wps needed).
- For "normal/default pace" reset pass direction='normal'.
- For specific rates ("talk at 3 words per second") pass direction='absolute' and wps=<number>.
- NOT for asking the user to slow down or speeding up a task — only for the assistant's own speech.
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    direction = (args.get("direction") or "").strip().lower()
    current = float(srv.session.settings.get("wps_score") or 0.0)
    base = current if current > 0 else WPS_NUDGE_BASE

    if direction == "faster":
        new = min(WPS_MAX, base + WPS_STEP)
    elif direction == "slower":
        new = max(WPS_MIN, base - WPS_STEP)
    elif direction == "normal":
        new = WPS_NORMAL
    elif direction == "absolute":
        try:
            new = float(args.get("wps"))
        except (TypeError, ValueError):
            return
        new = max(WPS_MIN, min(WPS_MAX, new))
    else:
        return

    new = round(new * 2) / 2  # snap to slider step (0.5)
    if new == current:
        return

    srv.session.settings["wps_score"] = new
    _slog(f"[set_speech_rate] wps_score {current} -> {new} ({direction})")

    # cond_bias is fixed at prompt-prefill; push set_cond so the change
    # applies to text generated from here on (past KV stays at old wps).
    srv.tts_cmd_queue.put(
        {
            "cmd": "set_cond",
            "wps": new,
            "sq": srv.session.settings.get("sq_scores"),
        }
    )

    # Push to the UI so the slider reflects the new value.
    ws = srv.session.ws
    if ws and not ws.closed:
        asyncio.ensure_future(
            ws.send_json({"type": "settings", "settings": srv.session.settings})
        )
