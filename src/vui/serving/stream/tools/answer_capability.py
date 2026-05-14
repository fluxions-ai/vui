"""Answer capability questions ('can you access X?', 'what tools do you have?')."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "answer_capability",
        "description": (
            "Answer a question about what the assistant can do, based ONLY "
            "on the AVAILABLE TOOLS list in your system prompt. Use for: "
            "'can you access my emails?', 'do you have Slack?', "
            "'what can you do?', 'what tools do you have?'. Provide a short, "
            "spoken-style sentence with conversational fillers. If asked "
            "about a single tool, say yes/no briefly. If asked broadly, "
            "list the tools naturally."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Spoken reply, short and natural. E.g. "
                        "'Yeah, Slack is good.' or 'No, no calendar I'm afraid.' "
                        "or 'Yeah, I've got email, calendar, Slack, Drive, web "
                        "search, plus I can remember things and manage tasks.'"
                    ),
                }
            },
            "required": ["text"],
        },
    },
}

RULE = """\
answer_capability: questions ABOUT what the assistant can do — "can you access my emails?", "are you able to search?", "do you have Slack?", "what can you do?", "what tools do you have?", "do you have access to X?".
- These ask IF you can — they do NOT ask you to DO it. NEVER delegate.
- Look at AVAILABLE TOOLS in your system prompt. If the asked-about thing is in the list, say yes briefly. If not, say no briefly.
- For broad "what can you do" / "what tools do you have" questions, list the tools naturally in one short spoken sentence.
- Use conversational fillers ("yeah", "um", "hmm").
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    text = (args.get("text") or "").strip()
    if not text:
        return
    srv = ctx.srv
    await ctx._wait_generation_done()
    await ctx._speak(text)
    srv.session.ready = not srv._ready_blockers
