"""Generate a new thoughts-stream tool live, by delegating to claude-task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


# Path-of-this-file -> repo root. propose_tool.py lives at
# <repo>/src/vui/serving/stream/tools/propose_tool.py, so 5 parents up is
# the repo root. We pass this as `cwd` to the codegen task so Claude's
# relative paths (`src/vui/serving/stream/tools/...`) resolve.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_TOOLS_REL = "src/vui/serving/stream/tools"


SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_tool",
        "description": (
            "Generate a NEW thoughts-stream tool by delegating codegen to "
            "Claude. Use ONLY when the user asks you to add/build/create a "
            "new capability or tool — not for normal task delegation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "snake_case tool name, e.g. 'set_timer', 'toggle_light'.",
                },
                "description": {
                    "type": "string",
                    "description": "What the tool should do, in plain English. Pass the user's request through.",
                },
            },
            "required": ["name", "description"],
        },
    },
}

RULE = """\
propose_tool: user explicitly asks you to ADD / BUILD / CREATE / WRITE a new tool, capability, or feature.
- Triggers: "add a tool that ...", "can you make a tool to ...", "build me a ...", "I want a tool for ...", "write a new tool that ...".
- Pick a snake_case name from the request (e.g. "set a timer" -> "set_timer").
- NOT for normal requests — "check my emails" is delegate, not propose_tool.
- NOT for fixing/changing an existing tool (that's a code task, ask the user to do it manually).
"""


_CODEGEN_SYSTEM_PROMPT = """\
You are extending a voice assistant by adding ONE new tool to its thoughts-stream router.

1. Read {tools_rel}/SPEC.md — that is the authoring contract. Follow it exactly.
2. Read {tools_rel}/add_memory.py — a small working example of the file shape.
3. Write the new tool to {tools_rel}/{name}.py per the contract.
4. Stop. Do not run anything else, do not test, do not modify other files.

Constraints:
- The file MUST export SCHEMA, RULE, and `async def handle(ctx, **args)`.
- SCHEMA.function.name MUST equal "{name}" (matches the filename).
- RULE should be 2-5 lines: trigger phrases plus 1-2 negative cases.
- Keep the handler small. Side-effects only; no long-running blocking work.
- Do NOT modify SPEC.md, add_memory.py, or any other tool file.

Reply with one short sentence confirming what you wrote (e.g. "Wrote {name}.py — fires on ..."). This sentence will be spoken aloud by TTS, so keep it conversational.
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    srv = ctx.srv
    name = (args.get("name") or "").strip()
    description = (args.get("description") or "").strip()
    if not name or not description:
        _slog("[propose_tool] missing name or description, skipping")
        return

    # Hygiene: snake_case identifier only. Reject anything else so a typo'd
    # name from the LLM can't write a weirdly-named file.
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        _slog(f"[propose_tool] rejected name {name!r}")
        return

    target = _REPO_ROOT / _TOOLS_REL / f"{name}.py"
    if target.exists():
        _slog(f"[propose_tool] {name}.py already exists, replying instead")
        await ctx._cancel_conversation()
        await ctx._speak(
            f"I've already got something similar — a {name.replace('_', ' ')} tool. Try asking me to use it directly."
        )
        return

    system_prompt = _CODEGEN_SYSTEM_PROMPT.format(tools_rel=_TOOLS_REL, name=name)
    user_prompt = f"User request: {description}\n\nWrite the tool now."

    _slog(f"[propose_tool] delegating codegen for '{name}': {description[:60]}")

    task_id = await srv._create_task(
        user_prompt,
        description=f"Adding tool: {name}",
        system_prompt=system_prompt,
        cwd=str(_REPO_ROOT),
        max_turns=8,
        is_codegen=True,
    )

    if task_id:
        ws = srv.session.ws
        if ws and not ws.closed:
            await ws.send_json(
                {"type": "status", "text": f"Writing tool '{name}'..."}
            )
        srv._pending_task_id = task_id
