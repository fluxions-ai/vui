# Thoughts-stream tool spec

A "thoughts tool" is one Python file in `src/vui/serving/stream/tools/`. The local thinking LLM sees the SCHEMA, picks one tool per turn, and the streaming server runs your `handle()`.

**Three required exports.** That's it.

```python
"""<one-line description>"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",                                # MUST match the filename
        "description": "<short, model-facing>",
        "parameters": {
            "type": "object",
            "properties": {
                "foo": {"type": "string", "description": "..."},
            },
            "required": ["foo"],
        },
    },
}

RULE = """\
my_tool: <when this tool fires, in plain language the LLM reads>.
- Trigger phrase examples ("do X", "set Y").
- NOT for: <negative cases that would otherwise look similar>.
"""


async def handle(ctx: "ThoughtsStream", **args) -> None:
    foo = args.get("foo", "")
    # ... do the thing
```

## Rules of the contract

1. **Filename = tool name.** `my_tool.py` defines `my_tool`. The loader rejects mismatches.
2. **`SCHEMA`** is OpenAI-style function-call JSON. The LLM only sees this — make `description` directive ("Use when…") not narrative.
3. **`RULE`** is appended into the system prompt under "RULES — read carefully:". One short block, optionally with negative cases. Skip it (set `RULE = ""`) if your `description` is unambiguous on its own.
4. **`handle(ctx, **args)`** is `async`. `args` come straight from the model's tool call — be defensive (`args.get("x", default)`).

## What `ctx` gives you

`ctx` is a `ThoughtsStream` instance. Common reach-throughs:

| Access | What it is |
|---|---|
| `ctx.srv` | The `StreamServer` — full session state, conversation, tasks dict, memory store, websocket. |
| `await ctx._speak(text)` | Announce text via the relay pipeline: the conversation LLM paraphrases your string into a natural utterance, speaks it, and appends the paraphrased reply to history. Waits for the system to be idle before firing. **Use sparingly** — most tools should let the conversation LLM's normal reply do the talking. |
| `await ctx._cancel_conversation()` | Cancel the in-flight conversation reply (rare; only when the side-effect makes the reply wrong). |
| `await ctx._wait_generation_done()` | Wait for current TTS to finish naturally. Use before tools that *replace* what's being said (e.g. `clear_context`). |

For server-state mutations, prefer existing helpers (`vui.serving.stream.memories.add_memory`, `vui.serving.stream.tasks.create_task`, etc.) — `handle()` should mostly be a thin adapter.

## Minimal example: `set_timer`

```python
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
                "label": {"type": "string", "description": "Optional label, e.g. 'pasta'."},
            },
            "required": ["seconds"],
        },
    },
}

RULE = """\
set_timer: user says "set a timer for X", "remind me in X minutes", "X minute timer".
- Convert all units to seconds before calling.
- NOT for calendar reminders (those are delegate).
"""


async def handle(ctx, **args) -> None:
    seconds = int(args.get("seconds", 0) or 0)
    label = (args.get("label") or "").strip()
    if seconds <= 0:
        return
    desc = f"{label} timer" if label else "timer"
    _slog(f"[set_timer] {desc} for {seconds}s")

    async def _fire():
        await asyncio.sleep(seconds)
        await ctx._speak(f"Your {desc} is up.")

    _spawn(_fire(), f"timer_{desc}")
```

Drop the file in this directory, then `curl -X POST http://localhost:8080/tools/reload` (or restart). Say *"set a timer for 30 seconds"* — done.

## Reload semantics

- `POST /tools/reload` re-walks this directory, re-imports every file, rebuilds the registry. The next thoughts-stream call sees the new tool list.
- A broken file (import error, missing export, name mismatch) is logged and skipped — other tools keep working.
- Removing a file removes the tool on the next reload. In-flight thoughts evaluations finish on the old registry; no need to drain.

## When NOT to write a tool here

- **Anything needing live web/MCP data** — use `delegate` and let `claude-task` handle it.
- **Anything that needs multi-step reasoning** — `handle()` runs once per turn, no agent loop.
- **State you'd want UI affordances for** — first-class tasks (`tasks.py`) already have create/cancel/list/check; don't reinvent them as one-shot tools.

See `docs/thoughts-tools.md` for the full overview.
