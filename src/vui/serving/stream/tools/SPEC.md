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
5. **`TASK`** (optional) opts the tool into the UI tasks panel — see below.

## Surfacing as a task (optional `TASK` block)

Tools fire silently by default. To make a call visible — countdown, query, result, with the same × cancel affordance as `delegate` — export a `TASK` dict:

```python
TASK = {
    "surface": True,        # required for the row to appear
    "auto_done": True,      # default — mark row "done" when `handle()` returns
}
```

When `TASK` is present, the dispatcher allocates a row *before* invoking `handle()` and attaches it to `ctx.task` (a `LocalTask`). The tool reads/writes the row through that handle:

| Call | Effect |
|---|---|
| `ctx.task.update(description=..., status=..., result=..., error=...)` | Push any subset of fields to `srv._tasks[task_id]` and emit a `task_update` ws event. |
| `ctx.task.on_cancel(cb)` | Register a sync/async callback to run when the user cancels (voice or × button). Typically `lambda: bg_task.cancel()`. |

The initial description is auto-derived from the first non-empty arg matching `subject`/`query`/`task`/`description`/`label`, falling back to the tool name. Override it with `ctx.task.update(description=...)` if you want something prettier (e.g. `"pasta timer (300s)"`).

`auto_done: False` keeps the row "running" after `handle()` returns — use it when the work continues in a spawned background task (e.g. `set_timer`'s countdown). The background task is responsible for calling `ctx.task.update(status="done"/"cancelled"/"error", result=...)` when it finishes.

Tools without a `TASK` block see `ctx.task is None` and run invisibly (existing behaviour).

### Status values & limits

`LocalTask.update(status=...)` accepts any string, but the dispatcher, UI, and duplicate-detection logic in `thoughts.py` only understand four:

| Status | When to set it |
|---|---|
| `running` | Default on creation. Nothing to set manually. |
| `done` | The unit of work finished successfully. Pair with `result=...` to attach the text the relay LLM will paraphrase. With `auto_done=True` (the default) the dispatcher sets this for you. |
| `cancelled` | User-driven abort — the `on_cancel` callback fires, then your background task should set this. |
| `error` | Anything that failed. Pair with `error="..."` for a short message; an uncaught exception inside `handle()` is auto-mapped to `error` for you. |

Other gotchas:

- **Descriptions are truncated to 100 chars** in `start_local_task` and on every `update(description=...)` (`tasks.py:89, 121`). Anything longer gets clipped silently.
- **Cancelled/error rows don't gate duplicate detection** — the next user request that fuzzy-matches a `cancelled` row will start fresh work, not get suppressed (`thoughts.py:_build_tasks_context`).
- **`result` text feeds the relay LLM** verbatim — the conversation LLM paraphrases it into a spoken reply. Keep it concise and speakable; no markdown, no digit symbols, no URLs you don't want read aloud.

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
                "subject": {"type": "string", "description": "What the timer is for, e.g. 'pasta'."},
            },
            "required": ["seconds", "subject"],
        },
    },
}

RULE = """\
set_timer: user says "set a timer for X", "remind me in X minutes", "X minute timer".
- Convert all units to seconds before calling.
- NOT for calendar reminders (those are delegate).
"""

# Show the timer as a UI task row so the user can cancel it; keep the row
# "running" until `_fire` finishes (auto_done=False).
TASK = {"surface": True, "auto_done": False}


async def handle(ctx, **args) -> None:
    seconds = int(args.get("seconds", 0) or 0)
    subject = (args.get("subject") or "reminder").strip()
    task = ctx.task
    if seconds <= 0:
        if task:
            task.update(status="error", error="invalid duration")
        return
    desc = f"{subject} timer"
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
            task.update(status="done", result=f"{desc} fired")
        await ctx._speak(f"Your {desc} is up.")

    fire_task = _spawn(_fire(), f"timer_{desc}")
    if task:
        task.on_cancel(lambda: fire_task.cancel())
```

Drop the file in this directory, then `curl -X POST http://localhost:8080/tools/reload` (or restart). Say *"set a timer for 30 seconds"* — done.

## Reload semantics

- `POST /tools/reload` re-walks this directory, re-imports every file, rebuilds the registry. The next thoughts-stream call sees the new tool list.
- A broken file (import error, missing export, name mismatch) is logged and skipped — other tools keep working.
- Removing a file removes the tool on the next reload. In-flight thoughts evaluations finish on the old registry; no need to drain.

## When NOT to write a tool here

- **Single-query factual web lookups** — already covered by the built-in `web_search` tool (Serper / Brave / Tavily backends). Don't write a parallel one.
- **Multi-step research or anything needing MCP integrations** (Gmail, Calendar, Slack, …) — use `delegate` and let `claude-task` handle it.
- **Anything that needs multi-step reasoning** — `handle()` runs once per turn, no agent loop.
- **State you'd want UI affordances for** — declare `TASK` (above) instead of building parallel tracking; the existing tasks panel handles create/cancel/list/check.

See `docs/thoughts-tools.md` for the full overview.
