# The thoughts stream

The **thoughts stream** (`src/vui/serving/stream/thoughts.py`) is a parallel LLM call running on the local Ollama model that already serves the conversation. On every user turn it gets the conversation + memories + current tasks and **must call exactly one tool**. Most turns it picks `no_action` and the conversation LLM speaks normally; otherwise it routes to a memory op, a task op, or `delegate` (hand off to `claude-task`).

The soul shapes how the assistant *talks*; the thoughts stream decides what it *does*. Same Ollama model, different prompt, never speaks. Forced to emit exactly one tool call from the registry.

## How the prompt is assembled

`ThoughtsStream._build_system_prompt` composes the prompt from four sources:

1. **Preamble** (`_THOUGHTS_PREAMBLE`) — the "you are an internal reasoner" framing plus the global rules: when to pick `no_action` (most turns; casual chat, general knowledge, follow-ups answerable from context), when to delegate, how to interpret ASR errors ("male" → "email"), and the hard requirement to call exactly one tool with no free text.
2. **AVAILABLE TOOLS list** — the local capabilities (`remember things…`, `manage background tasks…`) plus the live capability groups fetched from the optional Claude task server. This is what `answer_capability` reads from when the user asks "do you have access to my Slack?".
3. **CURRENT MEMORIES** — the same memory list the soul sees, so the evaluator can answer "what do you know about me?" with `no_action` (the conversation LLM handles it from context).
4. **Per-tool RULES** — concatenated `RULE` strings from every registered tool (`tools_registry.rules_block()`). Each tool ships its own one-paragraph block of when-to-pick-this-and-when-not-to examples; the registry stitches them into the system prompt at boot.

A second system message — **CURRENT TASKS** (`_TASKS_CONTEXT`) — is appended only when there are running or completed tasks. Each `[done]` entry includes a 300-char result excerpt so follow-up questions ("what was the second one?", "tell me more") map to `no_action` instead of re-delegating the same lookup. This is the main lever against duplicate Claude task spawns.

The final message is a sentinel `[evaluate]` user turn that tells the model "decide now". The reply comes back as a `tool_calls` array; only the first call is honoured. Temperature is `0.0` — this is a router, not a brainstormer.

Two performance notes worth knowing if you're tuning latency:

- **KV warming.** On every ASR partial transcript, `speculative_prefill` re-prefills the thoughts context with the in-progress turn so the evaluator's response after the user stops talking is mostly a decode, not a full prefill. The message order matches `_evaluate` exactly to avoid cache misses.
- **Tools aren't sent during prefill.** Only the conversation prefix is — keeping prefill cheap. The tool list goes in at `_evaluate` time alongside the `[evaluate]` sentinel.

# Adding a tool to the thoughts stream

Adding a tool here is the right move when:

- The action is **fast and local** — a few hundred ms of LLM tool-routing + a sync Python handler beats a 2–10s round-trip through the Claude task server.
- It needs **no MCP**, no agent loop, no web access — just access to the streaming server's process state.
- The Ollama model is **smart enough to pick it reliably** from the system prompt (qwen3.5:4b handles 10–12 tools fine; past that, recall drops).

Anything that needs live web data, multi-step reasoning, or an MCP integration belongs in `claude-task` via `delegate`. The thoughts stream is for the snappy, deterministic stuff.

## Latency budget

| Path | Round-trip | Examples |
|---|---|---|
| Thoughts tool (local) | ~150–400ms | `add_memory`, `cancel_task`, `clear_context`, "set a timer" |
| `delegate` → `claude-task` | ~2–10s + filler | "check my emails", "what's the weather" |

The thoughts call runs **in parallel with the conversation reply**, so for silent tools (memory ops) the user notices nothing — the assistant just speaks its natural acknowledgement and the side-effect happens before the turn ends.

## One file per tool

Every thoughts tool is a single file in `src/vui/serving/stream/tools/`, exporting three things:

| Export | What it is |
|---|---|
| `SCHEMA: dict` | OpenAI-style function-call JSON the local LLM sees. `function.name` MUST match the filename (`my_tool.py` → `"my_tool"`). |
| `RULE: str` | A short routing rule appended into the system prompt under "RULES — read carefully:". Set to `""` if the schema description is unambiguous on its own. |
| `async def handle(ctx, **args)` | The handler. `ctx` is a `ThoughtsStream`; `args` come from the model's tool call. |

Drop the file in, hit `POST /tools/reload`, done. No edits to `thoughts.py`, no dispatch branch to maintain. The full authoring contract lives at `src/vui/serving/stream/tools/SPEC.md`.

### What `ctx` gives you

`ctx` is a `ThoughtsStream` instance. The reach-throughs you'll actually use:

- `ctx.srv` — the `StreamServer` (session, conversation, memories, tasks dict, websocket).
- `await ctx._speak(text)` — speak text via TTS. Most tools should *not* call this; let the conversation LLM's natural reply do the talking. Double-speaking is the most common bug.
- `await ctx._cancel_conversation()` — cancel the in-flight reply. Rare; only when the side-effect makes the reply wrong.
- `await ctx._wait_generation_done()` — wait for current TTS to finish naturally. Use before tools that *replace* what's being said (e.g. `clear_context`).

For state mutations, prefer the existing helpers (`vui.serving.stream.memories.add_memory`, `vui.serving.stream.tasks.create_task`, …) — `handle()` should mostly be a thin adapter.

## Worked example: `set_timer`

A timer fits the criteria perfectly — purely local, instant, and the conversation LLM can't do it on its own (no clock-side-effect access).

`src/vui/serving/stream/tools/set_timer.py`:

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
- NOT for "what time is it" (that's no_action).
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

Then:

```sh
curl -X POST http://localhost:8080/tools/reload
```

Say *"set a timer for 30 seconds"* — the conversation LLM says "Sure, 30 seconds starting now", the thoughts stream fires `set_timer({"seconds": 30})`, and 30s later the speak call delivers the alarm.

## Testing your new tool

The thoughts evaluation is logged under the `[thoughts]` prefix:

```sh
docker compose logs -f vui-stream | grep -E '\[thoughts|\[tools'
```

You'll see one of:

```
[tools] registered 12: add_memory, ..., set_timer
[thoughts] no_action (180ms)
[thoughts] set_timer({"seconds": 30}) (220ms)
```

`GET /tools` returns the live registry as JSON if you want to inspect what the model sees.

If the model keeps picking `no_action` when you wanted your tool, the rule isn't strong enough — add explicit trigger phrases and a **negative example** (when *not* to pick it). The system prompt is the only lever; sampling is `temperature=0.0` so you're not fighting randomness, you're fighting prompt clarity.

If it picks your tool when it shouldn't (false positive), add the failing case as a `no_action` example in the preamble of `_THOUGHTS_PREAMBLE` (in `thoughts.py`) — that's where decision-policy negative cases live, separate from per-tool RULE blocks. Capability questions ("can you set a timer?") are the classic false-positive trap; they should be `no_action`, not the tool itself.

## Asking Vui to write one for you

There's a built-in `propose_tool` thoughts action. Say *"add a tool that sets a timer"* or *"can you build a tool to toggle my desk lamp"* and Vui delegates codegen to `claude-task`:

1. Thoughts stream picks `propose_tool({"name": "set_timer", "description": "..."})`.
2. The streaming server creates a delegated task with a tight system prompt:
   - Read `src/vui/serving/stream/tools/SPEC.md` (the contract).
   - Read `src/vui/serving/stream/tools/add_memory.py` (a working example).
   - Write `src/vui/serving/stream/tools/<name>.py`.
   - Stop.
3. The codegen task runs with the **full default tool set** (`Read`, `Write`, `Glob`, `Grep`, `Bash`, `WebSearch`, `WebFetch`, plus any MCPs). The system prompt + `cwd` + `max_turns=8` are what keep it on-script — not an allowlist.
4. `cwd` is set to the repo root so the relative paths resolve.
5. When the task callback arrives, the server auto-runs `tools_registry.load_tools()` and the new tool goes live in the same turn — no `/tools/reload` call needed.

**Safety posture.** The constraints are: a tight system prompt that says "read SPEC.md, write one file, stop"; a snake_case name validation in `propose_tool.handle()` so a typo'd name can't write a weirdly-pathed file; a refusal to overwrite an existing tool file; and the registry loader logs+skips broken files so a bad codegen doesn't break the rest. It is **not** a sandbox — Claude has shell + web access during codegen, same as any normal `delegate` call. If you don't trust the voice loop with write access to a subdir of your repo, disable `propose_tool` by deleting `tools/propose_tool.py`.

**Repo hygiene.** Every new tool lands as a working-tree change (`git status` will show it). Treat it like any other code: review, edit if needed, commit if you want it to stick. Tools you generated and don't want to keep can be deleted; the next reload drops them from the registry.

**Docker.** The `propose_tool` path needs `src/vui/serving/stream/tools/` mounted rw into both `vui-stream` and `claude-task` (already configured in the shipped `docker-compose.yml`). Without that shared mount, Claude writes into the claude-task container's ephemeral copy and `vui-stream` never sees the file.

## Hot reload

`POST /tools/reload` re-walks the tools directory, re-imports every file, rebuilds the registry. The next thoughts evaluation sees the updated tool list — no server restart needed. Broken files (import error, missing export, schema/filename mismatch) are logged and skipped; the rest of the registry keeps working.

Removing a file removes the tool on next reload. In-flight thoughts evaluations finish on the old registry, so no draining needed.

## Local tools vs. `claude-task`

In principle most of what `claude-task` does could be reimplemented as local thoughts tools — `add_memory`, `cancel_task`, etc. already are. The blocker is **reliability at scale**: we haven't tested how well the small local thinking model holds up past ~15 tools, and tool-selection accuracy is the kind of thing that degrades silently before it falls off a cliff. Running the stack with a larger model as the thinking LLM (a 14B/30B-class model in place of qwen3.5:4b) should push that ceiling up — that's the obvious next experiment if you want to migrate more functionality local.

For now: **if you don't mind waiting a couple of seconds, `claude-task` works great** and gives you the full MCP surface for free. Reach for a thoughts tool when latency genuinely matters or the action is something Claude can't reach (process-local state like timers, playback, or session controls).

We're planning more work in this area — better small-model evals for tool routing, a richer set of bundled local tools, and probably a thinking-model upsize. **PRs welcome** for tools you'd like to see in the standard set: open an issue or PR on the repo with the use case and we'll help land it.
