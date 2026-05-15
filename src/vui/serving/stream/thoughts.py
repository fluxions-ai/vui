"""Internal thoughts stream -- LLM-driven reasoning alongside voice conversation.

The thoughts stream is a separate LLM context that observes every user turn,
sees the full conversation + current memories, and autonomously decides what
to do: add/remove memories, delegate to Claude, search the web, manage tasks,
or (most often) do nothing and let the conversation stream handle it.

Tool definitions live in `vui.serving.stream.tools` (one file per tool).
See `tools/SPEC.md` for the authoring contract.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from vui.serving.stream import tools as tools_registry
from vui.serving.stream._log import _slog, _spawn

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


def _initial_task_desc(args: dict | None) -> str:
    """Pick a reasonable first-pass description for a local task row from
    the model's tool args. The tool's `handle()` typically overwrites this
    with something prettier, but we want the row to land in the UI with
    *some* content the instant dispatch begins."""
    if not args:
        return ""
    for k in ("subject", "query", "task", "description", "label"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


_LOCAL_CAPABILITIES = [
    "remember things about you",
    "forget specific memories or wipe them all",
    "manage background tasks (list, check on, cancel, delete)",
    "reset our conversation",
    "look things up on the web (weather, news, prices, scores, definitions)",
]


_THOUGHTS_PREAMBLE = """\
You are an internal reasoning process observing a voice conversation. You see each complete turn — the user's message AND the assistant's response — then decide what action to take.

AVAILABLE TOOLS (what the assistant can actually do right now — use this list when answering capability questions):
{capabilities}

CURRENT MEMORIES:
{memories}

The user's text comes from speech recognition and may contain errors. Interpret intent, not literal words (e.g. "website" = "web search", "male" = "email").

no_action (MOST turns):
- Casual chat, greetings, opinions, stories, venting, complaints ("CI is slow", "Docs is laggy") -> no_action.
- GENERAL KNOWLEDGE the assistant can answer from training data -> no_action. This includes: recipes, how-to guides, explanations, definitions, maths, science, coding concepts, history, trivia. If the answer exists in a textbook or Wikipedia, the assistant knows it. NEVER delegate general knowledge.
- Questions ABOUT capabilities -> answer_capability (NOT no_action, NOT delegate): "can you access my emails?", "are you able to search?", "do you have access to Slack?", "what can you do?", "what tools do you have?", "do you have access to X?". Look at AVAILABLE TOOLS above and answer based on it. NEVER delegate a capability question.
- "What do you remember/know about me?" -> no_action (conversation answers from memory context).
- Opinions/speculation ("do you think it'll rain?") -> no_action.
- Vague future intentions ("I need to reply at some point") -> no_action. Only delegate when the user wants it done NOW.
- FOLLOW-UP QUESTIONS about results already visible in the conversation -> no_action. If a task result was summarised by the assistant above, questions about those details ("what was the second one?", "what time was Lisa's?", "which one was from Dave?", "tell me more about the first one") are answerable from context. NEVER re-delegate for information already in the conversation. Only delegate for genuinely NEW external actions (reply to an email, check a different day, send a message).

CRITICAL — task results already in conversation:
- If the assistant already told the user the results (emails, weather, calendar, search results etc.) in a message above, then ANY follow-up question about those results is no_action. The conversation LLM can re-read its own messages. Examples: "what was the second email?", "what time was Lisa's meeting?", "which one was from Dave?", "tell me more about the first one". Do NOT use check_task, delegate, or any other tool — use no_action.
- Only delegate if the user asks for a genuinely NEW external action: "reply to Sarah's email", "check tomorrow's calendar", "do another search".

CRITICAL — assistant is asking the user to clarify, do not act yet:
- If the assistant's MOST RECENT reply (the one just above [evaluate]) is a clarifying question about what the user actually wants — not a filler/acknowledgement — pick no_action. Wait for the user's clarification next turn, then fire the action then.
- Clarifying-question signals: "did you mean...", "are you sure...", "X or Y?", "to be clear...", "wait, you want...", any "?" that questions the user's intent rather than offering a follow-up.
- ACT signals (NOT clarifying — proceed with the tool): short fillers like "yeah hold on", "lemme check", "one sec", "pulling it up". These mean the assistant accepted the request.
- Applies to ALL state-changing / background-work tools: set_timer, delegate, web_search, add_memory, remove_memory, clear_context, clear_memories, set_speech_rate. Read-only tools (answer_capability, list_tasks, check_task) are fine either way.
- Example: User: "Set a timer for the pasta app in five minutes." Assistant: "Did you mean the cooking app, or are you actually cooking?" -> no_action. Next turn, after user clarifies, set_timer fires normally.

RULES — read carefully:

{tool_rules}

Other rules:
- Memory ops are based on what the USER said, not the assistant's response.
- When you see [evaluate], decide what action to take for the conversation above.
- You MUST call exactly one tool. Never respond with text."""


_TASKS_CONTEXT = """\
CURRENT TASKS (check BEFORE choosing delegate — each [done] task includes its result excerpt):
{tasks}

CRITICAL — match the user's request to the task list above:
- A [running] task covering the user's request -> no_action.
- A [done] task whose result excerpt ALREADY answers the user's request -> no_action. The conversation LLM can re-read the existing result. Examples: "what was the second one?", "tell me more", "remind me what you found", "didn't you check that?", "can you check the results?", "what did you find?".
- Different temporal scope = NEW task. "Today's calendar" done does NOT cover "tomorrow's calendar" — that's delegate. "Latest news" done does NOT cover "news from yesterday" — that's delegate. Compare the user's specific time/scope against the task description AND the result excerpt.
- Different account / different person / different specific subject = NEW task -> delegate.
- Reply / send / write / create actions are always NEW -> delegate (even if a related lookup is already done).
- Same intent rephrased = SAME task -> no_action. "Look up solar costs" and "find solar panel prices" cover the same thing — don't duplicate. Match by intent and entity, not exact wording.
- Re-running an identical lookup the user has already received is the bug to avoid — pick no_action there."""


# Load the tool registry on import. Re-callable via `tools_registry.load_tools()`
# from the /tools/reload endpoint to pick up new files without a restart.
tools_registry.load_tools()


class ThoughtsStream:
    """LLM-driven reasoning that observes conversation and acts autonomously."""

    def __init__(self, srv: StreamServer):
        self.srv = srv
        self._task: asyncio.Task | None = None
        self._last_prefill_text: str = ""
        self._last_prefill_t: float = 0.0
        self._prefill_inflight: bool = False
        # Currently-dispatching tool's local task row (None for tools that
        # don't declare a TASK block). Tool authors read this as `ctx.task`
        # and call `task.update(...)` to push UI state.
        self.task = None

    @property
    def conversation(self) -> list[dict]:
        return list(self.srv.session.conversation)

    def _build_system_prompt(self) -> str:
        memories = getattr(self.srv, "_memories", []) or []
        if memories:
            mem_block = "\n".join(f"[{i + 1}] {m}" for i, m in enumerate(memories))
        else:
            mem_block = "(none)"
        remote_caps = getattr(self.srv, "_task_server_capabilities", []) or []
        all_caps = _LOCAL_CAPABILITIES + list(remote_caps)
        cap_block = "\n".join(f"- {c}" for c in all_caps)
        return _THOUGHTS_PREAMBLE.format(
            capabilities=cap_block,
            memories=mem_block,
            tool_rules=tools_registry.rules_block(),
        )

    def _build_tasks_context(self) -> str | None:
        tasks = getattr(self.srv, "_tasks", {})
        if not tasks:
            return None
        # Only [running] and [done] tasks gate duplicate detection — a
        # [cancelled] or [error] row is dead work and must NOT shadow a new
        # request that fuzzy-matches its description (e.g. user cancels
        # "pasta timer", then asks for a fresh 20s pasta timer — without
        # this filter the evaluator sees the stale cancelled row and picks
        # no_action). Cancellation outcomes still live in the conversation
        # history, so the model can still answer "did you cancel that?".
        task_lines = []
        for info in tasks.values():
            status = info.get("status", "unknown")
            if status not in ("running", "done"):
                continue
            desc = info.get("description", "unknown")
            line = f"- [{status}] {desc}"
            # For [done] tasks, append a short result excerpt so the
            # evaluator can answer follow-ups from cached data instead
            # of re-delegating. The full result is also in the relay
            # paraphrase that lives in conversation, but the evaluator
            # is more reliable when the data is right next to the
            # description it would key off of.
            if status == "done":
                result = (info.get("result") or "").strip()
                if result:
                    excerpt = result.replace("\n", " ")[:300]
                    line += f"\n    result: {excerpt}"
            task_lines.append(line)
        if not task_lines:
            return None
        return _TASKS_CONTEXT.format(tasks="\n".join(task_lines))

    async def speculative_prefill(self, partial_text: str):
        """Warm the thoughts KV cache during user speech (called from ASR drain)."""
        # Single in-flight gate — same reason as the convo LLM prefill:
        # slow models otherwise stack up several useless prefills per turn.
        if getattr(self, "_prefill_inflight", False):
            return
        now = time.monotonic()
        if now - self._last_prefill_t < 2.0 and partial_text == self._last_prefill_text:
            return
        self._last_prefill_text = partial_text
        self._last_prefill_t = now
        self._prefill_inflight = True
        t0 = now
        try:
            from vui.serving.stream.llm_backend import get_backend

            messages = (
                [{"role": "system", "content": self._build_system_prompt()}]
                + self.srv.session.conversation
                + [{"role": "user", "content": partial_text}]
            )
            # Note: backend.prefill currently doesn't pass tools. The
            # tools list is part of the prompt the model conditions on
            # for thoughts evaluation, but for KV warming the bulk of
            # the cost is the conversation prefix — adding tools would
            # marginally improve cache hit if _evaluate also sends them.
            # Trade-off: keep prefill cheap and identical to evaluate's
            # message order; tools are passed in evaluate() below.
            await get_backend().prefill(messages)
            _slog(
                f"[thoughts.prefill] done {(time.monotonic()-t0)*1000:.0f}ms "
                f"text='{partial_text[:40]}'"
            )
        except Exception as e:
            import httpx

            if isinstance(e, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
                await self.srv._set_llm_available(False)
        finally:
            self._prefill_inflight = False

    async def on_user_turn(self, asr_text: str):
        """Fire-and-forget: launches parallel evaluation."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = _spawn(self._evaluate(asr_text), "thoughts_evaluate")

    async def _evaluate(self, user_text: str):
        t0 = time.monotonic()
        try:
            from vui.serving.stream.llm_backend import get_backend

            messages = [
                {"role": "system", "content": self._build_system_prompt()},
            ] + self.conversation
            tasks_ctx = self._build_tasks_context()
            if tasks_ctx:
                messages.append({"role": "system", "content": tasks_ctx})
            messages.append({"role": "user", "content": "[evaluate]"})

            stats: dict = {}
            res = await get_backend().complete(
                messages,
                tools=tools_registry.tools_list(),
                temperature=0.0,
                stats=stats,
            )

            self.srv.thoughts_ctx = stats.get("ctx_used", 0)
            self.srv.thoughts_ctx_max = stats.get("ctx_max", 8192)

            elapsed_ms = (time.monotonic() - t0) * 1000
            tool_calls = res.get("tool_calls") or []

            if not tool_calls:
                _slog(f"[thoughts] no tool call ({elapsed_ms:.0f}ms)")
                return

            tc = tool_calls[0]
            name = tc["function"]["name"]
            args = tc["function"].get("arguments", {})

            self.srv._log_conv(
                "thoughts_eval",
                action=name,
                args=args,
                user_text=user_text,
                elapsed_ms=round(elapsed_ms),
            )

            if name == "no_action":
                _slog(f"[thoughts] no_action ({elapsed_ms:.0f}ms)")
                return

            handle = tools_registry.dispatch(name)
            if handle is None:
                _slog(f"[thoughts] unknown tool {name!r} ({elapsed_ms:.0f}ms)")
                return

            args_preview = json.dumps(args)[:80] if args else ""
            _slog(f"[thoughts] {name}({args_preview}) ({elapsed_ms:.0f}ms)")
            await self.srv._log(f"tool: {name}({args_preview})", "info")

            # If this tool opted into a UI task row via `TASK = {...}`, create
            # the row before dispatch so the panel reflects work-in-progress.
            # `ctx.task` is the LocalTask handle; tools that don't declare
            # TASK see `ctx.task is None`.
            meta = tools_registry.task_meta(name)
            local_task = None
            if meta and meta.get("surface", True):
                from vui.serving.stream.tasks import start_local_task

                initial_desc = _initial_task_desc(args)
                local_task = start_local_task(
                    self.srv, name, initial_desc or name
                )
            self.task = local_task

            auto_done = bool(meta.get("auto_done", True)) if meta else False
            try:
                await handle(self, **(args or {}))
            except TypeError as e:
                # Schema/handler mismatch: the model sent kwargs the handler
                # doesn't accept. Surface clearly so authors fix the schema.
                _slog(f"[thoughts] tool {name} arg mismatch: {e}")
                await self.srv._log(f"tool {name} arg mismatch: {e}", "error")
                if local_task:
                    local_task.update(status="error", error=str(e))
            except Exception as e:
                if local_task:
                    local_task.update(status="error", error=str(e))
                raise
            else:
                if local_task and auto_done:
                    info = self.srv._tasks.get(local_task.task_id, {})
                    if info.get("status") == "running":
                        local_task.update(status="done")
            finally:
                self.task = None

        except asyncio.CancelledError:
            return
        except Exception as e:
            import httpx

            if isinstance(e, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
                _slog(f"[thoughts] LLM unreachable: {e}")
                await self.srv._set_llm_available(False)
                return
            _slog(f"[thoughts] error: {e}")
            import traceback

            traceback.print_exc()

    async def _wait_generation_done(self):
        """Wait for current TTS generation to finish naturally (no cancel/rewind)."""
        srv = self.srv
        for _ in range(300):
            if srv._generates_done >= srv._generates_sent and not srv._llm_streaming:
                break
            await asyncio.sleep(0.1)

    async def _cancel_conversation(self):
        srv = self.srv
        srv.session.cancel_generation = True
        srv.tts_cancel_event.set()
        rewind_T = getattr(srv, "_pre_turn_T", 0)
        srv.tts_cmd_queue.put({"cmd": "cancel", "rewind_to": rewind_T})
        if srv.session.playback_track:
            srv.session.playback_track.flush()
        await asyncio.sleep(0.05)
        srv.session.cancel_generation = False

    async def _prefill_conversation_kv(self):
        """Warm the conversation LLM's KV while waiting for task result.

        After delegation, the conversation has: [...turns, user question,
        assistant filler]. Prefilling this into Ollama means when the task
        result arrives and gets appended as a user message, only the new
        result text needs processing — not the entire history.
        """
        try:
            from vui.serving.stream.llm import llm_prefill_user

            t0 = time.monotonic()
            await llm_prefill_user(
                self.srv.session.conversation,
                self.srv.session.soul,
                self.srv.ollama_model,
            )
            _slog(
                f"[thoughts] conv KV prefilled for result delivery "
                f"({(time.monotonic()-t0)*1000:.0f}ms)"
            )
        except Exception as e:
            _slog(f"[thoughts] conv KV prefill failed: {e}")

    async def _speak(self, text: str):
        # Route tool-driven utterances through the same synthetic-result
        # pipeline real task lookups use: queue the string as a pending
        # result, then trigger delivery. The relay LLM paraphrases it
        # conversationally and the deliverer appends the paraphrased reply
        # to session.conversation — so a fixed-string tool announcement
        # ("Your timer is up.") comes out naturally and is recorded once.
        # `deliver_pending_task_results` waits for the system to be idle
        # before speaking, which gracefully handles timers firing mid-turn.
        from vui.serving.stream.tasks import (
            _relay,
            deliver_pending_task_results,
        )

        if not text or not text.strip():
            return
        _relay(self.srv, text)
        await deliver_pending_task_results(self.srv)
