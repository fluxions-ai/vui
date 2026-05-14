"""Task server integration: create, cancel, check, list tasks + voice handlers."""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import time
from typing import TYPE_CHECKING

import httpx
from aiohttp import web

from vui.serving.stream._log import _spawn, _spawn_response
from vui.serving.stream.prompts import TASK_SERVER_URL

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


TASKS_PERSIST_PATH = pathlib.Path.home() / ".vui" / "tasks.json"


def save_tasks(srv: StreamServer) -> None:
    """Write `srv._tasks` to disk so cancelled/finished entries survive a
    server restart. Best-effort — failures log but don't break the flow."""
    try:
        TASKS_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASKS_PERSIST_PATH.write_text(json.dumps(srv._tasks))
    except Exception as e:
        print(f"[tasks] failed to persist: {e}")


def load_tasks() -> dict[str, dict]:
    """Load previously-persisted tasks. Any task still marked `running` is
    reclassified to `cancelled` (the background work was lost on restart)."""
    if not TASKS_PERSIST_PATH.exists():
        return {}
    try:
        data = json.loads(TASKS_PERSIST_PATH.read_text())
        if not isinstance(data, dict):
            return {}
        for info in data.values():
            if info.get("status") == "running":
                info["status"] = "cancelled"
                info["error"] = info.get("error") or "interrupted by restart"
        return data
    except Exception as e:
        print(f"[tasks] failed to load: {e}")
        return {}


def _push_task_update(srv: StreamServer, task_id: str):
    save_tasks(srv)
    info = srv._tasks.get(task_id, {})
    ws = srv.session.ws
    if ws and not ws.closed:
        asyncio.ensure_future(
            ws.send_json(
                {
                    "type": "task_update",
                    "task_id": task_id,
                    "description": info.get("description", ""),
                    "status": info.get("status", "unknown"),
                    "result": info.get("result", ""),
                    "error": info.get("error"),
                    "created": info.get("created", 0),
                }
            )
        )


def _push_tasks_cleared(srv: StreamServer):
    save_tasks(srv)
    ws = srv.session.ws
    if ws and not ws.closed:
        asyncio.ensure_future(ws.send_json({"type": "tasks_cleared"}))


def _push_task_removed(srv: StreamServer, task_id: str):
    save_tasks(srv)
    ws = srv.session.ws
    if ws and not ws.closed:
        asyncio.ensure_future(
            ws.send_json({"type": "task_removed", "task_id": task_id})
        )


def push_all_tasks(srv: StreamServer):
    for task_id in srv._tasks:
        _push_task_update(srv, task_id)


async def create_task(
    srv: StreamServer,
    prompt: str,
    description: str = "",
    *,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    max_turns: int | None = None,
    is_codegen: bool = False,
) -> str | None:
    """Create a task on the claude-task server.

    Optional kwargs let the caller override Claude's behaviour for a single
    task — e.g. a tighter `system_prompt` + narrowed `allowed_tools` + `cwd`
    for codegen tasks. `is_codegen` is a local-only flag that triggers
    `tools_registry.load_tools()` once the task callback arrives.
    """
    payload: dict = {
        "prompt": prompt,
        "callback_url": "http://localhost:8080/task_done",
    }
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if allowed_tools is not None:
        payload["allowed_tools"] = allowed_tools
    if cwd is not None:
        payload["cwd"] = cwd
    if max_turns is not None:
        payload["max_turns"] = max_turns

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(f"{TASK_SERVER_URL}/task", json=payload)
            data = resp.json()
            task_id = data.get("task_id")
            clean_desc = (description.strip() or prompt.strip())[:100]
            srv._tasks[task_id] = {
                "description": clean_desc,
                "status": "running",
                "created": time.time(),
                "is_codegen": is_codegen,
            }
            _push_task_update(srv, task_id)
            print(f"[main] Task created: {task_id} '{clean_desc[:60]}'")
            return task_id
    except Exception as e:
        print(f"[main] Task creation failed: {e}")
        return None


# Stop words filtered from cancel/delete/check queries before matching.
# Without this, common short words ("a", "the", "is") substring-match
# almost every task description and turn "delete the unrelated nonsense"
# into a hit on whichever task happens to be most recent. Request verbs
# ("cancel", "delete", ...) are also dropped — the LLM sometimes passes
# the user's full phrase rather than just the noun phrase, and counting
# the verb toward query length blocks legitimate single-word matches
# (e.g. "cancel France" → ["france"] → 1-of-1 match passes).
_TASK_QUERY_STOP_WORDS = frozenset({
    # generic stop words
    "a", "an", "the", "is", "of", "to", "for", "in", "on", "at", "that",
    "this", "it", "and", "or", "but", "with", "from", "by", "my", "me",
    "your", "you", "as", "be", "was", "are", "do", "please", "can",
    "could", "would", "should", "i", "we", "us", "they", "them", "task",
    # request verbs the user might say to the assistant
    "cancel", "delete", "remove", "stop", "abort", "drop", "kill",
    "check", "get", "show", "tell",
})


def find_task_by_description(
    srv: StreamServer,
    desc: str,
    statuses: frozenset[str] | None = None,
) -> str | None:
    """Match by overlapping words; empty `desc` falls back to the most
    recent task (so 'get the results from the task' works without naming
    one). Stop words are stripped from the query before matching so a
    miss can't fuzzy-hit unrelated tasks via "the"/"a"/etc.

    `statuses` filters which tasks are eligible — pass `{"running"}` for
    cancel so a done/cancelled/error entry can't shadow the live task
    the user actually means. Default `None` considers all statuses.
    """
    def _eligible(info: dict) -> bool:
        return statuses is None or info.get("status") in statuses

    desc_lower = desc.lower().strip()
    if not desc_lower:
        latest_id = None
        latest_time = 0.0
        for tid, info in srv._tasks.items():
            if not _eligible(info):
                continue
            t = info.get("created", 0)
            if t > latest_time:
                latest_time = t
                latest_id = tid
        return latest_id
    query_words = [
        w for w in re.findall(r"\w+", desc_lower)
        if w not in _TASK_QUERY_STOP_WORDS and len(w) > 1
    ]
    if not query_words:
        return None
    best_id = None
    best_score = 0
    best_time = 0
    for tid, info in srv._tasks.items():
        if not _eligible(info):
            continue
        task_desc = info["description"].lower()
        # Score = number of query words that hit this task. Substring
        # match keeps "email" → "emails". Higher score beats recency, so
        # "cancel the search for the UK" against [UK-news, France-news]
        # picks UK (2 hits) over France (1 hit on "search"); recency
        # only breaks ties.
        score = sum(1 for w in query_words if w in task_desc)
        if score == 0:
            continue
        if score > best_score or (
            score == best_score and info["created"] > best_time
        ):
            best_score = score
            best_time = info["created"]
            best_id = tid
    # Reject weak matches: when the query has multiple meaningful words
    # (e.g. "UK search") the best match must hit either all of them or
    # at least 2. A 1-of-2 match means the user named something specific
    # ("UK") that didn't appear anywhere — refuse to fall back onto the
    # remaining word ("search") and cancel the wrong task.
    if best_id is not None and len(query_words) >= 2 and best_score < 2:
        return None
    return best_id


async def cancel_task(srv: StreamServer, task_id: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.delete(f"{TASK_SERVER_URL}/task/{task_id}")
            if resp.status_code == 200:
                if task_id in srv._tasks:
                    srv._tasks[task_id]["status"] = "cancelled"
                    _push_task_update(srv, task_id)
                return True
    except Exception as e:
        print(f"[main] Task cancel failed: {e}")
    return False


async def get_task_status(srv: StreamServer, task_id: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{TASK_SERVER_URL}/task/{task_id}")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"[main] Task status failed: {e}")
    return None


async def refresh_task_statuses(srv: StreamServer) -> None:
    """Pull fresh statuses from the task server and sync `srv._tasks`.
    Best-effort: if the task server is unreachable or slow, leave the
    local cache as-is — `/task/done` callbacks keep it roughly fresh
    anyway. Short timeout because this runs in the voice-cancel hot path.
    """
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(f"{TASK_SERVER_URL}/tasks")
            if resp.status_code != 200:
                return
            remote = resp.json()
    except Exception as e:
        print(f"[tasks] refresh failed: {e}")
        return
    by_id = {t["id"]: t for t in remote if isinstance(t, dict) and "id" in t}
    for tid, info in srv._tasks.items():
        remote_info = by_id.get(tid)
        if remote_info and "status" in remote_info:
            info["status"] = remote_info["status"]


def is_truly_idle(srv: StreamServer) -> bool:
    if not srv.session.ready:
        return False
    rs = srv.session.recording_sink
    if rs and rs.recording:
        return False
    if srv._endpointing_task and not srv._endpointing_task.done():
        return False
    if srv._llm_streaming:
        return False
    if srv._generates_done < srv._generates_sent:
        return False
    pt = srv.session.playback_track
    if pt and pt.can_pause:
        return False
    return True


async def deliver_pending_task_results(srv: StreamServer):
    while srv._pending_task_results:
        for _ in range(600):
            if is_truly_idle(srv):
                break
            await asyncio.sleep(0.1)
        else:
            print("[main] Task result delivery: gave up waiting for idle")
            break

        if not srv._pending_task_results:
            break

        item = srv._pending_task_results.pop(0)
        item.get("desc", "that")
        result = item.get("result")
        error = item.get("error")

        # User content is JUST the raw result text (or a short status note
        # for failure / empty). The task description is bound into the
        # system prompt as context only — never given to the LLM in user
        # position, otherwise it parrots "Search the web for X" verbatim.
        if error:
            content = f"(lookup failed: {error})"
        elif not result:
            content = "(no results came back)"
        else:
            content = result

        print(f"[main] Injecting task result ({len(content)} chars)")
        srv._log_conv("task_result_inject", text=content)

        result_conv = [{"role": "user", "content": content}]
        reply = await srv._stream_llm_to_tts(
            result_conv,
            system_prompt=(
                "Below is the result of a lookup the user requested. Relay "
                "it naturally, like you're telling a friend. Cover the "
                "important details but don't read it word for word.\n\n"
                "You are talking DIRECTLY to the user. Use second person "
                "('you'). NEVER refer to the user by name in the third "
                "person, even if the search result mentions them. If the "
                "result says 'Harry is working on X', you say 'You're "
                "working on X' or 'oh yeah, that thing you're working on'. "
                "Bad: 'Harry has been working on a speech model.' Good: "
                "'Yeah, you've been grinding on that speech model.'\n\n"
                "OPEN with a brief subject preface so the user knows what's "
                "coming — 'Here's what I found on <subject>' or a natural "
                "variant ('So on <subject>...', 'Right, the <subject> stuff...', "
                "'Okay, <subject>...'). Keep the preface to one short clause, "
                "then dive into the actual content. The subject is the topic "
                "the user asked about (UK news, your emails, today's calendar, "
                "the Spurs match), NOT a literal restatement of the query.\n\n"
                "Good: 'Here's what I found on the UK news. So the threat "
                "level got bumped to severe after that Golders Green attack...'\n"
                "Good: 'Right, your emails. You've got three. Sarah wants...'\n"
                "Good: 'So on the Spurs match — oh mate. Two-two...'\n"
                "Bad: 'Here are the latest stories from the UK. The threat "
                "level has been raised...' (too formal, sounds like a report)\n"
                "Bad: 'So the threat level got bumped to severe...' (no "
                "preface — jarring when it follows unrelated chat)\n\n"
                "Don't say 'The results show...', 'I looked up...', 'I "
                "checked...' — the preface names the subject, it doesn't "
                "narrate the lookup.\n\n"
                "CRITICAL — output rules (this is a voice assistant — TTS will "
                "speak your reply):\n"
                "- ALWAYS spell out EVERYTHING phonetically: numbers, times, "
                "dates, units, percentages, money, model names, abbreviations.\n"
                "- NEVER write digits, colons, decimals, currency symbols, "
                "unit symbols, asterisks, or markdown.\n"
                '- Bad: "9:00 AM", "$213.42", "1.3%", "800MW", "60Hz", '
                '"15mph", "14°C", "$1599", "RTX 4090", "1993", "2024".\n'
                '- Good: "nine in the morning", "two hundred and thirteen '
                'dollars forty two cents", "one point three percent", "eight '
                'hundred megawatts", "sixty hertz", "fifteen miles an hour", '
                '"fourteen degrees", "fifteen ninety nine", "RTX forty '
                'ninety", "nineteen ninety three", "twenty twenty four".\n'
                "- Years and 4-digit model numbers: read in pairs ('1993' -> "
                "'nineteen ninety three', '4090' -> 'forty ninety').\n"
                "- Times: natural spoken. Drop ':00' on the hour ('11:00 AM' "
                "-> 'eleven AM', '9:00' -> 'nine'). Use natural phrasing for "
                "half/quarter ('half eleven', 'quarter past two', 'quarter to "
                "five'). Say 'seven PM', 'noon', 'midnight'. Never say "
                "'eleven oh oh' or 'eleven hundred'.\n"
                "- This applies to EVERY digit and symbol from the source text. "
                "Convert ALL of them before speaking."
            ),
            append_to_history=False,
            # Empty user_text_for_tts: don't seed the TTS [user_spk] block
            # with the task description. Otherwise the description sits in
            # KV as user-side conditioning right before the agent generates,
            # and the TTS sometimes echoes it forward.
            user_text_for_tts="",
        )
        if reply:
            srv.session.conversation.append({"role": "assistant", "content": reply})
        srv._log_conv("task_result_reply", text=reply)


def _relay(srv: StreamServer, content: str, desc: str = "your tasks"):
    """Synthetic 'task result': push a string into the pending-results
    queue so it gets paraphrased by the same relay LLM that handles real
    lookup results."""
    srv._pending_task_results.append({"result": content, "desc": desc})


async def handle_list_tasks_voice(srv: StreamServer):
    """Treat the local task list as a synthetic delegate result — push it
    through the relay pipeline so the response sounds identical to a
    real lookup."""
    running = [
        info["description"]
        for info in srv._tasks.values()
        if info.get("status") == "running" and info.get("description")
    ]
    done = [
        info["description"]
        for info in srv._tasks.values()
        if info.get("status") == "done" and info.get("description")
    ]

    if not running and not done:
        content = "No tasks are running. Nothing finished either."
    else:
        parts: list[str] = []
        if running:
            parts.append(f"Running ({len(running)}): " + "; ".join(running))
        if done:
            parts.append(f"Finished ({len(done)}): " + "; ".join(done))
        content = ". ".join(parts) + "."

    _relay(srv, content, desc="your tasks")
    await deliver_pending_task_results(srv)


_RUNNING = frozenset({"running"})


async def handle_cancel_task_voice(srv: StreamServer, description: str):
    desc = description.strip()
    if not desc:
        # No specifics — "just cancel it". Wipe all so the user isn't stuck
        # with running work they no longer want.
        await handle_clear_tasks_voice(srv)
        return
    # Refresh from /tasks so the running-only filter sees up-to-date
    # statuses — without this a task that finished between the user
    # speaking and us matching would still look "running" locally.
    await refresh_task_statuses(srv)
    # Restrict to running tasks: cancelling done/cancelled/error is a no-op
    # but would prevent the running task the user actually meant from
    # getting cancelled if a stale entry happens to score higher.
    task_id = find_task_by_description(srv, desc, statuses=_RUNNING)
    if not task_id:
        # Specific request that didn't match any running task. NEVER fall
        # through to clear_all here — a fuzzy-match miss must not nuke
        # unrelated tasks. Just log and no-op.
        print(f"[tasks] cancel: no running match for {desc!r}, ignoring")
        return
    ok = await cancel_task(srv, task_id)
    if ok and srv._pending_task_id == task_id:
        srv._pending_task_id = None


async def delete_task(srv: StreamServer, task_id: str) -> bool:
    """Remove a task from local state (and cancel it on the task server if
    it's still running). Always pushes a `task_removed` ws event so the UI
    clears the row even when our in-memory cache has drifted from what the
    browser thinks (e.g. after a reset wiped `srv._tasks` without notifying
    the client). Returns True if there was anything to remove locally.
    """
    info = srv._tasks.get(task_id)
    if info and info.get("status") == "running":
        await cancel_task(srv, task_id)
    existed = srv._tasks.pop(task_id, None) is not None
    if srv._pending_task_id == task_id:
        srv._pending_task_id = None
    _push_task_removed(srv, task_id)
    return existed


async def handle_delete_task_voice(srv: StreamServer, description: str):
    """Voice-side: 'delete the X task' or 'remove that one'. Empty
    description falls through to clear_tasks (delete all). A non-empty
    description that doesn't match anything is a no-op — never wipe all
    on a fuzzy-match miss."""
    desc = description.strip()
    if not desc:
        await handle_clear_tasks_voice(srv)
        return
    task_id = find_task_by_description(srv, desc)
    if not task_id:
        print(f"[tasks] delete: no match for {desc!r}, ignoring")
        return
    await delete_task(srv, task_id)


async def handle_check_task_voice(srv: StreamServer, description: str):
    task_id = find_task_by_description(srv, description)
    if not task_id:
        _relay(
            srv,
            f"No matching task found for {description!r}.",
            desc="task lookup",
        )
        await deliver_pending_task_results(srv)
        return

    # Use the LOCAL cached state first — no need to round-trip to the
    # task server when we already have the result from /task_done.
    local = srv._tasks.get(task_id, {})
    local_status = local.get("status")
    local_result = local.get("result")
    desc = local.get("description", description)

    if local_status == "done" and local_result:
        _relay(srv, local_result, desc=desc)
        await deliver_pending_task_results(srv)
        return

    info = await get_task_status(srv, task_id)
    if not info:
        _relay(
            srv,
            f"Couldn't reach the task server to check status of {desc}.",
            desc=desc,
        )
        await deliver_pending_task_results(srv)
        return
    status = info.get("status", local_status or "unknown")
    if status == "done" and info.get("result"):
        srv._tasks.setdefault(task_id, {}).update(
            {"status": "done", "result": info["result"]}
        )
        _relay(srv, info["result"], desc=desc)
    elif status == "running":
        tools_used = info.get("tool_uses", [])
        tools_note = f" Used {len(tools_used)} tools so far." if tools_used else ""
        _relay(srv, f"The {desc} task is still running.{tools_note}", desc=desc)
    elif status == "error":
        _relay(
            srv,
            f"The {desc} task failed: {info.get('error', 'unknown error')}.",
            desc=desc,
        )
    else:
        _relay(srv, f"The {desc} task is in status: {status}.", desc=desc)
    await deliver_pending_task_results(srv)


async def handle_clear_tasks_voice(srv: StreamServer):
    srv._tasks.clear()
    _push_tasks_cleared(srv)
    srv._pending_task_id = None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(f"{TASK_SERVER_URL}/tasks/clear")
    except Exception:
        pass


async def handle_task_done(srv: StreamServer, request: web.Request) -> web.Response:
    data = await request.json()
    task_id = data.get("task_id", "?")
    status = data.get("status", "unknown")
    result = data.get("result", "")
    error = data.get("error")
    prompt = data.get("prompt", "")

    if task_id in srv._tasks:
        srv._tasks[task_id]["status"] = status
        srv._tasks[task_id]["result"] = result
        srv._tasks[task_id]["error"] = error
        _push_task_update(srv, task_id)

    already_spoken = srv._tasks.get(task_id, {}).get("spoken", False)
    if already_spoken:
        print(f"[main] Task {task_id} callback (already spoken, skipping)")
        return web.json_response({"ok": True})

    if task_id in srv._tasks:
        srv._tasks[task_id]["spoken"] = True

    if srv._pending_task_id == task_id:
        srv._pending_task_id = None

    desc = (srv._tasks.get(task_id, {}).get("description") or prompt)[:100]
    print(f"[main] Task {task_id} done: status={status}")
    await srv._log(
        f"Task {task_id} complete ({data.get('duration_ms', 0):.0f}ms)", "info"
    )

    # Codegen tasks (proposed via the propose_tool thoughts action) write
    # a new file into src/vui/serving/stream/tools/ and need the registry
    # rebuilt so the next thoughts evaluation sees the new tool.
    if status == "done" and srv._tasks.get(task_id, {}).get("is_codegen"):
        from vui.serving.stream import tools as tools_registry

        before = set(tools_registry._HANDLES)
        tools_registry.load_tools()
        after = set(tools_registry._HANDLES)
        added = sorted(after - before)
        if added:
            print(f"[main] Tools reloaded; new: {', '.join(added)}")
            srv._tasks[task_id]["new_tools"] = added

    if status == "done" and result:
        srv._pending_task_results.append({"result": result, "desc": desc})
    elif status == "done" and not result:
        srv._pending_task_results.append({"result": None, "desc": desc})
    elif status == "error":
        srv._pending_task_results.append(
            {"result": None, "desc": desc, "error": error or "Something went wrong."}
        )
    else:
        result = None

    if srv._pending_task_results:
        srv._log_conv(
            "task_result_queued",
            task_id=task_id,
            desc=desc,
            idle=is_truly_idle(srv),
            ready=srv.session.ready,
            llm_streaming=srv._llm_streaming,
            gen_sent=srv._generates_sent,
            gen_done=srv._generates_done,
        )
        _spawn_response(srv, deliver_pending_task_results(srv), "deliver_task_results")

    return web.json_response({"ok": True})


async def handle_delete_task_http(
    srv: StreamServer, request: web.Request
) -> web.Response:
    """`POST /tasks/delete {task_id: "..."}` — remove a task from local
    state (cancelling first if running). Also exists as a voice command;
    this is for the UI's per-task × button."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    task_id = data.get("task_id")
    if not task_id:
        return web.json_response({"error": "missing task_id"}, status=400)
    await delete_task(srv, task_id)
    return web.json_response({"ok": True})


async def handle_clear_tasks_http(
    srv: StreamServer, request: web.Request
) -> web.Response:
    """`POST /tasks/clear` — wipe all local tasks."""
    await handle_clear_tasks_voice(srv)
    return web.json_response({"ok": True})


def bind(cls):
    cls._create_task = lambda self, *a, **kw: create_task(self, *a, **kw)
    cls._find_task_by_description = lambda self, *a, **kw: find_task_by_description(
        self, *a, **kw
    )
    cls._cancel_task = lambda self, *a, **kw: cancel_task(self, *a, **kw)
    cls._get_task_status = lambda self, *a, **kw: get_task_status(self, *a, **kw)
    cls._is_truly_idle = lambda self: is_truly_idle(self)
    cls._deliver_pending_task_results = lambda self: deliver_pending_task_results(self)
    cls._handle_list_tasks_voice = lambda self: handle_list_tasks_voice(self)
    cls._handle_cancel_task_voice = lambda self, *a, **kw: handle_cancel_task_voice(
        self, *a, **kw
    )
    cls._delete_task = lambda self, *a, **kw: delete_task(self, *a, **kw)
    cls._handle_delete_task_voice = lambda self, *a, **kw: handle_delete_task_voice(
        self, *a, **kw
    )
    cls._handle_check_task_voice = lambda self, *a, **kw: handle_check_task_voice(
        self, *a, **kw
    )
    cls._handle_clear_tasks_voice = lambda self: handle_clear_tasks_voice(self)
    cls.handle_task_done = lambda self, *a, **kw: handle_task_done(self, *a, **kw)
    cls.handle_delete_task_http = lambda self, *a, **kw: handle_delete_task_http(
        self, *a, **kw
    )
    cls.handle_clear_tasks_http = lambda self, *a, **kw: handle_clear_tasks_http(
        self, *a, **kw
    )
