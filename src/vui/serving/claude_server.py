"""Background task server for voice assistant.

Runs multi-step Claude agent tasks (mail/calendar/Slack reads, file ops,
deeper research) asynchronously. When complete, results are pushed to the
voice server via callback so TTS can speak them. One-shot factual web
lookups don't come through here — the streaming server's local `web_search`
tool handles those.

Endpoints:
    POST   /task          - Create and run a background task
    GET    /task/{id}     - Poll task status/result
    DELETE /task/{id}     - Cancel a task
    GET    /tasks         - List all tasks
    GET    /capabilities  - List available capability groups + raw tools
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import claude_agent_sdk as sdk
import httpx
from aiohttp import web


def _check_claude_auth() -> bool:
    """Verify Claude Code subscription / API auth is reachable.

    Returns True if any auth source is found. Prints a clear hint otherwise."""
    creds_paths = [
        Path.home() / ".claude" / ".credentials.json",
        Path.home() / ".claude" / "credentials.json",
    ]
    has_creds = any(p.exists() for p in creds_paths)
    has_api_key = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    )
    if has_creds or has_api_key:
        return True
    print("=" * 70, file=sys.stderr)
    print("WARNING: No Claude Code subscription or API key found.", file=sys.stderr)
    print("", file=sys.stderr)
    print("The task server needs one of:", file=sys.stderr)
    print("  1. A Claude Code subscription — run `claude` once and log in.", file=sys.stderr)
    print("     Credentials live in ~/.claude/.credentials.json", file=sys.stderr)
    print("  2. An API key in an env var:", file=sys.stderr)
    print("       export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
    print("     or  export CLAUDE_CODE_OAUTH_TOKEN=...", file=sys.stderr)
    print("", file=sys.stderr)
    print("Starting anyway — task requests will fail at runtime.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return False


def _default_system() -> str:
    today = time.strftime("%A, %B %d, %Y")
    return (
        f"Today's date is {today}. "
        "You are a helpful assistant used via a voice interface. "
        "You have MCP tools for Gmail, Google Calendar, Slack, and Google Drive. "
        "Always use the MCP tools (prefixed mcp__claude_ai_) to answer questions "
        "about email, calendar, messages, and files. Never use Bash for these. "
        "Your response will be spoken aloud by a TTS system. "
        "NEVER use markdown, asterisks, bullet points, numbered lists, or any formatting. "
        "Write plain conversational sentences only. "
        "Spell out all numbers, dates, times, and abbreviations in words. "
        "Answer the user's request directly. Use the minimum number of tool calls "
        "needed. Do NOT search broadly or speculatively — only look up what was "
        "specifically asked. Keep your answer to two or three sentences."
    )


MODEL = "claude-haiku-4-5-20251001"

BUILTIN_TOOLS = ["WebSearch", "WebFetch", "Bash", "Read", "Glob", "Grep"]

_SERVER_NAME_MAP = {
    "Gmail": "Gmail",
    "Google Calendar": "Google_Calendar",
    "Google Drive": "Google_Drive",
    "Slack": "Slack",
}


async def discover_mcp_tools() -> list[str]:
    """Query Claude CLI for connected MCP servers and return tool names."""
    tools = []
    try:
        client = sdk.ClaudeSDKClient()
        await client.connect()
        status = await client.get_mcp_status()
        for server in status.get("mcpServers", []):
            if server["status"] != "connected":
                continue
            raw_name = server["name"].removeprefix("claude.ai ")
            prefix = _SERVER_NAME_MAP.get(raw_name, raw_name.replace(" ", "_"))
            for t in server.get("tools", []):
                tools.append(f"mcp__claude_ai_{prefix}__{t['name']}")
        await client.disconnect()
    except Exception as e:
        print(f"[boot] MCP discovery failed: {e}")
    if tools:
        print(f"[boot] Discovered {len(tools)} MCP tools")
    return tools


DEFAULT_TOOLS: list[str] = list(BUILTIN_TOOLS)


def _capability_groups(tools: list[str]) -> list[str]:
    """Map raw tool names to user-facing capability labels."""
    groups = []
    if any("__Gmail__" in t for t in tools):
        groups.append("email")
    if any("__Google_Calendar__" in t for t in tools):
        groups.append("calendar")
    if any("__Slack__" in t for t in tools):
        groups.append("Slack")
    if any("__Google_Drive__" in t for t in tools):
        groups.append("Google Drive")
    if "WebSearch" in tools or "WebFetch" in tools:
        # Labelled as "research" rather than "web search" so the thoughts
        # stream doesn't route single-query factual lookups here — those go
        # to the streaming server's local `web_search` tool. claude-task is
        # for multi-step / chained-call investigations.
        groups.append("multi-step web research")
    if "Bash" in tools or "Read" in tools:
        groups.append("code execution")
    return groups


def log(task_id: str, msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}][task:{task_id}] {msg}")


class Task:
    def __init__(
        self,
        task_id: str,
        prompt: str,
        callback_url: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
        cwd: str | None = None,
    ):
        self.id = task_id
        self.prompt = prompt
        self.callback_url = callback_url
        self.system_prompt = system_prompt or _default_system()
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or []
        self.cwd = cwd
        self.status = "pending"
        self.result: str | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.duration_ms: float | None = None
        self.cost_usd: float | None = None
        self.model: str | None = None
        self.turns_used: int = 0
        self.tool_uses: list[str] = []
        self._cancel = False

    def info(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "prompt": self.prompt,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "turns_used": self.turns_used,
            "tool_uses": self.tool_uses,
        }


class TaskServer:
    def __init__(self, port: int = 8642):
        self.port = port
        self.tasks: dict[str, Task] = {}
        self._bg_tasks: dict[str, asyncio.Task] = {}
        self._session_id: str | None = (
            None  # persistent Claude session for conversation continuity
        )

    async def _run_task(self, task: Task):
        task.status = "running"
        log(task.id, f"Running: '{task.prompt[:80]}'")
        text_parts = []
        first_text_sent = False
        t0 = time.monotonic()
        try:
            opts = sdk.ClaudeAgentOptions(
                system_prompt=task.system_prompt,
                max_turns=task.max_turns,
                allowed_tools=task.allowed_tools or DEFAULT_TOOLS,
                model=MODEL,
                cwd=task.cwd,
            )
            if self._session_id:
                opts.resume = self._session_id
                opts.continue_conversation = True
                log(task.id, f"Resuming session {self._session_id}")

            async for msg in sdk.query(
                prompt=task.prompt,
                options=opts,
            ):
                if task._cancel:
                    task.status = "cancelled"
                    log(task.id, "Cancelled by user")
                    return

                if isinstance(msg, sdk.AssistantMessage):
                    task.model = msg.model
                    if msg.content:
                        for block in msg.content:
                            if isinstance(block, sdk.TextBlock):
                                text_parts.append(block.text)
                                log(task.id, f"Text: {block.text[:100]}")
                                # Send first text immediately so voice can start speaking
                                if (
                                    not first_text_sent
                                    and task.callback_url
                                    and block.text.strip()
                                ):
                                    first_text_sent = True
                                    elapsed = (time.monotonic() - t0) * 1000
                                    log(task.id, f"Early callback at {elapsed:.0f}ms")
                                    await self._send_callback_partial(
                                        task, block.text.strip(), elapsed
                                    )
                            elif isinstance(block, sdk.ToolUseBlock):
                                tool_name = block.name
                                tool_input = json.dumps(block.input)[:200]
                                task.tool_uses.append(tool_name)
                                log(task.id, f"Tool call: {tool_name}({tool_input})")
                            elif isinstance(block, sdk.ToolResultBlock):
                                content = (
                                    block.content
                                    if isinstance(block.content, str)
                                    else (
                                        json.dumps(block.content)[:300]
                                        if block.content
                                        else ""
                                    )
                                )
                                err = " [ERROR]" if block.is_error else ""
                                log(task.id, f"Tool result{err}: {content[:300]}")
                            elif isinstance(block, sdk.ThinkingBlock):
                                log(task.id, f"Thinking: {str(block.thinking)[:80]}")

                elif isinstance(msg, sdk.ResultMessage):
                    task.duration_ms = msg.duration_ms
                    task.cost_usd = msg.total_cost_usd
                    task.turns_used = msg.num_turns
                    self._session_id = msg.session_id
                    log(
                        task.id,
                        f"Result: {msg.num_turns} turns, {msg.duration_ms:.0f}ms, ${msg.total_cost_usd:.4f}, stop={msg.stop_reason}, session={msg.session_id}",
                    )

                elif isinstance(msg, sdk.SystemMessage):
                    log(task.id, f"System: {msg.subtype}")

            task.result = "\n".join(text_parts).strip()
            task.status = "done"
            elapsed = (time.monotonic() - t0) * 1000
            result_preview = (
                task.result[:120].replace("\n", " ") if task.result else "(empty)"
            )
            log(task.id, f"Done in {elapsed:.0f}ms: {result_preview}")

        except Exception as e:
            task.error = str(e)
            task.status = "error"
            log(task.id, f"Error: {e}")
        finally:
            task.finished_at = time.time()

        # Send final callback (full result) if we haven't sent anything yet,
        # or if there's more text than the early callback had
        if task.callback_url and (not first_text_sent or len(text_parts) > 1):
            await self._send_callback(task)

    async def _send_callback_partial(self, task: Task, text: str, elapsed_ms: float):
        payload = {
            "task_id": task.id,
            "status": "done",
            "result": text,
            "error": None,
            "prompt": task.prompt,
            "duration_ms": elapsed_ms,
            "cost_usd": None,
            "early": True,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(task.callback_url, json=payload)
                log(task.id, f"Early callback -> {resp.status_code}")
        except Exception as e:
            log(task.id, f"Early callback failed: {e}")

    async def _send_callback(self, task: Task):
        payload = {
            "task_id": task.id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "prompt": task.prompt,
            "duration_ms": task.duration_ms,
            "cost_usd": task.cost_usd,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(task.callback_url, json=payload)
                log(task.id, f"Callback -> {resp.status_code}")
        except Exception as e:
            log(task.id, f"Callback failed: {e}")

    async def handle_create_task(self, request: web.Request) -> web.Response:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise web.HTTPBadRequest(text='{"error": "prompt required"}')

        task_id = data.get("id", str(uuid.uuid4())[:8])
        task = Task(
            task_id=task_id,
            prompt=prompt,
            callback_url=data.get("callback_url"),
            system_prompt=data.get("system_prompt", _default_system()),
            max_turns=data.get("max_turns", 3),
            allowed_tools=data.get("allowed_tools"),
            cwd=data.get("cwd"),
        )
        self.tasks[task_id] = task
        bg = asyncio.create_task(self._run_task(task))
        self._bg_tasks[task_id] = bg
        log(task_id, f"Created (callback={task.callback_url or 'none'})")
        return web.json_response({"task_id": task_id, "status": "pending"}, status=202)

    async def handle_get_task(self, request: web.Request) -> web.Response:
        task_id = request.match_info["id"]
        if task_id not in self.tasks:
            raise web.HTTPNotFound(text='{"error": "task not found"}')
        return web.json_response(self.tasks[task_id].info())

    async def handle_cancel_task(self, request: web.Request) -> web.Response:
        task_id = request.match_info["id"]
        if task_id not in self.tasks:
            raise web.HTTPNotFound(text='{"error": "task not found"}')
        task = self.tasks[task_id]
        task._cancel = True
        if task_id in self._bg_tasks:
            self._bg_tasks[task_id].cancel()
        log(task_id, "Cancel requested")
        return web.json_response({"task_id": task_id, "status": "cancelling"})

    async def handle_list_tasks(self, request: web.Request) -> web.Response:
        return web.json_response([t.info() for t in self.tasks.values()])

    async def handle_capabilities(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"groups": _capability_groups(DEFAULT_TOOLS), "tools": DEFAULT_TOOLS}
        )

    async def handle_clear_session(self, request: web.Request) -> web.Response:
        old = self._session_id
        self._session_id = None
        log("*", f"Session cleared (was {old})")
        return web.json_response({"ok": True, "old_session": old})

    async def handle_clear_tasks(self, request: web.Request) -> web.Response:
        # Cancel any running tasks
        for tid, bg in self._bg_tasks.items():
            if not bg.done():
                self.tasks[tid]._cancel = True
                bg.cancel()
        count = len(self.tasks)
        self.tasks.clear()
        self._bg_tasks.clear()
        log("*", f"Cleared {count} tasks")
        return web.json_response({"ok": True, "cleared": count})

    async def _warmup(self, app):
        """Discover MCP tools and warm up Claude CLI."""
        global DEFAULT_TOOLS
        mcp_tools = await discover_mcp_tools()
        if mcp_tools:
            DEFAULT_TOOLS = mcp_tools + BUILTIN_TOOLS
            for t in mcp_tools:
                print(f"  {t}")

        print(f"[warmup] Warming up Claude ({MODEL})...")
        t0 = time.monotonic()
        try:
            async for msg in sdk.query(
                prompt="Say OK",
                options=sdk.ClaudeAgentOptions(
                    max_turns=1,
                    allowed_tools=[],
                    model=MODEL,
                ),
            ):
                if isinstance(msg, sdk.ResultMessage):
                    elapsed = (time.monotonic() - t0) * 1000
                    print(
                        f"[warmup] Ready in {elapsed:.0f}ms (${msg.total_cost_usd:.4f})"
                    )
        except Exception as e:
            print(f"[warmup] Failed: {e}")

    def run(self):
        app = web.Application()
        app.router.add_post("/task", self.handle_create_task)
        app.router.add_get("/task/{id}", self.handle_get_task)
        app.router.add_delete("/task/{id}", self.handle_cancel_task)
        app.router.add_get("/tasks", self.handle_list_tasks)
        app.router.add_post("/tasks/clear", self.handle_clear_tasks)
        app.router.add_post("/session/clear", self.handle_clear_session)
        app.router.add_get("/capabilities", self.handle_capabilities)
        app.on_startup.append(self._warmup)
        print(f"Task server on http://0.0.0.0:{self.port} (model={MODEL})")
        web.run_app(app, host="0.0.0.0", port=self.port)


if __name__ == "__main__":
    _check_claude_auth()
    port = int(os.environ.get("VUI_TASK_PORT", "8642"))
    TaskServer(port=port).run()
