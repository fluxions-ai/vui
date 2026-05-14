import asyncio
import time
import traceback
from collections.abc import Coroutine
from pathlib import Path

_USER_PREFILL_MIN_WORDS = 8

_SRV_T0 = time.monotonic()
_SRV_LOG_DIR = Path("debug_dump")
_SRV_LOG_DIR.mkdir(exist_ok=True)
_SRV_LOG_PATH = _SRV_LOG_DIR / "server.log"
_SRV_LOG_F = None


def _slog(msg: str) -> None:
    global _SRV_LOG_F
    stamp = f"[t={int((time.monotonic() - _SRV_T0) * 1000):>7d}ms] "
    line = stamp + msg
    print(line, flush=True)
    if _SRV_LOG_F is None:
        _SRV_LOG_F = open(_SRV_LOG_PATH, "a", buffering=1)
        _SRV_LOG_F.write(
            f"\n--- server start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
    _SRV_LOG_F.write(line + "\n")


def _spawn(coro: Coroutine, name: str) -> asyncio.Task:
    """Fire-and-forget task with crash-visible logging.

    Plain `asyncio.create_task` lets exceptions die with a single
    `Task exception was never retrieved` warning that's easy to miss in
    server logs and that has bitten this codebase before (drain death).
    Use this for any background coroutine whose result we don't await.
    """
    task = asyncio.create_task(coro, name=name)

    def _on_done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is None:
            return
        _slog(f"[task:{name}] crashed: {exc!r}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    task.add_done_callback(_on_done)
    return task


def _spawn_response(srv, coro: Coroutine, name: str) -> asyncio.Task:
    """`_spawn` + register the task in `srv._response_tasks`.

    Use for any task that streams `reply` chunks to the frontend WS. On
    `_reset_session_state` these are cancelled and awaited so no late
    reply lands after `context_cleared` and re-populates the chat.
    """
    task = _spawn(coro, name)
    srv._response_tasks.add(task)
    task.add_done_callback(srv._response_tasks.discard)
    return task
