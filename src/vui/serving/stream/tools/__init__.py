"""Thoughts-stream tool registry.

Each `<name>.py` in this directory defines one tool the local thinking LLM
can pick. See `SPEC.md` for the authoring contract.

The registry is rebuilt from disk on boot and on `POST /tools/reload`, so
hot-adding a tool file (by hand or via codegen) is a one-liner — no server
restart needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

ToolHandle = Callable[..., Awaitable[Any]]

_TOOLS_DIR = Path(__file__).parent
_SCHEMAS: list[dict] = []
_RULES: dict[str, str] = {}
_HANDLES: dict[str, ToolHandle] = {}
_TASKS_META: dict[str, dict] = {}
_REQUIRES_TASK_SERVER: set[str] = set()


def _validate(
    name: str, mod
) -> tuple[dict, str, ToolHandle, dict | None, bool] | None:
    schema = getattr(mod, "SCHEMA", None)
    rule = getattr(mod, "RULE", "")
    handle = getattr(mod, "handle", None)
    task = getattr(mod, "TASK", None)
    requires_task_server = bool(getattr(mod, "REQUIRES_TASK_SERVER", False))
    if not isinstance(schema, dict):
        print(f"[tools] {name}: missing or invalid SCHEMA, skipping")
        return None
    fn = schema.get("function") or {}
    schema_name = fn.get("name")
    if schema_name != name:
        print(
            f"[tools] {name}: SCHEMA.function.name={schema_name!r} doesn't match "
            f"file basename — skipping"
        )
        return None
    if not callable(handle):
        print(f"[tools] {name}: missing async `handle` callable, skipping")
        return None
    task_meta = task if isinstance(task, dict) else None
    return schema, rule or "", handle, task_meta, requires_task_server


def load_tools() -> int:
    """Walk the tools dir, import every `<name>.py`, populate the registry.

    Returns the count of registered tools. Safe to call repeatedly — clears
    and rebuilds in place so callers reading `tools_list()` see the new set.
    """
    _SCHEMAS.clear()
    _RULES.clear()
    _HANDLES.clear()
    _TASKS_META.clear()
    _REQUIRES_TASK_SERVER.clear()

    # Each tool is either a `<name>.py` file or a `<name>/` package with an
    # `__init__.py`. Packages let a tool ship its own helpers / multiple
    # backends without polluting the registry (only `<name>` is registered).
    names: list[str] = []
    for p in _TOOLS_DIR.iterdir():
        if p.name.startswith("_") or p.name == "SPEC.md":
            continue
        if p.is_file() and p.suffix == ".py":
            names.append(p.stem)
        elif p.is_dir() and (p / "__init__.py").exists():
            names.append(p.name)

    for name in sorted(names):
        mod_name = f"vui.serving.stream.tools.{name}"
        try:
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[tools] {name}: import failed: {e}")
            continue

        result = _validate(name, mod)
        if result is None:
            continue
        schema, rule, handle, task_meta, requires_task_server = result
        _SCHEMAS.append(schema)
        if rule.strip():
            _RULES[name] = rule.strip()
        _HANDLES[name] = handle
        if task_meta is not None:
            _TASKS_META[name] = task_meta
        if requires_task_server:
            _REQUIRES_TASK_SERVER.add(name)

    print(f"[tools] registered {len(_HANDLES)}: {', '.join(sorted(_HANDLES))}")
    return len(_HANDLES)


def _schema_name(schema: dict) -> str:
    return (schema.get("function") or {}).get("name", "")


def tools_list(task_server_available: bool = True) -> list[dict]:
    """OpenAI-style tool schemas in stable (alphabetical) order.

    When `task_server_available` is False, tools that declared
    `REQUIRES_TASK_SERVER = True` are filtered out so the evaluator LLM
    doesn't pick a tool whose handler would immediately fail at
    `srv._create_task(...)`.
    """
    if task_server_available:
        return list(_SCHEMAS)
    return [s for s in _SCHEMAS if _schema_name(s) not in _REQUIRES_TASK_SERVER]


def rules_block(task_server_available: bool = True) -> str:
    """Per-tool RULE strings concatenated for the system prompt.

    Tools with empty RULE (e.g. no_action, whose policy lives in the
    preamble) are skipped to avoid blank gaps. When the task server is
    unreachable, RULEs for task-server-only tools are also dropped so the
    prompt doesn't advertise capabilities the evaluator can't act on.
    """
    names = sorted(_RULES)
    if not task_server_available:
        names = [n for n in names if n not in _REQUIRES_TASK_SERVER]
    return "\n\n".join(_RULES[name] for name in names)


def dispatch(name: str) -> ToolHandle | None:
    """Return the handle for a tool name, or None if unknown."""
    return _HANDLES.get(name)


def task_meta(name: str) -> dict | None:
    """Return the optional TASK metadata for a tool, or None.

    A tool exporting `TASK = {"surface": True, ...}` opts into having a
    local-only row pushed to the UI tasks panel each time it fires. See
    `SPEC.md` for the supported keys.
    """
    return _TASKS_META.get(name)


def has_tool(name: str) -> bool:
    return name in _HANDLES
