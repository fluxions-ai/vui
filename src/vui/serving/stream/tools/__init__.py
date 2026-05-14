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


def _validate(name: str, mod) -> tuple[dict, str, ToolHandle] | None:
    schema = getattr(mod, "SCHEMA", None)
    rule = getattr(mod, "RULE", "")
    handle = getattr(mod, "handle", None)
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
    return schema, rule or "", handle


def load_tools() -> int:
    """Walk the tools dir, import every `<name>.py`, populate the registry.

    Returns the count of registered tools. Safe to call repeatedly — clears
    and rebuilds in place so callers reading `tools_list()` see the new set.
    """
    _SCHEMAS.clear()
    _RULES.clear()
    _HANDLES.clear()

    files = sorted(p for p in _TOOLS_DIR.glob("*.py") if not p.name.startswith("_"))
    for path in files:
        name = path.stem
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
        schema, rule, handle = result
        _SCHEMAS.append(schema)
        if rule.strip():
            _RULES[name] = rule.strip()
        _HANDLES[name] = handle

    print(f"[tools] registered {len(_HANDLES)}: {', '.join(sorted(_HANDLES))}")
    return len(_HANDLES)


def tools_list() -> list[dict]:
    """OpenAI-style tool schemas in stable (alphabetical) order."""
    return list(_SCHEMAS)


def rules_block() -> str:
    """Per-tool RULE strings concatenated for the system prompt.

    Tools with empty RULE (e.g. no_action, whose policy lives in the
    preamble) are skipped to avoid blank gaps.
    """
    return "\n\n".join(_RULES[name] for name in sorted(_RULES))


def dispatch(name: str) -> ToolHandle | None:
    """Return the handle for a tool name, or None if unknown."""
    return _HANDLES.get(name)


def has_tool(name: str) -> bool:
    return name in _HANDLES
