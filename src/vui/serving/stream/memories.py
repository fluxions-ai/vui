"""Persistent voice assistant memories.

Simple file-backed memory store. Memories are short strings that persist
across conversations and server restarts. They appear in the LLM system
prompt via the state block so the conversation model can reference them.

Storage: ~/.vui/memories.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from vui.serving.stream._log import _slog

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer

MEMORIES_PATH = Path.home() / ".vui" / "memories.json"
SEEDED_GEO_MARKER = Path.home() / ".vui" / ".seeded_geo"
MAX_MEMORIES = 15


def load_memories() -> list[dict]:
    if not MEMORIES_PATH.exists():
        return []
    try:
        data = json.loads(MEMORIES_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception as e:
        _slog(f"[memories] load failed: {e}")
        return []


def save_memories(memories: list[dict]):
    MEMORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORIES_PATH.write_text(json.dumps(memories, indent=2))


def _relative_time(ts: float) -> str:
    delta = time.time() - ts
    if delta < 120:
        return "just now"
    if delta < 3600:
        mins = int(delta / 60)
        return f"{mins} min ago"
    if delta < 86400:
        hrs = int(delta / 3600)
        return f"{hrs}h ago" if hrs == 1 else f"{hrs}h ago"
    days = int(delta / 86400)
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        weeks = days // 7
        return f"{weeks}w ago" if weeks == 1 else f"{weeks}w ago"
    months = days // 30
    return f"{months}mo ago" if months == 1 else f"{months}mo ago"


def memories_to_strings(memories: list[dict]) -> list[str]:
    out = []
    for m in memories:
        ts = m.get("created")
        if ts:
            out.append(f"({_relative_time(ts)}) {m['text']}")
        else:
            out.append(m["text"])
    return out


def add_memory(srv: StreamServer, text: str, replaces: str | None = None) -> str:
    text = text.strip()
    if not text:
        return "Nothing to remember."

    for m in srv._memories_store:
        if m["text"].lower() == text.lower():
            return "I already know that."

    if replaces:
        replaces_lower = replaces.lower().strip()
        for i, m in enumerate(srv._memories_store):
            if (
                replaces_lower in m["text"].lower()
                or m["text"].lower() in replaces_lower
            ):
                old = srv._memories_store.pop(i)
                _slog(f"[memories] replacing '{old['text'][:40]}' with '{text[:40]}'")
                break

    if len(srv._memories_store) >= MAX_MEMORIES:
        oldest = srv._memories_store.pop(0)
        _slog(f"[memories] evicted oldest: '{oldest['text'][:40]}'")

    srv._memories_store.append(
        {
            "text": text,
            "created": time.time(),
        }
    )
    _sync(srv)
    _slog(f"[memories] added: '{text[:60]}'")
    return f"Got it, I'll remember that."


_STOP = {
    "a", "an", "the", "my", "of", "is", "i", "am", "i'm", "about", "that",
    "to", "and", "any", "all", "me", "for", "on", "with", "have", "had",
    "from", "in", "or", "be", "are", "was", "were", "this", "these", "those",
}


def remove_memories_by_indices(srv: StreamServer, indices: list[int]) -> str:
    """Remove memories at the given 1-based indices (as shown to the
    thoughts LLM). Out-of-range indices are silently ignored.
    """
    if not indices:
        return "No memories specified."
    # Dedupe + convert to 0-based + clamp.
    zero = sorted({int(i) - 1 for i in indices}, reverse=True)
    removed: list[str] = []
    for z in zero:
        if 0 <= z < len(srv._memories_store):
            removed.append(srv._memories_store.pop(z)["text"])
    if not removed:
        return "I couldn't find those memories."
    removed.reverse()
    _sync(srv)
    _slog(f"[memories] removed by index {len(removed)}: {removed}")
    if len(removed) == 1:
        return f"Done, I forgot about: {removed[0]}"
    return f"Done, I forgot {len(removed)} things."


def remove_memory(srv: StreamServer, query: str) -> str:
    """Remove every memory matching `query`.

    Supports both targeted ("forget I have a dog") and topical bulk
    ("remove memories about my job") removals. Earlier behaviour returned a
    single best match; that meant "memories about my job" only cleared one
    job-related memory at a time.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return "What should I forget?"

    # Pass 1 — substring match either direction: high-confidence, takes ALL.
    matches: list[int] = []
    for i, m in enumerate(srv._memories_store):
        text_lower = m["text"].lower()
        if query_lower in text_lower or text_lower in query_lower:
            matches.append(i)

    # Pass 2 — shared-word overlap. Drop stopwords so "my", "i", "to" don't
    # trigger false positives. Match if ANY non-stopword query word appears
    # in the memory. The LLM is expected to broaden the query for topical
    # asks ("memories about my job" → query="job work career employer").
    if not matches:
        words = [w for w in query_lower.split() if w not in _STOP]
        if not words:
            words = query_lower.split()
        for i, m in enumerate(srv._memories_store):
            text_lower = m["text"].lower()
            if any(w in text_lower for w in words):
                matches.append(i)

    if not matches:
        return "I couldn't find a matching memory."

    # Pop in reverse so indices stay valid.
    removed_texts: list[str] = []
    for i in sorted(matches, reverse=True):
        removed_texts.append(srv._memories_store.pop(i)["text"])
    removed_texts.reverse()  # display order matches storage order
    _sync(srv)
    _slog(f"[memories] removed {len(removed_texts)}: {removed_texts}")
    if len(removed_texts) == 1:
        return f"Done, I forgot about: {removed_texts[0]}"
    return f"Done, I forgot {len(removed_texts)} things."


def list_memories(srv: StreamServer) -> str | None:
    if not srv._memories_store:
        return "I don't have any memories stored yet."
    return None


def clear_memories(srv: StreamServer) -> str:
    count = len(srv._memories_store)
    srv._memories_store.clear()
    _sync(srv)
    _slog(f"[memories] cleared {count} memories")
    if count:
        return f"Done, I cleared all {count} memories."
    return "There were no memories to clear."


def _sync(srv: StreamServer):
    save_memories(srv._memories_store)
    srv._memories = memories_to_strings(srv._memories_store)
    _push_to_client(srv)


def seed_geo_memory(store: list[dict]) -> bool:
    """Idempotently add a 'user is based in X' memory derived from the host
    locale/timezone. Called once on server startup, before the store goes
    into the system prompt. Returns True if a memory was added.

    Respects deletion: if the seed has been added before (marker file
    present) and the user removed it, we leave it removed.
    """
    from vui.geo import country_name, detect_country

    if any(m.get("source") == "system_geo" for m in store):
        return False  # already there
    if SEEDED_GEO_MARKER.exists():
        return False  # was there, user deleted — respect that

    cc = detect_country()
    text = f"User is based in {country_name(cc)}."
    store.append({"text": text, "created": time.time(), "source": "system_geo"})
    save_memories(store)
    try:
        SEEDED_GEO_MARKER.parent.mkdir(parents=True, exist_ok=True)
        SEEDED_GEO_MARKER.touch()
    except OSError as e:
        _slog(f"[memories] couldn't write geo marker: {e}")
    _slog(f"[memories] seeded geo: {text!r}")
    return True


def _push_to_client(srv: StreamServer):
    ws = srv.session.ws
    if ws and not ws.closed:
        import asyncio

        asyncio.ensure_future(
            ws.send_json(
                {
                    "type": "memories",
                    "memories": srv._memories,
                }
            )
        )


# --- HTTP routes ---


async def handle_list_memories(srv: StreamServer, request) -> web.Response:
    return web.json_response(
        {
            "memories": srv._memories_store,
            "count": len(srv._memories_store),
            "max": MAX_MEMORIES,
        }
    )


async def handle_add_memory(srv: StreamServer, request) -> web.Response:
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return web.json_response({"ok": False, "error": "text required"}, status=400)
    msg = add_memory(srv, text)
    return web.json_response(
        {"ok": True, "message": msg, "count": len(srv._memories_store)}
    )


async def handle_remove_memory(srv: StreamServer, request) -> web.Response:
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        return web.json_response({"ok": False, "error": "query required"}, status=400)
    msg = remove_memory(srv, query)
    return web.json_response(
        {"ok": True, "message": msg, "count": len(srv._memories_store)}
    )


async def handle_clear_memories(srv: StreamServer, request) -> web.Response:
    msg = clear_memories(srv)
    return web.json_response({"ok": True, "message": msg})


async def handle_delete_memory_by_index(srv: StreamServer, request) -> web.Response:
    """`POST /memories/delete {index: N}` — remove the Nth memory (0-based,
    matches the order pushed to the UI). For the per-row × button."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid json"}, status=400)
    try:
        idx = int(data.get("index"))
    except (TypeError, ValueError):
        return web.json_response({"ok": False, "error": "index required"}, status=400)
    if not (0 <= idx < len(srv._memories_store)):
        return web.json_response({"ok": False, "error": "out of range"}, status=404)
    msg = remove_memories_by_indices(srv, [idx + 1])
    return web.json_response(
        {"ok": True, "message": msg, "count": len(srv._memories_store)}
    )


def bind(cls):
    cls.handle_list_memories = lambda self, *a, **kw: handle_list_memories(
        self, *a, **kw
    )
    cls.handle_add_memory = lambda self, *a, **kw: handle_add_memory(self, *a, **kw)
    cls.handle_remove_memory = lambda self, *a, **kw: handle_remove_memory(
        self, *a, **kw
    )
    cls.handle_clear_memories = lambda self, *a, **kw: handle_clear_memories(
        self, *a, **kw
    )
    cls.handle_delete_memory_by_index = (
        lambda self, *a, **kw: handle_delete_memory_by_index(self, *a, **kw)
    )
