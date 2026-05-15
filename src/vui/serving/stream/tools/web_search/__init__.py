"""Quick factual web lookup, pluggable backend.

Backend is picked at call time from `VUI_SEARCH_PROVIDER` (`serper` | `brave`
| `tavily`), defaulting to whichever backend has its API key set.

Add a new backend by dropping a `<name>.py` next to this file that exposes
`async def search(query: str) -> str | None` returning the best speakable
snippet (or `None` for no usable answer), then list it in `_PROVIDERS`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Awaitable, Callable

from vui.serving.stream._log import _slog
from vui.serving.stream.tools.web_search import brave, serper, tavily

if TYPE_CHECKING:
    from vui.serving.stream.thoughts import ThoughtsStream


Provider = Callable[[str], Awaitable[str | None]]

# Provider name -> (search fn, env var that gates availability)
_PROVIDERS: dict[str, tuple[Provider, str]] = {
    "serper": (serper.search, "SERPER_API_KEY"),
    "brave": (brave.search, "BRAVE_API_KEY"),
    "tavily": (tavily.search, "TAVILY_API_KEY"),
}


def _pick_provider() -> tuple[str, Provider] | None:
    chosen = os.environ.get("VUI_SEARCH_PROVIDER", "").strip().lower()
    if chosen:
        entry = _PROVIDERS.get(chosen)
        if entry and os.environ.get(entry[1]):
            return chosen, entry[0]
        # Explicit choice but key missing — fall through to auto-pick so
        # the assistant can still answer instead of going dark.
        _slog(f"[web_search] VUI_SEARCH_PROVIDER={chosen!r} but {entry[1] if entry else '?'} unset, auto-picking")
    for name, (fn, env) in _PROVIDERS.items():
        if os.environ.get(env):
            return name, fn
    return None


SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Look up a single factual question on the web. Use for short, "
            "live data the assistant cannot answer from training: current "
            "weather, prices, scores, news, definitions, 'who is X'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural search query, e.g. 'weather in london today' "
                        "or 'tesla stock price'. One question, not a brief."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# Surface searches as a task row so the user can see the query and snippet
# after it finishes (defaults: auto_done=True, so dispatcher marks "done"
# when `handle()` returns without manually setting status).
TASK = {"surface": True}


RULE = """\
web_search: user asks for a SHORT, SINGLE-QUERY factual lookup on the public web — something one search engine query would answer.
- MUST fire when the user literally says "search the web", "google X", "look up X", "what's the latest news/headlines", "what's happening in X", "who won X", "what's the score", "what's the weather", "price of X", "what time is it in X", "who is X", "what is X".
- Use this over delegate for ANY single-query factual lookup, even if the user adds polite framing ("could you...", "please..."). Faster than delegate (one API round-trip vs. agent loop).
- NOT for: anything needing PERSONAL account access (email, calendar, files) — that's delegate. NOT for general knowledge already in training (recipes, how-tos, history, coding explanations) — answer directly.
- NOT for multi-step research briefs: if the request implies looking up MULTIPLE entities, COMPARING things, SUMMARISING across sources, or chaining results ("research top 5 X and summarise", "compare A vs B in depth", "find info on these 3 companies and tell me which is best", "do a writeup on X") — that's delegate. The giveaway is plural entities + verbs like "research", "compare", "summarise", "writeup", "analyse", "investigate". A single search engine query won't answer those.
"""


def provider_status() -> str:
    """One-line summary of which providers have a key set. Printed at boot
    so a missing key surfaces immediately, not at first 'search the web'."""
    avail = [name for name, (_, env) in _PROVIDERS.items() if os.environ.get(env)]
    if not avail:
        return (
            "[web_search] WARN: no provider key configured — searches will fail. "
            "Set one of: " + ", ".join(env for _, env in _PROVIDERS.values())
        )
    override = os.environ.get("VUI_SEARCH_PROVIDER", "").strip().lower()
    active = override if (override in avail) else avail[0]
    extra = f" (override via VUI_SEARCH_PROVIDER={'|'.join(avail)})" if len(avail) > 1 else ""
    return f"[web_search] active={active} available={avail}{extra}"


# Surface key-presence at boot. Imported via load_tools(), so this fires once
# during server startup (and once on each /tools/reload).
print(provider_status())


async def handle(ctx: "ThoughtsStream", **args) -> None:
    query = (args.get("query") or "").strip()
    task = ctx.task
    if not query:
        if task:
            task.update(status="error", error="empty query")
        return

    if task:
        task.update(description=f"web: {query}")

    pick = _pick_provider()
    if pick is None:
        _slog("[web_search] no provider key configured")
        if task:
            task.update(status="error", error="no provider key configured")
        await ctx._speak("I can't search the web — no search API key is configured.")
        return

    name, search = pick
    _slog(f"[web_search] provider={name} query={query!r}")
    try:
        snippet = await search(query)
    except Exception as e:
        _slog(f"[web_search] {name} error: {e}")
        if task:
            task.update(status="error", error=f"{name}: {e}")
        await ctx._speak("The web search failed.")
        return

    if not snippet:
        _slog(f"[web_search] {name} returned no usable snippet")
        if task:
            task.update(status="done", result="(no usable answer)")
        await ctx._speak("I didn't find a clear answer.")
        return

    _slog(f"[web_search] {name} snippet ({len(snippet)} chars): {snippet[:120]}")
    if task:
        task.update(status="done", result=snippet)
    # The relay LLM paraphrases this into a soul-styled spoken utterance.
    await ctx._speak(snippet)
