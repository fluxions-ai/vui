"""Serper.dev backend — Google SERP via $SERPER_API_KEY.

Routes news-y queries ("news", "headlines", "what's happening") to the
`/news` endpoint, which returns dated headlines with sources. Everything
else hits `/search` and is extracted by precedence:
answerBox -> knowledgeGraph.description -> top-3 organic snippets.
"""

from __future__ import annotations

import os
import re

import httpx

from vui.geo import detect_country

SEARCH_ENDPOINT = "https://google.serper.dev/search"
NEWS_ENDPOINT = "https://google.serper.dev/news"
TIMEOUT = 5.0
NUM_RESULTS = 5
NEWS_TOP_N = 5  # headlines to read out
# `gl` (geolocation) biases results toward this country. Explicit override
# wins; otherwise we auto-detect from locale/timezone.
COUNTRY = os.environ.get("VUI_SERPER_GL") or detect_country()

# Heuristic: phrases that mean the user wants current headlines, not a
# generic web lookup. We don't match bare "news" alone — it appears in
# product searches like "best news app", so we require it to be paired
# with a temporal/scope cue ("the news", "news today", "news in london",
# "latest news", etc.).
_NEWS_PATTERN = re.compile(
    r"\b(?:"
    r"headlines?|"
    r"breaking news|"
    r"top stories|"
    r"(?:the|any|latest|today'?s|this\s+(?:morning|afternoon|evening))\s+news|"
    r"news\s+(?:today|tonight|this\s+(?:morning|afternoon|evening|week)|"
    r"in|from|on|about|for|update)|"
    r"what'?s\s+(?:happening|going on|in the news)|"
    r"what\s+is\s+happening|"
    r"what\s+happened\s+today"
    r")\b",
    re.IGNORECASE,
)


def _is_news_query(query: str) -> bool:
    return bool(_NEWS_PATTERN.search(query))


def _extract_search(payload: dict) -> str:
    ab = payload.get("answerBox") or {}
    if isinstance(ab, dict):
        if ans := ab.get("answer"):
            snip = ab.get("snippet") or ""
            return f"{ans}. {snip}" if snip and snip.lower() != ans.lower() else ans
        if snip := ab.get("snippet"):
            return snip

    kg = payload.get("knowledgeGraph") or {}
    if isinstance(kg, dict) and (desc := kg.get("description")):
        return desc

    organic = payload.get("organic") or []
    return " ".join(
        o.get("snippet", "").strip()
        for o in organic[:3]
        if isinstance(o, dict) and o.get("snippet")
    ).strip()


def _extract_news(payload: dict) -> str:
    """Render top-N headlines as a single speakable string for the relay LLM
    to paraphrase. Includes source so the LLM can attribute if it wants."""
    items = payload.get("news") or []
    lines: list[str] = []
    for item in items[:NEWS_TOP_N]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        source = (item.get("source") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        # Compact "Title (Source): snippet" — the relay LLM strips/keeps as fits.
        parts = [title]
        if source:
            parts.append(f"({source})")
        prefix = " ".join(parts)
        lines.append(f"{prefix}: {snippet}" if snippet else prefix)
    return "\n".join(lines)


async def search(query: str) -> str | None:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return None

    use_news = _is_news_query(query)
    endpoint = NEWS_ENDPOINT if use_news else SEARCH_ENDPOINT
    body: dict = {"q": query, "gl": COUNTRY}
    if not use_news:
        body["num"] = NUM_RESULTS

    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(
            endpoint,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        payload = r.json()

    extracted = _extract_news(payload) if use_news else _extract_search(payload)
    return extracted or None
