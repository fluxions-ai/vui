"""Tavily backend — $TAVILY_API_KEY.

Tavily is designed for AI agents — its `answer` field is a pre-summarised
response, so the relay paraphrase has less work to do. Free tier: 1000
queries/month at https://tavily.com.
"""

from __future__ import annotations

import os

import httpx

ENDPOINT = "https://api.tavily.com/search"
TIMEOUT = 8.0  # Tavily's answer synthesis adds latency vs. raw search APIs.


def _extract(payload: dict) -> str:
    if ans := payload.get("answer"):
        return ans
    results = payload.get("results") or []
    return " ".join(
        r.get("content", "").strip()
        for r in results[:3]
        if isinstance(r, dict) and r.get("content")
    ).strip()


async def search(query: str) -> str | None:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(
            ENDPOINT,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 5,
            },
        )
        r.raise_for_status()
        return _extract(r.json()) or None
