"""Brave Search backend — $BRAVE_API_KEY.

Free tier: 2000 queries/month at https://api.search.brave.com/. Returns the
best speakable snippet, preferring infobox/discussion -> first web result.
"""

from __future__ import annotations

import os

import httpx

ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TIMEOUT = 5.0
COUNTRY = os.environ.get("VUI_BRAVE_COUNTRY", "GB")


def _extract(payload: dict) -> str:
    # Infobox: Brave's "knowledge panel" — usually the cleanest direct answer.
    infobox = payload.get("infobox") or {}
    if isinstance(infobox, dict):
        results = infobox.get("results") or []
        if results and isinstance(results[0], dict):
            if desc := results[0].get("description"):
                return desc

    # FAQ-style instant answers.
    faq = (payload.get("faq") or {}).get("results") or []
    if faq and isinstance(faq[0], dict):
        if answer := faq[0].get("answer"):
            return answer

    web = (payload.get("web") or {}).get("results") or []
    return " ".join(
        r.get("description", "").strip()
        for r in web[:3]
        if isinstance(r, dict) and r.get("description")
    ).strip()


async def search(query: str) -> str | None:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return None
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get(
            ENDPOINT,
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            params={"q": query, "country": COUNTRY, "count": 5},
        )
        r.raise_for_status()
        return _extract(r.json()) or None
