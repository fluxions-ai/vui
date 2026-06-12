"""Best-effort host country detection (ISO-3166 alpha-2, lowercase).

Resolution order (first hit wins):
  1. $VUI_COUNTRY explicit override
  2. $LANG locale suffix (en_GB.UTF-8 -> gb)
  3. /etc/timezone looked up in /usr/share/zoneinfo/zone.tab
  4. /etc/localtime symlink target (macOS / BSD)
  5. caller-supplied default

Everything is local — no network. Result is cached for the process.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_ZONE_TAB = Path("/usr/share/zoneinfo/zone.tab")
_TZ_FILE = Path("/etc/timezone")
_LANG_RE = re.compile(r"^[a-z]{2,3}_([A-Z]{2})")


@lru_cache(maxsize=1)
def _zone_tab_map() -> dict[str, str]:
    """Parse zone.tab once into {timezone: cc_lowercase}."""
    if not _ZONE_TAB.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        for line in _ZONE_TAB.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                out[parts[2].strip()] = parts[0].strip().lower()
    except OSError:
        return {}
    return out


def _from_timezone() -> str | None:
    tz: str | None = None
    if _TZ_FILE.is_file():
        try:
            tz = _TZ_FILE.read_text().strip() or None
        except OSError:
            pass
    if tz is None:
        # macOS / BSD: /etc/localtime is a symlink into zoneinfo/<TZ>.
        try:
            link = str(Path("/etc/localtime").resolve())
        except OSError:
            return None
        if "zoneinfo/" not in link:
            return None
        tz = link.rsplit("zoneinfo/", 1)[-1]
    return _zone_tab_map().get(tz)


@lru_cache(maxsize=4)
def detect_country(default: str = "gb") -> str:
    if cc := os.environ.get("VUI_COUNTRY", "").strip():
        return cc.lower()
    if m := _LANG_RE.match(os.environ.get("LANG", "")):
        return m.group(1).lower()
    if cc := _from_timezone():
        return cc
    return default


# Common-country name table for human-readable memory text. Anything not
# listed falls back to the uppercase ISO code, which the LLM understands.
_COUNTRY_NAMES = {
    "gb": "the United Kingdom", "us": "the United States", "ca": "Canada",
    "au": "Australia", "nz": "New Zealand", "ie": "Ireland",
    "fr": "France", "de": "Germany", "es": "Spain", "it": "Italy",
    "nl": "the Netherlands", "se": "Sweden", "no": "Norway",
    "dk": "Denmark", "fi": "Finland", "be": "Belgium", "ch": "Switzerland",
    "at": "Austria", "pt": "Portugal", "pl": "Poland", "cz": "Czechia",
    "gr": "Greece", "ro": "Romania", "hu": "Hungary",
    "jp": "Japan", "kr": "South Korea", "cn": "China", "tw": "Taiwan",
    "hk": "Hong Kong", "sg": "Singapore", "in": "India",
    "br": "Brazil", "mx": "Mexico", "ar": "Argentina",
    "ru": "Russia", "ua": "Ukraine", "tr": "Turkey", "il": "Israel",
    "za": "South Africa", "eg": "Egypt", "ng": "Nigeria",
    "ae": "the United Arab Emirates", "sa": "Saudi Arabia",
}


def country_name(cc: str) -> str:
    """Human-readable country name for an ISO-3166 alpha-2 code."""
    return _COUNTRY_NAMES.get(cc.lower(), cc.upper())
