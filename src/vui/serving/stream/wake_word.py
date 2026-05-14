"""Detect whether a transcript opens with the assistant's wake word.

Used by listen mode: the assistant keeps listening but only responds when
the user prefixes the turn with its name. Whisper transcribes short
proper nouns poorly, so we accept a curated set of variants for "Vui";
custom names fall back to literal matching.
"""

from __future__ import annotations

import re

# Whisper hears "Vui" all over the place. These are the variants observed
# in real transcripts of users calling it.
_VUI_VARIANTS = {
    "vui", "vooey", "voi", "vue", "vuey", "voee", "vooi", "wee", "weeey",
    "veuy", "voo", "ouey", "ooey",
}

# Prefixes the LLM should tolerate before the wake word.
_ADDRESS_PREFIXES = {"hey", "hi", "ok", "okay", "yo", "oi"}

# Explicit wake phrases that don't require the assistant's name.
_WAKE_PHRASES = (
    "wake up",
    "come back",
    "you can talk again",
    "you can chat",
    "you can chat again",
    "alright you can talk",
    "stop being quiet",
    "talk again",
    "speak again",
)

_PUNCT_TRIM = re.compile(r"[,.!?;:]+$")

# Phrases that engage listen mode. Matched as substrings on the lowercased
# transcript — same approach as _WAKE_PHRASES but for the inverse direction.
# Kept narrow on purpose: only phrases the user almost certainly addresses
# to the assistant, not to a third party. "Leave me alone" / "just listen"
# are intentionally excluded because they're ambiguous out of context.
_SHUT_UP_PHRASES = (
    "shut up",
    "be quiet",
    "stop talking",
    "stop replying",
    "stop responding",
    "go silent",
    "be silent",
    "go quiet",
)


def _norm(tok: str) -> str:
    return _PUNCT_TRIM.sub("", tok.lower())


def shut_up_match(text: str) -> bool:
    """True iff the transcript contains a clear request for listen mode."""
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in _SHUT_UP_PHRASES)


def match(text: str, name: str) -> tuple[bool, str]:
    """Return (matched, stripped_text).

    Matches the assistant's name as a vocative ANYWHERE in the transcript:
    "Vui, what time?", "What time is it, Vui?", "Hey Baz can you help", and
    "Listen Baz, the thing is..." all match. Also matches explicit wake
    phrases ("wake up", "come back", "you can talk again") regardless of
    name. Case-insensitive; trims surrounding punctuation. Default "Vui"
    accepts the Whisper variants above; any other name matches literally.
    """
    if not text:
        return False, text

    # Explicit wake phrases — keyword match, no name required.
    low = text.lower()
    for phrase in _WAKE_PHRASES:
        if phrase in low:
            return True, ""

    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return False, text
    name_lower = (name or "").lower().strip()
    variants = _VUI_VARIANTS if name_lower in {"", "vui"} else {name_lower}

    # Find the first occurrence of the name (optionally preceded by a
    # leading "hey"/"hi"/etc.) and remove that span from the text.
    for i, tok in enumerate(tokens):
        if tok.lower() not in variants:
            continue
        start_idx = i
        if i > 0 and tokens[i - 1].lower() in _ADDRESS_PREFIXES:
            start_idx = i - 1
        # Map each token back to its character offsets in the original
        # text so we can preserve spacing when we cut.
        offsets = []
        pos = 0
        for t in tokens:
            j = text.lower().find(t.lower(), pos)
            offsets.append((j, j + len(t)))
            pos = j + len(t)
        cut_from = offsets[start_idx][0]
        cut_to = offsets[i][1]
        stripped = (text[:cut_from] + text[cut_to:]).strip()
        # Tidy dangling punctuation: leading punct/comma, double spaces,
        # and any orphaned ",?" / ", ." sequences caused by removing
        # a vocative like "What do you think, Baz?".
        stripped = re.sub(r"^[\s,.!?;:]+", "", stripped)
        stripped = re.sub(r"\s+([,.!?;:])", r"\1", stripped)
        stripped = re.sub(r"\s+", " ", stripped)
        stripped = re.sub(r",\s*([?!.])", r"\1", stripped)
        return True, stripped
    return False, text
