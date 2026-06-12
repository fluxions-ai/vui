"""Text helpers shared across the streaming server + test routes."""

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U0000fe00-\U0000fe0f"
    "\U0000200d"
    "\U000023e9-\U000023f3"
    "\U0000231a-\U0000231b"
    "\U00002934-\U00002935"
    "\U000025aa-\U000025fe"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "\U0000203c-\U00003299"
    "]+",
    re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()
