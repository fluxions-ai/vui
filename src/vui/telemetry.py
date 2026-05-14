import asyncio
import os

import httpx

APP = "vui"
TELEMETRY_URL = os.environ.get(
    "VUI_TELEMETRY_URL",
    "https://fluxions-xyz--fluxions-telemetry-telemetry-track.modal.run",
)
MAX_PENDING = 32

_client: httpx.AsyncClient | None = None
_background_tasks: set[asyncio.Task] = set()
_current_voice: str | None = None

KNOWN_VOICES = {"maeve", "abraham", "rhian", "harry"}


def enabled() -> bool:
    return os.environ.get("VUI_TELEMETRY", "1").lower() not in ("0", "false", "no", "off")


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=5)
    return _client


async def _post(payload: dict) -> None:
    try:
        await _get_client().post(TELEMETRY_URL, json=payload)
    except Exception:
        pass


def record(event_type: str, **meta) -> None:
    if not enabled():
        return
    if len(_background_tasks) >= MAX_PENDING:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    payload = {
        "app": APP,
        "event_type": event_type,
        **{k: v for k, v in meta.items() if v is not None},
    }
    task = loop.create_task(_post(payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def record_voice_load(voice: str) -> None:
    global _current_voice
    _current_voice = voice if voice in KNOWN_VOICES else "clone"


def record_render(seconds: float) -> None:
    record("render", seconds=round(seconds, 3), voice=_current_voice)
