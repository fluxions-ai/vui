"""RealtimeAdapter: holds protocol state + the response state machine.

Outbound and inbound event translation live in `outbound.py` / `inbound.py`
as free functions taking an adapter — same pattern as the rest of the
streaming server (drains.py, voice_turn.py).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiohttp import web

from vui.serving.stream._log import _slog

from .audio import (
    audio_to_pcm16_b64,
    list_voices,
    new_event_id,
    new_item_id,
    new_response_id,
    new_session_id,
)

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


class RealtimeAdapter:
    def __init__(self, ws: web.WebSocketResponse, srv: "StreamServer"):
        self.ws = ws
        self.srv = srv
        self.session_id = new_session_id()

        self.voice: str | None = None
        self.instructions: str | None = None
        self.tools: list[dict] = []

        self.response_id: str | None = None
        self.assistant_item_id: str | None = None
        self.user_item_id: str | None = None
        self.transcript_buffer: str = ""
        self.audio_started: bool = False
        self.last_partial_emitted: str = ""

        self._out_queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self._pump_task: asyncio.Task | None = None

    # --- pump / send ---

    def start_pump(self):
        self._pump_task = asyncio.create_task(self._pump())

    async def stop_pump(self):
        if self._pump_task:
            self._out_queue.put_nowait(None)
            try:
                await asyncio.wait_for(self._pump_task, timeout=1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._pump_task.cancel()

    async def _pump(self):
        while True:
            event = await self._out_queue.get()
            if event is None:
                return
            try:
                await self.ws.send_json(event)
            except ConnectionResetError:
                return
            except Exception as e:
                _slog(f"[realtime] pump send error: {e}")
                return

    def _enqueue(self, event: dict):
        if "event_id" not in event:
            event["event_id"] = new_event_id()
        try:
            self._out_queue.put_nowait(event)
        except asyncio.QueueFull:
            _slog("[realtime] out queue full, dropping event")

    async def send(self, event: dict):
        self._enqueue(event)

    # --- session.created/updated payloads ---

    def session_obj(self) -> dict:
        return {
            "id": self.session_id,
            "object": "realtime.session",
            "model": "vui-stream-1",
            "modalities": ["audio", "text"],
            "instructions": self.instructions or self.srv.session.soul,
            "voice": self.voice or "default",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": (
                {"type": "server_vad"} if self.srv._server_vad else None
            ),
            "tools": self.tools,
            "tool_choice": "auto" if self.tools else "none",
            "voices_available": list_voices(),
        }

    async def send_session_created(self):
        await self.send({"type": "session.created", "session": self.session_obj()})

    async def send_session_updated(self):
        await self.send({"type": "session.updated", "session": self.session_obj()})

    async def send_error(self, code: str, message: str, event_id: str | None = None):
        await self.send(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": code,
                    "message": message,
                    "event_id": event_id,
                },
            }
        )

    # --- audio passthrough (sync — invoked from playback sink) ---

    def on_tts_audio(self, data):
        if not self.response_id or not self.assistant_item_id:
            return
        self.audio_started = True
        b64 = audio_to_pcm16_b64(data)
        self._enqueue(
            {
                "type": "response.audio.delta",
                "response_id": self.response_id,
                "item_id": self.assistant_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": b64,
            }
        )

    # --- response state-machine helpers (used by both directions) ---

    async def ensure_user_item(self, transcript_so_far: str):
        if self.user_item_id:
            return
        self.user_item_id = new_item_id()
        await self.send(
            {
                "type": "conversation.item.created",
                "item": {
                    "id": self.user_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "transcript": transcript_so_far}
                    ],
                },
            }
        )

    async def emit_transcription_delta(self, text: str):
        if not text:
            return
        await self.ensure_user_item(text)
        prev = self.last_partial_emitted
        if text == prev:
            return
        delta = text[len(prev) :] if text.startswith(prev) else text
        self.last_partial_emitted = text
        if not delta:
            return
        await self.send(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": self.user_item_id,
                "content_index": 0,
                "delta": delta,
            }
        )

    async def begin_response(self):
        if self.response_id:
            return
        self.response_id = new_response_id()
        self.assistant_item_id = new_item_id()
        self.audio_started = False
        self.transcript_buffer = ""
        await self.send(
            {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                },
            }
        )
        await self.send(
            {
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": 0,
                "item": {
                    "id": self.assistant_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                },
            }
        )
        await self.send(
            {
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": self.assistant_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio"},
            }
        )

    async def end_response(self, status: str = "completed"):
        if not self.response_id:
            return
        rid = self.response_id
        aid = self.assistant_item_id
        transcript = self.transcript_buffer
        self.response_id = None
        self.assistant_item_id = None
        self.user_item_id = None

        await self.send(
            {
                "type": "response.audio.done",
                "response_id": rid,
                "item_id": aid,
                "output_index": 0,
                "content_index": 0,
            }
        )
        await self.send(
            {
                "type": "response.audio_transcript.done",
                "response_id": rid,
                "item_id": aid,
                "output_index": 0,
                "content_index": 0,
                "transcript": transcript,
            }
        )
        await self.send(
            {
                "type": "response.content_part.done",
                "response_id": rid,
                "item_id": aid,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio", "transcript": transcript},
            }
        )
        await self.send(
            {
                "type": "response.output_item.done",
                "response_id": rid,
                "output_index": 0,
                "item": {
                    "id": aid,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": status,
                    "content": [{"type": "audio", "transcript": transcript}],
                },
            }
        )
        await self.send(
            {
                "type": "response.done",
                "response": {
                    "id": rid,
                    "object": "realtime.response",
                    "status": status,
                    "output": [
                        {
                            "id": aid,
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "audio", "transcript": transcript}],
                        }
                    ],
                },
            }
        )
