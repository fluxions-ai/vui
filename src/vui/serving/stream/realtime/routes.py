"""WebSocket entrypoint for `/v1/realtime`.

Hijacks `srv.session.ws` / `playback_track` / `recording_sink` for the
duration of the connection so the existing pipeline drives the realtime
client without modification. Single-tenant: one connection at a time.
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import web

import vui.serving.stream.server as _srv_mod

from .adapter import RealtimeAdapter
from .inbound import handle_client_event
from .outbound import handle_internal_event
from .sinks import RealtimePlaybackSink, RealtimeRecordingSink


class RealtimeWSProxy:
    """Stand-in for aiohttp.WebSocketResponse from the pipeline's POV.
    Routes outgoing internal events through the adapter for translation."""

    def __init__(self, ws: web.WebSocketResponse, adapter: RealtimeAdapter):
        self._ws = ws
        self._adapter = adapter

    @property
    def closed(self) -> bool:
        return self._ws.closed

    async def send_json(self, msg: dict):
        await handle_internal_event(self._adapter, msg)

    async def send_str(self, s: str):
        await self._ws.send_str(s)


_active_lock = asyncio.Lock()


async def handle_realtime_ws(srv, request):
    # TODO(auth): validate `Authorization: Bearer <token>` against an env
    # var (e.g. VUI_REALTIME_TOKEN) before upgrading. For now, open access —
    # intended to run behind localhost / a trusted proxy only.

    if _active_lock.locked():
        return web.Response(
            status=409, text="Another realtime client is already connected"
        )

    async with _active_lock:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        adapter = RealtimeAdapter(ws, srv)
        adapter.start_pump()

        saved_ws = srv.session.ws
        saved_pt = srv.session.playback_track
        saved_sink = srv.session.recording_sink
        saved_server_vad = srv._server_vad

        srv.session.ws = RealtimeWSProxy(ws, adapter)  # type: ignore[assignment]
        srv.session.playback_track = RealtimePlaybackSink(adapter.on_tts_audio)  # type: ignore[assignment]

        # If a real AudioRecordingSink was created via /offer, leave it; otherwise
        # install our stub so vad_start can drive the codec stream lifecycle.
        if saved_sink is None or not isinstance(
            saved_sink, _srv_mod.AudioRecordingSink
        ):
            srv.session.recording_sink = RealtimeRecordingSink(srv)  # type: ignore[assignment]

        # Each realtime connection is a fresh session (OpenAI semantics) —
        # clear conversation history + rewind TTS KV to the loaded prompt so
        # state from prior connections doesn't leak in.
        srv.session.conversation.clear()
        srv.conv_ctx = 0
        srv.thoughts_ctx = 0
        # Zero turn counters BEFORE rewind: the worker emits a "done" reply
        # to rewind, and drain_tts_audio would otherwise see stale counters
        # from a prior turn (sent==done==N, llm_streaming=False) and spawn
        # turn_done — which races with the next begin_response and fires
        # end_response immediately, killing all output for the new session.
        srv._generates_sent = 0
        srv._generates_done = 0
        srv._llm_streaming = False
        srv._pending_done = None
        srv.tts_cmd_queue.put({"cmd": "rewind"})
        srv.session.cancel_generation = False

        srv._server_vad = True
        if srv.session.recording_sink:
            srv.session.recording_sink.vad_enabled = True
        srv.asr_cmd_queue.put({"cmd": "vad_enable"})
        await srv._unblock_ready("mic")

        print(f"[realtime] connected, session={adapter.session_id}")
        await adapter.send_session_created()

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                    except Exception:
                        await adapter.send_error("invalid_json", "Could not parse")
                        continue
                    await handle_client_event(adapter, event)
                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            print("[realtime] disconnected")
            await adapter.stop_pump()
            srv.session.ws = saved_ws
            srv.session.playback_track = saved_pt
            srv.session.recording_sink = saved_sink
            srv._server_vad = saved_server_vad

        return ws
