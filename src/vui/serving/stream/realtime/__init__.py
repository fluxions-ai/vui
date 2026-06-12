"""OpenAI-compatible /v1/realtime endpoint for the streaming server.

Public surface: `handle_realtime_ws(srv, request)` — registered in
`server.py` as `GET /v1/realtime`. The package internals are split as:

  audio.py     encoders + voice discovery
  sinks.py     stub PlaybackSink + RecordingSink (mimic WebRTC tracks)
  adapter.py   RealtimeAdapter — protocol state + response state machine
  outbound.py  internal pipeline events → OpenAI events
  inbound.py   client OpenAI events → vui pipeline calls
  routes.py    RealtimeWSProxy + handle_realtime_ws entrypoint
"""

from .routes import handle_realtime_ws

__all__ = ["handle_realtime_ws"]
