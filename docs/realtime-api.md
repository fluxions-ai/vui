# Realtime API

Vui exposes an **OpenAI Realtime-compatible WebSocket** at `/v1/realtime`. Clients written for OpenAI's Realtime API can point at the Vui server and (mostly) just work — same event names, same audio format, same flow.

This is the recommended way to integrate Vui into your own application. The browser UI also uses the older `/ws` + `/offer` (WebSocket + WebRTC) protocol, but that's an internal surface and may change.

## Connect

```
ws://localhost:8080/v1/realtime
```

- Audio is **PCM16 little-endian, 24 kHz, mono, base64-encoded** in `input_audio_buffer.append` and `response.audio.delta` payloads — same as OpenAI's Realtime API.
- **Single-tenant**: one connection at a time. A second connection while one is open returns HTTP 409.
- **No auth currently** (`TODO(auth)` in `realtime/routes.py`). Run behind a trusted proxy or localhost only.
- Each connection is a fresh session: conversation history clears, server VAD turns on, the loaded voice prompt's KV is rewound to its prefill state.

## Supported client events

| Event | Status | Notes |
|---|---|---|
| `session.update` | ✅ | Updates `voice`, `instructions`, `turn_detection`, `tools`, `temperature`. |
| `input_audio_buffer.append` | ✅ | PCM16 b64 @ 24 kHz. Resampled to 16 kHz internally for ASR. |
| `input_audio_buffer.commit` | ⚠️ ignored | Server VAD drives commits; manual commit is no-op. |
| `input_audio_buffer.clear` | ✅ | Hard-resets the ASR session. |
| `response.create` | ✅ | Triggers reply on the most recent user message. `response.instructions` overrides session prompt for that turn. |
| `response.cancel` | ✅ | Aborts in-flight TTS, rewinds KV, drops the half-generated assistant message. |
| `conversation.item.create` (`message`) | ✅ | Append a message to history. Useful for seeding context without speaking. |
| `conversation.item.create` (`function_call_output`) | ⚠️ partial | Appended raw to history; tool-call wiring to come. |
| `conversation.item.delete` | ✅ | Only the last user item, by id. |
| `conversation.item.truncate` | ❌ no-op | Pipeline doesn't model per-item audio rewind. |

## Server events emitted

`session.created`, `session.updated`, `error`,
`input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`,
`conversation.item.created`, `conversation.item.deleted`,
`conversation.item.input_audio_transcription.delta`, `conversation.item.input_audio_transcription.completed`,
`response.created`,
`response.audio.delta`, `response.audio.done`,
`response.audio_transcript.delta`, `response.audio_transcript.done`,
`response.done`.

## `session.update` parameters

```json
{
  "type": "session.update",
  "session": {
    "voice": "alloy",                       // filename in prompts/ without .pt
    "instructions": "You are a helpful...", // system prompt
    "turn_detection": {"type": "server_vad"},  // or null for manual (commit currently ignored)
    "temperature": 0.7,                     // TTS sampling temperature
    "tools": [...]                          // forwarded to the LLM tool-use call
  }
}
```

`voice` resolves to `prompts/<name>.pt` (the same files saved from the browser UI). Available voices on the running server can be listed at startup or by inspecting `prompts/`. An unknown voice returns `error: voice_not_found`.

## Minimal client (Python)

```python
import asyncio, base64, json, websockets, soundfile as sf

async def main():
    async with websockets.connect("ws://localhost:8080/v1/realtime") as ws:
        # Load a saved voice + system prompt
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "my_voice",
                "instructions": "Reply in one short sentence.",
                "turn_detection": {"type": "server_vad"},
            }
        }))

        # Stream a wav into the input buffer, 100ms chunks
        audio, sr = sf.read("input.wav", dtype="int16")
        assert sr == 24000
        chunk = sr // 10
        for i in range(0, len(audio), chunk):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio[i:i+chunk].tobytes()).decode(),
            }))

        # Read events; collect output audio
        out = bytearray()
        async for msg in ws:
            ev = json.loads(msg)
            if ev["type"] == "response.audio.delta":
                out += base64.b64decode(ev["delta"])
            elif ev["type"] == "response.done":
                break
        with open("reply.pcm", "wb") as f:
            f.write(out)

asyncio.run(main())
```

## What's not implemented (yet)

- **Manual turn detection** (`turn_detection: null`) — the connection accepts the setting but `input_audio_buffer.commit` is a no-op; server VAD continues to drive commits.
- **Function-call round-trip** — `tools` are forwarded to the LLM and tool calls are detected by the existing thoughts router, but full OpenAI-style function call → `function_call_output` → continued response is partial.
- **Auth** — token validation is a `TODO(auth)` in `realtime/routes.py`. Don't expose this endpoint to the open internet.

## OpenClaw integration (WebSocket)

[OpenClaw](https://github.com/openclaw/openclaw)'s Talk mode talks to the OpenAI Realtime API over WebSocket. Because Vui implements the same protocol, you can point OpenClaw's `openai` realtime provider at the local Vui server and use Vui as the voice front-end of your OpenClaw assistant — your own voice, fully local, no OpenAI key.

In your OpenClaw config:

```jsonc
{
  "plugins": {
    "entries": {
      "voice-call": {
        "config": {
          "realtime": {
            "provider": "openai",
            "providers": {
              "openai": {
                "baseUrl": "ws://localhost:8080/v1/realtime",
                "apiKey": "not-needed",
                "model": "vui",
                "voice": "abraham"
              }
            }
          }
        }
      }
    }
  }
}
```

If your OpenClaw build doesn't honour `baseUrl` for the realtime path (the docs only show `azureEndpoint` as an override), the same trick works via Azure-style config:

```jsonc
"realtime.providers.openai.azureEndpoint": "ws://localhost:8080/v1/realtime"
```

Notes / caveats:
- **Audio format**: Vui sends/receives PCM16 @ 24 kHz. OpenAI's realtime default is also PCM16 (G.711 µ-law is opt-in). If OpenClaw's provider negotiates µ-law, override with `"input_audio_format": "pcm16", "output_audio_format": "pcm16"` in its `session.update`.
- **Single-tenant**: Vui's realtime endpoint accepts one connection at a time (HTTP 409 on the second). Fine for a personal OpenClaw, not for shared deployments.
- **No auth on Vui**: keep this on `localhost` or behind a trusted proxy. OpenClaw will send some `apiKey` value — Vui ignores it.
- **Voice**: `voice` resolves to `prompts/<name>.pt` on the Vui side. Save your voice in the browser UI first or drop a `.pt` into `prompts/`.

If neither override is exposed in your OpenClaw build, the fallback is a 50-line `aiohttp` WS proxy that listens where OpenClaw expects OpenAI and forwards to Vui — the protocol is identical, so it's a passthrough.

## One-shot voice-note endpoint (`POST /v1/voice-note`)

For voice-note style flows — push-to-talk, async transcribe-and-reply, walkie-talkie bridges — Vui exposes a synchronous REST endpoint that runs the whole **ASR → LLM → TTS** pipeline in one HTTP call. Audio in, JSON out (rendered WAV inline as base64, alongside the transcript and reply text).

```sh
curl -sS -X POST http://localhost:8080/v1/voice-note \
  -F "audio=@my-question.wav" \
  | jq -r '.audio' | base64 -d > reply.wav
```

Or in Python:

```python
import base64, requests
r = requests.post(
    "http://localhost:8080/v1/voice-note",
    files={"audio": open("my-question.wav", "rb")},
).json()
print("you:", r["asr_text"])
print("vui:", r["reply_text"])
open("reply.wav", "wb").write(base64.b64decode(r["audio"]))
```

- **Request**: `multipart/form-data` with an `audio` field. Any format `torchaudio` can decode (wav/flac/mp3/ogg/m4a/…), any sample rate — resampled internally.
- **Response**: `application/json` — `{ok, asr_text, reply_text, duration_sec, audio (base64), audio_format: "wav", sample_rate: 24000}`.
- **Pipeline**: same workers as the streaming server — faster-whisper / Moonshine for ASR, the configured Ollama model for the LLM, Vui for TTS. Text chunking (`chunk_text`) is applied automatically inside the TTS worker, so long replies don't blow past `max_per_turn`.
- **Single-tenant**: holds a lock for the duration. Returns **409** if another voice-note is in flight, **or** if a WebRTC `/offer` / `/v1/realtime` client is currently bound (running through would clobber its TTS KV + persistent ASR session).
- **No auth**: keep on `localhost` or behind a trusted proxy. Same posture as `/v1/realtime`.

Useful targets for this endpoint: a Telegram / WhatsApp / iMessage voice-note bot, an OpenClaw skill that records a clip and posts it, a Shortcuts action on iOS, or Home Assistant's "send voice command" automations.
