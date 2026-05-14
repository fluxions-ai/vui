# Configuration & UI reference

Three things in one doc:

1. [UI controls](#ui-controls) — every knob and button in the browser
2. [Supported models](#supported-models) — TTS / LLM / ASR
3. [Pointing at a custom model server](#custom-model-server) — env vars + worked examples

---

## UI controls

The browser UI (served at `http://localhost:8080`) is one page divided into sections. Top to bottom:

### Status pills

| Pill | Meaning |
|---|---|
| `tts` | TTS worker subprocess. Green = ready, red = down, yellow = warming up. |
| `asr` | ASR worker subprocess (faster-whisper / Moonshine / mlx-whisper). |
| `mic` | Mic capture + WebRTC pipe + Silero VAD. Green when first frame arrives. |
| `warmup` | Initial prefill / prompt load. Shows reason text on busy. |
| `task server` | Optional Claude task delegation server (`VUI_TASK_SERVER_URL`). Down if unreachable; the UI keeps working without it. |

### Voice prompt

A "voice prompt" is a short reference clip (~3–10s) that primes the TTS to speak in that voice / style for the rest of the session.

| Control | What it does |
|---|---|
| **Saved Prompts** dropdown + **Load** | Reload one previously saved under `prompts/`. The last-used prompt auto-selects on reconnect. |
| **Upload New** | Pick an audio file (`.wav`, `.mp3`, …). Server runs ASR for the transcript, encodes to codec codes, and prefills it as the speaker reference. Encoding is roughly 1 s wall-time per 1 s audio. |
| **Name to save as…** + **Save** | After uploading, type a name and click Save to write `prompts/<name>.pt` for future reuse. |
| **Reset** | Clears the conversation but preserves the loaded prompt and re-prefills it. |
| **Prompt transcript** + **Re-align** | Shows the auto-transcribed text. If ASR misheard a word, edit and click Re-align to re-encode the prompt with the corrected text. Improves cloning quality on hard names / domain words. |
| **Prompt player** | Plays back the loaded reference clip so you can sanity-check it. |

### Talk / mute / VAD

| Control | What it does |
|---|---|
| **Hold to Talk / VAD Active** | Push-to-talk button. With VAD off, hold to record. With VAD on, the button just shows state. **Spacebar** mirrors push-to-talk when VAD is off. |
| **VAD** toggle | Green = automatic turn taking via Silero VAD (default). Grey = manual push-to-talk only. |
| **🔊 / 🔇** | Mute/unmute the assistant's playback locally. Server keeps generating. |
| **Type text to speak…** + **Speak** | Direct TTS: skip ASR/LLM, render the typed text in the current voice. Useful for testing prompts. |
| **Cancel** | Aborts the current generation (LLM + TTS). Appears only while generating. |

### Meters

Three context bars and a mic-level bar:

| Bar | Counts |
|---|---|
| Mic level | Real-time RMS of mic input (green normally, red while recording). |
| **TTS** | Frames used in the TTS context vs. trained ceiling. Audio is ~12.5 frames/sec. Default cap is 180 s (`VUI_MAX_CONTEXT_SECS`). |
| **Conv** | Tokens used / max for the main LLM. From `usage` in the streaming response. |
| **Thoughts** | Tokens used / max for the parallel "thoughts" tool-router LLM call. |

Bars turn yellow at 60% and red at 80%.

### Conversation

Per-message playback: each user/assistant bubble gets a **Play** button that replays the captured WebRTC audio for that turn (the user's own voice for user turns, the synthesized voice for assistant turns).

**Download Audio** stitches every recorded blob into a single WAV (silence-trimmed, 50 ms padding between turns) and downloads it.

### Tasks (auto-shown when active)

Appears only when the assistant has dispatched background tasks via the Claude task server. Each task shows a status pill (`running` / `done` / `cancelled` / `error`) and click-to-expand to see the result text.

### Memories (auto-shown when present)

Persisted per-session facts the LLM has chosen to remember (via the `remember` tool). Stored on the server in plain text. Surface only — there is no edit UI here.

### System Prompt

Free-form textarea. Edits persist on blur. Used as the system message for both the conversation LLM and the thoughts LLM.

### ASR Model

Dropdown of every ASR backend the server can load (see [supported ASR models](#asr-models)). Switch is live — no reconnect needed. Selection is saved in `localStorage` as `vui-asr-model`. The 🔄 refreshes the list.

### LLM Model

Dropdown of models *currently loaded* on the Ollama server (`/api/ps`). Switch is live. The text input + **Pull** button calls `/api/pull` on the server to download a new model — progress streams to the line below.

> Note: this UI talks only to the Ollama backend. With `VUI_LLM_BACKEND=vllm`, the dropdown shows the served model and switching is disabled (vLLM serves one model per process).

### Log

Last 200 server-side log entries, classified `info` (grey), `timing` (blue), `warn` (yellow), `error` (red). Includes TTFR (time to first reply text) and TTFA (time to first audio) per turn.

### Settings

Live-tunable; each change sends a `{type: "settings", ...}` WebSocket message and applies on the next turn.

#### Sampling

| Setting | Default | Effect |
|---|---|---|
| **Temperature** | 0.7 | TTS sampling temperature on the cb0 (semantic) head. Lower = more monotone, higher = more varied. |
| **Top-K** | 50 | Top-k truncation on cb0 logits. |
| **Rep Penalty** | 1.1 | Multiplicative penalty on tokens already seen in the rep window. 1.0 disables. |
| **Rep Window** | 24 | History length (frames) over which Rep Penalty applies (≈2 s of cb0 history at 12.5 Hz). Enough to break stuck-loop artefacts without blocking natural repetition. 0 = off. |
| **EOS Thresh** | 0.4 | Sigmoid threshold for the EOS-of-turn head. Lower = the model ends turns sooner (more clipped); higher = it talks longer. |

#### Conditioning (additive bias on text embeddings — guides the model toward target qualities)

| Setting | Default | Effect |
|---|---|---|
| **WPS** | 0 | Target words-per-second. 0 disables. ~3.0 = brisk, ~1.8 = slow. |
| **DNS Signal / Background** | 0 / 0 | DNSMOS targets (1-5). Bias toward cleaner speech / less background. 0 = off. |
| **NISQA Noise / Disc. / Color. / Loudness** | 0 / 0 / 0 / 5 | NISQA component targets (1-5). Loudness 5 = near-broadcast level. |

These are training-time conditioning labels — the model was trained on labelled clips so setting them at inference biases generation. 0 disables that axis.

#### Streaming / chunking

| Setting | Default | Effect |
|---|---|---|
| **Chunk Words** | 60 | Max words per TTS chunk. Smaller chunks = lower latency but more chunk boundaries (occasional prosody bumps). |
| **First Chunk Words** | 8 | If >0, the very first chunk uses this smaller word count for low TTFB. The remaining text uses Chunk Words. |
| **Codebooks** | 16 | Number of RVQ levels to decode. Lower = faster + lower quality. 16 = full quality. |

#### Behavioural toggles

| Setting | Default | Effect |
|---|---|---|
| **User audio codes** | on | Encode the user's mic audio into Qwen codec codes and feed alongside the ASR transcript. Improves prosody matching of the reply. Off = text-only conditioning, slightly faster. |
| **Incremental align** | off | Re-align the prompt's text/codes alignment between turns. Niche; leave off unless prompt drift is visible. |
| **Keep context** | off | If on, multi-turn audio context is kept across turns (uses `VUI_MAX_CONTEXT_SECS`). Off = each turn re-prefills with prompt only. Off is recommended unless you need cross-turn prosody continuity. |
| **Tool check** | on | Run the parallel "thoughts" LLM call that checks for tool intents (`remember`, `ask_claude`). Off = pure conversation, no memories or task delegation. |

### Settings not in the UI

These live in `DEFAULT_SETTINGS` (`src/vui/serving/stream/server.py`) and are tunable over the WebSocket but not surfaced as inputs:

- `max_duration` (s, default 120) — hard cap on a single TTS turn.
- `vad_stop_secs` (s, 0.3) — Silero silence-to-stop time. Lower = snappier turn-taking, more false cuts.
- `asr_settle_s` (s, 0.12) — wait after VAD stop for the last fwhisper interim.
- `trailing_off_delay` (s, 0.7) — extra wait when ASR text doesn't end in `.?!`.

Send via the `/test/settings` HTTP endpoint or extend the UI.

---

## Supported models

### TTS (the Vui model itself)

One model: **`fluxions/vui` → `vui-nano.safetensors`** (305M params, Llama-style + RQ-Transformer head, Qwen3-TTS codec, 16 kHz output). Auto-downloaded from Hugging Face on first run via `vui.hf.download`. Pass a different path / repo as the first CLI arg to `python -m vui.serving.stream <path-or-repo>`.

### LLM backends

Two backends, selected by env var:

| Backend | `VUI_LLM_BACKEND=` | Default URL | Default model | Notes |
|---|---|---|---|---|
| Ollama | `ollama` (default) | `http://localhost:11434` | `qwen3.5:4b` | Default, GGUF-quantized. UI dropdown can hot-swap and pull new models. On Apple Silicon, an MLX-quantized variant (`qwen3.5-4b-mlx`) is auto-created via `ollama create --experimental --quantize int4` for ~1.9× faster decode. |
| vLLM (or any OpenAI-compatible server) | `vllm` | `http://localhost:8000` | `Qwen/Qwen3.5-4B` | Single model per process; UI dropdown is read-only. Sends Qwen-specific `chat_template_kwargs.enable_thinking=false` — set to `true` only if you want chain-of-thought (kills voice TTFB). |

The default sampling pinned in `llm_backend.py:DEFAULT_SAMPLING` (`temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5`) mirrors the qwen3.5:4b Ollama Modelfile, so vLLM and Ollama produce comparable replies. Override per-call as needed.

**Recommended models** (anything that runs on Ollama or vLLM with the OpenAI chat API works):

- `qwen3:4b` / `qwen3.5:4b` — primary target. Smart enough for tool use, fast TTFB.
- `qwen3:8b`, `qwen3.5:8b` — better quality if you have the VRAM.
- `llama3.2:3b`, `llama3.2:1b` — tested, faster, less smart on tools.
- `gemma3:4b`, `phi4:mini` — work but tool-use accuracy varies.

Two LLM calls per turn: the **conversation** LLM streams the reply text; the **thoughts** LLM runs in parallel for memory / task tool routing (gated by the **Tool check** setting).

### ASR models

Selected from the UI dropdown (live-switchable). Backed by `ASR_MODELS` in `src/vui/serving/stream/asr_worker.py`:

| Key | Backend | Where it runs | Notes |
|---|---|---|---|
| `moonshine.tiny` | moonshine (ONNX) | CPU | Tiny, fastest CPU option. |
| `moonshine.small` | moonshine | CPU | Default-ish CPU choice. |
| `moonshine.medium` | moonshine | CPU | Slowest CPU, best quality of the trio. |
| `fwhisper.distil-small.en` | faster-whisper | GPU | **Default on first start.** English-only, fast, accurate. |
| `fwhisper.small.en` | faster-whisper | GPU | English-only. |
| `fwhisper.distil-medium.en` | faster-whisper | GPU | English-only, larger. |
| `fwhisper.medium.en` | faster-whisper | GPU | English-only. |
| `fwhisper.distil-large-v3` | faster-whisper | GPU | Multilingual, best quality. |
| `fwhisper.turbo` | faster-whisper | GPU | Multilingual, ~similar accuracy to large-v3, faster. |
| `mlx-whisper.small` | mlx-whisper | Apple Silicon | Requires `uv sync --extra mlx`. |
| `mlx-whisper.turbo` | mlx-whisper | Apple Silicon | Requires `uv sync --extra mlx`. |

Default at startup is set in code by `DEFAULT_ASR_MODEL = "fwhisper.distil-small.en"`. The UI remembers the last selection in `localStorage` and switches to it on reconnect.

You can also set startup defaults via env (these only affect the *initial* backend choice; the UI selection takes over after first switch):

```sh
VUI_ASR=moonshine VUI_MOONSHINE_ARCH=4 python -m vui.serving.stream
VUI_ASR=fwhisper VUI_FWHISPER_MODEL=turbo VUI_FWHISPER_DEVICE=cuda python -m vui.serving.stream
```

---

## Custom model server

The most common case: you've got an LLM running somewhere else (a remote vLLM box, a shared Ollama instance, a private OpenAI-compatible endpoint) and want Vui to use it instead of the bundled Ollama.

### Pointing at a remote Ollama

Two env vars (the codebase has both — set them to the same URL):

```sh
export OLLAMA_URL="http://gpu-box.lan:11434"
export VUI_OLLAMA_URL="http://gpu-box.lan:11434"
export VUI_OLLAMA_MODEL="qwen3:8b"

python -m vui.serving.stream
```

`OLLAMA_URL` is used by the model-listing / pull helpers (`llm.py`); `VUI_OLLAMA_URL` is used by the streaming/completion path (`llm_backend.py`). Setting only one will half-work — symptoms include the dropdown listing a different set of models than the one actually used for replies.

### Pointing at vLLM (or any OpenAI-compatible server)

```sh
export VUI_LLM_BACKEND=vllm
export VUI_VLLM_URL="http://my-vllm-host:8000"
export VUI_VLLM_MODEL="Qwen/Qwen3.5-4B"

python -m vui.serving.stream
```

The backend hits `${VUI_VLLM_URL}/v1/chat/completions`, so anything that speaks the OpenAI chat completion API (vLLM, sglang, LM Studio in server mode, llama.cpp `--api`, OpenAI itself) should work. Tools, streaming, and `usage` are decoded.

Caveats:
- The body sets `chat_template_kwargs.enable_thinking=false`. Servers that don't recognise this key just ignore it; servers that pass it through to a non-Qwen template may complain. Patch `_body` in `llm_backend.py:VLLMBackend` if you hit issues.
- Hot-swapping models from the UI is disabled in this mode (vLLM serves one model per process).

### docker-compose

`docker-compose.yml` sets `OLLAMA_URL=http://ollama:11434` for the `vui-stream` service, pointing at the bundled Ollama container. To point at an external server, edit the `vui-stream.environment` block:

```yaml
environment:
  VUI_LLM_BACKEND: "vllm"
  VUI_VLLM_URL: "http://my-vllm-host:8000"
  VUI_VLLM_MODEL: "Qwen/Qwen3.5-4B"
  # Drop OLLAMA_URL if the bundled ollama service isn't running.
```

…and remove the `depends_on: [ollama]` if you're not running the bundled Ollama.

### All env vars at a glance

| Var | Default | Purpose |
|---|---|---|
| `VUI_LLM_BACKEND` | `ollama` | Backend select: `ollama` or `vllm`. |
| `VUI_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (used by the chat backend). |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (used by `/api/ps`, `/api/pull`, MLX detection). Set to the same value as `VUI_OLLAMA_URL`. |
| `VUI_OLLAMA_MODEL` | `qwen3.5:4b` | Initial Ollama model. UI can switch live. |
| `VUI_VLLM_URL` | `http://localhost:8000` | vLLM (or OpenAI-compatible) base URL. |
| `VUI_VLLM_MODEL` | `Qwen/Qwen3.5-4B` | Model id sent to vLLM. |
| `VUI_ASR` | `moonshine` | Initial ASR backend if no model key is requested: `moonshine`, `fwhisper`, `mlx_whisper`. |
| `VUI_MOONSHINE_ARCH` | `4` | Moonshine variant: `0` tiny, `2` tiny-streaming, `4` small-streaming, `5` medium-streaming. |
| `VUI_FWHISPER_MODEL` | `distil-small.en` | faster-whisper model id. |
| `VUI_FWHISPER_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `VUI_TASK_SERVER_URL` | `http://localhost:8642` | Optional Claude task server. |
| `VUI_TASK_PORT` | `8642` | Port the Claude task server itself binds. |
| `VUI_MAX_CONTEXT_SECS` | `180` | TTS context cap in seconds of audio. Trained ceiling is ~240s. |
| `VUI_DEBUG` | `1` | TTS worker per-turn `.pt` dumps + sequence logs. Set `0` for fast mode. |
