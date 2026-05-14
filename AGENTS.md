# AGENTS.md

Orientation for AI coding agents working on Vui — a streaming conversational voice assistant (ASR → LLM → TTS) built around a 305M speech transformer over the Qwen3-TTS-12Hz codec.

For end-user setup (Docker, voices, hardware), read `README.md`. This file is for editing the code.

## Setup

```sh
uv sync                    # base + flash-attn prebuilt wheel (Linux/CUDA)
uv sync --extra mlx        # add Apple Silicon backend
uv sync --extra claude     # add Claude task-server deps
```

Python 3.12 only. Package manager is `uv` — `pip` ignores `[tool.uv.sources]` and will try to source-build flash-attn, which fails. Imports use setuptools layout (`from vui.x import y`); no `src.` prefix, no `sys.path` hacks.

## Run

| Command | What it does |
|---|---|
| `python -m vui.serving.stream` | Streaming server on `:8080` (browser UI at `/`) |
| `python -m vui.serving.claude_server` | Optional Claude task sidecar on `:8642` |
| `python demo.py` | Gradio TTS playground |
| `python demo.py --render --prompt prompts/abraham.wav` | CLI render with a preset voice |
| `docker compose up -d` | Full stack via compose (see `docker-compose.yml`, `docker/`) |

No test suite committed. Verify changes by running the relevant entry point and exercising the path end-to-end. For UI/streaming changes, hit the browser UI; for model changes, render via `demo.py`.

## Repo layout

### Model + inference (`src/vui/`)

| File | Role |
|---|---|
| `model.py` | Vui transformer + RQ-Transformer head. 768 dim, 22 layers, 8 heads. Where `sq_proj` / `wps_proj` conditioning lives. |
| `engine.py` | High-level `Engine` / `GenConfig` / `Row` API. Used by `demo.py` and the TTS worker. WPS estimation around `engine.py:749`. |
| `inference.py` | Lower-level inference helpers: `InferenceState`, `render_audio_stream`, `asr`, `simple_clean`. |
| `qwen_codec.py` | Pure-PyTorch reimpl of Qwen3-TTS-12Hz codec (16 codebooks × 2048, 12.5 Hz, 24 kHz). |
| `qwen_spk_enc.py` | ECAPA-TDNN speaker encoder from Qwen3-TTS. |
| `rope.py`, `sampling.py` | Standard transformer plumbing. |
| `tokenizer.py` | Text tokeniser. |
| `align.py` | Forced alignment for prompt re-alignment in the UI. |
| `prompt_utils.py` | Voice-prompt loading / encoding helpers. |
| `streaming.py` | Streaming-decode utilities used by the engine. |
| `hf.py` | Hugging Face checkpoint download. |
| `config.py` | Model config dataclass. |
| `demo/cli.py` | Interactive CLI demo (separate from `demo.py` Gradio). |

### Apple Silicon backend (`src/vui/mlx/`)

| Path | Role |
|---|---|
| `mlx/tts/` | MLX port of the TTS stack — `model.py`, `codec.py`, `generate.py`, `stream.py`, `weights.py`. |
| `mlx/asr/` | MLX-Moonshine ASR (`load.py`, `model.py`). |

CUDA path is the polished one; MLX path is rougher (see README "Help wanted" note).

### Streaming server (`src/vui/serving/stream/`)

Three OS processes connected by `torch.multiprocessing.Queue`. Main aiohttp/WebRTC server, GPU TTS worker, ASR worker.

| File | Role |
|---|---|
| `__main__.py` | `python -m vui.serving.stream` entry — boots `server.py`. |
| `server.py` | aiohttp app, route registration, worker lifecycle, `DEFAULT_SETTINGS` (incl. `n_codebooks` at ~`server.py:228`). |
| `connection.py` | Per-client WebRTC/WS state machine. |
| `voice_turn.py` | Per-turn orchestration: ASR → LLM → TTS with thoughts stream in parallel. |
| `tts_worker.py` | CUDA TTS worker — Vui + RQ-Transformer + Qwen codec, CUDA graphs. The big one. |
| `tts_worker_mlx.py` | MLX TTS worker (Apple Silicon). |
| `asr_worker.py` | ASR worker process; backend selected via `asr/`. |
| `audio_in_worker.py` | Mic ingest worker. |
| `codec_worker.py` | Codec encode/decode worker. |
| `vad.py` | Silero VAD (ONNX model at `silero_vad.onnx`). |
| `playback.py` | Outbound audio scheduling / backpressure. |
| `drains.py` | Sentence-level TTS chunking + drain coordination. |
| `llm.py`, `llm_backend.py` | LLM streaming (Ollama default; vLLM/OpenAI-compatible via env). |
| `thoughts.py` | Parallel "thoughts" LLM that routes tool intents. |
| `tools/` | One file per intent tool — `add_memory`, `delegate`, `cancel_task`, `set_timer`, `propose_tool`, etc. `SPEC.md` describes the contract. |
| `memories.py` | `~/.vui/memories.json` persistence. |
| `tasks.py` | In-memory task list (delegated work). |
| `protocol.py` | WS event schema. |
| `prompts.py`, `prompt_routes.py` | Voice-prompt management + HTTP routes. |
| `model_routes.py` | Live model swap routes. |
| `voice_note_routes.py` | `POST /v1/voice-note` synchronous endpoint. |
| `test_routes.py` | Dev/test endpoints. |
| `realtime/` | OpenAI Realtime API adapter — `routes.py` (`/v1/realtime` WS), `inbound.py`, `outbound.py`, `audio.py`, `adapter.py`, `sinks.py`. |
| `asr/` | ASR backend abstractions — `base.py`, `fwhisper.py`, `moonshine.py`, `mlx_whisper.py`. |
| `frontend.py`, `index.html`, `visualizer.js` | Browser UI (single-page, no build step). |
| `text_utils.py`, `_log.py` | Helpers. |

### Claude task server (`src/vui/serving/claude_server.py`)

Standalone aiohttp service on `:8642`. Wraps the Claude Agent SDK as a long-lived agent loop and exposes `POST /task`. `discover_mcp_tools()` reflects MCP servers connected to the host's Claude Code config (`~/.claude`). Default model is the `MODEL` constant near the top.

### Other top-level

| Path | Role |
|---|---|
| `demo.py` | Gradio TTS playground — separate from the streaming server. |
| `prompts/` | Preset voice prompts (`.safetensors` + `.txt`). |
| `docs/` | `configuration.md`, `realtime-api.md`, `memory-budget.md`, `thoughts-tools.md`. Long-form references — link to these from code rather than duplicating. |
| `docker/` | `Dockerfile.stream` (vui-stream), `Dockerfile.claude` (claude-task). |
| `docker-compose.yml` | Full stack incl. optional `ollama` and `claude-task` profiles. |
| `pyproject.toml` | Dependencies, optional extras, flash-attn wheel pin. |

## Conventions

- **No emojis** in code or strings.
- **Modern typing**: `list[str]`, `dict[str, X]`, `str | None`. Don't use `Optional` / `List` / `Dict`.
- **Minimal comments** — let typing and naming carry intent. A comment should explain a non-obvious *why*, not a *what*.
- **Imports** are flat: `from vui.engine import Engine`. Never `from src.vui...`.
- **Env vars** are the public API for deployment knobs (see `docs/configuration.md`). Prefix with `VUI_`. Read once at startup, not per-request.
- **Single-tenant**: streaming server, realtime endpoint, and `/v1/voice-note` all assume one active client. Don't introduce shared mutable state without a lock.
- **Three-process boundary**: anything crossing main ↔ TTS worker ↔ ASR worker goes via `torch.multiprocessing.Queue`. Keep payloads small and picklable; large tensors flow as codec codes, not raw audio.
- **Hot paths**: `tts_worker.py` and `voice_turn.py` are latency-sensitive. Avoid allocations in the per-frame loop; reuse buffers; respect existing CUDA-graph capture boundaries.

## Where to look first

| If you're touching… | Start in |
|---|---|
| The model itself | `src/vui/model.py`, `src/vui/engine.py` |
| TTS streaming / latency | `src/vui/serving/stream/tts_worker.py`, `drains.py`, `playback.py` |
| Turn taking / VAD | `src/vui/serving/stream/vad.py`, `voice_turn.py` |
| LLM swap / backend | `src/vui/serving/stream/llm_backend.py`, `llm.py` |
| Tool routing | `src/vui/serving/stream/thoughts.py`, `tools/SPEC.md`, `tools/*.py` |
| OpenAI Realtime compat | `src/vui/serving/stream/realtime/` |
| Browser UI | `src/vui/serving/stream/index.html`, `visualizer.js`, `frontend.py` |
| Apple Silicon | `src/vui/mlx/`, `tts_worker_mlx.py` |
| Claude task delegation | `src/vui/serving/claude_server.py` |
