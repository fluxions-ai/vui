# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-14

First production release. Vui shifts from a standalone TTS model to a full
streaming conversational voice assistant.

### Added

- **Vui Nano (300M)** — new flagship model. Llama-style decoder + RQ-Transformer
  head over the Qwen3-TTS-12Hz codec. bf16 inference, CUDA graphs, ~9× realtime
  streaming on a 4090.
- **Streaming server** (`python -m vui.serving.stream`) — WebRTC + WebSocket
  pipeline (ASR → LLM → TTS) with browser UI, VAD-driven turn-taking,
  speculative LLM prefill, sentence-level TTS chunking with backpressure, and
  barge-in.
- **OpenAI Realtime API compatibility** — drop-in `ws://…/v1/realtime` with the
  standard event surface (`session.update`, `input_audio_buffer.append`,
  `response.create`, `response.audio.delta`, …) and PCM16 @ 24 kHz.
- **`POST /v1/voice-note`** — synchronous REST endpoint that runs the full
  ASR → LLM → TTS pipeline in a single HTTP call.
- **Voice cloning + fine-tuned presets** — `maeve`, `abraham`, `rhian`, `harry`
  shipped in `prompts/`; arbitrary speakers cloneable from a `.wav` sample.
- **SQ / WPS conditioning** — six speech-quality channels and words-per-second,
  fed through `sq_proj` / `wps_proj` and added to the text embeddings.
- **Pluggable ASR** — faster-whisper (GPU, default) and Moonshine (CPU, ONNX),
  switchable live from the UI.
- **Pluggable LLM backends** — Ollama, vLLM, any OpenAI-compatible endpoint.
- **Memories** — assistant remembers facts across sessions, persisted to
  `~/.vui/memories.json`.
- **Thoughts stream** — parallel LLM that routes voice intent to ~10 tools
  (memory ops, task control, web search, delegation) without a wake-word
  grammar; pluggable for user-defined local tools
  (`src/vui/serving/stream/tools/`).
- **Built-in `web_search` tool** — single-query factual lookups via a pluggable
  backend (Serper, Brave, or Tavily — first one with a key wins, or pin with
  `VUI_SEARCH_PROVIDER`). One HTTP round-trip, no `claude-task` needed; falls
  through to `delegate` for multi-step research or account-bound queries.
- **Claude task server** (optional sidecar) — handles slow/agentic work
  (Gmail, Calendar, Drive, Slack, multi-step web research) via the host's
  Claude Code MCPs.
  Auto-discovered on boot. Speaks Anthropic's `/v1/messages`; can be backed by
  Ollama, z.ai, DeepSeek, vLLM, LM Studio, or LiteLLM via `ANTHROPIC_BASE_URL`.
- **Apple Silicon (MLX) backend** — auto-detected; first-run auto-setup of
  `qwen3.5-4b-mlx` via `ollama create --experimental --quantize int4`. Marked
  WIP.
- **Mobile support** — documented cloudflared and Tailscale paths for phone
  access with mic over HTTPS (`docs/mobile.md`).
- **Docker compose** — one-file stack (streaming server + optional bundled
  Ollama + optional Claude task server).
- **One-liner installer** — `curl -fsSL https://install.fluxions.ai | bash`,
  auto-detects Docker vs. native and pulls the model.
- **Standalone TTS demo** (`demo.py`) — Gradio playground with voice-prompt
  upload, SQ/WPS sliders, and CLI render mode.
- **Telemetry** — anonymous `{voice, seconds}` events per render; disable with
  `VUI_TELEMETRY=0`.
- **Documentation** — `docs/configuration.md`, `docs/realtime-api.md`,
  `docs/claude-task-server.md`, `docs/thoughts-tools.md`, `docs/soul.md`,
  `docs/memory-budget.md`, `docs/mobile.md`.

### Changed

- **Audio codec**: Fluac (modified DAC with FSQ, ~21.5 Hz) replaced by
  Qwen3-TTS-Tokenizer-12Hz (16 codebooks of 2048 entries at 12.5 Hz, 24 kHz
  decoded audio).
- **Speaker encoder**: ECAPA-TDNN from `Qwen3-TTS-12Hz-0.6B-Base` (8.9M params,
  1024-dim) replaces the previous codec-coupled speaker path.
- **Text tokenization**: byT5 byte-level tokenizer replaced by tiktoken-based
  tokenizer (`src/vui/tokenizer.py`).
- **Python**: pinned to `>=3.12,<3.13` (was `==3.12.3`).
- **Dependencies**: streaming/server stack pulled in (`aiohttp`, `aiortc`,
  `av`, `faster-whisper`, `onnxruntime`, `huggingface_hub`, `safetensors`,
  `claude-agent-sdk`, `flash-attn`); strict version pins relaxed to ranges.

### Removed

- `src/vui/fluac.py` — Fluac codec module (replaced by `qwen_codec.py`).
- `src/vui/patterns.py`, `src/vui/tok.py`, `src/vui/notebook.py`,
  `src/vui/utils.py`, `src/vui/vad.py` — superseded by the new
  `engine.py` / `tokenizer.py` / `streaming.py` / serving stack.
- `inference.py`, `inference.ipynb` — replaced by `engine.py` and the
  streaming server.
- `Vui.BASE`, `Vui.ABRAHAM`, `Vui.COHOST` checkpoints — superseded by Vui Nano.
  Voices `abraham` (and three others) live on as `.wav` prompts in `prompts/`
  rather than separate checkpoints.

## [0.1.0] - 2026-02-25

Initial public release of **Vui — 100M Parameter On-Device Conversational
Text-to-Speech**.

### Added

- **Vui 100M** — Llama-style causal transformer (6 layers, 512 dim, 8 heads,
  RMSNorm, SiLU, RoPE) predicting audio tokens from text. Trained on 40,000
  hours of real audio conversations.
- **Fluac codec** — modified [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
  using Finite Scalar Quantization (9 codebooks × 1000 entries), ~21.5 Hz token
  rate (4× reduction vs standard DAC at 86 Hz).
- **ByT5 byte-level text tokenizer**.
- **Three checkpoints** — `Vui.BASE` (40k-hour pretrain), `Vui.ABRAHAM`
  (single-speaker, context-aware replies), `Vui.COHOST` (two-speaker dialogue).
- **Voice cloning** from short audio samples (base model).
- **Streaming synthesis** with KV caching and CUDA-graph acceleration.
- **Non-verbal sound tags** — inline `[breath]`, `[laugh]`, `[sigh]`,
  `[hesitate]`, `[tut]`.
- **Gradio demo** + Hugging Face Spaces hosted demo.

[1.0.0]: https://github.com/fluxions-ai/vui/releases/tag/v1.0.0
[0.1.0]: https://github.com/fluxions-ai/vui/releases/tag/v0.1.0
