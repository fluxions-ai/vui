# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Runs on more than Ampere-and-newer.** The model hardcoded bf16 and the
  install hardcoded FlashAttention-2, so anything below compute capability 8.0
  failed — bf16 with no native support, and a flash-attn wheel whose cubins are
  `sm_80/90/100/120` with no PTX, which dies at the first decode step. Now:
  - New `vui.hardware` resolves dtype from the device: bf16 at 8.0+, fp16 on
    Turing/Volta, fp32 off-GPU. Override with `VUI_DTYPE=bf16|fp16|fp32`. The
    13 hardcoded `torch.bfloat16` sites now go through it.
  - `vui.flash_compat` checks compute capability on the first call and goes
    straight to SDPA below 8.0, instead of letting a kernel launch fail and
    catching the error. The launch-error catch stays as a backstop for Jetson,
    where the capability looks fine but the wheel has no matching cubin.
  - **flash-attn is now an optional extra** (`uv sync --extra flash`) rather
    than a hard dependency, since the SDPA path is a correct substitute.
    `install.sh` adds it only at 8.0+.
  - `pyproject.toml` sets `[tool.uv] torch-backend = "auto"`, so `uv sync`
    picks the CUDA build from the driver. That resolves from the driver
    version, not the compute capability, so `install.sh` still pins `cu126` for
    pre-Turing cards, whose kernels recent wheels omit entirely.
- **`python -m vui.doctor`** — preflight report: GPU and compute capability,
  whether the installed torch has kernels for it, resolved dtype, active
  attention path, whether torchcodec can load ffmpeg, LLM reachability, and
  free disk — each with a remedy. Non-zero exit only for blocking problems.
  `install.sh` runs it before starting the server, so hardware mismatches
  surface immediately rather than minutes into model loading.
- `tests/test_hardware.py` — dtype selection and attention dispatch across
  faked compute capabilities (Volta through Blackwell), so the decisions that
  need a shelf of old GPUs to exercise are covered on any box.

- **Rootless install — `./install.sh --native` now needs no sudo and no Docker.**
  See [`docs/rootless-install.md`](docs/rootless-install.md). Two things used to
  require root, and one of them was a mistake:
  - The ffmpeg gate tested `command -v ffmpeg`, but Vui never runs the ffmpeg
    *binary* — `torchcodec` needs the shared libraries, which it reaches via
    `NEEDED` entries on `libtorchcodec_core{4..8}.so` (one per ffmpeg major).
    The old check therefore passed on hosts with a static ffmpeg while
    torchcodec was still broken, and failed on hosts that were fine. It now
    tests by importing torchcodec, and on Linux fetches an LGPL shared build
    into `~/.cache/vui/ffmpeg` when there's nothing usable. New
    `vui.ffmpeg_libs` preloads it with `ctypes.CDLL(RTLD_GLOBAL)` — the same
    trick `vui._preload_nvidia_npp` already used — so no `LD_LIBRARY_PATH` is
    needed for processes you start by hand later. Inert when the system
    supplies ffmpeg. Knobs: `VUI_FFMPEG_DIR`, `VUI_FFMPEG_VERSION`.
  - The installer no longer installs Ollama at all (its installer requires
    root, writes `/usr/local`, adds a system user and a systemd unit — while
    `install.sh` went on to background `ollama serve` itself anyway). It now
    uses an Ollama you already run, or defaults to vLLM, which is pip
    installable. New `--llm vllm|ollama`; `VUI_MODE` env alias for the
    docker/native choice.
- `tests/test_llm_backend.py` and `tests/test_model_routes.py` — backend and
  route coverage against mocked HTTP, so they run without a GPU or a live LLM.

### Changed

- **vLLM is now a first-class backend, not just a supported one.** Its hot path
  was already at parity; the management surface was hardcoded to Ollama:
  - `LLMBackend` gained `health()`, `loaded_models()`, `pull()`, and
    `supports_model_switch` / `supports_pull` capability flags.
  - The model routes moved to `/llm/*` (old `/ollama/*` paths still work) and
    now go through the backend. The UI hides **Pull** and disables the selector
    according to the backend's capabilities, and shows which backend is active.
  - `srv.ollama_model` → `srv.llm_model`, derived from `backend.model` rather
    than stored alongside it. The vestigial `model=` parameter is gone from the
    `llm_*` helpers, which had been `del`-ing it for some time.
  - `VUI_LLM_BACKEND` is validated eagerly at startup with a backend banner,
    instead of raising inside a spawned warmup task where it was swallowed.
  - `VUI_OLLAMA_URL` and `OLLAMA_URL` no longer address separate code paths;
    either alone is now sufficient.
  - Default vLLM model is `google/gemma-4-E4B-it`.

### Fixed

- **The UI's LLM model dropdown only ever changed a label.**
  `handle_ollama_set_model` never called into the backend, and the re-prefill
  that followed warmed the *old* model — so the switch appeared to succeed
  while the previous model kept answering. Affected Ollama too, not just vLLM.
- **The `llm` status pill was permanently red under vLLM.** `probe_llm()` asked
  the backend for its base URL and then hit `/api/version`, which only Ollama
  serves. It's backend-dispatched now.
- Under vLLM the model dropdown rendered empty and **Pull** returned HTTP 500;
  the latter is now a 409, since having no model registry is a capability gap
  rather than a server fault. A non-2xx model list falls back to the current
  model instead of an empty dropdown.
- On Apple Silicon, `ensure_mlx_model()` ran regardless of backend — a
  multi-GB download plus an int4 quantize, blocking warmup, for a model the
  vLLM path would never reference. Now gated on the Ollama backend.
- The `OLLAMA_NUM_PARALLEL` warning (which shells out to
  `systemctl`/`pgrep`/`launchctl`) is likewise gated on Ollama.
- `install.sh --dry-run` could `die` on a host without ffmpeg, so it wasn't
  side-effect-free. `~/.local/bin` is now added to `PATH` unconditionally in
  the native path, rather than only inside the "uv wasn't found" branch — a
  second run couldn't otherwise see what the first had installed.

- **ARM64 Linux: `ModuleNotFoundError: No module named 'flash_attn'`.** Only an
  x86_64 wheel was pinned, so the TTS worker crashed on the first decode step on
  ARM hosts. Two changes:
  - `pyproject.toml` now pins an **aarch64** flash-attn wheel as well
    (2.8.3 + cu130 + torch 2.11 + cp312), so Grace-Hopper / Grace-Blackwell
    servers run the real kernel.
  - New `vui.flash_compat` provides a pure-PyTorch SDPA implementation of
    `flash_attn_with_kvcache` (same semantics, CUDA-graph safe) for everywhere
    the kernel can't run: CPU/macOS, and Jetson parts (Orin sm_87, Thor sm_110)
    whose compute capability isn't built into the aarch64 wheel — the launch
    failure is caught once and attention falls back for the rest of the process.
    Force it anywhere with `VUI_ATTN=torch`.

  `docker/Dockerfile.stream` also stopped requesting the non-existent `[cuda]`
  extra.

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
