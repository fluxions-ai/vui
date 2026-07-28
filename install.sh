#!/usr/bin/env bash
# Vui installer — run from inside a Vui checkout (setup / upgrade / launch).
# For a fresh machine, bootstrap with:
#     curl -fsSL https://install.fluxions.ai | bash
# (that clones the repo and execs this script inside it.)
#
# Modes (auto-detect; flags override):
#   ./install.sh                  # set up + launch (docker if available, else native)
#   ./install.sh --docker         # force docker compose path
#   ./install.sh --native         # force native path, no prompt
#   ./install.sh --upgrade        # git pull, re-sync, then launch
#   ./install.sh --no-claude      # skip the Claude task server
#   ./install.sh --no-launch      # set up only, don't start anything
#   ./install.sh --llm vllm       # force the LLM backend (vllm | ollama)
#   ./install.sh --model qwen3:8b # alternate model (backend-specific id)
#   ./install.sh --dry-run        # print the plan, change nothing
#   ./install.sh --help
#
# Env knobs:
#   VUI_REF         git ref for --upgrade (default: main; pin a tag for stability)
#   OLLAMA_HOST     remote Ollama endpoint (e.g. gpu-box.lan:11434)
#   VUI_VLLM_URL    vLLM endpoint (default: http://localhost:8000)
#   VUI_TASK_PORT   port for the Claude task server (default: 8642)
#   VUI_MODE        "native" or "docker" — same as the flags, for curl | bash
#   VUI_FFMPEG_DIR  where to cache ffmpeg shared libs (default: ~/.cache/vui/ffmpeg)
#   VUI_FFMPEG_VERSION  ffmpeg major line to fetch (default: 7.1)
#
# This script never uses sudo. On Linux it needs no root: ffmpeg shared libs are
# cached under $HOME, and the LLM is either one you already run or a pip-installed
# vLLM. See docs/rootless-install.md.

set -euo pipefail

MODEL=""            # resolved per-backend once LLM_BACKEND is known
LLM_BACKEND=""      # "", "ollama", "vllm"
LAUNCH=1
WITH_CLAUDE=1
UPGRADE=0
DRY_RUN=0
MODE=""   # "", "docker", "native"

show_help() {
    cat <<'EOF'
Vui installer — run from inside a Vui checkout (setup / upgrade / launch).
For a fresh machine: curl -fsSL https://install.fluxions.ai | bash

Modes (auto-detect; flags override):
  ./install.sh                  set up + launch (docker if available, else native)
  ./install.sh --docker         force docker compose path
  ./install.sh --native         force native path, no prompt
  ./install.sh --upgrade        git pull, re-sync, then launch
  ./install.sh --no-claude      skip the Claude task server
  ./install.sh --no-launch      set up only, don't start anything
  ./install.sh --llm vllm       force the LLM backend (vllm | ollama)
  ./install.sh --model qwen3:8b alternate model (backend-specific id)
  ./install.sh --dry-run        print the plan, change nothing
  ./install.sh --help

Env knobs:
  VUI_REF         git ref for --upgrade (default: main; pin a tag for stability)
  OLLAMA_HOST     remote Ollama endpoint (e.g. gpu-box.lan:11434)
  VUI_VLLM_URL    vLLM endpoint (default: http://localhost:8000)
  VUI_TASK_PORT   port for the Claude task server (default: 8642)
  VUI_MODE        "native" or "docker" — same as the flags, for curl | bash
  VUI_FFMPEG_DIR  where to cache ffmpeg shared libs (default: ~/.cache/vui/ffmpeg)
  VUI_FFMPEG_VERSION  ffmpeg major line to fetch (default: 7.1)

This script never uses sudo. The native path needs no root — see
docs/rootless-install.md.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --no-launch) LAUNCH=0; shift ;;
        --no-claude) WITH_CLAUDE=0; shift ;;
        --docker) MODE="docker"; shift ;;
        --native) MODE="native"; shift ;;
        --llm) LLM_BACKEND="$2"; shift 2 ;;
        --upgrade) UPGRADE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "unknown arg: $1" >&2; echo "see --help" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;36m>>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ "$DRY_RUN" -eq 1 ]]; then printf '   $ %s\n' "$*"; else eval "$@"; fi; }

VUI_REF="${VUI_REF:-main}"

cd "$(dirname "$0")"

# Refuse to run from outside a Vui checkout.
if ! { [[ -f pyproject.toml && -d src/vui ]] && grep -q '^name = "vui"' pyproject.toml 2>/dev/null; }; then
    die "Not a Vui checkout. Bootstrap with: curl -fsSL https://install.fluxions.ai | bash"
fi

# --upgrade: git pull the existing checkout before doing anything else.
if [[ "$UPGRADE" -eq 1 ]]; then
    [[ -d .git ]] || die "--upgrade needs a git checkout (no .git here)."
    if ! git diff --quiet HEAD 2>/dev/null; then
        die "Uncommitted changes — commit/stash them before --upgrade."
    fi
    log "Upgrading: fetch + checkout $VUI_REF + pull"
    run "git fetch --tags origin"
    run "git checkout '$VUI_REF'"
    run "git pull --ff-only origin '$VUI_REF' || true"
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

# Resolve Ollama endpoint (used by both paths).
RAW_HOST="${OLLAMA_HOST:-localhost:11434}"
case "$RAW_HOST" in
    http://*|https://*) OLLAMA_URL_RESOLVED="$RAW_HOST" ;;
    *)                  OLLAMA_URL_RESOLVED="http://$RAW_HOST" ;;
esac
REMOTE_OLLAMA=0
[[ -n "${OLLAMA_HOST:-}" && "$RAW_HOST" != "localhost:11434" && "$RAW_HOST" != "127.0.0.1:11434" ]] && REMOTE_OLLAMA=1

ollama_up() { curl -fsS "$OLLAMA_URL_RESOLVED/api/version" >/dev/null 2>&1; }

VLLM_URL_RESOLVED="${VUI_VLLM_URL:-http://localhost:8000}"
vllm_up() { curl -fsS "$VLLM_URL_RESOLVED/v1/models" >/dev/null 2>&1; }

# Pick the LLM backend. We never *install* Ollama (its installer requires root);
# we use one that's already running, else vLLM, which is pip-installable and so
# works without root. Neither being up is fine — the server serves TTS/ASR
# regardless and the UI's llm pill goes green on its own once a backend appears.
resolve_llm_backend() {
    [[ -n "$LLM_BACKEND" ]] && return 0
    if [[ -n "${VUI_LLM_BACKEND:-}" ]]; then
        LLM_BACKEND="$VUI_LLM_BACKEND"
    elif ollama_up; then
        LLM_BACKEND="ollama"
    elif vllm_up; then
        LLM_BACKEND="vllm"
    elif [[ "$REMOTE_OLLAMA" -eq 1 ]]; then
        die "OLLAMA_HOST=$RAW_HOST set but unreachable at $OLLAMA_URL_RESOLVED."
    else
        LLM_BACKEND="vllm"
    fi
}

docker_usable() {
    command -v docker >/dev/null 2>&1 || return 1
    docker compose version >/dev/null 2>&1 || return 1
    docker info >/dev/null 2>&1 || return 1
    return 0
}

# Decide docker vs native if not forced. VUI_MODE lets `curl … | bash` force one
# without the `-s --` incantation.
MODE="${MODE:-${VUI_MODE:-}}"
if [[ -z "$MODE" ]]; then
    if docker_usable; then
        if [[ -t 0 ]]; then
            read -r -p ">> Docker detected. Use docker compose? [Y/n] " ans
            case "$ans" in
                n|N|no|NO) MODE="native" ;;
                *) MODE="docker" ;;
            esac
        else
            log "Docker detected (non-interactive) — using docker compose. Pass --native to override."
            MODE="docker"
        fi
    else
        MODE="native"
    fi
fi

# -------------------------------------------------------------------- GPU
#
# Two things depend on the card's compute capability:
#
#   * which CUDA build of torch to install — the cu130 wheels carry no sm_70
#     kernels, so Volta needs a cu12 build;
#   * whether to install flash-attn at all — its wheels are sm_80+ with no PTX,
#     and vui.flash_compat falls back to SDPA below that anyway.
#
# `nvidia-smi --query-gpu=compute_cap` needs no root and no CUDA toolkit.

GPU_CAP=""      # e.g. "8.6"; empty when there's no NVIDIA GPU

detect_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d '[:space:]')"
    [[ -n "$GPU_CAP" ]] && log "GPU compute capability: $GPU_CAP"
    return 0
}

# Numeric compare on "major.minor" without bc.
cap_at_least() {
    local want="$1" have="${GPU_CAP:-0.0}"
    [[ -z "$GPU_CAP" ]] && return 1
    local hw_major="${have%%.*}" hw_minor="${have##*.}"
    local wt_major="${want%%.*}" wt_minor="${want##*.}"
    (( hw_major > wt_major )) && return 0
    (( hw_major < wt_major )) && return 1
    (( hw_minor >= wt_minor ))
}

# pyproject.toml sets `[tool.uv] torch-backend = "auto"`, so uv already picks a
# CUDA build per the installed driver — nothing to do in the common case.
#
# What `auto` can't see is the compute capability: a Volta box with a current
# driver resolves to a recent build that has no sm_70 kernels. That's the one
# case worth overriding here.
resolve_torch_backend() {
    [[ -n "${UV_TORCH_BACKEND:-}" ]] && return 0   # respect an explicit choice
    [[ -z "$GPU_CAP" ]] && return 0
    cap_at_least 7.5 && return 0

    export UV_TORCH_BACKEND=cu126
    warn "Compute capability $GPU_CAP predates Turing — pinning a CUDA 12.6 torch,"
    warn "since recent wheels carry no kernels for it. Expect SDPA attention and"
    warn "fp16 rather than bf16. This combination is not well tested; if it"
    warn "misbehaves, 'python -m vui.doctor' reports what was resolved."
}

# ---------------------------------------------------------------- ffmpeg libs
#
# torchcodec dlopens libtorchcodec_core{4..8}.so — one per ffmpeg major — and
# each NEEDEDs one generation's sonames (core7 -> libavcodec.so.61, ...). So the
# dependency is the ffmpeg *shared libraries*, not the binary: Vui never shells
# out to ffmpeg. A static ffmpeg on PATH satisfies `command -v` and still leaves
# torchcodec broken, which is why we test by importing it instead.
#
# vui/__init__.py preloads whatever lands in $VUI_FFMPEG_DIR, so no
# LD_LIBRARY_PATH is needed for processes the user starts later by hand.

FFMPEG_DIR="${VUI_FFMPEG_DIR:-$HOME/.cache/vui/ffmpeg}"
FFMPEG_VER="${VUI_FFMPEG_VERSION:-7.1}"

FFMPEG_CHECK="import vui.ffmpeg_libs as m; raise SystemExit(m.selftest())"
torchcodec_ok() { uv run python -c "$FFMPEG_CHECK" >/dev/null 2>&1; }

ensure_ffmpeg_libs() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Would verify torchcodec can load ffmpeg; fetch into $FFMPEG_DIR if not."
        return 0
    fi

    if torchcodec_ok; then
        log "ffmpeg OK ($(uv run python -c "$FFMPEG_CHECK" 2>/dev/null))"
        return 0
    fi

    if [[ "$OS" != "Linux" ]]; then
        die "torchcodec can't load ffmpeg. On macOS: brew install ffmpeg"
    fi

    local tag
    case "$ARCH" in
        x86_64)          tag="linux64" ;;
        aarch64|arm64)   tag="linuxarm64" ;;
        *) die "No prebuilt ffmpeg for $ARCH. Install ffmpeg, or build one — see docs/rootless-install.md" ;;
    esac

    local asset="ffmpeg-n${FFMPEG_VER}-latest-${tag}-lgpl-shared-${FFMPEG_VER}.tar.xz"
    local url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${asset}"
    local tmp="${FFMPEG_DIR}.tmp.$$"

    log "No usable ffmpeg — fetching shared libs into $FFMPEG_DIR (~170 MB)..."
    mkdir -p "$tmp"
    # Extract to a temp dir and mv, so an interrupted fetch can't leave a
    # half-populated dir that the preload would happily find.
    if ! curl -fL --retry 3 "$url" | tar -xJ --strip-components=1 -C "$tmp"; then
        rm -rf "$tmp"
        die "ffmpeg download failed: $url
   Install ffmpeg yourself, point VUI_FFMPEG_DIR at an existing prefix, or
   build one — see docs/rootless-install.md"
    fi
    rm -rf "$FFMPEG_DIR"
    mkdir -p "$(dirname "$FFMPEG_DIR")"
    mv "$tmp" "$FFMPEG_DIR"
    echo "$FFMPEG_VER" > "$FFMPEG_DIR/.vui-version"

    torchcodec_ok || die "Fetched ffmpeg into $FFMPEG_DIR but torchcodec still can't load it.
   Details: $(uv run python -c "$FFMPEG_CHECK" 2>&1 | tail -2)
   This usually means the prebuilt libs need a newer glibc than this host has.
   Build ffmpeg from source instead — see docs/rootless-install.md"
    log "ffmpeg ready in $FFMPEG_DIR"
}

check_claude_creds() {
    [[ "$WITH_CLAUDE" -eq 1 ]] || return 0
    local has=0
    [[ -f "$HOME/.claude/.credentials.json" ]] && has=1
    [[ -n "${ANTHROPIC_API_KEY:-}" ]]         && has=1
    [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]   && has=1
    if [[ "$has" -eq 0 ]]; then
        warn "Could not detect Claude Code credentials."
        warn "  - No ~/.claude/.credentials.json (run 'claude' once to log in with a Pro/Max plan)"
        warn "  - No \$ANTHROPIC_API_KEY or \$CLAUDE_CODE_OAUTH_TOKEN in this shell"
        if [[ -t 0 ]]; then
            read -r -p "Run without the Claude task server? [y/N] " ans
            case "$ans" in
                y|Y|yes|YES) WITH_CLAUDE=0 ;;
                *) die "Aborted. Run 'claude' to log in, export an API key, or re-run with --no-claude." ;;
            esac
        else
            warn "Non-interactive — defaulting to --no-claude."
            WITH_CLAUDE=0
        fi
    fi
}

run_docker() {
    log "Using docker compose path."
    # The bundled compose stack is Ollama-based.
    [[ -z "$MODEL" ]] && MODEL="qwen3.5:4b"
    check_claude_creds

    local profiles=() services=(vui-stream)
    if ollama_up; then
        log "Ollama reachable at $OLLAMA_URL_RESOLVED — using it (no bundled container)."
    elif [[ "$REMOTE_OLLAMA" -eq 1 ]]; then
        die "OLLAMA_HOST=$RAW_HOST set but unreachable. Start it there or unset OLLAMA_HOST."
    else
        log "No host Ollama — enabling bundled ollama service (--profile ollama)."
        profiles+=(--profile ollama)
        services=(ollama "${services[@]}")
    fi
    [[ "$WITH_CLAUDE" -eq 1 ]] && services+=(claude-task)

    if [[ "$LAUNCH" -eq 0 ]]; then
        log "Setup-only mode. Would run: docker compose ${profiles[*]} up -d ${services[*]}"
        exit 0
    fi

    log "Bringing up: ${services[*]}"
    run "docker compose ${profiles[*]} up -d ${services[*]}"

    if [[ " ${services[*]} " == *" ollama "* ]]; then
        log "Pulling $MODEL inside the bundled ollama container..."
        run "docker compose exec -T ollama ollama pull '$MODEL'"
    fi

    log "Up. Open http://localhost:8080 — follow logs with: docker compose logs -f vui-stream"
}

run_native() {
    log "Using native (no-Docker, no-sudo) path."

    # Do this once, unconditionally: everything we install lands here, and a
    # second run must be able to see what the first one did even if the user's
    # login PATH doesn't include it.
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        log "Installing uv..."
        run "curl -LsSf https://astral.sh/uv/install.sh | sh"
        command -v uv >/dev/null 2>&1 || die "uv install failed — add ~/.local/bin to PATH and retry."
    fi

    detect_gpu
    resolve_torch_backend

    local extras=()
    [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]] && extras+=(--extra mlx)
    # flash-attn only where its kernels exist (Ampere+). Below that
    # vui.flash_compat uses SDPA, so installing it would just waste the
    # download.
    if cap_at_least 8.0; then
        extras+=(--extra flash)
    elif [[ -n "$GPU_CAP" ]]; then
        log "Skipping flash-attn (needs compute 8.0+, this GPU is $GPU_CAP) — using SDPA."
    fi

    log "Syncing Python env (${extras[*]:-base})${UV_TORCH_BACKEND:+, torch=$UV_TORCH_BACKEND}..."
    run "uv sync ${extras[*]}"

    # After uv sync: the only honest ffmpeg test is importing torchcodec, which
    # has to be installed first.
    ensure_ffmpeg_libs

    resolve_llm_backend
    if [[ "$LLM_BACKEND" == "ollama" ]]; then
        [[ -z "$MODEL" ]] && MODEL="qwen3.5:4b"
        if ollama_up; then
            log "Using the Ollama already running at $OLLAMA_URL_RESOLVED"
            log "Pulling Ollama model: $MODEL"
            run "ollama pull '$MODEL'"
        else
            # We never install Ollama — its installer needs root.
            warn "Backend is ollama but nothing is listening at $OLLAMA_URL_RESOLVED."
            warn "Start it (ollama serve), or use --llm vllm. Continuing without an LLM."
        fi
    else
        [[ -z "$MODEL" ]] && MODEL="google/gemma-4-E4B-it"
        if vllm_up; then
            log "Using the vLLM already running at $VLLM_URL_RESOLVED"
        else
            warn "No LLM at $VLLM_URL_RESOLVED. TTS and ASR work without one; the"
            warn "llm pill turns green on its own once a backend appears. Start one with:"
            warn "  uv run --with 'vllm==0.26.0' python -m vllm.entrypoints.openai.api_server \\"
            warn "      --model $MODEL --max-model-len 8192 \\"
            warn "      --max-num-seqs 1 --enforce-eager --gpu-memory-utilization 0.6 \\"
            warn "      --enable-auto-tool-choice --tool-call-parser gemma4 --port 8000"
            warn "--max-num-seqs 1 + --enforce-eager keep the KV pool and CUDA graphs to"
            warn "one request's worth; --gpu-memory-utilization leaves the rest of the"
            warn "card for the TTS/ASR workers. See docs/rootless-install.md."
        fi
    fi

    if [[ "$WITH_CLAUDE" -eq 1 ]]; then
        if ! command -v claude >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/claude" ]]; then
            log "Installing Claude Code CLI..."
            case "$OS" in
                Linux|Darwin) run "curl -fsSL https://claude.ai/install.sh | bash" ;;
                *) warn "Unknown OS ($OS) — install Claude Code manually." ;;
            esac
        fi
    fi
    check_claude_creds

    if [[ "$LAUNCH" -eq 0 || "$DRY_RUN" -eq 1 ]]; then
        log "Setup done."
        [[ "$DRY_RUN" -eq 0 ]] && uv run python -m vui.doctor || true
        if [[ "$LLM_BACKEND" == "ollama" ]]; then
            log "  Stream: VUI_OLLAMA_MODEL=$MODEL VUI_OLLAMA_URL=$OLLAMA_URL_RESOLVED uv run python -m vui.serving.stream"
        else
            log "  Stream: VUI_LLM_BACKEND=vllm VUI_VLLM_MODEL=$MODEL VUI_VLLM_URL=$VLLM_URL_RESOLVED uv run python -m vui.serving.stream"
        fi
        [[ "$WITH_CLAUDE" -eq 1 ]] && log "  Claude task: uv run python -m vui.serving.claude_server"
        exit 0
    fi

    CLAUDE_PID=""
    cleanup() { [[ -n "$CLAUDE_PID" ]] && kill "$CLAUDE_PID" 2>/dev/null || true; }
    trap cleanup EXIT INT TERM

    if [[ "$WITH_CLAUDE" -eq 1 ]]; then
        log "Starting Claude task server on :${VUI_TASK_PORT:-8642} (logs: /tmp/vui-claude.log) ..."
        uv run python -m vui.serving.claude_server >/tmp/vui-claude.log 2>&1 &
        CLAUDE_PID=$!
    fi

    # Surface hardware/ffmpeg/LLM problems here rather than as a stack trace
    # three minutes into model loading. Non-fatal: the report is advisory.
    uv run python -m vui.doctor || true

    log "Starting Vui streaming server on http://localhost:8080 ..."
    export VUI_LLM_BACKEND="$LLM_BACKEND"
    if [[ "$LLM_BACKEND" == "ollama" ]]; then
        export VUI_OLLAMA_MODEL="$MODEL"
        export VUI_OLLAMA_URL="$OLLAMA_URL_RESOLVED"
        export OLLAMA_URL="$OLLAMA_URL_RESOLVED"
    else
        export VUI_VLLM_MODEL="$MODEL"
        export VUI_VLLM_URL="$VLLM_URL_RESOLVED"
    fi
    uv run python -m vui.serving.stream
}

case "$MODE" in
    docker) run_docker ;;
    native) run_native ;;
    *) die "internal: unknown mode '$MODE'" ;;
esac
