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
#   ./install.sh --model qwen3:8b # alternate Ollama model
#   ./install.sh --dry-run        # print the plan, change nothing
#   ./install.sh --help
#
# Env knobs:
#   VUI_REF         git ref for --upgrade (default: main; pin a tag for stability)
#   OLLAMA_HOST     remote Ollama endpoint (e.g. gpu-box.lan:11434)
#   VUI_TASK_PORT   port for the Claude task server (default: 8642)

set -euo pipefail

MODEL="qwen3.5:4b"
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
  ./install.sh --model qwen3:8b alternate Ollama model
  ./install.sh --dry-run        print the plan, change nothing
  ./install.sh --help

Env knobs:
  VUI_REF         git ref for --upgrade (default: main; pin a tag for stability)
  OLLAMA_HOST     remote Ollama endpoint (e.g. gpu-box.lan:11434)
  VUI_TASK_PORT   port for the Claude task server (default: 8642)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --no-launch) LAUNCH=0; shift ;;
        --no-claude) WITH_CLAUDE=0; shift ;;
        --docker) MODE="docker"; shift ;;
        --native) MODE="native"; shift ;;
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

docker_usable() {
    command -v docker >/dev/null 2>&1 || return 1
    docker compose version >/dev/null 2>&1 || return 1
    docker info >/dev/null 2>&1 || return 1
    return 0
}

# Decide docker vs native if not forced.
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
    log "Using native (no-Docker) path."

    if ! command -v ffmpeg >/dev/null 2>&1; then
        case "$OS" in
            Linux)  die "ffmpeg not found. Install: sudo apt install ffmpeg" ;;
            Darwin) die "ffmpeg not found. Install: brew install ffmpeg" ;;
            *)      die "ffmpeg not found and unknown OS ($OS) — install it manually." ;;
        esac
    fi

    if ! command -v uv >/dev/null 2>&1; then
        log "Installing uv..."
        run "curl -LsSf https://astral.sh/uv/install.sh | sh"
        export PATH="$HOME/.local/bin:$PATH"
        command -v uv >/dev/null 2>&1 || die "uv install failed — add ~/.local/bin to PATH and retry."
    fi

    if ! command -v ollama >/dev/null 2>&1; then
        log "Installing Ollama CLI..."
        case "$OS" in
            Linux)  run "curl -fsSL https://ollama.com/install.sh | sh" ;;
            Darwin) die "Ollama not found. Install from https://ollama.com/download or: brew install --cask ollama" ;;
            *)      die "Unknown OS ($OS) — install Ollama manually." ;;
        esac
    fi

    local extras=()
    [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]] && extras+=(--extra mlx)
    log "Syncing Python env (${extras[*]:-base})..."
    run "uv sync ${extras[*]}"

    if ollama_up; then
        log "Ollama already up at $OLLAMA_URL_RESOLVED"
    elif [[ "$REMOTE_OLLAMA" -eq 1 ]]; then
        die "OLLAMA_HOST=$RAW_HOST is set but unreachable at $OLLAMA_URL_RESOLVED."
    else
        log "Starting ollama serve in the background..."
        if [[ "$DRY_RUN" -eq 0 ]]; then
            nohup ollama serve >/tmp/vui-ollama.log 2>&1 &
            for _ in $(seq 1 20); do
                sleep 0.5
                ollama_up && break
            done
            ollama_up || die "ollama serve didn't come up — see /tmp/vui-ollama.log"
        fi
    fi

    log "Pulling Ollama model: $MODEL"
    run "ollama pull '$MODEL'"

    if [[ "$WITH_CLAUDE" -eq 1 ]]; then
        if ! command -v claude >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/claude" ]]; then
            log "Installing Claude Code CLI..."
            case "$OS" in
                Linux|Darwin) run "curl -fsSL https://claude.ai/install.sh | bash" ;;
                *) warn "Unknown OS ($OS) — install Claude Code manually." ;;
            esac
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
    check_claude_creds

    if [[ "$LAUNCH" -eq 0 || "$DRY_RUN" -eq 1 ]]; then
        log "Setup done."
        log "  Stream: VUI_OLLAMA_MODEL=$MODEL VUI_OLLAMA_URL=$OLLAMA_URL_RESOLVED uv run python -m vui.serving.stream"
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

    log "Starting Vui streaming server on http://localhost:8080 ..."
    export VUI_OLLAMA_MODEL="$MODEL"
    export VUI_OLLAMA_URL="$OLLAMA_URL_RESOLVED"
    export OLLAMA_URL="$OLLAMA_URL_RESOLVED"
    uv run python -m vui.serving.stream
}

case "$MODE" in
    docker) run_docker ;;
    native) run_native ;;
    *) die "internal: unknown mode '$MODE'" ;;
esac
