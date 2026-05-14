#!/usr/bin/env bash
# Vui bootstrap — hosted at https://install.fluxions.ai
#
#     curl -fsSL https://install.fluxions.ai | bash
#     curl -fsSL https://install.fluxions.ai | bash -s -- --docker
#
# Clones Vui into $VUI_HOME (default ~/vui), then execs ./install.sh inside it.
# All args after `--` are forwarded to install.sh — see `./install.sh --help`.
#
# Env knobs:
#   VUI_HOME  target clone dir (default: ~/vui)
#   VUI_REPO  git repo URL (default: https://github.com/fluxions-ai/vui)
#   VUI_REF   git ref to check out (default: main; pin a tag for stability)

set -euo pipefail

VUI_HOME="${VUI_HOME:-$HOME/vui}"
VUI_REPO="${VUI_REPO:-https://github.com/fluxions-ai/vui}"
VUI_REF="${VUI_REF:-main}"

log() { printf '\033[1;36m>>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required. Install it and retry."

if [[ -d "$VUI_HOME/.git" ]]; then
    log "Existing Vui clone at $VUI_HOME — fetching latest."
    git -C "$VUI_HOME" fetch --tags origin
    if ! git -C "$VUI_HOME" diff --quiet HEAD 2>/dev/null; then
        die "Uncommitted changes in $VUI_HOME. Commit/stash them, or set VUI_HOME elsewhere."
    fi
    git -C "$VUI_HOME" checkout "$VUI_REF"
    # Tags don't fast-forward; let it fail silently.
    git -C "$VUI_HOME" pull --ff-only origin "$VUI_REF" || true
elif [[ -e "$VUI_HOME" ]]; then
    die "$VUI_HOME exists but isn't a git repo. Move it aside or set VUI_HOME=<other path>."
else
    log "Cloning Vui into $VUI_HOME (ref: $VUI_REF)..."
    git clone --branch "$VUI_REF" "$VUI_REPO" "$VUI_HOME"
fi

cd "$VUI_HOME"
exec bash "$VUI_HOME/install.sh" "$@"
