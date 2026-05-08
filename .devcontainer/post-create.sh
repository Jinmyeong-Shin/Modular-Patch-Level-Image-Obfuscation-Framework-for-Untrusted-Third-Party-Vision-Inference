#!/usr/bin/env bash
set -u
set -o pipefail

log()  { printf '\n=== %s ===\n' "$*"; }
warn() { printf '\n!!! %s !!!\n' "$*" >&2; }

export DEBIAN_FRONTEND=noninteractive

#------------------------------------------------------------------------------
# 1. Project Python deps
#------------------------------------------------------------------------------
log "Installing Python dependencies"
if [ -f /workspace/requirements.txt ]; then
    if ! pip install -r /workspace/requirements.txt; then
        warn "pip install failed. Fix requirements.txt and rerun:  bash /workspace/.devcontainer/post-create.sh"
    fi
else
    warn "/workspace/requirements.txt not found, skipping pip install."
fi

#------------------------------------------------------------------------------
# 2. Claude Code (only if host credentials exist AND CPU supports AVX)
#------------------------------------------------------------------------------
log "Checking Claude Code prerequisites"

if [ ! -s /root/.claude/.credentials.json ]; then
    log "No host Claude credentials found at ~/.claude/.credentials.json. Skipping Claude Code install."
    log "To use Claude Code: run 'claude' on your host, complete /login, then rebuild this container."
    exit 0
fi

if ! grep -qm1 '\bavx\b' /proc/cpuinfo; then
    warn "This CPU does not support AVX. Claude Code's bundled runtime requires AVX and will crash with SIGILL."
    warn "Skipping Claude Code install. Use a node pool with a newer CPU, or run Claude Code from your host."
    exit 0
fi

log "Installing Node.js 22"
if ! curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    || ! apt-get install -y --no-install-recommends nodejs; then
    warn "Node.js install failed. Skipping Claude Code install."
    exit 0
fi

log "Installing Claude Code"
if ! npm install -g @anthropic-ai/claude-code; then
    warn "npm install of Claude Code failed."
    exit 0
fi

if command -v claude >/dev/null 2>&1; then
    log "Claude Code installed at: $(command -v claude)"
    log "Verifying binary architecture"
    file "$(command -v claude)" || true
else
    warn "Claude Code installer ran but 'claude' is not on PATH."
fi

log "Setup complete."
