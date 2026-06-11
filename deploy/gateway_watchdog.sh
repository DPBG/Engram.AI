#!/usr/bin/env bash
# gateway_watchdog.sh -- External cron supervisor for the sensory gateway.
# Checks: (1) tmux session alive, (2) gateway process alive, (3) NATS observation flow.
# If any check fails, kills the old session and starts a fresh one.
#
# Install: crontab -e -> */5 * * * * /data/gateway_watchdog.sh >> /data/logs/gateway_watchdog.log 2>&1
# Requires: tmux, nats CLI, /data/start_gateway.sh

set -euo pipefail

LOGPREFIX="[gateway-watchdog $(date '+%Y-%m-%d %H:%M:%S')]"
TMUX_SESSION="gateway"
STARTUP_SCRIPT="/data/start_gateway.sh"
NATS_URL="nats://127.0.0.1:4222"
# Minimum observations expected in 10s sample window
MIN_OBS=5
SAMPLE_SECONDS=10

log() { echo "$LOGPREFIX $*"; }

restart_gateway() {
    log "RESTARTING gateway -- reason: $1"
    # Kill existing tmux session (ignore error if already dead)
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2
    # Start fresh
    tmux new-session -d -s "$TMUX_SESSION" "bash $STARTUP_SCRIPT"
    log "New tmux session '$TMUX_SESSION' started."
}

# --- Check 1: tmux session exists ---
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    restart_gateway "tmux session '$TMUX_SESSION' not found"
    exit 0
fi

# --- Check 2: at least one python process in the session ---
TMUX_PID=$(tmux list-panes -t "$TMUX_SESSION" -F '#{pane_pid}' 2>/dev/null | head -1)
if [ -z "$TMUX_PID" ]; then
    restart_gateway "no pane PID found in tmux session"
    exit 0
fi

# Check if the pane's process tree contains a python process
CHILD_PIDS=$(pgrep -P "$TMUX_PID" 2>/dev/null || true)
HAS_PYTHON=0
for pid in $TMUX_PID $CHILD_PIDS; do
    if grep -q python "/proc/$pid/cmdline" 2>/dev/null; then
        HAS_PYTHON=1
        break
    fi
done

if [ "$HAS_PYTHON" -eq 0 ]; then
    restart_gateway "no python process found under tmux pane PID $TMUX_PID"
    exit 0
fi

# --- Check 3: NATS observation flow ---
# Use nats CLI to subscribe and count messages over SAMPLE_SECONDS
if command -v nats >/dev/null 2>&1; then
    OBS_COUNT=$(timeout "${SAMPLE_SECONDS}s" nats sub "observation.>" --server "$NATS_URL" --count "$MIN_OBS" 2>/dev/null | grep -c "Received" || echo "0")
    if [ "$OBS_COUNT" -lt "$MIN_OBS" ]; then
        # Double-check: maybe brain is just between steps. Try once more.
        sleep 5
        OBS_COUNT2=$(timeout "${SAMPLE_SECONDS}s" nats sub "observation.>" --server "$NATS_URL" --count "$MIN_OBS" 2>/dev/null | grep -c "Received" || echo "0")
        if [ "$OBS_COUNT2" -lt "$MIN_OBS" ]; then
            restart_gateway "NATS observation flow too low ($OBS_COUNT + $OBS_COUNT2 msgs in ${SAMPLE_SECONDS}s x2)"
            exit 0
        fi
    fi
    log "OK -- tmux alive, python running, NATS flow healthy ($OBS_COUNT msgs in ${SAMPLE_SECONDS}s)"
else
    log "OK -- tmux alive, python running (nats CLI not installed, skipping flow check)"
fi
