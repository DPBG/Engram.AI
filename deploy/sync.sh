#!/usr/bin/env bash
# Sync codebase to cloud server, build, and restart
#
# Usage:
#   ./deploy/sync.sh user@server
#   ./deploy/sync.sh user@server --build   # also rebuild containers
#   ./deploy/sync.sh user@server --restart # rebuild + restart

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@server [--build] [--restart]"
    exit 1
fi

SERVER="$1"
shift
DO_BUILD=false
DO_RESTART=false
for arg in "$@"; do
    case "$arg" in
        --build)   DO_BUILD=true ;;
        --restart) DO_BUILD=true; DO_RESTART=true ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-~/engram}"

echo "=== Syncing to $SERVER ==="
echo "Local:  $PROJECT_DIR"
echo "Remote: $REMOTE_DIR"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    --exclude '.uv' \
    --exclude 'cache/' \
    --exclude 'sensory-gateway/queue_state.json' \
    --exclude 'sensory-gateway/models/*.onnx' \
    --exclude 'neuromorphic/benchmarks/' \
    "$PROJECT_DIR/" "$SERVER:$REMOTE_DIR/"

echo "Sync complete."
echo "NOTE: ONNX models are excluded from sync."
echo "  To generate on server: cd sensory-gateway && python scripts/export_mobilenet.py"

if $DO_BUILD; then
    echo "=== Building containers ==="
    ssh "$SERVER" "cd $REMOTE_DIR && docker compose -f docker-compose.yml -f deploy/docker-compose.cloud.yml build"
fi

if $DO_RESTART; then
    echo "=== Restarting services ==="
    ssh "$SERVER" "cd $REMOTE_DIR && docker compose -f docker-compose.yml -f deploy/docker-compose.cloud.yml up -d"
fi

echo "=== Done ==="
