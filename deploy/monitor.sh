#!/usr/bin/env bash
set -eo pipefail
# Monitor training progress on cloud server
#
# Usage:
#   ./deploy/monitor.sh user@server           # show live logs
#   ./deploy/monitor.sh user@server --stats   # show resource usage
#   ./deploy/monitor.sh user@server --metrics # show brain metrics

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@server [--stats|--metrics|--logs]"
    exit 1
fi

SERVER="$1"
MODE="${2:---logs}"
REMOTE_DIR="${REMOTE_DIR:-~/engram}"

case "$MODE" in
    --stats)
        echo "=== Resource Usage on $SERVER ==="
        ssh "$SERVER" "
            echo '--- Memory ---'
            free -g
            echo ''
            echo '--- CPU ---'
            uptime
            echo ''
            echo '--- Docker containers ---'
            docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}'
            echo ''
            echo '--- Disk ---'
            df -h /data 2>/dev/null || df -h /
        "
        ;;
    --metrics)
        echo "=== Brain Metrics from $SERVER ==="
        ssh "$SERVER" "
            cd $REMOTE_DIR
            docker exec activelearning-neuromorphic python -c '
import asyncio, json
from neuromorphic.persistence import NeuromorphicPersistence
async def main():
    p = NeuromorphicPersistence(\"/data/sqlite/neuromorphic.db\")
    await p.open()
    try:
        state = await p.load_state()
        if state:
            print(json.dumps({
                \"step_count\": state.get(\"step_count\", 0),
                \"phase\": state.get(\"neuromodulation\", {}).get(\"phase\", \"unknown\"),
            }, indent=2))
        else:
            print(\"No saved state\")
    finally:
        await p.close()
asyncio.run(main())
' 2>/dev/null || echo 'Could not read metrics'
        "
        ;;
    --logs|*)
        echo "=== Live logs from $SERVER (Ctrl+C to stop) ==="
        ssh "$SERVER" "docker compose -f $REMOTE_DIR/docker-compose.yml logs -f neuromorphic --tail 50"
        ;;
esac
