#!/usr/bin/env bash
# Backup brain state (SQLite) from cloud to local machine
#
# Usage:
#   ./deploy/backup.sh user@server                    # backup to ./backups/
#   ./deploy/backup.sh user@server /path/to/local/dir # custom local dir

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@server [local_backup_dir]"
    exit 1
fi

SERVER="$1"
LOCAL_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOCAL_DIR"

echo "=== Backing up brain state from $SERVER ==="

# Create a consistent snapshot (SQLite WAL checkpoint first)
ssh "$SERVER" "sqlite3 /data/sqlite/neuromorphic.db 'PRAGMA wal_checkpoint(TRUNCATE);'" 2>/dev/null || true

# Download
BACKUP_FILE="$LOCAL_DIR/neuromorphic_${TIMESTAMP}.db"
scp "$SERVER:/data/sqlite/neuromorphic.db" "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup saved: $BACKUP_FILE ($SIZE)"

# Also grab latest benchmark if exists
ssh "$SERVER" "ls ${REMOTE_DIR:-~/engram}/neuromorphic/benchmarks/*.json 2>/dev/null | tail -1" | while read -r f; do
    if [ -n "$f" ]; then
        scp "$SERVER:$f" "$LOCAL_DIR/"
        echo "Benchmark saved: $(basename "$f")"
    fi
done

echo "=== Backup complete ==="
ls -lh "$LOCAL_DIR"/*_${TIMESTAMP}* 2>/dev/null || true
