#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

if [ $# -ge 1 ]; then
    TARGET="$1"
else
    TARGET=$(ls -t "$LOGS_DIR"/*.out "$LOGS_DIR"/*.log 2>/dev/null | head -1 || true)
    if [ -z "$TARGET" ]; then
        echo "No log files found in $LOGS_DIR"
        exit 1
    fi
fi

echo "=== Tailing: $TARGET ==="
tail -f "$TARGET"
