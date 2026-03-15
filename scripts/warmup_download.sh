#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/runtime/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT_ROOT/runtime/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJECT_ROOT/runtime/hf/transformers}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"

echo "=== warmup_download.sh ==="
echo "MODEL_ID:             $MODEL_ID"
echo "HF_HOME:              $HF_HOME"
echo "HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
echo ""

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python - <<PY
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download("${MODEL_ID}")
    print(f"Download complete: {path}")
except PermissionError:
    print("ERROR: Authentication failed. Run: bash scripts/login_hf.sh")
    sys.exit(1)
except OSError as e:
    if "disk" in str(e).lower() or "space" in str(e).lower():
        print(f"ERROR: Disk space issue: {e}")
    else:
        print(f"ERROR: Network/IO issue: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
PY

echo ""
echo "=== Warmup download complete ==="
