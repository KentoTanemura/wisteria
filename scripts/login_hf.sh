#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== Hugging Face Login ==="
echo "This is an interactive login. You will be prompted for your token."
echo "Token is NOT stored in this repo."
echo ""

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli login
else
    python -c "from huggingface_hub import login; login()"
fi
