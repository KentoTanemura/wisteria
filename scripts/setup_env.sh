#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== setup_env.sh ==="
echo "PROJECT_ROOT: $PROJECT_ROOT"

# Python version check
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Python binary: $(which "$PYTHON_BIN")"
"$PYTHON_BIN" --version

# Create venv
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Activated venv: $(which python)"

# Upgrade base tools
python -m pip install -U pip setuptools wheel uv

# Install packages via uv
uv pip install -U vllm
uv pip install -U transformers huggingface_hub openai accelerate

# Import check
echo ""
echo "=== Import check ==="
python - <<'PY'
import sys
errors = []
for mod in ["torch", "vllm", "transformers", "openai"]:
    try:
        __import__(mod)
        print(f"  {mod}: OK")
    except ImportError as e:
        print(f"  {mod}: FAIL ({e})")
        errors.append(mod)
try:
    from huggingface_hub import snapshot_download
    print("  huggingface_hub.snapshot_download: OK")
except ImportError as e:
    print(f"  huggingface_hub: FAIL ({e})")
    errors.append("huggingface_hub")

if errors:
    print(f"\nFailed imports: {errors}")
    print(f"Python path: {sys.executable}")
    sys.exit(1)
print("\nimports_ok")
PY

echo ""
echo "=== Setup complete ==="
