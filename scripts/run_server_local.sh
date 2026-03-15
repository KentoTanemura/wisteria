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
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$PROJECT_ROOT/runtime/vllm}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/tmp}"

MODEL_ID="${MODEL_ID:?MODEL_ID is not set}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
API_KEY="${API_KEY:-EMPTY}"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/runtime" "$PROJECT_ROOT/tmp"

CMD="vllm serve ${MODEL_ID} \
  --host ${HOST} \
  --port ${PORT} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --dtype ${DTYPE} \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  --reasoning-parser qwen3 \
  --language-model-only \
  --api-key ${API_KEY}"

echo "=== run_server_local.sh ==="
echo "Executing:"
echo "  $CMD"
echo ""

exec $CMD
