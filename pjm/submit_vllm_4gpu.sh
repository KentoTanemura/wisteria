#!/usr/bin/env bash
#PJM -L rscgrp=share-short
#PJM -g EDIT_ME                 # ← .env の PJM_PROJECT に合わせて書き換える
#PJM -L gpu=4
#PJM -L elapse=06:00:00
#PJM -j
#PJM -o logs/pjm_vllm_4gpu.out
set -euo pipefail

# --- Resolve PROJECT_ROOT from .env ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PJM_DIR="$SCRIPT_DIR"
# .env is one level up from pjm/
source "$(dirname "$PJM_DIR")/.env"

source "$PROJECT_ROOT/.venv/bin/activate"

module load gcc/12.2.0

export HF_HOME="$PROJECT_ROOT/runtime/hf"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/runtime/hf/hub"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/runtime/hf/transformers"
export VLLM_CONFIG_ROOT="$PROJECT_ROOT/runtime/vllm"
export TMPDIR="$PROJECT_ROOT/tmp"
export HOME="$PROJECT_ROOT/tmp"
export TRITON_CACHE_DIR="$PROJECT_ROOT/tmp/triton"
export XDG_CACHE_HOME="$PROJECT_ROOT/tmp/cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/work/opt/local/x86_64/cores/cuda/12.9
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$PROJECT_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/runtime" "$PROJECT_ROOT/tmp/triton" "$PROJECT_ROOT/tmp/cache"

echo "=== PJM Job Start ==="
echo "hostname: $(hostname)"
echo "date:     $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')"
python -c "import vllm; print(f'vllm={vllm.__version__}')"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
API_KEY="${API_KEY:-EMPTY}"

echo ""
echo "=== Starting vLLM server (max_model_len=$MAX_MODEL_LEN) ==="
exec vllm serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --api-key "${API_KEY}"
