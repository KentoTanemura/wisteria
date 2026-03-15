#!/usr/bin/env bash
#PJM -L rscgrp=share-short
#PJM -g EDIT_ME                 # ← .env の PJM_PROJECT に合わせて書き換える
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -j
#PJM -o logs/build_vllm.out
set -euo pipefail

# --- Resolve PROJECT_ROOT from .env ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PJM_DIR="$SCRIPT_DIR"
source "$(dirname "$PJM_DIR")/.env"

VLLM_SRC="${WORK_DIR}/vllm-src"

module load gcc/12.2.0

source "$PROJECT_ROOT/.venv/bin/activate"
export PATH="$PROJECT_ROOT/.venv/bin:$HOME/.local/bin:/work/opt/local/x86_64/cores/cuda/12.9/bin:$PATH"
export CUDA_HOME=/work/opt/local/x86_64/cores/cuda/12.9
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=4
export CC=/work/opt/local/x86_64/cores/gcc/12.2.0/bin/gcc
export CXX=/work/opt/local/x86_64/cores/gcc/12.2.0/bin/g++
export CUDAHOSTCXX=/work/opt/local/x86_64/cores/gcc/12.2.0/bin/g++

echo "=== Build Environment ==="
echo "hostname: $(hostname)"
echo "date:     $(date -Iseconds)"
echo "gcc:      $(which gcc) -> $(gcc --version | head -1)"
echo "CC=$CC  CXX=$CXX"
echo "cmake:    $(which cmake) -> $(cmake --version | head -1)"
echo "nvcc:     $(which nvcc) -> $(nvcc --version | tail -1)"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

echo "=== Building vllm v0.17.1 ==="
cd "$VLLM_SRC"
pip install . --no-build-isolation 2>&1 | tee "$PROJECT_ROOT/logs/build_full.log"

echo ""
echo "=== Verify ==="
python -c "import vllm; print(f'vllm={vllm.__version__}')" || echo "IMPORT FAILED"
echo "=== Done: $(date -Iseconds) ==="
