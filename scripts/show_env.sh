#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

echo "=== Environment ==="
echo "MODEL_ID:              ${MODEL_ID:-<unset>}"
echo "HOST:                  ${HOST:-<unset>}"
echo "PORT:                  ${PORT:-<unset>}"
echo "TENSOR_PARALLEL_SIZE:  ${TENSOR_PARALLEL_SIZE:-<unset>}"
echo "DTYPE:                 ${DTYPE:-<unset>}"
echo "MAX_MODEL_LEN:         ${MAX_MODEL_LEN:-<unset>}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-<unset>}"
echo ""
echo "HF_HOME:               ${HF_HOME:-<unset>}"
echo "HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE:-<unset>}"
echo "TRANSFORMERS_CACHE:    ${TRANSFORMERS_CACHE:-<unset>}"
echo "VLLM_CONFIG_ROOT:      ${VLLM_CONFIG_ROOT:-<unset>}"
echo "TMPDIR:                ${TMPDIR:-<unset>}"
echo "VENV_DIR:              ${VENV_DIR:-<unset>}"
echo ""
echo "PJM_RSCGRP:            ${PJM_RSCGRP:-<unset>}"
echo "PJM_PROJECT:           ${PJM_PROJECT:-<unset>}"
echo "PJM_ELAPSE:            ${PJM_ELAPSE:-<unset>}"
