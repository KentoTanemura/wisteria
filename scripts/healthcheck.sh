#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
BASE_URL="http://${HOST}:${PORT}"

echo "=== Health Check ==="
echo "Target: $BASE_URL"
echo ""

# 1. /health
echo "--- /health ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" 2>/dev/null || true)
if [ "$HTTP_CODE" = "200" ]; then
    echo "  OK (HTTP $HTTP_CODE)"
else
    echo "  FAIL (HTTP $HTTP_CODE)"
    echo "  Server may not be running at $BASE_URL"
    exit 1
fi

# 2. /v1/models
echo "--- /v1/models ---"
MODELS_RESP=$(curl -s -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/v1/models" 2>/dev/null || true)
if [ -z "$MODELS_RESP" ]; then
    echo "  FAIL: No response"
    exit 1
fi
if command -v jq &>/dev/null; then
    echo "$MODELS_RESP" | jq .
else
    echo "  $MODELS_RESP"
fi

# 3. Simple chat completion
echo "--- Chat Completion ---"
CHAT_RESP=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{
        "model": "'"${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"'",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 32
    }' 2>/dev/null || true)

if [ -z "$CHAT_RESP" ]; then
    echo "  FAIL: No response"
    exit 1
fi
if command -v jq &>/dev/null; then
    echo "$CHAT_RESP" | jq '.choices[0].message.content // .'
else
    echo "  $CHAT_RESP"
fi

echo ""
echo "=== All checks passed ==="
