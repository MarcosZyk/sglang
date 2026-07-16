#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

"${SGLANG_PYTHON}" -m sglang.srt.disaggregation.mini_lb \
    --prefill "${PREFILL_URL:-http://10.87.79.111:30000}" \
    --decode \
    "${DECODE_URL_1:-http://10.87.79.112:30000}" \
    "${DECODE_URL_2:-http://10.87.79.113:30000}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-12347}"
