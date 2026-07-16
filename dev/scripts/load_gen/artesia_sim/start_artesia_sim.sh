#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/sgl-workspace/sglang/.venv/bin/python}"

exec "${PYTHON}" "${SCRIPT_DIR}/main.py" \
    --host 0.0.0.0 \
    --port 50353 \
    --workers 1 \
    --cpu-gpu-bandwidth-gbps 3.5 \
    --gpu-sglang-bandwidth-gbps 490 \
    --prefill-throughput-tps 31000 \
    --decode-throughput-tps 65 \
    --kv-cache-kb-per-token 144 \
    --decode-max-concurrency 9 \
    --gpu-capacity-gb 196 \
    --tokenizer /artesia-workspace/models/GLM-4.7 \
    --enable-artesia \
    --eviction-policy mru
