#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_GEN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/sgl-workspace/sglang/.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-12306}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
HTTP_LIMIT_CONCURRENCY="${HTTP_LIMIT_CONCURRENCY:-500}"
CONTEXTCAKE_BASE_URL="${CONTEXTCAKE_BASE_URL:-http://127.0.0.1:50350}"
RESULT_DIR="${RESULT_DIR:-${LOAD_GEN_ROOT}/result}"

mkdir -p "${RESULT_DIR}"

printf '[run_server] 启动 main_art.py\n'
printf '[run_server] 监听地址：%s:%s\n' "${HOST}" "${PORT}"
printf '[run_server] ContextCake：%s\n' "${CONTEXTCAKE_BASE_URL}"
printf '[run_server] 最大并发：%s\n' "${MAX_CONCURRENT}"
printf '[run_server] 结果目录：%s\n' "${RESULT_DIR}"

exec "${PYTHON}" "${SCRIPT_DIR}/main_art.py" \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1 \
    --max-concurrent "${MAX_CONCURRENT}" \
    --http-limit-concurrency "${HTTP_LIMIT_CONCURRENCY}" \
    --contextcake-base-url "${CONTEXTCAKE_BASE_URL}" \
    --result-dir "${RESULT_DIR}"
