#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_GEN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/sgl-workspace/sglang/.venv/bin/python}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:12306}"
MODEL_PATH="${MODEL_PATH:-/artesia-workspace/models/GLM-4.7}"
RPS="${RPS:-0.08}"
REQUESTS="${REQUESTS:-96}"
RECORDED_REQUESTS="${RECORDED_REQUESTS:-60}"
MAX_CONCURRENT="${MAX_CONCURRENT:-100}"
SUPPORT_SUBMIT_WORKERS="${SUPPORT_SUBMIT_WORKERS:-8}"
JSON_FILE="${JSON_FILE:-${LOAD_GEN_ROOT}/output_json_flatten/test1.json}"
JSON_FILE_A="${JSON_FILE_A:-}"
JSON_FILE_B="${JSON_FILE_B:-}"
JSON_RATIO="${JSON_RATIO:-1:1}"
TRACE_SEED="${TRACE_SEED:-0}"

args=(
    "${PYTHON}" "${SCRIPT_DIR}/load_generator.py"
    --url "${SERVER_URL}"
    --rps "${RPS}"
    --requests "${REQUESTS}"
    --recorded-requests "${RECORDED_REQUESTS}"
    --max-concurrent "${MAX_CONCURRENT}"
    --support-submit-workers "${SUPPORT_SUBMIT_WORKERS}"
    --model "${MODEL_PATH}"
)

if [[ -n "${JSON_FILE_A}" || -n "${JSON_FILE_B}" ]]; then
    if [[ -z "${JSON_FILE_A}" || -z "${JSON_FILE_B}" ]]; then
        printf 'JSON_FILE_A 和 JSON_FILE_B 必须同时设置\n' >&2
        exit 2
    fi
    args+=(
        --json-file-a "${JSON_FILE_A}"
        --json-file-b "${JSON_FILE_B}"
        --json-ratio "${JSON_RATIO}"
        --trace-seed "${TRACE_SEED}"
    )
else
    args+=(--json-file "${JSON_FILE}")
fi

printf '[run_client] 服务地址：%s\n' "${SERVER_URL}"
printf '[run_client] 模型：%s\n' "${MODEL_PATH}"
printf '[run_client] 最大发送请求数：%s，记录请求数：%s，RPS：%s\n' \
    "${REQUESTS}" "${RECORDED_REQUESTS}" "${RPS}"

exec "${args[@]}"
