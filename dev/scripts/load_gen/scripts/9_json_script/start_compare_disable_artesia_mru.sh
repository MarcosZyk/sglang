#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
METHOD_NAME="disable-artesia-mru"
RESULT_BUCKET="result_128G_9"

PYTHON="${PYTHON:-/sgl-workspace/sglang/.venv/bin/python}"

ARTESIA_PORT=50373
SERVER_PORT=12323
RESULT_DIR="$ROOT_DIR/${RESULT_BUCKET}/result-disable-artesia-mru"
LOG_DIR="$ROOT_DIR/logs/$METHOD_NAME/$RESULT_BUCKET"
JSON_FILE="$ROOT_DIR/output_json_flatten/test9.json"

CPU_GPU_BW_GBPS=5
GPU_SGLANG_BW_GBPS=240
PREFILL_TPS=6100
DECODE_TPS=65
DECODE_MAX_CONCURRENCY=9
KV_CACHE_KB_PER_TOKEN=144
GPU_CAPACITY_GB=128
CPU_CAPACITY_GB=256
TOKENIZER_NAME="/artesia-workspace/models/GLM-4.7"
CLIENT_RPS=0.08
CLIENT_REQUESTS=96
CLIENT_URL="http://127.0.0.1:${SERVER_PORT}"
ARTESIA_BASE_URL="http://127.0.0.1:${ARTESIA_PORT}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

ARTESIA_LOG="$LOG_DIR/artesia_sim.log"
SERVER_LOG="$LOG_DIR/main_art.log"
CLIENT_LOG="$LOG_DIR/load_generator.log"

start_process() {
    local log_file="$1"
    shift
    nohup "$@" >"$log_file" 2>&1 &
    echo $!
}

echo "Starting ${METHOD_NAME}"

ARTESIA_PID="$(
    start_process \
        "$ARTESIA_LOG" \
        bash -lc "cd '$ROOT_DIR/artesia_sim' && PYTHONUNBUFFERED=1 ${PYTHON} main.py \
            --host 0.0.0.0 \
            --port ${ARTESIA_PORT} \
            --workers 1 \
            --cpu-gpu-bandwidth-gbps ${CPU_GPU_BW_GBPS} \
            --gpu-sglang-bandwidth-gbps ${GPU_SGLANG_BW_GBPS} \
            --prefill-throughput-tps ${PREFILL_TPS} \
            --decode-throughput-tps ${DECODE_TPS} \
            --decode-max-concurrency ${DECODE_MAX_CONCURRENCY} \
            --kv-cache-kb-per-token ${KV_CACHE_KB_PER_TOKEN} \
            --gpu-capacity-gb ${GPU_CAPACITY_GB} \
            --cpu-capacity-gb ${CPU_CAPACITY_GB} \
            --eviction-policy mru \
            --tokenizer ${TOKENIZER_NAME}"
)"
echo "artesia_sim pid=${ARTESIA_PID} log=${ARTESIA_LOG}"

sleep 3

SERVER_PID="$(
    start_process \
        "$SERVER_LOG" \
        bash -lc "cd '$ROOT_DIR/server' && PYTHONUNBUFFERED=1 ${PYTHON} main_art.py \
            --host 0.0.0.0 \
            --port ${SERVER_PORT} \
            --workers 1 \
            --contextcake-base-url ${ARTESIA_BASE_URL} \
            --result-dir '$RESULT_DIR'"
)"
echo "main_art pid=${SERVER_PID} log=${SERVER_LOG}"

sleep 3

CLIENT_PID="$(
    start_process \
        "$CLIENT_LOG" \
        bash -lc "cd '$ROOT_DIR/client' && PYTHONUNBUFFERED=1 ${PYTHON} load_generator.py \
            --url ${CLIENT_URL} \
            --rps ${CLIENT_RPS} \
            --requests ${CLIENT_REQUESTS} \
            --model ${TOKENIZER_NAME} \
            --json-file '$JSON_FILE'"
)"
echo "load_generator pid=${CLIENT_PID} log=${CLIENT_LOG}"

echo "Result directory: ${RESULT_DIR}"
echo "Replay JSON: ${JSON_FILE}"
