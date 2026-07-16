#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
MC_GID_INDEX="${MC_GID_INDEX:-3}" \
MC_TE_METRIC="${MC_TE_METRIC:-true}" \
SGLANG_MOONCAKE_TRANS_THREAD="${SGLANG_MOONCAKE_TRANS_THREAD:-16}" \
python -m sglang.launch_server \
    --model-path "${MODEL_PATH:-/artesia-workspace/models/GLM-4.7}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-30000}" \
    --tp-size "${TP_SIZE:-4}" \
    --dtype bfloat16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --context-length "${CONTEXT_LENGTH:-32768}" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-prefill-pp 1 \
    --disaggregation-ib-device \
    "${IB_DEVICE:-mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3}" \
    --disable-overlap-schedule
