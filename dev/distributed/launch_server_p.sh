#!/usr/bin/env bash

# Run this script manually on the prefill node (10.87.79.111).
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MC_GID_INDEX=3 \
MC_TE_METRIC=true \
SGLANG_MOONCAKE_TRANS_THREAD=8 \
python -m sglang.launch_server \
    --model-path /artesia-workspace/models/GLM-4.7 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 4 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-decode-tp 4 \
    --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
    --disable-overlap-schedule

