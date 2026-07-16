#!/usr/bin/env bash

# Run this script manually on each decode node (10.87.79.112 and 10.87.79.113).
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MC_GID_INDEX=3 \
MC_TE_METRIC=true \
SGLANG_MOONCAKE_TRANS_THREAD=16 \
python -m sglang.launch_server \
    --model-path /artesia-workspace/models/GLM-4.7 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 4 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-prefill-pp 1 \
    --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
    --disable-overlap-schedule

