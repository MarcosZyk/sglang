CUDA_VISIBLE_DEVICES=$1 python3 -m sglang.launch_server \
    --mem-fraction-static 0.6 \
    --context-length 131071 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 8999 \
    --model-path /artesia-workspace/models/GLM-4.7 \
    --port $2 \
    --enable-mixed-chunk \
    --chunked-prefill-size 8192 \
    --enable-cache-report \
    --disable-overlap-schedule \
    #--max-total-tokens 4096 \
