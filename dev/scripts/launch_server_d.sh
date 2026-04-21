CUDA_VISIBLE_DEVICES=$1 python3 -m sglang.launch_server \
    --mem-fraction-static 0.6 \
    --context-length 131071 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 8999 \
    --json-model-override-args '{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
    --model-path /artesia-workspace/models/Qwen3-8B \
    --port $2 \
    --enable-mixed-chunk \
    --chunked-prefill-size 8192 \
    --enable-cache-report \
    --disable-overlap-schedule \
    #--max-total-tokens 4096 \
