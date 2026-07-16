CUDA_VISIBLE_DEVICES=$1 python3 -m sglang.launch_server \
    --mem-fraction-static 0.6 \
    --context-length 131071 \
    --json-model-override-args '{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
    --model-path /artesia-workspace/models/GLM-4.7 \
    --port $2 \
    --enable-mixed-chunk \
    --chunked-prefill-size 8192 \
    --enable-cache-report \
    --disable-overlap-schedule \
    --enable-artesia
    #--max-total-tokens 4096 \
