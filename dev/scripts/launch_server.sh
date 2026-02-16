python3 -m sglang.launch_server \
    --mem-fraction-static 0.15 \
    --model-path Qwen/Qwen3-8B \
    --port 12306 \
    --enable-mixed-chunk \
    --chuned-prefill-size 8192 \
    --enable-cache-report \
    --enable-artesia
    #--max-total-tokens 4096 \
