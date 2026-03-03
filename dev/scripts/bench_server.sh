python3 -m sglang.bench_one_batch_server \
    --model-path /sgl-workspace/models/DeepSeek-V2-Lite-Chat \
    --base-url http://127.0.0.1:30010 \
    --batch-size 1 \
    --input-len 4096 \
    --output-len 256
