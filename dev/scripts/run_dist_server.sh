NODE_RANK=$1
HOST_PORT=$2

python3 -m sglang.launch_server \
    --device cpu \
    --mem-fraction-static 0.012 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --model-path Qwen/Qwen3-0.6B-FP8 \
    --tp 2 \
    --dist-init-addr 127.0.0.1:20000 \
    --host 0.0.0.0 --port $HOST_PORT \
    --nnodes 2 \
    --node-rank $NODE_RANK
