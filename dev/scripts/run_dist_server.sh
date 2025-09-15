NODE_RANK=$1
HOST_PORT=$2

python3 -m sglang.launch_server \
    --device cpu \
    --mem-fraction-static 0.016 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --tp 3 \
    --dist-init-addr 127.0.0.1:20000 \
    --host 0.0.0.0 --port $HOST_PORT \
    --nnodes 3 \
    --node-rank $NODE_RANK
