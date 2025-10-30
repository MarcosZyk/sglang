NODE_RANK=$1
HOST_PORT=$2

export AMX_PARALLEL=1

python3 -m sglang.launch_server \
    --device cpu \
    --mem-fraction-static 0.048 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --model-path /sgl-workspace/models/DeepSeek-V2-Lite-Chat \
    --enable-torch-compile \
    --torch-compile-max-bs 4 \
    --tp 2 \
    --dist-init-addr 127.0.0.1:20000 \
    --host 0.0.0.0 --port $HOST_PORT \
    --nnodes 2 \
    --node-rank $NODE_RANK
