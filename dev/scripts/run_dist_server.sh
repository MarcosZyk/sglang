NODE_RANK=$1
HOST_PORT=$2
BIND_NUMA=$3

export SGLANG_CPU_OMP_THREADS_BIND=$BIND_NUMA
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
export AMX_KERNEL_AUTO_TUNE=1

python3 -m sglang.launch_server \
    --device cpu \
    --mem-fraction-static 0.048 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size=-1 \
    --model-path /sgl-workspace/models/DeepSeek-V2-Lite-Chat \
    --tp 2 \
    --dist-init-addr 127.0.0.1:20000 \
    --host 0.0.0.0 --port $HOST_PORT \
    --nnodes 2 \
    --node-rank $NODE_RANK \
     --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
