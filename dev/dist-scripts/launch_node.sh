# TP 0:
export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
# change the path to your conda env
export LD_PRELOAD="/root/miniforge3/envs/sglamx/lib/libiomp5.so:/root/miniforge3/envs/sglamx/lib/libtcmalloc.so:/root/miniforge3/envs/sglamx/lib/libtbbmalloc.so.2"

SGLANG_CPU_OMP_THREADS_BIND='0-47' python3 -m sglang.launch_server --model /home/dchen/models/qwen3-32b  \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 2 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 30001 --host 33.198.112.137  \
    --dist-init-addr 33.198.112.137:20000 --nnodes 2 --node-rank 0


# TP 1:
export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
# change the path to your conda env
export LD_PRELOAD="/root/miniforge3/envs/sglamx/lib/libiomp5.so:/root/miniforge3/envs/sglamx/lib/libtcmalloc.so:/root/miniforge3/envs/sglamx/lib/libtbbmalloc.so.2"

SGLANG_CPU_OMP_THREADS_BIND='0-47' python3 -m sglang.launch_server --model /home/dchen/models/qwen3-32b \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 2 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 30001 --host 33.198.90.160  \
    --dist-init-addr 33.198.112.137:20000 --nnodes 2 --node-rank 1

# launch requests:
# Option 1:
curl -X POST http://33.198.112.137:30001/generate -H "Content-Type: application/json" -d '{
  "text": "Let me tell you a long story more than 200 words, do not ship the detail, show your math skills ",
  "sampling_params": {
    "temperature": 1
  }
}'

# Option 2:
python3 -m sglang.bench_serving --dataset-path /home/dchen/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random --random-input 1024 --random-output 64 --num-prompts 1 --request-rate 0.5 \
    --random-range-ratio 1.0 --max-concurrency 10 --host 33.198.112.137 --port 30000 --profile --flush-cache
