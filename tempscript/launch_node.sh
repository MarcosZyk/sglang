# TP 0:
export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=0
# change the path to your conda env
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

modelpath="/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

SGLANG_CPU_OMP_THREADS_BIND='0-39|40-79' python3 -m sglang.launch_server --model "$modelpath"  \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 2 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 30001 --host 33.197.201.175  \
    --dist-init-addr 33.197.201.175:20000 --nnodes 4 --node-rank 0

# TP 1:
export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
# change the path to your conda env
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

modelpath="/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

SGLANG_CPU_OMP_THREADS_BIND='40-79' python3 -m sglang.launch_server --model "$modelpath"  \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 4 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 31001 --host 33.197.201.175  \
    --dist-init-addr 33.197.201.175:20000 --nnodes 4 --node-rank 1

# TP 2:
export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
# change the path to your conda env
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

modelpath="/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

SGLANG_CPU_OMP_THREADS_BIND='0-39' python3 -m sglang.launch_server --model "$modelpath"  \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 4 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 32001 --host 33.197.201.116  \
    --dist-init-addr 33.197.201.175:20000 --nnodes 4 --node-rank 2


export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
# change the path to your conda env
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

modelpath="/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

SGLANG_CPU_OMP_THREADS_BIND='40-79' python3 -m sglang.launch_server --model "$modelpath"  \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 4 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 33001 --host 33.197.201.116  \
    --dist-init-addr 33.197.201.175:20000 --nnodes 4 --node-rank 3


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



###############################
# TP 0:
export SGLANG_USE_CPU_ENGINE=1
# enable this env var to disable the amx all reduce
# and it will fall back to torch distrubuted all reduce
export SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1
# change the path to your conda env
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

modelpath="/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

SGLANG_CPU_OMP_THREADS_BIND='0-39' python3 -m sglang.launch_server --model "$modelpath"  \
    --trust-remote-code --device cpu --disable-overlap-schedule --disable-radix-cache --tp 4 \
    --mem-fraction-static 0.5 --max-total-tokens 63356  --port 30001 --host 33.197.201.175  \
    --dist-init-addr 33.197.201.175:5000 --nnodes 4 --node-rank 0 --trust-remote


# rank 0 tp 2:
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

UCX_NET_DEVICES=mlx5_23:1 SGLANG_USE_CPU_ENGINE=1 SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1 SGLANG_CPU_OMP_THREADS_BIND='0-39' \
    python3 -m sglang.launch_server --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B  --trust-remote-code \
    --device cpu --disable-overlap-schedule --disable-radix-cache --tp 4 --mem-fraction-static 0.5 --max-total-tokens 63356  \
    --port 30010 --host 33.197.201.175 --dist-init-addr 33.197.201.175:5000 --nnodes 4 --node-rank 0 --trust-remote


# rank 1 tp 2:
condaenv="sglamx-dong1"
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

UCX_NET_DEVICES=mlx5_22:1 SGLANG_USE_CPU_ENGINE=1 SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1 SGLANG_CPU_OMP_THREADS_BIND='0-39' \
    python3 -m sglang.launch_server --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B  --trust-remote-code --device cpu \
    --disable-overlap-schedule --disable-radix-cache --tp 4 --mem-fraction-static 0.5 --max-total-tokens 63356  \
    --port 30010 --dist-init-addr 33.197.201.175:5000 --nnodes 4 --node-rank 1 --trust-remote
--host 33.197.201.116

# rank 2 tp 2:
UCX_NET_DEVICES=mlx5_89:1 SGLANG_USE_CPU_ENGINE=1 SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1 SGLANG_CPU_OMP_THREADS_BIND='0-39' \
    python3 -m sglang.launch_server --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B  --trust-remote-code --device cpu \
    --disable-overlap-schedule --disable-radix-cache --tp 4 --mem-fraction-static 0.5 --max-total-tokens 63356  \
    --port 30010 --dist-init-addr 33.197.201.175:5000 --nnodes 4 --node-rank 2 --trust-remote
--host 33.197.201.155

# rank 3 tp 2:
UCX_NET_DEVICES=mlx5_47:1 SGLANG_USE_CPU_ENGINE=1 SGLANG_USE_AMX_DEFAULT_ALLREDUCE=1 SGLANG_CPU_OMP_THREADS_BIND='0-39' \
    python3 -m sglang.launch_server --model /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B  --trust-remote-code --device cpu \
    --disable-overlap-schedule --disable-radix-cache --tp 4 --mem-fraction-static 0.5 --max-total-tokens 63356  \
    --port 30010 --dist-init-addr 33.197.201.175:5000 --nnodes 4 --node-rank 3 --trust-remote
--host 33.197.201.149