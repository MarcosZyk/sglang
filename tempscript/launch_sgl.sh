condaenv="sglamx-dong1"

export SGLANG_USE_CPU_ENGINE=1
export LD_PRELOAD=/root/miniforge3/envs/$condaenv/lib/libiomp5.so:/root/miniforge3/envs/$condaenv/lib/libtcmalloc.so:/root/miniforge3/envs/$condaenv/lib/libtbbmalloc.so.2

# model path:
# /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B 
# /root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Instruct-2507
# /root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-0528   

modelpath="/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

SGLANG_CPU_OMP_THREADS_BIND='0-39|40-79' python3 -m sglang.launch_server \
    --model $modelpath --trust-remote-code --device cpu --disable-overlap-schedule \
    --tp 2 --mem-fraction-static 0.8 --max-total-tokens 63356 --port 30010 --disable-radix-cache #--enable-hierarchical-cache #