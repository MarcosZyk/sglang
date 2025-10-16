export SGLANG_USE_CPU_ENGINE=1
export LD_PRELOAD=/root/miniforge3/envs/sglamx-dong1/lib/libiomp5.so:/root/miniforge3/envs/sglamx-dong1/lib/libtcmalloc.so:/root/miniforge3/envs/sglamx-dong1/lib/libtbbmalloc.so.2

SGLANG_CPU_OMP_THREADS_BIND='0-39|40-79' python3 -m sglang.launch_server \
    --model  /home/dchen/models/deepseek-r1 --trust-remote-code --device cpu --disable-overlap-schedule \
    --tp 2 --mem-fraction-static 0.8 --max-total-tokens 63356 --port 30010 --disable-radix-cache # --enable-hierarchical-cache #
