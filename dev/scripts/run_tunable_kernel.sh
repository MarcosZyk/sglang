export SGLANG_CPU_OMP_THREADS_BIND="40-119"
export AMX_KERNEL_AUTO_TUNE=1

python3 -m sglang.launch_server \
    --device cpu \
    --mem-fraction-static 0.096 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size=-1 \
    --model-path /sgl-workspace/models/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 --port 30010
