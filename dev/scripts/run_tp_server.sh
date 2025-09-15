export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/profile_log

python3 -m sglang.launch_server \
    --disable-overlap-schedule \
    --device cpu \
    --disable-radix-cache \
    --tp 3 \
    --mem-fraction-static 0.016 \
    --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 --port 30000
