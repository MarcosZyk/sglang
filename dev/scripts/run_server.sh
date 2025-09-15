# Qwen/Qwen3-0.6B-FP8, Qwen/Qwen3-30B-A3B-Instruct-2507-FP8, deepseek-ai/DeepSeek-V2-Lite-Chat
python3 -m sglang.launch_server \
    --disable-overlap-schedule \
    --device cpu \
    --disable-radix-cache \
    --mem-fraction-static 0.016 \
    --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 --port 30000
