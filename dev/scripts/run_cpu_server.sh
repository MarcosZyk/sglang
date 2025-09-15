# Qwen/Qwen3-0.6B-FP8 Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

python3 -m sglang.launch_server \
    --disable-overlap-schedule \
    --device cpu \
    --quantization fp8 \
    --model-path Qwen/Qwen3-0.6B-FP8 \
    --host 0.0.0.0 --port 30000


# Qwen/Qwen3-0.6B-FP8 Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
#--quantization fp8 \
#    --disable-radix-cache \
python3 -m sglang.launch_server \
    --disable-overlap-schedule \
    --device cpu \
    --disable-radix-cache \
    --mem-fraction-static 0.016 \
    --model-path Qwen/Qwen3-0.6B-FP8 \
    --host 0.0.0.0 --port 30000
