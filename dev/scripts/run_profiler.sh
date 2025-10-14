HOST_PORT=$1

export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/profile_log

python3 -m sglang.profiler --url http://localhost:${HOST_PORT}
