#!/usr/bin/env bash

# Run this after the prefill and decode instances are healthy.
python -m sglang.srt.disaggregation.mini_lb \
    --prefill http://10.87.79.111:30000 \
    --decode http://10.87.79.112:30000 http://10.87.79.113:30000 \
    --host 0.0.0.0 \
    --port 8000

