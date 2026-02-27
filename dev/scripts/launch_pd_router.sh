python -m sglang.srt.disaggregation.launch_lb \
    --policy random \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --decode http://127.0.0.1:30002 \
    --decode http://127.0.0.1:30003 \
    --prefill-bootstrap-ports 8999 \
    --host 0.0.0.0 \
    --port 12347