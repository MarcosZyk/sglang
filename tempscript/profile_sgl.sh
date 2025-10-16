
dest_dir="/home/dchen/results/sglamx_spr96/"
mkdir -p "$dest_dir"

for input_len in 1024 2048 4096 8192 16384; do
    for output_len in 128; do
        for batch in 1 5 10; do
            python3 -m sglang.bench_serving  --dataset-path /root/ShareGPT_V3_unfiltered_cleaned_split.json \
                --dataset-name random --random-input $input_len --random-output $output_len --num-prompts $batch \
                --request-rate inf --random-range-ratio 1.0 --max-concurrency 50 --host 127.0.0.1 --port 30000 \
                > $dest_dir"sglang_benchmark_qwen235b_fp8_input${input_len}_output${output_len}_batch${batch}.log" 2>&1
        sleep 5
       	done
    done
done


python3 -m sglang.bench_serving  --dataset-path /home/dchen/ShareGPT_V3_unfiltered_cleaned_split.json \
                --dataset-name random --random-input 50 --random-output 50 --num-prompts 1 \
                --request-rate inf --random-range-ratio 1.0 --max-concurrency 50 --host 127.0.0.1 --port 30010