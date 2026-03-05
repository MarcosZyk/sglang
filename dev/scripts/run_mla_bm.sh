export OMP_SCHEDULE=STATIC

python bm_mla.py \
    --seq-len 8192 \
    --head-num 32 \
    --head-block-size 6 \
    --split-num 8 \
    --bind-numa "40-119" \
