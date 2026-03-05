import os

import torch
import time
import argparse
import sgl_kernel

torch.manual_seed(1234)

def bm_mla(seq_len: int, head_num: int, head_block_size: int, split_num: int, ) -> float:
    B = 1
    H_Q = head_num
    H_KV = 1
    D = 576
    D_V = 512

    dtype = torch.int8

    total_tokens = B * seq_len
    sm_scale = (128 + 64)**-0.5
    logit_cap = 0.0

    # fixed params
    loc = torch.randperm(total_tokens)[:B].to(torch.int64)
    req_to_token = torch.arange(total_tokens).reshape(B, seq_len).to(torch.int32)
    b_req_idx = torch.arange(B).to(torch.int64)
    b_seq_len = torch.full((B,), seq_len).to(torch.int64)

    # dynamic params

    round = 61
    param_list = []
    for _ in range(round):
        q = torch.randn(B, H_Q, D, dtype=torch.bfloat16)
        k_buffer = torch.randint(-127, 127, [total_tokens, H_KV, D], dtype=dtype)
        v_buffer = k_buffer.narrow(2, 0, D_V)
        o = torch.zeros(B, H_Q, D_V, dtype=dtype)
        key = torch.randint(-127, 127, [B, H_KV, D], dtype=dtype)
        value = key.narrow(2, 0, D_V)
        attn_logits = torch.empty(
            (B, H_Q, split_num, D_V + 1),
            dtype=torch.float32,
        )
        param_list.append((q, k_buffer, v_buffer, o, key, value, attn_logits))

    start_time = time.perf_counter()
    for param in param_list:
        q, k_buffer, v_buffer, o, key, value, attn_logits = param
        torch.ops.sgl_kernel.decode_attention_cpu_v2(
            q,
            k_buffer,
            v_buffer,
            o,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_seq_len,
            sm_scale,
            logit_cap,
            head_block_size,
        )
    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"Finish decoding {round} rounds in {duration * 1000} ms.", flush=True)
    print(f"Avg latency {duration * 1000 * 1000 / round} us.", flush=True)
    return duration * 1000 * 1000 / round

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--seq-len', '-l', type=int, default=4096, )
    parser.add_argument('--head-num', '-q', type=int, default=32, )
    parser.add_argument('--head-block-size', '-b', type=int, default=6, )
    parser.add_argument('--split-num', '-s', type=int, default=8, )
    parser.add_argument('--bind-numa', '-c', type=str, default="80-119", )

    # 解析参数
    args = parser.parse_args()
    seq_len = args.seq_len
    head_num = args.head_num
    head_block_size = args.head_block_size
    split_num = args.split_num

    torch.ops.sgl_kernel.init_cpu_threads_env(args.bind_numa)
    torch.ops.sgl_kernel.enable_timing(torch.empty([]))
    bm_mla(seq_len, head_num, head_block_size, split_num)
    torch.ops.sgl_kernel.export_timing(torch.empty([]))
