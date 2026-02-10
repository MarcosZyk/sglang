import torch
import argparse
import sgl_kernel
import numpy as np

from bm_mla import bm_mla

def run_batch(*args, **kwargs):
    results = []
    for _ in range(10):
        latency = bm_mla(*args, **kwargs)
        results.append(latency)
    return {
        "min": np.min(results).real,
        "max": np.max(results).real,
        "mean": np.mean(results).real,
        "std": np.std(results).real,
        "median": np.median(results).real,
    }

def bm_core_number(*args, **kwargs):
    result_list = []
    for i in range(12):
        bind_numa = f"0-{(i + 1) * 10 - 1}"
        torch.ops.sgl_kernel.init_cpu_threads_env(bind_numa)
        result = run_batch(*args, **kwargs)
        result_list.append(result)

    for i, result in enumerate(result_list):
        bind_numa = f"0-{(i + 1) * 10 - 1}"
        print(f"{bind_numa}\t{result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--seq-len', '-l', type=int, default=1024, )
    parser.add_argument('--head-num', '-q', type=int, default=128, )
    parser.add_argument('--head-block-size', '-b', type=int, default=6, )
    parser.add_argument('--split-num', '-s', type=int, default=8, )
    parser.add_argument('--bind-numa', '-c', type=str, default="0-59", )

    parser.add_argument('--mode', '-m', type=str, default=None, )

    # 解析参数
    args = parser.parse_args()
    seq_len = args.seq_len
    head_num = args.head_num
    head_block_size = args.head_block_size
    split_num = args.split_num

    basic_args = (seq_len, head_num, head_block_size, split_num)

    mode = args.mode

    if mode is None:
        bind_numa = args.bind_numa
        torch.ops.sgl_kernel.init_cpu_threads_env(bind_numa)
        run_batch(*basic_args)
    elif mode=="core":
        bm_core_number(*basic_args)
