#!/usr/bin/env python3
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

import sgl_kernel  # noqa: E402,F401

INVOCATIONS_PER_ROUND = 61
NUM_ROUNDS = 5

PRESETS: Dict[str, Dict[str, int]] = {
    "tiny": {
        "batch_size": 1,
        "num_heads": 8,
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "prefix_len": 512,
        "extend_len": 16,
    },
    "eagle_like": {
        "batch_size": 4,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_size": 128,
        "head_size_v": 128,
        "prefix_len": 2048,
        "extend_len": 20,
    },
    "long_prefix": {
        "batch_size": 1,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_size": 128,
        "head_size_v": 128,
        "prefix_len": 8192,
        "extend_len": 16,
    },
}


def percentile(values, q: float) -> float:
    assert 0.0 <= q <= 1.0
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    alpha = pos - lo
    return vals[lo] * (1.0 - alpha) + vals[hi] * alpha


def parse_args():
    parser = argparse.ArgumentParser("CPU extend_attention_treemask benchmark")
    parser.add_argument("--preset", type=str, default="tiny", choices=sorted(PRESETS.keys()))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-size", type=int, default=None)
    parser.add_argument("--head-size-v", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument(
        "--prefix-lens",
        type=str,
        default=None,
        help='Comma-separated per-request prefix lengths. Overrides --prefix-len. Example: "8192,16384,32768,8192".',
    )
    parser.add_argument("--extend-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--logit-cap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-bind", type=str, default=None)
    return parser.parse_args()


def resolve_shape(args):
    shape = PRESETS[args.preset].copy()
    for k in [
        "batch_size",
        "num_heads",
        "num_kv_heads",
        "head_size",
        "head_size_v",
        "prefix_len",
        "extend_len",
    ]:
        val = getattr(args, k)
        if val is not None:
            shape[k] = val
    return shape


def validate_shape(shape):
    assert shape["batch_size"] > 0
    assert shape["num_heads"] > 0
    assert shape["num_kv_heads"] > 0
    assert shape["head_size"] > 0 and shape["head_size"] % 32 == 0
    assert shape["head_size_v"] > 0 and shape["head_size_v"] % 32 == 0
    assert shape["prefix_len"] >= 0
    assert shape["extend_len"] > 0


def parse_int_list(value: str, name: str) -> List[int]:
    try:
        values = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{name} must be a comma-separated integer list, got: {value}") from e
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def resolve_prefix_lens(args, shape) -> List[int]:
    if args.prefix_lens is not None:
        prefix_lens = parse_int_list(args.prefix_lens, "--prefix-lens")
        if len(prefix_lens) != shape["batch_size"]:
            raise ValueError(
                f"--prefix-lens length ({len(prefix_lens)}) must equal batch_size ({shape['batch_size']})"
            )
        if any(x < 0 for x in prefix_lens):
            raise ValueError("--prefix-lens values must all be >= 0")
        return prefix_lens

    prefix_len = shape["prefix_len"]
    if prefix_len < 0:
        raise ValueError(f"--prefix-len must be >= 0, got {prefix_len}")
    return [prefix_len for _ in range(shape["batch_size"])]


def build_custom_mask(seq_lens: torch.Tensor, extend_seq_lens: torch.Tensor) -> torch.Tensor:
    # Packed layout per batch:
    # [extend_len_0 * total_kv_len_0, extend_len_1 * total_kv_len_1, ...]
    masks = []
    for b in range(seq_lens.numel()):
        p = int(seq_lens[b].item())
        e = int(extend_seq_lens[b].item())
        prefix_mask = torch.ones((e, p), dtype=torch.bool, device=seq_lens.device)
        causal_mask = torch.tril(torch.ones((e, e), dtype=torch.bool, device=seq_lens.device))
        masks.append(torch.cat([prefix_mask, causal_mask], dim=1).reshape(-1))
    return torch.cat(masks, dim=0).contiguous()


def make_treemask_bundle(shape, dtype, device, prefix_lens_list: List[int]):
    b = shape["batch_size"]
    hq = shape["num_heads"]
    hkv = shape["num_kv_heads"]
    d = shape["head_size"]
    dv = shape["head_size_v"]
    extend_len = shape["extend_len"]

    total_kv_lens = [p + extend_len for p in prefix_lens_list]
    max_context_len = max(total_kv_lens)
    total_tokens = sum(total_kv_lens)
    total_extend_tokens = b * extend_len
    starts: List[int] = []
    offset = 0
    for tl in total_kv_lens:
        starts.append(offset)
        offset += tl

    q_extend = torch.randn((total_extend_tokens, hq, d), dtype=dtype, device=device)
    o_extend = torch.empty((total_extend_tokens, hq, dv), dtype=dtype, device=device)
    k_buffer = torch.randn((total_tokens, hkv, d), dtype=dtype, device=device)
    v_buffer = torch.randn((total_tokens, hkv, dv), dtype=dtype, device=device)

    req_to_token = torch.empty((b, max_context_len), dtype=torch.int32, device=device)
    for i in range(b):
        s = starts[i]
        total_kv_len = total_kv_lens[i]
        req_to_token[i, :total_kv_len] = torch.arange(s, s + total_kv_len, dtype=torch.int32, device=device)
        if total_kv_len < max_context_len:
            req_to_token[i, total_kv_len:] = s

    req_pool_indices = torch.arange(b, dtype=torch.int64, device=device)
    seq_lens = torch.tensor(prefix_lens_list, dtype=torch.int64, device=device)
    extend_seq_lens = torch.full((b,), extend_len, dtype=torch.int32, device=device)
    extend_start_loc = torch.zeros((b,), dtype=torch.int32, device=device)
    if b > 1:
        extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    custom_mask = build_custom_mask(seq_lens, extend_seq_lens)

    return {
        "q_extend": q_extend,
        "o_extend": o_extend,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "req_to_token": req_to_token,
        "req_pool_indices": req_pool_indices,
        "seq_lens": seq_lens,
        "extend_seq_lens": extend_seq_lens,
        "extend_start_loc": extend_start_loc,
        "custom_mask": custom_mask,
        "max_len_extend": extend_len,
    }


def make_treemask_pool(shape, dtype, device, prefix_lens_list: List[int]) -> List[dict]:
    return [
        make_treemask_bundle(shape=shape, dtype=dtype, device=device, prefix_lens_list=prefix_lens_list)
        for _ in range(INVOCATIONS_PER_ROUND)
    ]


def run_invocations(pool):
    for tensors in pool:
        torch.ops.sgl_kernel.extend_attention_treemask_cpu(
            tensors["q_extend"],
            tensors["o_extend"],
            tensors["k_buffer"],
            tensors["v_buffer"],
            tensors["req_to_token"],
            tensors["req_pool_indices"],
            tensors["seq_lens"],
            tensors["extend_seq_lens"],
            tensors["extend_start_loc"],
            tensors["custom_mask"],
            tensors["max_len_extend"],
            tensors["sm_scale"],
            tensors["logit_cap"],
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    shape = resolve_shape(args)
    validate_shape(shape)
    prefix_lens_list = resolve_prefix_lens(args, shape)
    shape["prefix_len"] = max(prefix_lens_list)

    sm_scale = args.sm_scale if args.sm_scale is not None else (1.0 / math.sqrt(shape["head_size"]))
    logit_cap = args.logit_cap

    if args.cpu_bind:
        print(f"Applying CPU binding: {args.cpu_bind}")
        binding_info = torch.ops.sgl_kernel.init_cpu_threads_env(args.cpu_bind)
        print(binding_info, end="" if binding_info.endswith("\n") else "\n")

    pool = make_treemask_pool(shape=shape, dtype=dtype, device=device, prefix_lens_list=prefix_lens_list)
    for tensors in pool:
        tensors["sm_scale"] = sm_scale
        tensors["logit_cap"] = logit_cap

    if args.cpu_bind:
        print(f"Warmup round (unmeasured): {INVOCATIONS_PER_ROUND} invocations")
        with torch.inference_mode():
            run_invocations(pool)

    prefix_min = min(prefix_lens_list)
    prefix_p50 = percentile([float(x) for x in prefix_lens_list], 0.5)
    prefix_max = max(prefix_lens_list)
    print(
        "Config: "
        f"preset={args.preset} B={shape['batch_size']} H={shape['num_heads']} HKV={shape['num_kv_heads']} "
        f"D={shape['head_size']} DV={shape['head_size_v']} prefix={shape['prefix_len']} extend={shape['extend_len']} "
        f"dtype={args.dtype}"
    )
    print(
        "Hybrid prefix_lens: "
        f"min={prefix_min} p50={prefix_p50:.1f} max={prefix_max} list={','.join(str(x) for x in prefix_lens_list)}"
    )
    print(f"Measured rounds={NUM_ROUNDS}, invocations_per_round={INVOCATIONS_PER_ROUND}")

    round_times = []
    print("\nRound Results (latency in us)")
    print(f"{'Round':>8} | {'Total(us)':>14} | {'Avg/us per call':>16}")
    print("-" * 46)
    with torch.inference_mode():
        for r in range(NUM_ROUNDS):
            t0 = time.perf_counter()
            run_invocations(pool)
            dt = time.perf_counter() - t0
            round_times.append(dt)
            total_us = dt * 1e6
            avg_us = total_us / INVOCATIONS_PER_ROUND
            print(f"{r + 1:>8d} | {total_us:>14.3f} | {avg_us:>16.3f}")

    per_call_us = [(t / INVOCATIONS_PER_ROUND) * 1e6 for t in round_times]

    print("\nOverall Latency Statistics (us)")
    print("-" * 72)
    print(f"{'Min':24s}: {min(per_call_us):.3f}")
    print(f"{'P50':24s}: {percentile(per_call_us, 0.5):.3f}")
    print(f"{'P90':24s}: {percentile(per_call_us, 0.9):.3f}")
    print(f"{'P99':24s}: {percentile(per_call_us, 0.99):.3f}")
    print(f"{'Max':24s}: {max(per_call_us):.3f}")
    print(f"{'Avg':24s}: {sum(per_call_us) / len(per_call_us):.3f}")
    print("-" * 72)


if __name__ == "__main__":
    main()
