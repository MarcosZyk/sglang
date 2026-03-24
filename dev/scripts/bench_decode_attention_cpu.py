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
        "seq_len": 512,
        "num_kv_splits": 8,
    },
    "gqa": {
        "batch_size": 4,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "seq_len": 2048,
        "num_kv_splits": 8,
    },
    "mla_like": {
        "batch_size": 1,
        "num_heads": 22,
        "num_kv_heads": 1,
        "head_size": 576,
        "head_size_v": 512,
        "seq_len": 1024,
        "num_kv_splits": 8,
    },
    "mla": {
        "batch_size": 1,
        "num_heads": 22,
        "num_kv_heads": 1,
        "head_size": 576,
        "head_size_v": 512,
        "seq_len": 2048,
        "num_kv_splits": 8,
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
    parser = argparse.ArgumentParser("CPU decode_attention benchmark")
    parser.add_argument("--preset", type=str, default="tiny", choices=sorted(PRESETS.keys()))
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["sdm", "tdm", "adaptive"],
        default="sdm",
        help="Decode strategy: sdm uses decode_attention_cpu_tuned; tdm uses decode_attention_cpu_tdm; adaptive uses decode_attention_cpu_adaptive.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["auto", "mha", "gqa", "mla"],
        default="auto",
        help="Decode workload mode. 'mla' enforces shared K/V storage layout.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-size", type=int, default=None)
    parser.add_argument("--head-size-v", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=None,
        help='Comma-separated per-request sequence lengths. Overrides --seq-len. Example: "8192,16384,32768,8192".',
    )
    parser.add_argument("--num-kv-splits", type=int, default=None)
    parser.add_argument(
        "--head-block-num",
        type=int,
        default=8,
        help="Number of Q-head blocks for tuned decode strategies (sdm/tdm).",
    )
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--logit-cap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-bind", type=str, default=None)
    parser.add_argument(
        "--enable-decode-timer",
        action="store_true",
        help="Enable C++ decode timer for tuned/tdm/adaptive MLA + packed GQA paths.",
    )
    return parser.parse_args()


def resolve_shape(args):
    shape = PRESETS[args.preset].copy()
    for k in [
        "batch_size",
        "num_heads",
        "num_kv_heads",
        "head_size",
        "head_size_v",
        "seq_len",
        "num_kv_splits",
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
    assert shape["seq_len"] > 0
    assert shape["num_kv_splits"] > 0


def parse_int_list(value: str, name: str) -> List[int]:
    try:
        values = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{name} must be a comma-separated integer list, got: {value}") from e
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def resolve_seq_lens(args, shape) -> List[int]:
    if args.seq_lens is not None:
        seq_lens = parse_int_list(args.seq_lens, "--seq-lens")
        if len(seq_lens) != shape["batch_size"]:
            raise ValueError(
                f"--seq-lens length ({len(seq_lens)}) must equal batch_size ({shape['batch_size']})"
            )
        if any(x <= 0 for x in seq_lens):
            raise ValueError("--seq-lens values must all be > 0")
        return seq_lens

    seq_len = shape["seq_len"]
    if seq_len <= 0:
        raise ValueError(f"--seq-len must be > 0, got {seq_len}")
    return [seq_len for _ in range(shape["batch_size"])]


def make_decode_bundle(shape, dtype, device, use_mla: bool, seq_lens_list: List[int]):
    b = shape["batch_size"]
    hq = shape["num_heads"]
    hkv = shape["num_kv_heads"]
    d = shape["head_size"]
    dv = shape["head_size_v"]
    num_kv_splits = shape["num_kv_splits"]
    max_context_len = max(seq_lens_list)
    total_tokens = sum(seq_lens_list)
    starts: List[int] = []
    offset = 0
    for sl in seq_lens_list:
        starts.append(offset)
        offset += sl

    query = torch.randn((b, hq, d), dtype=dtype, device=device)
    if use_mla:
        k_buffer = torch.randn((total_tokens, hkv, d), dtype=dtype, device=device)
        v_buffer = k_buffer.narrow(2, 0, dv)
        key = torch.randn((b, hkv, d), dtype=dtype, device=device)
        value = key.narrow(2, 0, dv)
    else:
        key = torch.randn((b, hkv, d), dtype=dtype, device=device)
        value = torch.randn((b, hkv, dv), dtype=dtype, device=device)
        k_buffer = torch.randn((total_tokens, hkv, d), dtype=dtype, device=device)
        v_buffer = torch.randn((total_tokens, hkv, dv), dtype=dtype, device=device)
    output = torch.empty((b, hq, dv), dtype=dtype, device=device)
    loc = torch.tensor(
        [starts[i] + seq_lens_list[i] - 1 for i in range(b)],
        dtype=torch.int64,
        device=device,
    )
    attn_logits = torch.empty((b, hq, num_kv_splits, dv + 1), dtype=torch.float32, device=device)
    req_to_token = torch.empty((b, max_context_len), dtype=torch.int32, device=device)
    for i in range(b):
        start = starts[i]
        sl = seq_lens_list[i]
        req_to_token[i, :sl] = torch.arange(start, start + sl, dtype=torch.int32, device=device)
        if sl < max_context_len:
            req_to_token[i, sl:] = start
    req_pool_indices = torch.arange(b, dtype=torch.int64, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)

    return {
        "query": query,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "output": output,
        "key": key,
        "value": value,
        "loc": loc,
        "attn_logits": attn_logits,
        "req_to_token": req_to_token,
        "req_pool_indices": req_pool_indices,
        "seq_lens": seq_lens,
    }


def make_decode_pool(shape, dtype, device, use_mla: bool, seq_lens_list: List[int]) -> List[dict]:
    return [
        make_decode_bundle(
            shape=shape,
            dtype=dtype,
            device=device,
            use_mla=use_mla,
            seq_lens_list=seq_lens_list,
        )
        for _ in range(INVOCATIONS_PER_ROUND)
    ]


def run_invocations(pool):
    for tensors in pool:
        if tensors["strategy"] == "tdm":
            torch.ops.sgl_kernel.decode_attention_cpu_tdm(
                tensors["query"],
                tensors["k_buffer"],
                tensors["v_buffer"],
                tensors["output"],
                tensors["key"],
                tensors["value"],
                tensors["loc"],
                tensors["attn_logits"],
                tensors["req_to_token"],
                tensors["req_pool_indices"],
                tensors["seq_lens"],
                tensors["sm_scale"],
                tensors["logit_cap"],
                tensors["head_block_num"],
            )
        elif tensors["strategy"] == "adaptive":
            torch.ops.sgl_kernel.decode_attention_cpu_adaptive(
                tensors["query"],
                tensors["k_buffer"],
                tensors["v_buffer"],
                tensors["output"],
                tensors["key"],
                tensors["value"],
                tensors["loc"],
                tensors["attn_logits"],
                tensors["req_to_token"],
                tensors["req_pool_indices"],
                tensors["seq_lens"],
                tensors["sm_scale"],
                tensors["logit_cap"],
                tensors["head_block_num"],
            )
        else:
            torch.ops.sgl_kernel.decode_attention_cpu_tuned(
                tensors["query"],
                tensors["k_buffer"],
                tensors["v_buffer"],
                tensors["output"],
                tensors["key"],
                tensors["value"],
                tensors["loc"],
                tensors["attn_logits"],
                tensors["req_to_token"],
                tensors["req_pool_indices"],
                tensors["seq_lens"],
                tensors["sm_scale"],
                tensors["logit_cap"],
                tensors["head_block_num"],
                tensors["num_kv_splits"],
            )

def use_grouped_packed_decode_kernel_py(
    batch_size: int,
    num_heads: int,
    num_heads_kv: int,
    head_size: int,
    head_size_v: int,
    max_seq_len: int,
) -> bool:
    _ = batch_size
    _ = max_seq_len
    tile_k = 32
    if num_heads_kv <= 0 or num_heads % num_heads_kv != 0:
        return False
    num_groups = num_heads // num_heads_kv
    return (
        num_groups <= 8
        and head_size <= 256
        and head_size_v <= 256
        and head_size % tile_k == 0
        and head_size_v % tile_k == 0
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    shape = resolve_shape(args)
    validate_shape(shape)
    seq_lens_list = resolve_seq_lens(args, shape)
    shape["seq_len"] = max(seq_lens_list)

    sm_scale = args.sm_scale if args.sm_scale is not None else (1.0 / math.sqrt(shape["head_size"]))
    logit_cap = args.logit_cap
    is_mla_shape = shape["num_kv_heads"] == 1 and shape["head_size"] == shape["head_size_v"] + 64

    if args.workload == "auto":
        use_mla = is_mla_shape
    elif args.workload == "mla":
        if not is_mla_shape:
            raise ValueError(
                "MLA mode requires num_kv_heads==1 and head_size==head_size_v+64, "
                f"got num_kv_heads={shape['num_kv_heads']} head_size={shape['head_size']} head_size_v={shape['head_size_v']}"
            )
        if shape["num_heads"] == shape["num_kv_heads"]:
            raise ValueError(
                "MLA mode requires grouped-query setup (num_heads != num_kv_heads) "
                "to enter the MLA kernel path."
            )
        use_mla = True
    elif args.workload == "mha":
        if shape["num_heads"] != shape["num_kv_heads"]:
            raise ValueError(
                f"MHA mode requires num_heads==num_kv_heads, got {shape['num_heads']} and {shape['num_kv_heads']}"
            )
        use_mla = False
    else:  # gqa
        if shape["num_heads"] <= shape["num_kv_heads"]:
            raise ValueError(
                f"GQA mode requires num_heads > num_kv_heads, got {shape['num_heads']} and {shape['num_kv_heads']}"
            )
        use_mla = False

    if args.cpu_bind:
        print(f"Applying CPU binding: {args.cpu_bind}")
        binding_info = torch.ops.sgl_kernel.init_cpu_threads_env(args.cpu_bind)
        print(binding_info, end="" if binding_info.endswith("\n") else "\n")

    max_seq_len = max(seq_lens_list)
    use_grouped_packed = (
        not use_mla
        and shape["num_heads"] != shape["num_kv_heads"]
        and use_grouped_packed_decode_kernel_py(
            shape["batch_size"],
            shape["num_heads"],
            shape["num_kv_heads"],
            shape["head_size"],
            shape["head_size_v"],
            max_seq_len,
        )
    )
    adaptive_target = use_mla or use_grouped_packed

    if args.strategy == "tdm":
        num_threads = torch.get_num_threads()
        if num_threads <= 0:
            raise ValueError(f"TDM expects num_threads > 0, got {num_threads}")
        if num_threads % args.head_block_num != 0:
            raise ValueError(
                f"TDM expects num_threads % head_block_num == 0, got num_threads={num_threads}, "
                f"head_block_num={args.head_block_num}"
            )
        derived_num_kv_splits = num_threads // args.head_block_num
        shape["num_kv_splits"] = derived_num_kv_splits
    elif args.strategy == "adaptive":
        num_threads = torch.get_num_threads()
        if adaptive_target:
            if use_mla:
                denom = args.head_block_num
            else:
                denom = args.head_block_num * shape["num_kv_heads"]
            if denom <= 0:
                raise ValueError(f"Adaptive expects denom > 0, got {denom}")
            if num_threads % denom != 0:
                raise ValueError(
                    f"Adaptive expects num_threads % denom == 0, got num_threads={num_threads}, denom={denom}"
                )
            derived_num_kv_splits = num_threads // denom
            if derived_num_kv_splits <= 0:
                raise ValueError(
                    f"Adaptive expects derived split capacity > 0, got {derived_num_kv_splits}"
                )
            shape["num_kv_splits"] = derived_num_kv_splits
        else:
            # Non-target path falls back to tuned SDM in C++ adaptive op.
            derived_num_kv_splits = shape["num_kv_splits"]
    else:
        num_threads = torch.get_num_threads()
        derived_num_kv_splits = None

    pool = make_decode_pool(shape=shape, dtype=dtype, device=device, use_mla=use_mla, seq_lens_list=seq_lens_list)
    for tensors in pool:
        tensors["sm_scale"] = sm_scale
        tensors["logit_cap"] = logit_cap
        tensors["head_block_num"] = args.head_block_num
        tensors["num_kv_splits"] = shape["num_kv_splits"]
        tensors["strategy"] = args.strategy

    if args.cpu_bind:
        print(f"Warmup round (unmeasured): {INVOCATIONS_PER_ROUND} invocations")
        with torch.inference_mode():
            run_invocations(pool)

    if use_mla:
        derived_head_block_size = (shape["num_heads"] + args.head_block_num - 1) // args.head_block_num
    else:
        derived_head_block_size = ((shape["num_heads"] // shape["num_kv_heads"]) + args.head_block_num - 1) // args.head_block_num

    seq_min = min(seq_lens_list)
    seq_p50 = percentile([float(x) for x in seq_lens_list], 0.5)
    seq_max = max(seq_lens_list)
    print(
        "Config: "
        f"strategy={args.strategy} workload={args.workload} effective_layout={'mla' if use_mla else 'non-mla'} "
        f"head_block_num={args.head_block_num} derived_head_block_size={derived_head_block_size} "
        f"preset={args.preset} B={shape['batch_size']} H={shape['num_heads']} HKV={shape['num_kv_heads']} "
        f"D={shape['head_size']} DV={shape['head_size_v']} L={shape['seq_len']} splits={shape['num_kv_splits']} "
        f"dtype={args.dtype}"
    )
    print(
        "Hybrid seq_lens: "
        f"min={seq_min} p50={seq_p50:.1f} max={seq_max} list={','.join(str(x) for x in seq_lens_list)}"
    )
    if args.strategy == "tdm":
        print(f"TDM derived_num_kv_splits={derived_num_kv_splits} num_threads={num_threads}")
    elif args.strategy == "adaptive":
        print(
            f"Adaptive derived_split_capacity={derived_num_kv_splits} num_threads={num_threads} "
            f"target_path={'yes' if adaptive_target else 'no(fallback-to-sdm)'}"
        )
    print(f"Measured rounds={NUM_ROUNDS}, invocations_per_round={INVOCATIONS_PER_ROUND}")
    if args.enable_decode_timer:
        print("Decode timer: enabled")
        torch.ops.sgl_kernel.decode_timer_start(True)

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
    if args.enable_decode_timer:
        torch.ops.sgl_kernel.decode_timer_stop_and_print()

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
