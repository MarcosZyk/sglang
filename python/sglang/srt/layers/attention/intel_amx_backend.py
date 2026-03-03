from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)



class KernelAutoTuner:

    # L: sequence Length; l: split length;      s: split num
    # H: head num;        h: head block size;   b: block num
    # C: core num
    # Constraints: s * b = C, s * b means the num of parallel unit
    # MLA Target: min(head_dim * h * s + kv_dim * l * b)
    # MHA/GQA Target: min(head_dim * h * s + 2 * kv_dim * l * b)

    def __init__(
        self,
        q_head_num: int,
        q_dim: int,
        kv_dim: int,
        is_mla: bool,
        core_number: int,
    ):
        self.q_head_num = q_head_num
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.is_mla = is_mla
        self.core_number = core_number

        self.factor_pair_list = []
        for i in range(1, int(self.core_number ** 0.5) + 1):
            if self.core_number % i == 0:
                self.factor_pair_list.append((i, self.core_number // i))

    def calculate_config(self, seq_len: int) -> tuple[int, int]:
        # input sequence length, output (head block size, split num)
        index = 0
        minimal_overhead = -1
        for i, factor_pair in enumerate(self.factor_pair_list):
            block_num = factor_pair[0]
            split_num = factor_pair[1]

            block_size = math.ceil(self.q_head_num / block_num)
            split_size = math.ceil(seq_len / split_num)

            overhead = self.redundant_io_size(block_num, block_size, split_num, split_size)
            if minimal_overhead > overhead or minimal_overhead == -1:
                minimal_overhead = overhead
                index = i

        return self.factor_pair_list[index]


    def redundant_io_size(self, block_num: int, block_size: int, split_num: int, split_size: int) -> int:
        if self.is_mla:
            return self.q_dim * block_size + self.kv_dim * split_size
        else:
            return self.q_dim * block_size + 2 * self.kv_dim * split_size


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel

        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu_v2
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu

        cores = model_runner.local_omp_cpuid.split(",")
        core_number = 0
        for core in cores:
            if "-" in core:
                core = core.split("-")
                core_number += int(core[1]) - int(core[0]) + 1
            else:
                core_number += 1
        self.core_number = core_number

        self.auto_tune = int(os.getenv("AMX_KERNEL_AUTO_TUNE", 0)) == 1
        if model_runner.use_mla_backend:
            q_head_dim = model_runner.model_config.qk_nope_head_dim + model_runner.model_config.qk_rope_head_dim
            kv_dim = model_runner.model_config.kv_lora_rank + model_runner.model_config.qk_rope_head_dim
        else:
            q_head_dim = model_runner.model_config.head_dim
            kv_dim = model_runner.model_config.head_dim
        self.tuner = KernelAutoTuner(
            self.num_head,
            q_head_dim,
            kv_dim,
            model_runner.use_mla_backend,
            self.core_number
        )


    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        bs = forward_batch.batch_size
        if self.auto_tune:
            # todo consider bs > 1
            block_size, split_num = self.tuner.calculate_config(forward_batch.seq_lens_sum)
            forward_batch.split_num = split_num
            forward_batch.head_block_size = block_size
        else:
            forward_batch.split_num = 8
            forward_batch.head_block_size = 6 if bs == 1 else (22 if bs > 16 else 11)

        logger.debug(
            "Forward decode with cores=%s, split_num=%s, head_block_size=%s",
            self.core_number,
            forward_batch.split_num,
            forward_batch.head_block_size,
        )

        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                forward_batch.split_num,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
        self.forward_metadata = (attn_logits, max_extend_len)

    def get_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        attn_logits, _ = self.forward_metadata

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            forward_batch.out_cache_loc,
            attn_logits,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.scaling,
            layer.logit_cap,
            forward_batch.head_block_size
        )

        return o

    def support_triton(self):
        return False
