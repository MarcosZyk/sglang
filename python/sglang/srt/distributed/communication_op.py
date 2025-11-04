# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllToAllOptions,
)

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


_GROUP: Optional[torch.distributed.ProcessGroup] = None
_TP_SIZE = None

_input_split_sizes = []
_output_split_sizes = []

_ALL_GATHER_OUTPUT: Optional[torch.Tensor] = None
_ALL_TO_ALL_OUTPUT: Optional[torch.Tensor] = None

def init_amx_tp_group(batch_size, head_num, q_head_dim, attn_logits_dim):
    global _GROUP, _TP_SIZE, _ALL_GATHER_OUTPUT, _ALL_TO_ALL_OUTPUT
    _GROUP = get_tp_group().device_group
    _TP_SIZE = get_tp_group().world_size

    _ALL_GATHER_OUTPUT = torch.empty(head_num, batch_size, q_head_dim, dtype=torch.bfloat16, device="cpu")
    _ALL_TO_ALL_OUTPUT = torch.empty(head_num, batch_size, attn_logits_dim, dtype=torch.float32, device="cpu")


def parallel_amx_all_gather(input_: torch.Tensor,) -> torch.Tensor:
    global _GROUP, _ALL_GATHER_OUTPUT
    opts = AllgatherOptions()

    work = _GROUP._allgather_base(
        _ALL_GATHER_OUTPUT, input_, opts
    )

    work.wait()

    return _ALL_GATHER_OUTPUT

def parallel_amx_all_to_all(input_: torch.Tensor,) -> torch.Tensor:
    global _GROUP, _ALL_TO_ALL_OUTPUT
    opts = AllToAllOptions()

    work = _GROUP.alltoall_base(
        _ALL_TO_ALL_OUTPUT, input_, _output_split_sizes, _input_split_sizes, opts
    )

    work.wait()

    return _ALL_TO_ALL_OUTPUT
