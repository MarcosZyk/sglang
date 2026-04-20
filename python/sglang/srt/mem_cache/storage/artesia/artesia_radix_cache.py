from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, List, Optional
import time
import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

try:
    from artesia.connector.connector import (
        ArtesiaConnector,
        ModelDescription,
        ContextDescription,
        SemanticDescription,
    )
except ImportError as e:
    raise RuntimeError(
        "Artesia is not installed."
    ) from e

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

class RadixTreeLog:

    def __init__(self):
        self.dir_path = "./sgl_tree_log"
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path, exist_ok=True)

        self.file_path = os.path.join(self.dir_path, f"radix_tree-{time.time()}.log")
        self.log_file = open(self.file_path, "w", encoding="utf-8",)
        self.buffer = []
        self.buffer_capacity = 128
        self.last_ops_time = time.time()

    def log_store(self, node: TreeNode):
        if node.parent is None:
            parent_id = None
        else:
            parent_id = node.parent.id
        line = f"{time.time()},insert,{node.id},{parent_id},{node.key}\n"
        self._add_log(line)

    def log_remove(self, node: TreeNode):
        if node.parent is None:
            parent_id = None
        else:
            parent_id = node.parent.id
        line = f"{time.time()},remove,{node.id},{parent_id},\n"
        self._add_log(line)

    def _add_log(self, line: str):
        self.buffer.append(line)
        if len(self.buffer) >= self.buffer_capacity or time.time() - self.last_ops_time > 1:
            self.log_file.writelines(self.buffer)
            self.buffer = []
            self.log_file.flush()
            os.fsync(self.log_file.fileno())
            self.last_ops_time = time.time()


class ArtesiaRadixCache(RadixCache):
    """RadixCache + LMCache IO.

    This subclass adds:
      - LMCache connector setup (device/host buffers, TP rank/size)
      - Two CUDA streams for async load/store
      - Layer-wise transfer executor wiring to the KV cache
      - Overridden `match_prefix` to fetch missing prefix chunks from LMCache
      - Extended cache_finalization paths to store back into LMCache
      - Eviction barrier that respects any in-flight host->device stores
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        model_config: Optional["ModelConfig"] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        enable_tree_log: bool = False,
    ):
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            disable=disable,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.enable_tree_log = enable_tree_log
        if self.enable_tree_log:
            self.tree_log = RadixTreeLog()
        else:
            self.tree_log = None

        # Currently, we only support MHATokenToKVPool, which supports model using MHA/GQA/MQA
        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        self.k_pool = getattr(
            kvcache,
            "k_buffer",
            getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),
        )
        self.v_pool=getattr(
            kvcache,
            "v_buffer",
            getattr(self.token_to_kv_pool_allocator._kvcache, "v_buffer"),
        )
        self.kv_pool = [self.k_pool, self.v_pool]

        self.model_description = ModelDescription(
            model_name=f"{model_config.model_path}-{rank}",
            dtype=kvcache.store_dtype,
            layer_num=kvcache.layer_num,
            kv_shape=torch.Size([2, kvcache.head_num, kvcache.head_dim]),
        )

        device = self.k_pool[0].device
        self.artesia_connector = ArtesiaConnector(
            local_rank=device.index,
            device=device,
            model=self.model_description,
            kv_pool=self.kv_pool,
        )
        self.artesia_connector.open()
        logger.info(
            f"Open Artesia connection with config: "
            f"local_rank={device.index}, "
            f"device={device}, "
            f"model_description={self.model_description}"
        )

        self._node_lock = threading.Lock()

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:  # type: ignore[override]
        """Match cached prefix; if there's a tail miss, prefetch from LMCache.

        Reuses the base matching logic to obtain (value, last_node). If there
        remains a *page-aligned* uncached suffix and there is room (or after
        eviction), we allocate token slots and trigger an async LMCache load
        into those slots, then materialize a new child node for the retrieved
        chunk.
        """
        if self.disable or not key:
            return super().match_prefix(key, **kwargs)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(key, **kwargs)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        num_local_cache = value.numel()

        base_res = MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            num_local_cache=num_local_cache,
            num_global_cache=0
        )

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            #logger.info(f"uncached len is 0")
            return base_res

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.inc_lock_ref(last_node)
            # the matched prefix shall not be evicted
            self.evict(uncached_len)
            self.dec_lock_ref(last_node)

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            #logger.info(f"alloc token slots failed")
            return base_res

        logger.info(f"Try load {len(key)} tokens from Artesia with offset={value.numel()}")
        torch.cuda.synchronize()
        start_retrieve = time.perf_counter()

        context = ContextDescription(token_ids=key, offset=value.numel())
        tag_list = []
        if "agent_id" in kwargs:
            tag_list.append(("agent_id", kwargs["agent_id"]))
        if "task_id" in kwargs:
            tag_list.append(("task_id", kwargs["task_id"]))
        semantics = SemanticDescription(tag_list=tag_list)
        num_retrieved = self.artesia_connector.load_kv(
            context=context,
            semantics=semantics,
            kv_indices=torch.cat([value, token_slots]),
        )

        torch.cuda.synchronize()
        end_retrieve = time.perf_counter()

        logger.info(f"Retrieve Time: {end_retrieve - start_retrieve}s")

        if num_retrieved > 0:
            prefix_pad = num_retrieved % self.page_size
            fetched = num_retrieved - prefix_pad
            self.token_to_kv_pool_allocator.free(
                token_slots[fetched :]
            )
            logger.info("Retrieved token num %s from Artesia， used %s", num_retrieved, fetched)
            new_node = TreeNode()
            start = value.numel()
            end = start + fetched
            new_node.key = key[start:end]
            new_node.value = token_slots[:fetched]
            new_node.parent = last_node
            last_node.children[self.get_child_key_fn(new_node.key)] = new_node
            last_node = new_node

            value = torch.cat([value, token_slots[:fetched]])
            self.evictable_size_ += fetched

            self._record_store_event(new_node)

            return MatchResult(
                    device_indices=value,
                    last_device_node=last_node,
                    last_host_node=last_node,
                    num_local_cache=num_local_cache,
                    num_global_cache=fetched
                )
        else:
            logger.info("Retrieved token num 0 from Artesia")
            self.token_to_kv_pool_allocator.free(token_slots)
            #return base_res
            return MatchResult(
                    device_indices=value,
                    last_device_node=last_node,
                    last_host_node=last_node,
                    num_local_cache=num_local_cache,
                    num_global_cache=0
                )

    def cache_finished_req(self, req: "Req") -> None:  # type: ignore[override]
        """On request completion, insert device KV into radix and store to Artesia."""

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        logger.info(f"Start offload {len(token_ids)} tokens to Artesia")
        torch.cuda.synchronize()
        start_store = time.perf_counter()
        context = ContextDescription(token_ids=token_ids, offset=0)
        tag_list = []
        tag_list.append(("agent_id", req.agent_id))
        tag_list.append(("task_id", req.task_id))
        semantics = SemanticDescription(tag_list=tag_list)
        self.artesia_connector.offload_kv(
            context=context,
            semantics=semantics,
            kv_indices=kv_indices,
            call_id=getattr(req, "call_id", None),
        )
        torch.cuda.synchronize()
        end_store = time.perf_counter()

        logger.info(f'Offload time is {end_store - start_store}s')
        super().cache_finished_req(req)


    def _record_store_event(self, node: TreeNode):
        super()._record_store_event(node)
        if not self.enable_tree_log:
            return
        self.tree_log.log_store(node)

    def _record_remove_event(self, node: TreeNode):
        super()._record_remove_event(node)
        if not self.enable_tree_log:
            return
        self.tree_log.log_remove(node)

    def pretty_print(self):  # type: ignore[override]
        super().pretty_print()
        try:
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
            )
        except Exception:  # pragma: no cover
            pass
