from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

try:
    from artesia.connector.connector import (
        ArtesiaConnector,
        ContextDescription,
        ModelDescription,
        SemanticDescription,
        TPDescription,
    )
except ImportError as e:
    raise RuntimeError("Artesia is not installed.") from e

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class RadixTreeLog:
    def __init__(self):
        self.dir_path = "./sgl_tree_log"
        os.makedirs(self.dir_path, exist_ok=True)
        self.file_path = os.path.join(self.dir_path, f"radix_tree-{time.time()}.log")
        self.log_file = open(self.file_path, "w", encoding="utf-8")
        self.buffer = []
        self.buffer_capacity = 128
        self.last_ops_time = time.time()

    def log_store(self, node: TreeNode):
        parent_id = None if node.parent is None else node.parent.id
        self._add_log(f"{time.time()},insert,{node.id},{parent_id},{node.key}\n")

    def log_remove(self, node: TreeNode):
        parent_id = None if node.parent is None else node.parent.id
        self._add_log(f"{time.time()},remove,{node.id},{parent_id},\n")

    def _add_log(self, line: str):
        self.buffer.append(line)
        if (
            len(self.buffer) >= self.buffer_capacity
            or time.time() - self.last_ops_time > 1
        ):
            self.log_file.writelines(self.buffer)
            self.buffer.clear()
            self.log_file.flush()
            os.fsync(self.log_file.fileno())
            self.last_ops_time = time.time()


class ArtesiaRadixCache(RadixCache):
    """Radix cache backed by Artesia for MHA/GQA/MQA KV tensors."""

    def __init__(
        self,
        params: CacheInitParams,
        model_config: "ModelConfig",
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        enable_tree_log: bool = False,
    ):
        super().__init__(params)
        del tp_group  # Reserved for connector-side collective coordination.

        self.enable_tree_log = enable_tree_log
        self.tree_log = RadixTreeLog() if enable_tree_log else None

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        if not hasattr(kvcache, "k_buffer") or not hasattr(kvcache, "v_buffer"):
            raise ValueError(
                "Artesia currently supports only MHA/GQA/MQA KV cache pools."
            )

        self.k_pool = kvcache.k_buffer
        self.v_pool = kvcache.v_buffer
        tp_head_num = kvcache.head_num
        self.tp_description = TPDescription(
            rank=rank,
            head_start=rank * tp_head_num,
            head_end=(rank + 1) * tp_head_num,
        )
        self.model_description = ModelDescription(
            model_name=model_config.model_path,
            dtype=kvcache.store_dtype,
            layer_num=kvcache.layer_num,
            kv_shape=torch.Size([2, tp_head_num * tp_size, kvcache.head_dim]),
            tp_size=tp_size,
            tp_head_num=tp_head_num,
        )

        device = self.k_pool[0].device
        self.artesia_connector = ArtesiaConnector(
            local_rank=device.index,
            device=device,
            model=self.model_description,
            kv_pool=[self.k_pool, self.v_pool],
        )
        self.artesia_connector.open()
        self._node_lock = threading.Lock()
        logger.info(
            "Opened Artesia connection: local_rank=%s device=%s model=%s",
            device.index,
            device,
            self.model_description,
        )

    @staticmethod
    def _semantic_description(agent_id, task_id):
        labels = []
        if agent_id is not None:
            labels.append(("agent_id", agent_id))
        if task_id is not None:
            labels.append(("task_id", task_id))
        return SemanticDescription(labels=labels)

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        if self.disable or len(key) == 0:
            return super().match_prefix(key, **kwargs)

        if self.page_size != 1:
            key = key[: len(key) // self.page_size * self.page_size]
        if len(key) == 0:
            return super().match_prefix(key, **kwargs)

        base_res = super().match_prefix(key, **kwargs)
        local_indices = base_res.device_indices
        last_node = base_res.last_device_node
        num_local_cache = local_indices.numel()
        uncached_len = len(key) - num_local_cache

        base_res = MatchResult(
            device_indices=local_indices,
            last_device_node=last_node,
            last_host_node=last_node,
            num_local_cache=num_local_cache,
            num_global_cache=0,
        )
        if uncached_len <= 0:
            return base_res

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.inc_lock_ref(last_node)
            try:
                self.evict(uncached_len)
            finally:
                self.dec_lock_ref(last_node)

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return base_res

        logger.info(
            "Trying to load %d tokens from Artesia at offset=%d",
            len(key),
            num_local_cache,
        )
        torch.cuda.synchronize()
        start_retrieve = time.perf_counter()
        num_retrieved = self.artesia_connector.load_kv(
            context=ContextDescription(token_ids=key.token_ids, offset=num_local_cache),
            semantics=self._semantic_description(
                kwargs.get("agent_id"), kwargs.get("task_id")
            ),
            kv_indices=torch.cat([local_indices, token_slots]),
            tp_description=self.tp_description,
        )
        torch.cuda.synchronize()
        load_kv_elapsed = time.perf_counter() - start_retrieve

        req = kwargs.get("req")
        if req is not None:
            req.load_kv_elapsed = load_kv_elapsed

        fetched = min(num_retrieved, uncached_len)
        fetched -= fetched % self.page_size
        if fetched <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            logger.info("Retrieved 0 tokens from Artesia in %.6fs", load_kv_elapsed)
            return base_res

        self.token_to_kv_pool_allocator.free(token_slots[fetched:])
        fetched_slots = token_slots[:fetched]
        combined_indices = torch.cat([local_indices, fetched_slots])
        fetched_key = key[: num_local_cache + fetched]

        with self._node_lock:
            self.insert(fetched_key, combined_indices)
            matched = RadixCache.match_prefix(self, fetched_key)

        logger.info(
            "Retrieved %d tokens from Artesia in %.6fs", fetched, load_kv_elapsed
        )
        return MatchResult(
            device_indices=matched.device_indices,
            last_device_node=matched.last_device_node,
            last_host_node=matched.last_device_node,
            num_local_cache=num_local_cache,
            num_global_cache=fetched,
        )

    def cache_finished_req(self, req: "Req", is_insert: bool = True, **kwargs) -> None:
        committed_len = req.kv_committed_len
        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :committed_len
        ]

        if committed_len > 0:
            logger.info("Starting Artesia offload for %d tokens", committed_len)
            torch.cuda.synchronize()
            start_store = time.perf_counter()
            self.artesia_connector.offload_kv(
                context=ContextDescription(token_ids=token_ids, offset=0),
                semantics=self._semantic_description(req.agent_id, req.task_id),
                kv_indices=kv_indices,
                call_id=req.call_id,
                tp_description=self.tp_description,
            )
            torch.cuda.synchronize()
            req.offload_kv_elapsed = time.perf_counter() - start_store
            logger.info("Artesia offload took %.6fs", req.offload_kv_elapsed)

        super().cache_finished_req(req, is_insert=is_insert, **kwargs)

    def _record_store_event(self, node: TreeNode):
        super()._record_store_event(node)
        if self.tree_log is not None:
            self.tree_log.log_store(node)

    def _record_remove_event(self, node: TreeNode):
        super()._record_remove_event(node)
        if self.tree_log is not None:
            self.tree_log.log_remove(node)
