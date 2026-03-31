from __future__ import annotations

import heapq
import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import _key_match_page_size1, _key_match_paged

try:
    from artesia.connector.extension_connector import (
        ArtesiaExtensionConnector,
        PageOffloadSpec,
    )
    from artesia.service.common import (
        ContextDescription,
        ModelDescription,
        SemanticDescription,
    )
except ImportError as e:
    raise RuntimeError("Artesia is not installed.") from e

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class ArtesiaTreeNode:
    counter = 0

    def __init__(self):
        self.children: dict[Any, "ArtesiaTreeNode"] = {}
        self.parent: Optional["ArtesiaTreeNode"] = None
        self.key: list[int] = []
        self.value: Optional[torch.Tensor] = None
        self.page_ids: Optional[list[bytes]] = None
        self.page_end_offsets: Optional[list[int]] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.id = ArtesiaTreeNode.counter
        ArtesiaTreeNode.counter += 1

    def __lt__(self, other: "ArtesiaTreeNode") -> bool:
        return self.last_access_time < other.last_access_time

    def get_last_page_id(self) -> Optional[bytes]:
        if self.page_ids:
            return self.page_ids[-1]
        return None


class ArtesiaExtensionCache(BasePrefixCache):
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
        _ = tp_size, tp_group, enable_tree_log
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # Keep the constructor signature compatible with other cache
        # implementations, while keeping SGLang-local paging logic and
        # Artesia paging logic explicit.
        self.sglang_page_size = page_size
        self.artesia_page_size = 1
        self._configure_page_layout(self.sglang_page_size)

        self.k_pool = None
        self.v_pool = None
        self.kv_pool = None
        self.model_description = None
        self.artesia_connector: Optional[ArtesiaExtensionConnector] = None

        if not self.disable:
            if model_config is None:
                raise RuntimeError(
                    "ArtesiaExtensionCache requires model_config to build the "
                    "Artesia model description."
                )
            kvcache = self.token_to_kv_pool_allocator.get_kvcache()
            self.k_pool = getattr(
                kvcache,
                "k_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),
            )
            self.v_pool = getattr(
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
            local_rank = rank if device.index is None else device.index
            self.artesia_connector = ArtesiaExtensionConnector(
                local_rank=local_rank,
                device=device,
                model=self.model_description,
                kv_pool=self.kv_pool,
            )
            self.artesia_connector.open()
            self.artesia_page_size = self.artesia_connector.resolve_model_page_size(
                self.model_description
            )
            logger.info(
                "Open Artesia extension connection with config: local_rank=%s, "
                "device=%s, model_description=%s, sglang_page_size=%s, "
                "artesia_page_size=%s",
                local_rank,
                device,
                self.model_description,
                self.sglang_page_size,
                self.artesia_page_size,
            )

        self.reset()

    def _configure_page_layout(self, page_size: int) -> None:
        if page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])

    def reset(self):
        ArtesiaTreeNode.counter = 0
        self.root_node = ArtesiaTreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.page_ids = []
        self.root_node.page_end_offsets = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.sglang_page_size != 1:
            aligned_len = len(key) // self.sglang_page_size * self.sglang_page_size
            key = key[:aligned_len]

        value_parts, last_node = self._match_prefix_helper(self.root_node, key)
        if value_parts:
            value = torch.cat(value_parts)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        base_res = MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            num_local_cache=value.numel(),
            num_global_cache=0,
        )

        prefix_page_id, anchor_token_count = self._get_load_anchor(last_node)
        # Artesia suffix navigation must continue from the last completed
        # Artesia page, not from the current radix leaf boundary.
        suffix_token_ids = key[anchor_token_count:]
        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.inc_lock_ref(last_node)
            self.evict(uncached_len)
            self.dec_lock_ref(last_node)

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return base_res

        semantics = self._build_semantics(kwargs.get("agent_id"), kwargs.get("task_id"))
        try:
            load_result = self.artesia_connector.load_kv_by_suffix(
                prefix_page_id=prefix_page_id,
                suffix_token_ids=suffix_token_ids,
                semantics=semantics,
                kv_indices=token_slots,
            )
        except Exception:
            self.token_to_kv_pool_allocator.free(token_slots)
            raise

        fetched, fetched_page_ids = self._normalize_loaded_pages(
            num_retrieved=load_result.num_retrieved,
            page_ids=load_result.page_ids,
            max_tokens=token_slots.shape[0],
            prefix_token_count=value.numel(),
        )
        if fetched == 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return base_res

        self.token_to_kv_pool_allocator.free(token_slots[fetched:])

        new_node = ArtesiaTreeNode()
        start = value.numel()
        end = start + fetched
        new_node.key = key[start:end]
        new_node.value = token_slots[:fetched]
        new_node.parent = last_node
        page_end_offsets = self._page_end_offsets_for_span(start, fetched)
        page_count = min(len(page_end_offsets), len(fetched_page_ids))
        if page_count > 0:
            new_node.page_ids = list(fetched_page_ids[:page_count])
            new_node.page_end_offsets = page_end_offsets[:page_count]
        last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        self.evictable_size_ += fetched
        self._record_store_event(new_node)

        value = torch.cat([value, new_node.value])
        return MatchResult(
            device_indices=value,
            last_device_node=new_node,
            last_host_node=new_node,
            num_local_cache=base_res.num_local_cache,
            num_global_cache=fetched,
        )

    def insert(
        self,
        key: List[int],
        value: Optional[torch.Tensor] = None,
        page_ids: Optional[list[bytes]] = None,
    ) -> int:
        if self.disable:
            return 0

        if value is None:
            value = torch.tensor(key, dtype=torch.int64, device=self.device)
        return self._insert_helper(self.root_node, key, value, page_ids)

    def cache_finished_req(self, req: "Req") -> None:
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.sglang_page_size != 1:
            page_aligned_len = (
                len(kv_indices) // self.sglang_page_size * self.sglang_page_size
            )
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        aligned_token_ids = token_ids[:page_aligned_len]
        new_prefix_len = self.insert(aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        if new_prefix_len < page_aligned_len:
            canonical_indices, _, _, _, _, _ = self.match_prefix(aligned_token_ids)
            self._register_local_pages(
                aligned_token_ids,
                canonical_indices,
                req.agent_id,
                req.task_id,
            )

        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: "Req") -> None:
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.sglang_page_size != 1:
            page_aligned_len = (
                len(kv_indices) // self.sglang_page_size * self.sglang_page_size
            )
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
        page_aligned_token_ids = token_ids[:page_aligned_len]

        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        new_indices, new_last_node, _, _, _, _ = self.match_prefix(
            page_aligned_token_ids
        )
        self._register_local_pages(
            page_aligned_token_ids,
            new_indices,
            req.agent_id,
            req.task_id,
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if self.sglang_page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        selected_nodes: list[ArtesiaTreeNode] = []
        selected_ids: set[int] = set()
        selected_page_specs: list[PageOffloadSpec] = []
        planned_evicted = 0

        while planned_evicted < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node == self.root_node:
                break
            if node.lock_ref > 0:
                continue
            if node.value is None:
                continue

            selected_nodes.append(node)
            selected_ids.add(node.id)
            planned_evicted += len(node.value)
            if node.page_ids is not None:
                selected_page_specs.extend(self._build_offload_specs(node))

            parent = node.parent
            if parent is not None and parent != self.root_node:
                if all(child.id in selected_ids for child in parent.children.values()):
                    heapq.heappush(leaves, parent)

        if not selected_nodes:
            return

        self.artesia_connector.offload_pages(selected_page_specs)

        for node in sorted(selected_nodes, key=self._node_depth, reverse=True):
            self.token_to_kv_pool_allocator.free(node.value)
            self._delete_leaf(node)
            self._record_remove_event(node)

    def inc_lock_ref(self, node: ArtesiaTreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: ArtesiaTreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def total_size(self):
        return self._total_size_helper()

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def take_events(self):
        ret = self.kv_event_queue
        self.kv_event_queue = []
        return ret

    def _align_down(self, token_count: int, page_size: int) -> int:
        if page_size <= 1:
            return token_count
        return token_count // page_size * page_size

    def _normalize_loaded_pages(
        self,
        num_retrieved: int,
        page_ids: list[bytes],
        max_tokens: int,
        prefix_token_count: int,
    ) -> tuple[int, list[bytes]]:
        if num_retrieved <= 0 or len(page_ids) == 0 or max_tokens <= 0:
            return 0, []

        fetched = self._align_down(
            min(num_retrieved, max_tokens),
            self.sglang_page_size,
        )
        step = self.sglang_page_size if self.sglang_page_size > 1 else 1
        while fetched > 0:
            page_count = len(
                self._page_end_offsets_for_span(prefix_token_count, fetched)
            )
            if page_count <= len(page_ids):
                return fetched, list(page_ids[:page_count])
            fetched -= step
        return 0, []

    def _truncate_register_inputs(
        self,
        token_ids: list[int],
        kv_indices: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        register_len = self._align_down(
            min(len(token_ids), kv_indices.shape[0]),
            self.sglang_page_size,
        )
        if register_len == 0:
            return [], kv_indices[:0]
        return token_ids[:register_len], kv_indices[:register_len]

    def _build_semantics(
        self,
        agent_id: Optional[str],
        task_id: Optional[str],
    ) -> SemanticDescription:
        tag_list = []
        if agent_id is not None:
            tag_list.append(("agent_id", agent_id))
        if task_id is not None:
            tag_list.append(("task_id", task_id))
        return SemanticDescription(tag_list=tag_list)

    def _match_prefix_helper(self, node: ArtesiaTreeNode, key: List[int]):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return [], node

        child_key = self.get_child_key_fn(key)
        value = []
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break

            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return value, node

    def _split_node(
        self,
        key: list[int],
        child: ArtesiaTreeNode,
        split_len: int,
    ) -> ArtesiaTreeNode:
        self._record_remove_event(child)
        new_node = ArtesiaTreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        if child.page_ids is not None and child.page_end_offsets is not None:
            split_idx = 0
            while (
                split_idx < len(child.page_end_offsets)
                and child.page_end_offsets[split_idx] <= split_len
            ):
                split_idx += 1
            if split_idx > 0:
                new_node.page_ids = child.page_ids[:split_idx]
                new_node.page_end_offsets = child.page_end_offsets[:split_idx]
                child.page_ids = child.page_ids[split_idx:]
                child.page_end_offsets = [
                    offset - split_len for offset in child.page_end_offsets[split_idx:]
                ]
                if len(child.page_ids) == 0:
                    child.page_ids = None
                    child.page_end_offsets = None
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)
        return new_node

    def _insert_helper(
        self,
        node: ArtesiaTreeNode,
        key: List[int],
        value: torch.Tensor,
        page_ids: Optional[list[bytes]],
    ) -> int:
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        remaining_page_ids = list(page_ids) if page_ids is not None else None

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_start = self._node_global_start(node)
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if remaining_page_ids is not None:
                consumed = len(self._page_end_offsets_for_span(prefix_start, prefix_len))
                remaining_page_ids = remaining_page_ids[consumed:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = ArtesiaTreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            if remaining_page_ids is not None:
                prefix_start = self._node_global_end(node)
                page_end_offsets = self._page_end_offsets_for_span(prefix_start, len(key))
                page_count = min(len(page_end_offsets), len(remaining_page_ids))
                if page_count > 0:
                    new_node.page_ids = list(remaining_page_ids[:page_count])
                    new_node.page_end_offsets = page_end_offsets[:page_count]
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        return total_prefix_length

    def _bind_page_ids(self, key: list[int], page_ids: list[bytes]) -> None:
        if len(key) == 0 or len(page_ids) == 0:
            return

        node = self.root_node
        remaining_key = key
        remaining_page_ids = list(page_ids)
        while len(remaining_key) > 0:
            child_key = self.get_child_key_fn(remaining_key)
            if child_key not in node.children:
                raise RuntimeError("Failed to bind page ids: radix path not found.")
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, remaining_key)
            prefix_start = self._node_global_start(child)
            node_page_end_offsets = self._page_end_offsets_for_span(
                prefix_start, prefix_len
            )
            page_count = len(node_page_end_offsets)
            node_page_ids = remaining_page_ids[:page_count]
            if page_count > 0 and len(node_page_ids) > 0:
                if child.page_ids is None:
                    child.page_ids = []
                    child.page_end_offsets = []
                existing_count = len(child.page_ids)
                if existing_count < page_count:
                    child.page_ids.extend(node_page_ids[existing_count:page_count])
                    child.page_end_offsets.extend(
                        node_page_end_offsets[existing_count:page_count]
                    )

            remaining_key = remaining_key[prefix_len:]
            remaining_page_ids = remaining_page_ids[page_count:]
            if page_count > len(node_page_ids):
                break
            node = child

    def _register_local_pages(
        self,
        token_ids: list[int],
        kv_indices: torch.Tensor,
        agent_id: Optional[str],
        task_id: Optional[str],
    ) -> None:
        register_token_ids, register_kv_indices = self._truncate_register_inputs(
            token_ids, kv_indices
        )
        if len(register_token_ids) == 0:
            return

        context = ContextDescription(token_ids=register_token_ids, offset=0)
        semantics = self._build_semantics(agent_id, task_id)
        register_result = self.artesia_connector.register_pages(
            context, semantics, register_kv_indices
        )
        page_count = len(self._page_end_offsets_for_span(0, len(register_token_ids)))
        self._bind_page_ids(register_token_ids, register_result.page_ids[:page_count])

    def _build_offload_specs(self, node: ArtesiaTreeNode) -> list[PageOffloadSpec]:
        if node.page_ids is None or node.page_end_offsets is None:
            return []

        specs = []
        for page_id, page_end_offset in zip(node.page_ids, node.page_end_offsets):
            specs.append(
                PageOffloadSpec(
                    page_id=page_id,
                    kv_indices=self._collect_page_kv_indices(node, page_end_offset),
                )
            )
        return specs

    def _get_load_anchor(self, node: ArtesiaTreeNode) -> tuple[bytes, int]:
        current = node
        while current != self.root_node:
            if current.page_ids and current.page_end_offsets:
                return (
                    current.page_ids[-1],
                    self._node_global_start(current) + current.page_end_offsets[-1],
                )
            current = current.parent
        return b"", 0

    def _node_global_start(self, node: ArtesiaTreeNode) -> int:
        total = 0
        current = node.parent
        while current is not None:
            total += len(current.key)
            current = current.parent
        return total

    def _node_global_end(self, node: ArtesiaTreeNode) -> int:
        return self._node_global_start(node) + len(node.key)

    def _page_end_offsets_for_span(
        self,
        prefix_start: int,
        span_len: int,
    ) -> list[int]:
        if span_len <= 0:
            return []

        first_page_end = (
            (prefix_start + self.artesia_page_size - 1) // self.artesia_page_size
        ) * self.artesia_page_size
        if first_page_end <= prefix_start:
            first_page_end += self.artesia_page_size

        prefix_end = prefix_start + span_len
        page_end_offsets = []
        page_end = first_page_end
        while page_end <= prefix_end:
            page_end_offsets.append(page_end - prefix_start)
            page_end += self.artesia_page_size
        return page_end_offsets

    def _collect_page_kv_indices(
        self,
        node: ArtesiaTreeNode,
        page_end_offset: int,
    ) -> torch.Tensor:
        page_end = self._node_global_start(node) + page_end_offset
        page_begin = page_end - self.artesia_page_size
        value_parts = []
        current = node
        while current != self.root_node:
            current_start = self._node_global_start(current)
            current_end = current_start + len(current.key)
            overlap_begin = max(page_begin, current_start)
            overlap_end = min(page_end, current_end)
            if overlap_begin < overlap_end:
                local_begin = overlap_begin - current_start
                local_end = overlap_end - current_start
                value_parts.append(current.value[local_begin:local_end])
            current = current.parent

        if len(value_parts) == 1:
            return value_parts[0]
        value_parts.reverse()
        return torch.cat(value_parts)

    def _node_depth(self, node: ArtesiaTreeNode) -> int:
        depth = 0
        while node != self.root_node:
            node = node.parent
            depth += 1
        return depth

    def _collect_leaves(self) -> list[ArtesiaTreeNode]:
        ret = []
        stack = [self.root_node]
        while stack:
            current = stack.pop()
            if len(current.children) == 0:
                ret.append(current)
            else:
                stack.extend(current.children.values())
        return ret

    def _delete_leaf(self, node: ArtesiaTreeNode) -> None:
        for child_key, child in node.parent.children.items():
            if child == node:
                break
        del node.parent.children[child_key]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self) -> int:
        total_size = 0
        stack = [self.root_node]
        while stack:
            current = stack.pop()
            total_size += len(current.value)
            for child in current.children.values():
                stack.append(child)
        return total_size

    def _print_helper(self, node: ArtesiaTreeNode, indent: int) -> None:
        stack = [(node, indent)]
        while stack:
            current, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current.key),
                current.key[:10],
                f"r={current.lock_ref}",
                f"pages={0 if current.page_ids is None else len(current.page_ids)}",
            )
            for child_key, child in current.children.items():
                stack.append((child, current_indent + 2))
                assert child_key == self.get_child_key_fn(
                    child.key
                ), f"{child_key=}, {self.get_child_key_fn(child.key)=}"

    def _record_store_event(self, node: ArtesiaTreeNode) -> None:
        if self.enable_kv_cache_events:
            if node.parent is None:
                parent_block_hash = None
            else:
                last_page_start = (
                    (len(node.parent.key) - 1) // self.sglang_page_size
                ) * self.sglang_page_size
                parent_parent_tokens = node.parent.key[last_page_start:]
                parent_block_hash = hash(tuple(parent_parent_tokens))

            for start in range(0, len(node.key), self.sglang_page_size):
                page_tokens = node.key[start : start + self.sglang_page_size]
                if not page_tokens:
                    continue
                block_hash = hash(tuple(page_tokens))
                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                    )
                )
                parent_block_hash = block_hash

    def _record_remove_event(self, node: ArtesiaTreeNode) -> None:
        if self.enable_kv_cache_events:
            for start in range(0, len(node.key), self.sglang_page_size):
                page_tokens = node.key[start : start + self.sglang_page_size]
                if not page_tokens:
                    continue
                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[hash(tuple(page_tokens))])
                )

    def _record_all_cleared_event(self) -> None:
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())
