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
    from artesia.service.common import (
        ContextDescription,
        ModelDescription,
        SemanticDescription,
    )

    from sglang.srt.mem_cache.storage.artesia.artesia_extension_connector import (
        ArtesiaExtensionConnector,
        PageOffloadSpec,
    )
except ImportError as e:
    raise RuntimeError("Artesia is not installed.") from e

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class ExtensionTreeNode:
    counter = 0

    def __init__(self):
        self.children: dict[Any, "ExtensionTreeNode"] = {}
        self.parent: Optional["ExtensionTreeNode"] = None
        self.key: list[int] = []
        self.value: Optional[torch.Tensor] = None
        self.page_ids: Optional[list[bytes]] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.id = ExtensionTreeNode.counter
        ExtensionTreeNode.counter += 1

    def __lt__(self, other: "ExtensionTreeNode") -> bool:
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

        self.page_size = page_size
        self._configure_page_layout(page_size)

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
            self.artesia_connector = ArtesiaExtensionConnector(
                local_rank=device.index,
                device=device,
                model=self.model_description,
                kv_pool=self.kv_pool,
            )

            resolved_page_size = self.artesia_connector.resolve_model_page_size(
                self.model_description
            )
            if resolved_page_size != page_size:
                raise RuntimeError(
                    "ArtesiaExtensionCache page size mismatch: "
                    f"sglang={page_size}, artesia={resolved_page_size}. "
                    "Set SGLang and Artesia to the same page size before enabling "
                    "the extension cache."
                )
            self.page_size = resolved_page_size
            self._configure_page_layout(self.page_size)
            self.artesia_connector.open()
            logger.info(
                "Open Artesia extension connection with config: local_rank=%s, "
                "device=%s, model_description=%s, page_size=%s",
                device.index,
                device,
                self.model_description,
                self.page_size,
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
        ExtensionTreeNode.counter = 0
        self.root_node = ExtensionTreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.page_ids = []
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

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
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

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        if last_node == self.root_node:
            prefix_page_id = b""
        else:
            prefix_page_id = last_node.get_last_page_id()
            if prefix_page_id is None:
                logger.info(
                    "Skip Artesia extension load because matched node %s has no "
                    "bound page ids yet.",
                    last_node.id,
                )
                return base_res

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.inc_lock_ref(last_node)
            self.evict(uncached_len)
            self.dec_lock_ref(last_node)

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return base_res

        semantics = self._build_semantics(kwargs.get("agent_id"), kwargs.get("task_id"))
        suffix_token_ids = key[value.numel() :]
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

        num_retrieved = load_result.num_retrieved
        fetched = num_retrieved - (num_retrieved % self.page_size)
        if fetched == 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return base_res

        expected_page_count = self._page_count_for_tokens(fetched)
        if len(load_result.page_ids) != expected_page_count:
            self.token_to_kv_pool_allocator.free(token_slots)
            raise RuntimeError(
                "ArtesiaExtensionCache load returned mismatched page ids: "
                f"expected {expected_page_count}, got {len(load_result.page_ids)}."
            )

        self.token_to_kv_pool_allocator.free(token_slots[fetched:])

        new_node = ExtensionTreeNode()
        start = value.numel()
        end = start + fetched
        new_node.key = key[start:end]
        new_node.value = token_slots[:fetched]
        new_node.page_ids = list(load_result.page_ids)
        new_node.parent = last_node
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

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
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

        context = ContextDescription(token_ids=aligned_token_ids, offset=0)
        semantics = self._build_semantics(req.agent_id, req.task_id)
        register_result = self.artesia_connector.register_pages(context, semantics)
        expected_page_count = self._page_count_for_tokens(page_aligned_len)
        if len(register_result.page_ids) != expected_page_count:
            raise RuntimeError(
                "ArtesiaExtensionCache registration returned mismatched page ids: "
                f"expected {expected_page_count}, got {len(register_result.page_ids)}."
            )
        self._bind_page_ids(aligned_token_ids, register_result.page_ids)

        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: "Req") -> None:
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
        page_aligned_token_ids = token_ids[:page_aligned_len]

        self.insert(page_aligned_token_ids, page_aligned_kv_indices)

        new_indices, new_last_node, _, _, _, _ = self.match_prefix(page_aligned_token_ids)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if self.page_size != 1:
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

        selected_nodes: list[ExtensionTreeNode] = []
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
            if node.page_ids is None:
                raise RuntimeError(
                    "ArtesiaExtensionCache cannot evict a node without bound page ids. "
                    f"node_id={node.id}"
                )

            selected_nodes.append(node)
            selected_ids.add(node.id)
            planned_evicted += len(node.value)
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

    def inc_lock_ref(self, node: ExtensionTreeNode):
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

    def dec_lock_ref(
        self,
        node: ExtensionTreeNode,
        swa_uuid_for_lock: Optional[str] = None,
    ):
        _ = swa_uuid_for_lock
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

    def _page_count_for_tokens(self, token_count: int) -> int:
        if token_count == 0:
            return 0
        if token_count % self.page_size != 0:
            raise RuntimeError(
                f"Token count {token_count} is not aligned to page size {self.page_size}."
            )
        return token_count // self.page_size

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

    def _match_prefix_helper(self, node: ExtensionTreeNode, key: List[int]):
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
        child: ExtensionTreeNode,
        split_len: int,
    ) -> ExtensionTreeNode:
        self._record_remove_event(child)
        new_node = ExtensionTreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        prefix_page_count = self._page_count_for_tokens(split_len)
        if child.page_ids is not None:
            new_node.page_ids = child.page_ids[:prefix_page_count]
            child.page_ids = child.page_ids[prefix_page_count:]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)
        return new_node

    def _insert_helper(
        self,
        node: ExtensionTreeNode,
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
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if remaining_page_ids is not None:
                consumed = self._page_count_for_tokens(prefix_len)
                remaining_page_ids = remaining_page_ids[consumed:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = ExtensionTreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            new_node.page_ids = (
                list(remaining_page_ids) if remaining_page_ids is not None else None
            )
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        return total_prefix_length

    def _bind_page_ids(self, key: list[int], page_ids: list[bytes]) -> None:
        if len(key) == 0:
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
            if prefix_len != len(child.key):
                raise RuntimeError("Failed to bind page ids: non-exact radix match.")

            page_count = self._page_count_for_tokens(len(child.key))
            node_page_ids = remaining_page_ids[:page_count]
            if child.page_ids is None:
                child.page_ids = list(node_page_ids)
            elif child.page_ids != list(node_page_ids):
                raise RuntimeError(
                    f"Page-id mismatch while binding node {child.id}: "
                    "existing page ids differ from logical registration."
                )

            remaining_key = remaining_key[prefix_len:]
            remaining_page_ids = remaining_page_ids[page_count:]
            node = child

        if remaining_page_ids:
            raise RuntimeError("Failed to bind page ids: unconsumed page ids remain.")

    def _build_offload_specs(self, node: ExtensionTreeNode) -> list[PageOffloadSpec]:
        if node.page_ids is None:
            return []
        expected_page_count = self._page_count_for_tokens(len(node.key))
        if len(node.page_ids) != expected_page_count:
            raise RuntimeError(
                "ArtesiaExtensionCache found misaligned page ids on eviction: "
                f"node_id={node.id}, expected_pages={expected_page_count}, "
                f"actual_pages={len(node.page_ids)}."
            )

        specs = []
        for idx, page_id in enumerate(node.page_ids):
            begin = idx * self.page_size
            end = begin + self.page_size
            specs.append(
                PageOffloadSpec(
                    page_id=page_id,
                    kv_indices=node.value[begin:end],
                )
            )
        return specs

    def _node_depth(self, node: ExtensionTreeNode) -> int:
        depth = 0
        while node != self.root_node:
            node = node.parent
            depth += 1
        return depth

    def _collect_leaves(self) -> list[ExtensionTreeNode]:
        ret = []
        stack = [self.root_node]
        while stack:
            current = stack.pop()
            if len(current.children) == 0:
                ret.append(current)
            else:
                stack.extend(current.children.values())
        return ret

    def _delete_leaf(self, node: ExtensionTreeNode) -> None:
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

    def _print_helper(self, node: ExtensionTreeNode, indent: int) -> None:
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

    def _record_store_event(self, node: ExtensionTreeNode) -> None:
        if self.enable_kv_cache_events:
            if node.parent is None:
                parent_block_hash = None
            else:
                last_page_start = (
                    (len(node.parent.key) - 1) // self.page_size
                ) * self.page_size
                parent_parent_tokens = node.parent.key[last_page_start:]
                parent_block_hash = hash(tuple(parent_parent_tokens))

            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key[start : start + self.page_size]
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

    def _record_remove_event(self, node: ExtensionTreeNode) -> None:
        if self.enable_kv_cache_events:
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key[start : start + self.page_size]
                if not page_tokens:
                    continue
                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[hash(tuple(page_tokens))])
                )

    def _record_all_cleared_event(self) -> None:
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())
