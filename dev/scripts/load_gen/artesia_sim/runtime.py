from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Literal, Optional, Protocol

from transformers import AutoTokenizer

try:
    from models import ChatCompletionRequest, ChatCompletionResponse
except ImportError:
    from .models import ChatCompletionRequest, ChatCompletionResponse


Placement = Literal["gpu", "cpu"]
MessageState = Literal["durable", "suspend"]


class ArtesiaError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class TokenizerProtocol(Protocol):
    all_special_ids: List[int]
    vocab_size: int

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ...

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        ...

    def get_vocab(self) -> Dict[str, int]:
        ...


TokenizerLoader = Callable[[str], TokenizerProtocol]
SleepFunc = Callable[[float], Awaitable[None]]


@dataclass
class SimulatorConfig:
    cpu_gpu_bandwidth_gbps: float = 12.0
    gpu_sglang_bandwidth_gbps: float = 24.0
    prefill_throughput_tps: float = 8000.0
    decode_throughput_tps: float = 2000.0
    decode_max_concurrency: int = 1
    kv_cache_kb_per_token: float = 144.0
    gpu_capacity_gb: float = 24.0
    eviction_policy: Literal["lru", "mru"] = "lru"
    enable_artesia: bool = False
    tokenizer_name: Optional[str] = None
    seed: Optional[int] = None

    @property
    def cpu_gpu_bytes_per_second(self) -> float:
        return self.cpu_gpu_bandwidth_gbps * 1_000_000_000.0

    @property
    def gpu_sglang_bytes_per_second(self) -> float:
        return self.gpu_sglang_bandwidth_gbps * 1_000_000_000.0

    @property
    def kv_cache_bytes_per_token(self) -> int:
        return int(self.kv_cache_kb_per_token * 1_000)

    @property
    def gpu_capacity_bytes(self) -> int:
        return int(self.gpu_capacity_gb * 1_000_000_000)


@dataclass
class StoredMessage:
    message_id: str
    role: str
    token_count: int
    placement: Placement
    state: MessageState

    def kv_bytes(self, kv_cache_bytes_per_token: int) -> int:
        return self.token_count * kv_cache_bytes_per_token


@dataclass
class ContextState:
    context_id: str
    messages: List[StoredMessage] = field(default_factory=list)
    state: MessageState = "durable"
    state_changed_at_ns: int = 0
    pending_one_off_index: Optional[int] = None


def default_tokenizer_loader(model_name: str) -> TokenizerProtocol:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


class ArtesiaSimulator:
    def __init__(
        self,
        config: SimulatorConfig,
        tokenizer_loader: TokenizerLoader = default_tokenizer_loader,
        sleep_func: SleepFunc = asyncio.sleep,
    ) -> None:
        self.config = config
        self.tokenizer_loader = tokenizer_loader
        self.sleep_func = sleep_func
        self.contexts: Dict[str, ContextState] = {}
        self._archived_contexts: Dict[str, ContextState] = {}
        self._tokenizers: Dict[str, TokenizerProtocol] = {}
        self._tokenizer_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._prefill_lock = asyncio.Lock()
        self._decode_semaphore = asyncio.Semaphore(config.decode_max_concurrency)
        self._busy_contexts: set[str] = set()
        self._gpu_bytes_used = 0
        self._cpu_bytes_used = 0
        self._archive_seq = 0
        self._message_seq = 0
        self._random = random.Random(config.seed)

    @property
    def gpu_bytes_used(self) -> int:
        return self._gpu_bytes_used

    @property
    def cpu_bytes_used(self) -> int:
        return self._cpu_bytes_used

    async def create_context(self, context_id: str, context_type: str) -> Dict[str, Any]:
        if context_type != "durable":
            raise ArtesiaError(400, "only context_type=durable is supported")

        async with self._state_lock:
            if context_id in self.contexts:
                raise ArtesiaError(409, f"context {context_id} already exists")
            self.contexts[context_id] = ContextState(
                context_id=context_id,
                state="durable",
                state_changed_at_ns=self._now_ns(),
            )
            return {
                "status": "ok",
                "context_id": context_id,
                "context_type": context_type,
            }

    async def delete_context(self, context_id: str) -> Dict[str, Any]:
        async with self._state_lock:
            context = self._require_context(context_id)
            self._ensure_context_not_busy(context_id)
            if not self.config.enable_artesia:
                return {"status": "ok", "context_id": context_id}
            for message in context.messages:
                self._release_message_storage(message)
            del self.contexts[context_id]
            return {"status": "ok", "context_id": context_id}

    async def truncate(self, context_id: str, msg_index: int) -> Dict[str, Any]:
        async with self._state_lock:
            context = self._require_context(context_id)
            self._ensure_context_not_busy(context_id)
            if msg_index < 0 or msg_index > len(context.messages):
                raise ArtesiaError(
                    400,
                    f"msg_index must be in [0, {len(context.messages)}]",
                )
            if self.config.enable_artesia:
                removed = context.messages[msg_index:]
                context.messages = context.messages[:msg_index]
                for message in removed:
                    self._release_message_storage(message)
            else:
                self._archive_context_suffix(
                    context=context,
                    start_index=msg_index,
                    operation="truncate",
                )
            return {
                "status": "ok",
                "context_id": context_id,
                "kept_messages": len(context.messages),
            }

    async def one_off(self, context_id: str, msg_index: int) -> Dict[str, Any]:
        if msg_index < 0:
            raise ArtesiaError(400, "msg_index must be >= 0")

        async with self._state_lock:
            context = self._require_context(context_id)
            self._ensure_context_not_busy(context_id)
            context.pending_one_off_index = msg_index
            return {
                "status": "ok",
                "context_id": context_id,
                "msg_index": msg_index,
            }

    async def suspend(self, context_id: str) -> Dict[str, Any]:
        async with self._state_lock:
            context = self._require_context(context_id)
            self._ensure_context_not_busy(context_id)
            if not self.config.enable_artesia:
                return {
                    "status": "ok",
                    "context_id": context_id,
                    "num_messages": len(context.messages),
                }
            self._set_context_state(context, "suspend")
            return {
                "status": "ok",
                "context_id": context_id,
                "num_messages": len(context.messages),
            }

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise ArtesiaError(400, "stream is not supported")
        if request.context_id is None:
            raise ArtesiaError(400, "context_id is required")
        if request.max_tokens < 0:
            raise ArtesiaError(400, "max_tokens must be >= 0")
        if self.config.decode_max_concurrency <= 0:
            raise ArtesiaError(500, "decode_max_concurrency must be > 0")

        tokenizer_name = self.config.tokenizer_name or request.model
        tokenizer = await self._get_tokenizer(tokenizer_name)
        return await self._run_generate(request=request, tokenizer=tokenizer)

    async def _run_generate(
        self,
        *,
        request: ChatCompletionRequest,
        tokenizer: TokenizerProtocol,
    ) -> ChatCompletionResponse:
        prompt_token_ids: List[List[int]] = [
            tokenizer.encode(message.content, add_special_tokens=False)
            for message in request.messages
        ]
        prompt_roles = [message.role for message in request.messages]
        prompt_token_counts = [len(token_ids) for token_ids in prompt_token_ids]
        context_marked_busy = False
        try:
            async with self._state_lock:
                context = self._require_context(request.context_id)
                if request.context_id in self._busy_contexts:
                    raise ArtesiaError(409, f"context {request.context_id} is busy")
                self._busy_contexts.add(request.context_id)
                context_marked_busy = True
            async with self._prefill_lock:
                async with self._state_lock:
                    context = self._require_context(request.context_id)
                    if not self.config.enable_artesia:
                        self._touch_context_timestamp(context)
                    matched_prefix_len = self._count_matched_prefix(
                        context.messages,
                        prompt_token_counts,
                    )
                    matched_messages = context.messages[:matched_prefix_len]
                    num_local_cache = sum(
                        message.token_count
                        for message in matched_messages
                        if message.placement == "gpu"
                    )
                    num_global_cache = sum(message.token_count for message in matched_messages)
                    fetch_time, offload_for_fetch_time = self._ensure_prefix_available_on_gpu(
                        matched_messages=matched_messages
                    )
                    transfer_to_sglang_time = self._bytes_to_seconds(
                        sum(
                            message.kv_bytes(self.config.kv_cache_bytes_per_token)
                            for message in matched_messages
                        ),
                        self.config.gpu_sglang_bytes_per_second,
                    )
                    pending_one_off_index = context.pending_one_off_index

                prefix_time = fetch_time + offload_for_fetch_time + transfer_to_sglang_time
                uncached_prompt_tokens = sum(prompt_token_counts[matched_prefix_len:])
                prefill_compute_time = self._tokens_to_seconds(
                    uncached_prompt_tokens,
                    self.config.prefill_throughput_tps,
                )
                if prefix_time > 0:
                    await self.sleep_func(prefix_time)
                if self.config.enable_artesia:
                    async with self._state_lock:
                        context = self._require_context(request.context_id)
                        self._set_context_state(context, "durable")
                if prefill_compute_time > 0:
                    await self.sleep_func(prefill_compute_time)

            decode_time = self._tokens_to_seconds(
                request.max_tokens,
                self.config.decode_throughput_tps,
            )
            async with self._decode_semaphore:
                if decode_time > 0:
                    await self.sleep_func(decode_time)

            completion_token_ids = self._sample_token_ids(tokenizer, request.max_tokens)
            completion_text = tokenizer.decode(
                completion_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            completion_token_count = len(completion_token_ids)

            final_messages = [
                {"role": role, "token_count": token_count}
                for role, token_count in zip(prompt_roles, prompt_token_counts)
            ]
            final_messages.append(
                {
                    "role": "assistant",
                    "token_count": completion_token_count,
                }
            )
            if self.config.enable_artesia and pending_one_off_index is not None:
                target_messages = final_messages[:pending_one_off_index]
            else:
                target_messages = final_messages

            async with self._prefill_lock:
                async with self._state_lock:
                    context = self._require_context(request.context_id)
                    reuse_prefix_len = min(matched_prefix_len, len(target_messages))
                    stale_messages = context.messages[reuse_prefix_len:]
                    context.messages = context.messages[:reuse_prefix_len]
                    for message in stale_messages:
                        self._release_message_storage(message)

                    write_transfer_time, write_offload_time = self._append_target_messages(
                        context=context,
                        target_messages=target_messages[reuse_prefix_len:],
                    )
                    context.pending_one_off_index = None

                    if not self.config.enable_artesia and pending_one_off_index is not None:
                        self._archive_context_suffix(
                            context=context,
                            start_index=min(pending_one_off_index, len(context.messages)),
                            operation="one_off",
                        )

                write_time = write_transfer_time + write_offload_time
                if write_time > 0:
                    await self.sleep_func(write_time)

            prefill_time = prefix_time + prefill_compute_time + write_time
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion_text,
                        },
                        "finish_reason": "length",
                        "logprobs": None,
                    }
                ],
                usage={
                    "prompt_tokens": uncached_prompt_tokens,
                    "completion_tokens": completion_token_count,
                    "total_tokens": uncached_prompt_tokens + completion_token_count,
                    "prompt_tokens_details": {
                        "cached_tokens": num_global_cache,
                    },
                },
                prefill_time=prefill_time,
                decode_time=[decode_time],
                num_local_cache=num_local_cache,
                num_global_cache=num_global_cache,
            )
        finally:
            if context_marked_busy:
                async with self._state_lock:
                    self._busy_contexts.discard(request.context_id)

    async def _get_tokenizer(self, model_name: str) -> TokenizerProtocol:
        async with self._tokenizer_lock:
            if model_name not in self._tokenizers:
                self._tokenizers[model_name] = self.tokenizer_loader(model_name)
            return self._tokenizers[model_name]

    def _require_context(self, context_id: str) -> ContextState:
        context = self.contexts.get(context_id)
        if context is None:
            raise ArtesiaError(404, f"context {context_id} does not exist")
        return context

    def _ensure_context_not_busy(self, context_id: str) -> None:
        if context_id in self._busy_contexts:
            raise ArtesiaError(409, f"context {context_id} is busy")

    def _count_matched_prefix(
        self,
        stored_messages: List[StoredMessage],
        prompt_token_counts: List[int],
    ) -> int:
        matched = 0
        for index, token_count in enumerate(prompt_token_counts):
            if index >= len(stored_messages):
                break
            if stored_messages[index].token_count != token_count:
                break
            matched += 1
        return matched

    def _ensure_prefix_available_on_gpu(
        self,
        *,
        matched_messages: List[StoredMessage],
    ) -> tuple[float, float]:
        required_bytes = sum(
            message.kv_bytes(self.config.kv_cache_bytes_per_token)
            for message in matched_messages
        )
        if required_bytes > self.config.gpu_capacity_bytes:
            raise ArtesiaError(507, "matched prefix cannot fit in GPU memory")

        pinned_ids = {message.message_id for message in matched_messages}
        fetch_time = 0.0
        offload_time = 0.0
        for message in matched_messages:
            if message.placement == "gpu":
                continue
            message_bytes = message.kv_bytes(self.config.kv_cache_bytes_per_token)
            offload_time += self._make_room_for_gpu_bytes(
                required_bytes=message_bytes,
                pinned_message_ids=pinned_ids,
            )
            message.placement = "gpu"
            self._cpu_bytes_used -= message_bytes
            self._gpu_bytes_used += message_bytes
            fetch_time += self._bytes_to_seconds(
                message_bytes,
                self.config.cpu_gpu_bytes_per_second,
            )
        return fetch_time, offload_time

    def _append_target_messages(
        self,
        *,
        context: ContextState,
        target_messages: List[Dict[str, Any]],
    ) -> tuple[float, float]:
        transfer_time = 0.0
        offload_time = 0.0
        for message_spec in target_messages:
            token_count = int(message_spec["token_count"])
            message_bytes = token_count * self.config.kv_cache_bytes_per_token
            if message_bytes > self.config.gpu_capacity_bytes:
                raise ArtesiaError(507, "a single message cannot fit in GPU memory")

            offload_time += self._make_room_for_gpu_bytes(
                required_bytes=message_bytes,
                pinned_message_ids=set(),
            )
            stored_message = StoredMessage(
                message_id=self._next_message_id(),
                role=str(message_spec["role"]),
                token_count=token_count,
                placement="gpu",
                state="durable",
            )
            context.messages.append(stored_message)
            self._gpu_bytes_used += message_bytes
            transfer_time += self._bytes_to_seconds(
                message_bytes,
                self.config.gpu_sglang_bytes_per_second,
            )
        return transfer_time, offload_time

    def _make_room_for_gpu_bytes(
        self,
        *,
        required_bytes: int,
        pinned_message_ids: set[str],
    ) -> float:
        if required_bytes > self.config.gpu_capacity_bytes:
            raise ArtesiaError(507, "required data cannot fit in GPU memory")
        messages_to_offload = self._plan_offload_messages(
            required_bytes=required_bytes,
            pinned_message_ids=pinned_message_ids,
        )
        offload_time = 0.0
        for message in messages_to_offload:
            offload_time += self._offload_message_to_cpu(message)
        return offload_time

    def _plan_offload_messages(
        self,
        *,
        required_bytes: int,
        pinned_message_ids: set[str],
    ) -> List[StoredMessage]:
        overflow_bytes = self._gpu_bytes_used + required_bytes - self.config.gpu_capacity_bytes
        if overflow_bytes <= 0:
            return []

        planned_messages: List[StoredMessage] = []
        freed_bytes = 0
        for context in self._ordered_offload_contexts(pinned_message_ids):
            for message in self._iter_tail_gpu_messages(context, pinned_message_ids):
                planned_messages.append(message)
                freed_bytes += message.kv_bytes(self.config.kv_cache_bytes_per_token)
                if freed_bytes >= overflow_bytes:
                    return planned_messages
        raise ArtesiaError(507, "unable to free enough GPU memory")

    def _ordered_offload_contexts(self, pinned_message_ids: set[str]) -> List[ContextState]:
        candidate_contexts = [
            context
            for context in self._iter_all_contexts()
            if self._pick_tail_gpu_message(context, pinned_message_ids) is not None
        ]
        if not self.config.enable_artesia:
            if self.config.eviction_policy == "lru":
                return sorted(
                    candidate_contexts,
                    key=lambda context: (context.state_changed_at_ns, context.context_id),
                )
            else:
                return sorted(
                    candidate_contexts,
                    key=lambda context: (-context.state_changed_at_ns, context.context_id),
                )
            
        suspend_contexts = [context for context in candidate_contexts if context.state == "suspend"]
        durable_contexts = [context for context in candidate_contexts if context.state == "durable"]

        ordered_contexts: List[ContextState] = []
        for candidates in (suspend_contexts, durable_contexts):
            ordered_contexts.extend(self._sort_offload_contexts(candidates))
        return ordered_contexts

    def _sort_offload_contexts(self, contexts: List[ContextState]) -> List[ContextState]:
        if self.config.eviction_policy == "lru":
            return sorted(
                contexts,
                key=lambda context: (context.state_changed_at_ns, context.context_id),
            )
        return sorted(
            contexts,
            key=lambda context: (-context.state_changed_at_ns, context.context_id),
        )

    def _offload_message_to_cpu(self, message: StoredMessage) -> float:
        message_bytes = message.kv_bytes(self.config.kv_cache_bytes_per_token)
        message.placement = "cpu"
        self._gpu_bytes_used -= message_bytes
        self._cpu_bytes_used += message_bytes
        return self._bytes_to_seconds(
            message_bytes,
            self.config.cpu_gpu_bytes_per_second,
        )

    def _release_message_storage(self, message: StoredMessage) -> None:
        message_bytes = message.kv_bytes(self.config.kv_cache_bytes_per_token)
        if message.placement == "gpu":
            self._gpu_bytes_used -= message_bytes
        else:
            self._cpu_bytes_used -= message_bytes

    def _set_context_state(
        self,
        context: ContextState,
        state: MessageState,
    ) -> None:
        context.state = state
        self._touch_context_timestamp(context)
        for message in context.messages:
            message.state = state

    def _touch_context_timestamp(self, context: ContextState) -> None:
        context.state_changed_at_ns = self._now_ns()

    def _pick_tail_gpu_message(
        self,
        context: ContextState,
        pinned_message_ids: set[str],
    ) -> Optional[StoredMessage]:
        for message in self._iter_tail_gpu_messages(context, pinned_message_ids):
            return message
        return None

    def _iter_tail_gpu_messages(
        self,
        context: ContextState,
        pinned_message_ids: set[str],
    ) -> Iterable[StoredMessage]:
        for message in reversed(context.messages):
            if message.placement != "gpu":
                continue
            if message.message_id in pinned_message_ids:
                continue
            yield message

    def _iter_all_contexts(self) -> Iterable[ContextState]:
        yield from self.contexts.values()
        yield from self._archived_contexts.values()

    def _archive_context_suffix(
        self,
        *,
        context: ContextState,
        start_index: int,
        operation: str,
    ) -> Optional[ContextState]:
        start_index = max(0, min(start_index, len(context.messages)))
        suffix_messages = context.messages[start_index:]
        if not suffix_messages:
            return None

        context.messages = context.messages[:start_index]
        archive_id = self._next_archive_context_id(context.context_id, operation)
        archived_context = ContextState(
            context_id=archive_id,
            messages=suffix_messages,
            state=context.state,
            state_changed_at_ns=context.state_changed_at_ns,
        )
        self._archived_contexts[archive_id] = archived_context
        return archived_context

    def _sample_token_ids(self, tokenizer: TokenizerProtocol, count: int) -> List[int]:
        if count <= 0:
            return []

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(tokenizer.get_vocab())
        forbidden = set(getattr(tokenizer, "all_special_ids", []) or [])

        sampled: List[int] = []
        attempts = 0
        max_attempts = max(count * 10, 10)
        while len(sampled) < count and attempts < max_attempts:
            candidate = self._random.randrange(0, vocab_size)
            attempts += 1
            if candidate in forbidden:
                continue
            sampled.append(candidate)

        while len(sampled) < count:
            sampled.append(self._random.randrange(0, vocab_size))
        return sampled

    def _tokens_to_seconds(self, token_count: int, throughput_tps: float) -> float:
        if token_count <= 0:
            return 0.0
        if throughput_tps <= 0:
            raise ArtesiaError(500, "throughput must be > 0")
        return token_count / throughput_tps

    def _bytes_to_seconds(self, byte_count: int, bandwidth_bytes_per_second: float) -> float:
        if byte_count <= 0:
            return 0.0
        if bandwidth_bytes_per_second <= 0:
            raise ArtesiaError(500, "bandwidth must be > 0")
        return byte_count / bandwidth_bytes_per_second

    def _now_ns(self) -> int:
        return time.monotonic_ns()

    def _next_message_id(self) -> str:
        self._message_seq += 1
        return f"msg-{self._message_seq}"

    def _next_archive_context_id(self, source_context_id: str, operation: str) -> str:
        self._archive_seq += 1
        return f"archive:{source_context_id}:{operation}:{self._archive_seq}"
