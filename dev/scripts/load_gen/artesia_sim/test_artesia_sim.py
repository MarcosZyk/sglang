from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import create_app
from models import ChatCompletionRequest
from runtime import ArtesiaError, ArtesiaSimulator, SimulatorConfig


class FakeTokenizer:
    all_special_ids = [0]
    vocab_size = 10_000

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        if not text:
            return []
        return [abs(hash(token)) % 9999 + 1 for token in text.split()]

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def get_vocab(self):
        return {f"tok{i}": i for i in range(self.vocab_size)}


async def noop_sleep(seconds: float) -> None:
    del seconds
    await asyncio.sleep(0)


def build_runtime(
    *,
    eviction_policy: str = "lru",
    decode_throughput_tps: float = 10.0,
    prefill_throughput_tps: float = 1000.0,
    decode_max_concurrency: int = 1,
    gpu_capacity_gb: float = 0.000003,
    enable_artesia: bool = False,
    sleep_func=noop_sleep,
) -> ArtesiaSimulator:
    return ArtesiaSimulator(
        SimulatorConfig(
            cpu_gpu_bandwidth_gbps=0.001,
            gpu_sglang_bandwidth_gbps=0.001,
            prefill_throughput_tps=prefill_throughput_tps,
            decode_throughput_tps=decode_throughput_tps,
            decode_max_concurrency=decode_max_concurrency,
            kv_cache_kb_per_token=1.0,
            gpu_capacity_gb=gpu_capacity_gb,
            eviction_policy=eviction_policy,
            enable_artesia=enable_artesia,
            seed=123,
        ),
        tokenizer_loader=lambda _: FakeTokenizer(),
        sleep_func=sleep_func,
    )


def request_for(context_id: str, messages, max_tokens: int = 1) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="fake-model",
        context_id=context_id,
        messages=[{"role": role, "content": content} for role, content in messages],
        max_tokens=max_tokens,
        temperature=0.0,
    )


def attach_monotonic_counter(runtime: ArtesiaSimulator, start: int = 1) -> None:
    current = start - 1

    def next_ns() -> int:
        nonlocal current
        current += 1
        return current

    runtime._now_ns = next_ns  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_context_lifecycle_and_one_off() -> None:
    runtime = build_runtime(enable_artesia=True)
    await runtime.create_context("ctx", "durable")

    completion = await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d e")],
            max_tokens=2,
        )
    )
    assert completion.usage.prompt_tokens == 5
    assert len(runtime.contexts["ctx"].messages) == 3

    await runtime.suspend("ctx")
    assert runtime.contexts["ctx"].state == "suspend"
    assert all(message.state == "suspend" for message in runtime.contexts["ctx"].messages)

    await runtime.one_off("ctx", 1)
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d e"), ("assistant", "f g")],
            max_tokens=2,
        )
    )
    context = runtime.contexts["ctx"]
    assert len(context.messages) == 1
    assert context.pending_one_off_index is None

    await runtime.truncate("ctx", 1)
    assert len(runtime.contexts["ctx"].messages) == 1

    await runtime.delete_context("ctx")
    assert "ctx" not in runtime.contexts
    assert runtime.gpu_bytes_used == 0
    assert runtime.cpu_bytes_used == 0


@pytest.mark.asyncio
async def test_prefix_fetch_prefill_and_write_timing() -> None:
    runtime = build_runtime(
        enable_artesia=True,
        gpu_capacity_gb=0.00002,
        decode_throughput_tps=10.0,
        prefill_throughput_tps=10.0,
    )
    await runtime.create_context("ctx", "durable")

    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d e")],
            max_tokens=2,
        )
    )

    context = runtime.contexts["ctx"]
    first_message = context.messages[0]
    first_bytes = first_message.kv_bytes(runtime.config.kv_cache_bytes_per_token)
    first_message.placement = "cpu"
    runtime._gpu_bytes_used -= first_bytes
    runtime._cpu_bytes_used += first_bytes

    completion = await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d e")],
            max_tokens=2,
        )
    )

    assert completion.num_local_cache == 3
    assert completion.num_global_cache == 5
    assert completion.usage.prompt_tokens == 0
    assert completion.usage.prompt_tokens_details.cached_tokens == 5
    assert completion.decode_time == pytest.approx([0.2], rel=1e-6)
    assert completion.prefill_time == pytest.approx(0.009, rel=1e-6)

    partial_completion = await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d e f g")],
            max_tokens=2,
        )
    )
    assert partial_completion.num_local_cache == 2
    assert partial_completion.num_global_cache == 2
    assert partial_completion.usage.prompt_tokens == 4
    assert partial_completion.prefill_time == pytest.approx(0.012, rel=1e-6)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("eviction_policy", "expected_offloaded_context"),
    [
        ("lru", "ctx-a"),
        ("mru", "ctx-b"),
    ],
)
async def test_eviction_policy_uses_context_state_time_and_tail_order(
    eviction_policy: str,
    expected_offloaded_context: str,
) -> None:
    runtime = build_runtime(
        eviction_policy=eviction_policy,
        gpu_capacity_gb=0.000006,
        enable_artesia=True,
    )
    attach_monotonic_counter(runtime)

    await runtime.create_context("ctx-a", "durable")
    await runtime.generate(
        request_for(
            "ctx-a",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )
    await runtime.suspend("ctx-a")

    await runtime.create_context("ctx-b", "durable")
    await runtime.generate(
        request_for(
            "ctx-b",
            [("system", "c"), ("user", "d")],
            max_tokens=1,
        )
    )
    await runtime.suspend("ctx-b")

    await runtime.create_context("ctx-c", "durable")
    await runtime.generate(
        request_for(
            "ctx-c",
            [("system", "e")],
            max_tokens=1,
        )
    )

    offloaded_context = runtime.contexts[expected_offloaded_context]
    untouched_context_id = "ctx-b" if expected_offloaded_context == "ctx-a" else "ctx-a"
    untouched_context = runtime.contexts[untouched_context_id]

    assert offloaded_context.state == "suspend"
    assert [message.role for message in offloaded_context.messages if message.placement == "gpu"] == [
        "system"
    ]
    assert [message.role for message in untouched_context.messages if message.placement == "gpu"] == [
        "system",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_suspend_contexts_are_evicted_before_older_durable_contexts() -> None:
    runtime = build_runtime(eviction_policy="lru", gpu_capacity_gb=0.000006, enable_artesia=True)
    attach_monotonic_counter(runtime)

    await runtime.create_context("ctx-durable", "durable")
    await runtime.generate(
        request_for(
            "ctx-durable",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )

    await runtime.create_context("ctx-suspend", "durable")
    await runtime.generate(
        request_for(
            "ctx-suspend",
            [("system", "c"), ("user", "d")],
            max_tokens=1,
        )
    )
    await runtime.suspend("ctx-suspend")

    await runtime.create_context("ctx-new", "durable")
    await runtime.generate(
        request_for(
            "ctx-new",
            [("system", "e")],
            max_tokens=1,
        )
    )

    assert [message.role for message in runtime.contexts["ctx-suspend"].messages if message.placement == "gpu"] == [
        "system"
    ]
    assert [message.role for message in runtime.contexts["ctx-durable"].messages if message.placement == "gpu"] == [
        "system",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_pinned_prefix_keeps_head_and_evicts_tail_first() -> None:
    runtime = build_runtime(eviction_policy="lru", enable_artesia=True)
    attach_monotonic_counter(runtime)

    await runtime.create_context("ctx", "durable")
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )

    pinned_message_id = runtime.contexts["ctx"].messages[0].message_id
    offload_time = runtime._make_room_for_gpu_bytes(
        required_bytes=2 * runtime.config.kv_cache_bytes_per_token,
        pinned_message_ids={pinned_message_id},
    )

    assert offload_time == pytest.approx(0.002, rel=1e-6)
    assert [message.role for message in runtime.contexts["ctx"].messages if message.placement == "gpu"] == [
        "system"
    ]


@pytest.mark.asyncio
async def test_context_state_timestamp_refreshes_on_suspend_and_generate() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002, enable_artesia=True)
    attach_monotonic_counter(runtime)

    await runtime.create_context("ctx", "durable")
    created_at = runtime.contexts["ctx"].state_changed_at_ns

    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d")],
            max_tokens=1,
        )
    )
    after_generate = runtime.contexts["ctx"].state_changed_at_ns
    assert runtime.contexts["ctx"].state == "durable"
    assert after_generate > created_at

    await runtime.suspend("ctx")
    after_suspend = runtime.contexts["ctx"].state_changed_at_ns
    assert runtime.contexts["ctx"].state == "suspend"
    assert all(message.state == "suspend" for message in runtime.contexts["ctx"].messages)
    assert after_suspend > after_generate

    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d"), ("assistant", "e")],
            max_tokens=1,
        )
    )
    final_context = runtime.contexts["ctx"]
    assert final_context.state == "durable"
    assert final_context.state_changed_at_ns == after_suspend + 1
    assert all(message.state == "durable" for message in final_context.messages)


@pytest.mark.asyncio
async def test_non_artesia_delete_context_is_noop() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002)
    await runtime.create_context("ctx", "durable")
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a b"), ("user", "c d")],
            max_tokens=1,
        )
    )

    gpu_before = runtime.gpu_bytes_used
    cpu_before = runtime.cpu_bytes_used
    response = await runtime.delete_context("ctx")

    assert response == {"status": "ok", "context_id": "ctx"}
    assert "ctx" in runtime.contexts
    assert runtime.gpu_bytes_used == gpu_before
    assert runtime.cpu_bytes_used == cpu_before

    with pytest.raises(ArtesiaError) as exc_info:
        await runtime.create_context("ctx", "durable")
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_non_artesia_suspend_is_noop() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002)
    attach_monotonic_counter(runtime)
    await runtime.create_context("ctx", "durable")
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )

    context = runtime.contexts["ctx"]
    timestamp_before = context.state_changed_at_ns
    message_states_before = [message.state for message in context.messages]
    response = await runtime.suspend("ctx")

    assert response == {"status": "ok", "context_id": "ctx", "num_messages": 3}
    assert context.state == "durable"
    assert context.state_changed_at_ns == timestamp_before
    assert [message.state for message in context.messages] == message_states_before


@pytest.mark.asyncio
async def test_non_artesia_truncate_archives_suffix_without_freeing_storage() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002)
    attach_monotonic_counter(runtime)
    await runtime.create_context("ctx", "durable")
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )

    context = runtime.contexts["ctx"]
    timestamp_before = context.state_changed_at_ns
    gpu_before = runtime.gpu_bytes_used
    cpu_before = runtime.cpu_bytes_used

    response = await runtime.truncate("ctx", 1)

    assert response == {"status": "ok", "context_id": "ctx", "kept_messages": 1}
    assert [message.role for message in context.messages] == ["system"]
    assert context.state_changed_at_ns == timestamp_before
    assert runtime.gpu_bytes_used == gpu_before
    assert runtime.cpu_bytes_used == cpu_before
    assert len(runtime._archived_contexts) == 1
    archived_context = next(iter(runtime._archived_contexts.values()))
    assert archived_context.context_id.startswith("archive:ctx:truncate:")
    assert [message.role for message in archived_context.messages] == ["user", "assistant"]
    assert archived_context.state == context.state
    assert archived_context.state_changed_at_ns == timestamp_before


@pytest.mark.asyncio
async def test_non_artesia_one_off_archives_suffix_after_generate() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002)
    attach_monotonic_counter(runtime)
    await runtime.create_context("ctx", "durable")
    await runtime.one_off("ctx", 1)

    completion = await runtime.generate(
        request_for(
            "ctx",
            [("system", "a"), ("user", "b")],
            max_tokens=2,
        )
    )

    context = runtime.contexts["ctx"]
    assert completion.usage.prompt_tokens == 2
    assert [message.role for message in context.messages] == ["system"]
    assert context.pending_one_off_index is None
    assert len(runtime._archived_contexts) == 1
    archived_context = next(iter(runtime._archived_contexts.values()))
    assert archived_context.context_id.startswith("archive:ctx:one_off:")
    assert [message.role for message in archived_context.messages] == ["user", "assistant"]
    assert all(message.placement == "gpu" for message in archived_context.messages)
    assert archived_context.state_changed_at_ns == context.state_changed_at_ns


@pytest.mark.asyncio
async def test_non_artesia_prefill_touches_only_active_context_timestamp() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002)
    attach_monotonic_counter(runtime)
    await runtime.create_context("ctx", "durable")

    created_at = runtime.contexts["ctx"].state_changed_at_ns
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )
    after_first_generate = runtime.contexts["ctx"].state_changed_at_ns
    assert after_first_generate == created_at + 1

    await runtime.one_off("ctx", 1)
    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )
    active_context = runtime.contexts["ctx"]
    archived_context = next(iter(runtime._archived_contexts.values()))
    archived_timestamp = archived_context.state_changed_at_ns
    assert active_context.state_changed_at_ns == after_first_generate + 1

    await runtime.generate(
        request_for(
            "ctx",
            [("system", "a")],
            max_tokens=1,
        )
    )
    assert runtime.contexts["ctx"].state_changed_at_ns == active_context.state_changed_at_ns + 1
    assert archived_context.state_changed_at_ns == archived_timestamp


@pytest.mark.asyncio
async def test_non_artesia_offload_includes_archived_contexts_and_stops_when_enough() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.000005)
    attach_monotonic_counter(runtime)

    await runtime.create_context("ctx-old", "durable")
    await runtime.generate(
        request_for(
            "ctx-old",
            [("system", "a"), ("user", "b")],
            max_tokens=1,
        )
    )
    await runtime.truncate("ctx-old", 1)

    await runtime.create_context("ctx-new", "durable")
    await runtime.generate(
        request_for(
            "ctx-new",
            [("system", "c")],
            max_tokens=1,
        )
    )

    archived_context = next(iter(runtime._archived_contexts.values()))
    archived_gpu_roles_before = [
        message.role for message in archived_context.messages if message.placement == "gpu"
    ]
    assert archived_gpu_roles_before == ["user", "assistant"]

    offload_time = runtime._make_room_for_gpu_bytes(
        required_bytes=2 * runtime.config.kv_cache_bytes_per_token,
        pinned_message_ids=set(),
    )

    assert offload_time == pytest.approx(0.002, rel=1e-6)
    assert [message.role for message in archived_context.messages if message.placement == "gpu"] == []
    assert [message.role for message in runtime.contexts["ctx-new"].messages if message.placement == "gpu"] == [
        "system",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_busy_context_returns_409() -> None:
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def blocking_sleep(seconds: float) -> None:
        del seconds
        sleep_started.set()
        await release_sleep.wait()

    runtime = build_runtime(
        gpu_capacity_gb=0.00002,
        decode_throughput_tps=1.0,
        prefill_throughput_tps=100.0,
        sleep_func=blocking_sleep,
    )
    await runtime.create_context("ctx", "durable")

    task = asyncio.create_task(
        runtime.generate(
            request_for(
                "ctx",
                [("system", "a")],
                max_tokens=1,
            )
        )
    )
    await sleep_started.wait()

    with pytest.raises(ArtesiaError) as exc_info:
        await runtime.truncate("ctx", 0)
    assert exc_info.value.status_code == 409

    release_sleep.set()
    await task


@pytest.mark.asyncio
async def test_prefill_is_serial_and_second_prefill_can_overlap_first_decode() -> None:
    events = []

    async def tracked_sleep(seconds: float) -> None:
        rounded = round(seconds, 3)
        if rounded == 0.01:
            events.append(("prefill_start", time.perf_counter()))
            await asyncio.sleep(0.05)
            events.append(("prefill_end", time.perf_counter()))
            return
        if rounded == 0.2:
            events.append(("decode_start", time.perf_counter()))
            await asyncio.sleep(0.05)
            events.append(("decode_end", time.perf_counter()))
            return
        await asyncio.sleep(0)

    runtime = build_runtime(
        gpu_capacity_gb=0.00002,
        decode_throughput_tps=10.0,
        prefill_throughput_tps=100.0,
        decode_max_concurrency=2,
        sleep_func=tracked_sleep,
    )
    await runtime.create_context("ctx-a", "durable")
    await runtime.create_context("ctx-b", "durable")

    async def run_request(context_id: str) -> None:
        await runtime.generate(request_for(context_id, [("system", "a")], max_tokens=2))

    task_a = asyncio.create_task(run_request("ctx-a"))
    await asyncio.sleep(0.001)
    task_b = asyncio.create_task(run_request("ctx-b"))
    await asyncio.gather(task_a, task_b)

    prefill_starts = [ts for event, ts in events if event == "prefill_start"]
    prefill_ends = [ts for event, ts in events if event == "prefill_end"]
    decode_starts = [ts for event, ts in events if event == "decode_start"]
    decode_ends = [ts for event, ts in events if event == "decode_end"]

    assert len(prefill_starts) == 2
    assert len(prefill_ends) == 2
    assert len(decode_starts) == 2
    assert len(decode_ends) == 2
    assert prefill_starts[1] >= prefill_ends[0]
    assert prefill_starts[1] >= prefill_ends[0]
    assert prefill_starts[1] >= decode_starts[0]
    assert prefill_starts[1] < decode_ends[0]


@pytest.mark.asyncio
async def test_decode_queue_waits_without_blocking_prefill() -> None:
    stage_events = []

    async def tracked_sleep(seconds: float) -> None:
        rounded = round(seconds, 3)
        if rounded == 0.01:
            stage_events.append(("prefill", time.perf_counter()))
            await asyncio.sleep(0.01)
            return
        if rounded == 0.2:
            stage_events.append(("decode", time.perf_counter()))
            await asyncio.sleep(0.05)
            return
        await asyncio.sleep(0)

    runtime = build_runtime(
        gpu_capacity_gb=0.00002,
        decode_throughput_tps=10.0,
        prefill_throughput_tps=100.0,
        decode_max_concurrency=1,
        sleep_func=tracked_sleep,
    )
    await runtime.create_context("ctx-a", "durable")
    await runtime.create_context("ctx-b", "durable")

    async def run_request(context_id: str) -> None:
        await runtime.generate(request_for(context_id, [("system", "a")], max_tokens=2))

    task_a = asyncio.create_task(run_request("ctx-a"))
    await asyncio.sleep(0.001)
    task_b = asyncio.create_task(run_request("ctx-b"))
    await asyncio.gather(task_a, task_b)

    prefill_times = [ts for event, ts in stage_events if event == "prefill"]
    decode_times = [ts for event, ts in stage_events if event == "decode"]
    assert len(prefill_times) == 2
    assert len(decode_times) == 2
    assert prefill_times[1] < decode_times[1]


@pytest.mark.asyncio
async def test_decode_time_excludes_decode_queue_wait() -> None:
    runtime = build_runtime(
        gpu_capacity_gb=0.00002,
        decode_throughput_tps=10.0,
        prefill_throughput_tps=100.0,
        decode_max_concurrency=1,
        sleep_func=asyncio.sleep,
    )
    await runtime.create_context("ctx-a", "durable")
    await runtime.create_context("ctx-b", "durable")

    async def run_request(context_id: str):
        start = time.perf_counter()
        completion = await runtime.generate(request_for(context_id, [("system", "a")], max_tokens=2))
        elapsed = time.perf_counter() - start
        return completion, elapsed

    task_a = asyncio.create_task(run_request("ctx-a"))
    await asyncio.sleep(0.001)
    task_b = asyncio.create_task(run_request("ctx-b"))
    (_, elapsed_a), (completion_b, elapsed_b) = await asyncio.gather(task_a, task_b)

    assert completion_b.decode_time == pytest.approx([0.2], rel=1e-6)
    assert elapsed_b > elapsed_a
    assert elapsed_b > completion_b.prefill_time + completion_b.decode_time[0]


@pytest.mark.asyncio
async def test_configured_tokenizer_overrides_request_model() -> None:
    requested_tokenizers = []

    def tokenizer_loader(name: str):
        requested_tokenizers.append(name)
        return FakeTokenizer()

    runtime = ArtesiaSimulator(
        SimulatorConfig(
            cpu_gpu_bandwidth_gbps=0.001,
            gpu_sglang_bandwidth_gbps=0.001,
            prefill_throughput_tps=1000.0,
            decode_throughput_tps=10.0,
            decode_max_concurrency=1,
            kv_cache_kb_per_token=1.0,
            gpu_capacity_gb=0.00002,
            eviction_policy="lru",
            enable_artesia=False,
            tokenizer_name="configured-tokenizer",
            seed=123,
        ),
        tokenizer_loader=tokenizer_loader,
        sleep_func=noop_sleep,
    )
    await runtime.create_context("ctx", "durable")
    await runtime.generate(
        ChatCompletionRequest(
            model="request-model",
            context_id="ctx",
            messages=[{"role": "user", "content": "a b"}],
            max_tokens=1,
        )
    )
    assert requested_tokenizers == ["configured-tokenizer"]


@pytest.mark.asyncio
async def test_http_routes_smoke() -> None:
    runtime = build_runtime(gpu_capacity_gb=0.00002)
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.put(
            "/context/test-ctx",
            json={"context_type": "durable"},
        )
        assert create_response.status_code == 200

        health_response = await client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["decode_max_concurrency"] == runtime.config.decode_max_concurrency
        assert health_response.json()["enable_artesia"] is False

        missing_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "context_id": "missing",
                "messages": [{"role": "user", "content": "a"}],
                "max_tokens": 1,
            },
        )
        assert missing_response.status_code == 404

        chat_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "context_id": "test-ctx",
                "messages": [{"role": "user", "content": "a b"}],
                "max_tokens": 2,
            },
        )
        assert chat_response.status_code == 200
        body = chat_response.json()
        assert body["choices"][0]["message"]["content"]
        assert body["prefill_time"] >= 0
        assert body["decode_time"][0] >= 0
        assert "cached_tokens" in body["usage"]["prompt_tokens_details"]
        assert body["decode_time"] == [2 / runtime.config.decode_throughput_tps]
