# server.py - 优化版（保持轮次顺序，请求间并发）
import random
import asyncio
import time
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from openai import OpenAI  # 保持同步客户端
from transformers import AutoTokenizer
import uvicorn
import argparse
import uuid
import csv
import numpy as np
import json

try:
    from server.contextcake_client import (
        ContextCakeHttpClient,
        build_contextcake_openai_base_url,
        get_contextcake_base_url,
        set_contextcake_base_url,
    )
except ImportError:
    from contextcake_client import (
        ContextCakeHttpClient,
        build_contextcake_openai_base_url,
        get_contextcake_base_url,
        set_contextcake_base_url,
    )

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Agent Simulator")

RESULT_DIR = Path(__file__).resolve().parent.parent / "result"

SPECIAL_TASK_KEYWORDS = ["iFlow CLI"]

# ========== 关键优化：线程池 ==========
# 用于并发执行多个请求（每个请求内部保持顺序）
request_executor = ThreadPoolExecutor(
    max_workers=32,  # 根据 GPU 能力调整，支持 32 个并发请求
    thread_name_prefix="request_worker"
)

# 用于 Tokenizer 的 CPU 密集型操作
tokenizer_executor = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="tokenizer_worker"
)

# 全局 tokenizer 缓存（避免重复加载）
_tokenizer_cache: Dict[str, AutoTokenizer] = {}
_tokenizer_lock = asyncio.Lock()


# ========== Pydantic 请求模型 ==========
class SimRequest(BaseModel):
    n: int
    n_task: int
    s_list: Optional[List[int]] = None
    m_list: Optional[List[int]] = None
    a_list: Optional[List[List[int]]] = None
    b_list: Optional[List[List[int]]] = None
    roles_list: Optional[List[List[str]]] = None
    wait_time: Optional[List[float]] = None
    task_list: Optional[List[int]] = None
    tokenizer_name: Optional[str] = None
    openai_model: Optional[str] = None
    temperature: Optional[float] = 0.0
    task_type_list: Optional[List[str]] = None
    semantic_type_list: Optional[List[Optional[List[str]]]] = None


# ========== Helper Functions ==========

def get_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """获取或加载 tokenizer（带缓存，线程安全）"""
    global _tokenizer_cache
    if tokenizer_name not in _tokenizer_cache:
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        _tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )
    return _tokenizer_cache[tokenizer_name]


def random_id_sampler(tokenizer, k: int, forbidden: set):
    """从 tokenizer vocab 范围内随机生成 k 个 token ids"""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer.get_vocab())
        except Exception:
            raise RuntimeError("无法获取 tokenizer 的 vocab 大小")

    out = []
    attempts = 0
    max_attempts = k * 10

    while len(out) < k and attempts < max_attempts:
        cand = random.randrange(0, vocab_size)
        attempts += 1
        if cand not in forbidden:
            out.append(cand)

    # 如果 forbidden 太多，允许重复
    while len(out) < k:
        cand = random.randrange(0, vocab_size)
        out.append(cand)

    return out


def _sync_decode_token_ids(args):
    """在线程池中执行 tokenizer decode"""
    tokenizer, token_ids = args
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def _sync_encode_text(args):
    """在线程池中执行 tokenizer encode"""
    tokenizer, text = args
    return tokenizer.encode(text, add_special_tokens=False)


def normalize_semantic_type(value: Optional[List[str]]) -> Optional[str]:
    """返回首个非空 semantic type。"""
    if not value:
        return None
    for item in value:
        if item is None:
            continue
        normalized = item.strip()
        if normalized:
            return normalized
    return None


def slugify_identifier(value: str) -> str:
    """将文本规范化为稳定短 slug。"""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "main"


def build_task_hash(task_type: str) -> str:
    """基于 task_type 文本生成稳定短哈希。"""
    return hashlib.sha1(task_type.encode("utf-8")).hexdigest()[:12]


def build_context_id(agent_id: str, task_type: str, semantic_type: Optional[str]) -> str:
    semantic_slug = slugify_identifier(semantic_type) if semantic_type is not None else "main"
    task_hash = build_task_hash(task_type)
    return f"ctx_{agent_id}_{task_hash}_{semantic_slug}"


def build_fallback_context_id(task_type: int) -> str:
    """为无 Artesia 元数据的请求构造稳定的内部上下文 key。"""
    return f"task-{task_type}"


def build_call_id(agent_id: str, round_idx: int) -> str:
    return f"call_{agent_id}_{round_idx}"


def is_loop_task_type(task_type: str) -> bool:
    return any(keyword in task_type for keyword in SPECIAL_TASK_KEYWORDS)


def classify_round(task_type: str, semantic_type: Optional[str]) -> str:
    if not is_loop_task_type(task_type):
        return "non_loop"
    if semantic_type is None:
        return "loop_main"
    return "loop_derived"


def build_artesia_context(req: SimRequest, agent_id: str) -> Optional[Dict[str, Any]]:
    """校验并预计算 artesia 所需的 round/context 级元数据。"""
    presence = [
        req.task_type_list is not None,
        req.semantic_type_list is not None,
    ]
    if not any(presence):
        return None
    if not all(presence):
        raise HTTPException(
            status_code=400,
            detail="task_type_list and semantic_type_list must both be provided together",
        )

    if len(req.task_type_list) != req.n:
        raise HTTPException(status_code=400, detail="task_type_list length must equal n")
    if len(req.semantic_type_list) != req.n:
        raise HTTPException(status_code=400, detail="semantic_type_list length must equal n")

    normalized_semantic_list: List[Optional[str]] = [
        normalize_semantic_type(value) for value in req.semantic_type_list
    ]
    classification_list: List[str] = []
    context_id_list: List[str] = []
    context_states: Dict[str, Dict[str, int]] = {}

    for round_idx, task_type_text in enumerate(req.task_type_list):
        normalized_semantic = normalized_semantic_list[round_idx]
        classification = classify_round(task_type_text, normalized_semantic)
        context_id = build_context_id(agent_id, task_type_text, normalized_semantic)
        classification_list.append(classification)
        context_id_list.append(context_id)

        if context_id not in context_states:
            context_states[context_id] = {
                "first_round": round_idx,
                "last_round": round_idx,
            }
        else:
            context_states[context_id]["last_round"] = round_idx

    return {
        "normalized_semantic_list": normalized_semantic_list,
        "classification_list": classification_list,
        "context_id_list": context_id_list,
        "context_states": context_states,
    }


def find_rebase_index(
    a_row: List[int],
    b_row: List[int],
    m: int,
) -> int:
    """找到首个缓存未命中的 message 索引。"""
    for idx in range(m):
        if b_row[idx] < a_row[idx]:
            return idx
    return m

def resolve_context_cache_id(
    artesia_context: Optional[Dict[str, Any]],
    round_idx: int,
    task_type: int,
) -> str:
    if artesia_context is not None:
        return artesia_context["context_id_list"][round_idx]
    return build_fallback_context_id(task_type)


def get_prompt_tokens_details(usage: Any) -> Any:
    if usage is None:
        return None
    return getattr(usage, "prompt_tokens_details", None)


def get_cached_tokens_from_completion(completion: Any) -> int:
    usage = getattr(completion, "usage", None)
    prompt_tokens_details = get_prompt_tokens_details(usage)
    if prompt_tokens_details is None:
        return 0
    if isinstance(prompt_tokens_details, dict):
        return int(prompt_tokens_details.get("cached_tokens", 0) or 0)
    return int(getattr(prompt_tokens_details, "cached_tokens", 0) or 0)


def get_decode_time_stats(completion: Any) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    decode_time = getattr(completion, "decode_time", None)
    if decode_time is None:
        return None, None, None, None

    decode_time_array = np.asarray(decode_time, dtype=float)
    if decode_time_array.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.sum(decode_time_array)),
        float(np.average(decode_time_array)),
        float(np.percentile(decode_time_array, 50)),
        float(np.percentile(decode_time_array, 95)),
    )


def get_context_runtime_state(
    context_runtime: Dict[str, Dict[str, bool]],
    context_id: str,
) -> Dict[str, bool]:
    return context_runtime.setdefault(
        context_id,
        {
            "created": False,
            "suspended": False,
        },
    )


def execute_common_pre_commands(
    contextcake_client: ContextCakeHttpClient,
    artesia_context: Optional[Dict[str, Any]],
    context_runtime: Dict[str, Dict[str, bool]],
    round_idx: int,
    current_context_id: Optional[str],
) -> Optional[str]:
    if artesia_context is None:
        return current_context_id

    context_id = artesia_context["context_id_list"][round_idx]

    if current_context_id is not None and current_context_id != context_id:
        current_context_state = get_context_runtime_state(context_runtime, current_context_id)
        if not current_context_state["suspended"]:
            contextcake_client.suspend(round_idx, "pre", current_context_id)
            current_context_state["suspended"] = True

    context_state = get_context_runtime_state(context_runtime, context_id)
    if not context_state["created"]:
        contextcake_client.create_context(round_idx, context_id)
        context_state["created"] = True
    context_state["suspended"] = False

    return context_id


def execute_common_post_commands(
    contextcake_client: ContextCakeHttpClient,
    artesia_context: Optional[Dict[str, Any]],
    context_runtime: Dict[str, Dict[str, bool]],
    round_idx: int,
) -> None:
    if artesia_context is None:
        return

    context_id = artesia_context["context_id_list"][round_idx]
    context_state = artesia_context["context_states"][context_id]
    if round_idx == context_state["last_round"]:
        contextcake_client.delete_context(round_idx, context_id)
        context_runtime.pop(context_id, None)


def execute_artesia_commands_for_round(
    contextcake_client: ContextCakeHttpClient,
    artesia_context: Optional[Dict[str, Any]],
    context_runtime: Dict[str, Dict[str, bool]],
    round_idx: int,
    current_context_id: Optional[str],
    m: int,
    a_row: List[int],
    effective_b_row: List[int],
) -> Optional[str]:
    if artesia_context is None:
        return current_context_id

    current_context_id = execute_common_pre_commands(
        contextcake_client=contextcake_client,
        artesia_context=artesia_context,
        context_runtime=context_runtime,
        round_idx=round_idx,
        current_context_id=current_context_id,
    )

    context_id = artesia_context["context_id_list"][round_idx]
    classification = artesia_context["classification_list"][round_idx]

    if classification == "non_loop":
        contextcake_client.one_off(round_idx, context_id, 0)
        return current_context_id

    if classification == "loop_main":
        rebase_index = find_rebase_index(a_row, effective_b_row, m)
        contextcake_client.truncate(round_idx, "pre", context_id, rebase_index)
        return current_context_id

    if classification == "loop_derived":
        contextcake_client.one_off(round_idx, context_id, 1)

    return current_context_id


def execute_artesia_post_commands_for_round(
    contextcake_client: ContextCakeHttpClient,
    artesia_context: Optional[Dict[str, Any]],
    context_runtime: Dict[str, Dict[str, bool]],
    round_idx: int,
) -> Optional[str]:
    if artesia_context is None:
        return None

    context_id = artesia_context["context_id_list"][round_idx]
    classification = artesia_context["classification_list"][round_idx]
    context_state = get_context_runtime_state(context_runtime, context_id)

    if classification == "loop_derived":
        contextcake_client.truncate(round_idx, "post", context_id, 1)
    elif classification == "loop_main" and not context_state["suspended"]:
        contextcake_client.suspend(round_idx, "post", context_id)
        context_state["suspended"] = True

    execute_common_post_commands(
        contextcake_client=contextcake_client,
        artesia_context=artesia_context,
        context_runtime=context_runtime,
        round_idx=round_idx,
    )
    round_context_state = artesia_context["context_states"][context_id]
    if round_idx == round_context_state["last_round"]:
        return None
    return context_id


def simulate_sync(req_dict: Dict) -> Dict:
    """
    同步模拟函数 - 保持原有逻辑不变

    关键：这个函数会在线程池中执行，所以不会阻塞事件循环
    不同请求可以并行执行，但每个请求内部保持顺序
    """
    rid = str(uuid.uuid4())
    start_time = time.perf_counter()
    contextcake_client: Optional[ContextCakeHttpClient] = None
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULT_DIR / f"{rid}.csv", "a", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([
            "context_id",
            "num_prefill_tokens",
            "num_decode_tokens",
            "theoretical_cached_tokens",
            "num_cached_tokens",
            "prefill_time",
            "sum_decode_time",
            "tpot",
            "p50_tpot",
            "p95_tpot",
            "num_local_cache_tokens",
            "num_global_cached_tokens",
        ])

        # 从字典重建 SimRequest 对象
        req = SimRequest(**req_dict)

        # ========== 原有验证逻辑（保持不变） ==========
        if req.n <= 0:
            raise HTTPException(status_code=400, detail="n must be > 0")
        if req.n_task <= 0:
            raise HTTPException(status_code=400, detail="n_task must be > 0")

        tokenizer_name = req.tokenizer_name
        openai_model = req.openai_model

        # load tokenizer
        tokenizer = get_tokenizer(tokenizer_name)
        forbidden_ids = set(getattr(tokenizer, "all_special_ids", []))

        n = req.n

        if req.m_list:
            if len(req.m_list) != n:
                raise HTTPException(status_code=400, detail="m_list length must equal n")
            m_list = req.m_list
        else:
            raise HTTPException(status_code=400, detail="m_list length must equal n")

        if req.s_list:
            if len(req.s_list) != n:
                raise HTTPException(status_code=400, detail="s_list length must equal n")
            s_list = req.s_list
        else:
            raise HTTPException(status_code=400, detail="s_list length must equal n")

        a_list = req.a_list
        b_list = req.b_list
        roles_list = req.roles_list

        if a_list and len(a_list) != n:
            raise HTTPException(status_code=400, detail="a_list length must equal n if provided")
        if b_list and len(b_list) != n:
            raise HTTPException(status_code=400, detail="b_list length must equal n if provided")

        artesia_context = build_artesia_context(req, rid)
        context_runtime: Dict[str, Dict[str, bool]] = {}
        current_context_id: Optional[str] = None
        contextcake_base_url = get_contextcake_base_url()

        contextcake_client = ContextCakeHttpClient(
            base_url=contextcake_base_url,
        )

        # ========== 创建 OpenAI 客户端 ==========
        client = OpenAI(
            api_key="EMPTY",
            base_url=build_contextcake_openai_base_url(contextcake_base_url),
            timeout=1800,
            max_retries=0,
        )

        # ========== 历史记录 ==========
        history_rounds_by_context: Dict[str, List[List[Dict[str, str]]]] = {}
        round_prompt_token_ids_by_round: List[Optional[List[List[int]]]] = [None for _ in range(n)]
        last_round_idx_by_context: Dict[str, int] = {}

        try:
            # ========== 主循环 - 保持顺序执行 ==========
            for i in range(n):
                m = m_list[i]
                task_type = req.task_list[i]
                context_cache_id = resolve_context_cache_id(
                    artesia_context=artesia_context,
                    round_idx=i,
                    task_type=task_type,
                )

                a_row = a_list[i]
                b_row = b_list[i]
                roles_row = roles_list[i]

                if len(a_row) != m:
                    raise HTTPException(status_code=400, detail=f"a_list[{i}] length must be {m}")
                if len(b_row) != m:
                    raise HTTPException(status_code=400, detail=f"b_list[{i}] length must be {m}")
                if len(roles_row) != m:
                    raise HTTPException(status_code=400, detail=f"roles_list[{i}] length must be {m}")

                round_messages = []
                this_round_prompt_token_ids: List[List[int]] = []
                effective_b_row: List[int] = []
                source_round_idx = last_round_idx_by_context.get(context_cache_id)
                source_round_ids = (
                    round_prompt_token_ids_by_round[source_round_idx]
                    if source_round_idx is not None
                    else None
                )
                classification = (
                    artesia_context["classification_list"][i]
                    if artesia_context is not None
                    else None
                )

                for j in range(m):
                    a_ij = int(a_row[j])
                    b_ij = int(b_row[j])
                    role_j = roles_row[j]

                    reused_ids = []
                    if classification == "loop_derived":
                        if source_round_ids is not None and j == 0 and len(source_round_ids) > 0:
                            prev_ids = source_round_ids[0]
                            b_ij = min(b_ij, len(prev_ids), a_ij)
                            reused_ids = prev_ids[:b_ij]
                        else:
                            b_ij = 0
                            reused_ids = []
                    else:
                        if source_round_ids is not None:
                            if j < len(source_round_ids):
                                prev_ids = source_round_ids[j]
                                b_ij = min(b_ij, len(prev_ids), a_ij)
                                reused_ids = prev_ids[:b_ij]
                            else:
                                b_ij = 0
                                reused_ids = []
                        else:
                            b_ij = 0
                            reused_ids = []

                    new_count = a_ij - b_ij
                    new_ids = (
                        random_id_sampler(tokenizer, new_count, forbidden_ids)
                        if new_count > 0
                        else []
                    )
                    token_ids = reused_ids + new_ids

                    prompt_text = tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                    )

                    effective_b_row.append(b_ij)
                    this_round_prompt_token_ids.append(token_ids)
                    round_messages.append({role_j: prompt_text})

                call_messages = []
                for item in round_messages:
                    for key in item:
                        role = key
                        content = item[key]
                        call_messages.append({"role": role, "content": content})

                current_context_id = execute_artesia_commands_for_round(
                    contextcake_client=contextcake_client,
                    artesia_context=artesia_context,
                    context_runtime=context_runtime,
                    round_idx=i,
                    current_context_id=current_context_id,
                    m=m,
                    a_row=a_row,
                    effective_b_row=effective_b_row,
                )

                logger.info(f'rid {rid}, round{i} finish pre-exe commands, ready to call openai client')

                extra_body: Dict[str, Any] = {"ignore_eos": True}
                context_id_for_chat = context_cache_id if artesia_context is not None else None

                try:
                    completion_kwargs: Dict[str, Any] = {
                        "model": openai_model,
                        "messages": call_messages,
                        "max_tokens": s_list[i],
                        "logprobs": True,
                        "temperature": req.temperature,
                        "top_logprobs": 1,
                        "extra_body": extra_body,
                    }
                    if context_id_for_chat is not None:
                        completion_kwargs["context_id"] = context_id_for_chat

                    completion = client.chat.completions.create(**completion_kwargs)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {e}") from e

                if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
                    assistant_text = ""
                    for token_info in completion.choices[0].logprobs.content:
                        assistant_text += token_info.token
                else:
                    assistant_text = completion.choices[0].message.content or ""

                assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)

                round_messages.append({"assistant": assistant_text})
                history_rounds_by_context.setdefault(context_cache_id, []).append(round_messages)

                this_round_prompt_token_ids.append(assistant_token_ids)
                round_prompt_token_ids_by_round[i] = this_round_prompt_token_ids
                last_round_idx_by_context[context_cache_id] = i

                post_context_id = execute_artesia_post_commands_for_round(
                    contextcake_client=contextcake_client,
                    artesia_context=artesia_context,
                    context_runtime=context_runtime,
                    round_idx=i,
                )
                if artesia_context is not None:
                    current_context_id = post_context_id

                usage = getattr(completion, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                cached_tokens = get_cached_tokens_from_completion(completion)
                sum_decode_time, avg_decode_time, p50_decode_time, p95_decode_time = (
                    get_decode_time_stats(completion)
                )
                prefill_time = getattr(completion, "prefill_time", None)
                num_local_cache = getattr(completion, "num_local_cache", None)
                num_global_cache = getattr(completion, "num_global_cache", None)
                calculate_cached_tokens = int(np.sum(effective_b_row))

                write_result = [
                    context_cache_id,
                    prompt_tokens,
                    completion_tokens,
                    calculate_cached_tokens,
                    cached_tokens,
                    prefill_time,
                    sum_decode_time,
                    avg_decode_time,
                    p50_decode_time,
                    p95_decode_time,
                    num_local_cache,
                    num_global_cache,
                ]
                csv_writer.writerow(write_result)

                if req.wait_time and req.wait_time[i] > 0:
                    time.sleep(req.wait_time[i])
        finally:
            if contextcake_client is not None:
                contextcake_client.close()

    end_time = time.perf_counter()
    processing_time = end_time - start_time

    logger.info(f"Request completed in {processing_time:.3f}s")

    return {
        "history_rounds_by_context": history_rounds_by_context,
        "processing_time_s": processing_time,
    }


# ========== API Endpoints ==========

@app.post("/api/process")
async def process_request(req: SimRequest):
    """
    主处理接口 - 将请求放到线程池中执行

    关键优化：
    1. 使用 run_in_executor 将同步函数放到线程池
    2. 事件循环不被阻塞，可以处理其他请求
    3. 不同请求并行执行，每个请求内部保持顺序
    """
    # 将请求放到线程池中执行
    loop = asyncio.get_event_loop()
    #print(f"Recv Req at: {time.time()}", flush=True)

    try:
        # 关键：使用线程池执行同步的 simulate_sync 函数
        result = await loop.run_in_executor(
            request_executor,  # 使用请求线程池
            simulate_sync,
            req.model_dump()  # 转为字典传递给同步函数
        )
        return ORJSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """获取服务器统计信息"""
    return ORJSONResponse(content={
        "request_executor_workers": request_executor._max_workers,
        "request_executor_queued": request_executor._work_queue.qsize() if hasattr(request_executor, '_work_queue') else 0,
        "tokenizer_executor_workers": tokenizer_executor._max_workers,
        "status": "healthy"
    })


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return ORJSONResponse(content={"status": "healthy"})


@app.get("/")
async def root():
    """根路径"""
    return ORJSONResponse(content={
        "message": "LLM Agent Simulator - Optimized",
        "docs": "/docs",
        "concurrent_requests": request_executor._max_workers
    })


def run_server(
    host: str = "0.0.0.0",
    port: int = 12306,
    workers: int = 1,
    max_concurrent: Optional[int] = None,
    work_delay: Optional[float] = None,
    contextcake_base_url: Optional[str] = None,
):
    """启动服务器"""
    del max_concurrent, work_delay
    if contextcake_base_url is not None:
        set_contextcake_base_url(contextcake_base_url)

    uvicorn.run(
        "main_art:app",
        host=host,
        port=port,
        workers=workers,
        loop="asyncio",
        http="httptools",
        log_level="info",
        access_log=False,
        limit_concurrency=500,
        backlog=2048,
        timeout_keep_alive=1800,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12306)
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    parser.add_argument(
        "--contextcake-base-url",
        default=get_contextcake_base_url(),
        help="Base URL of the ContextCake server",
    )
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        contextcake_base_url=args.contextcake_base_url,
    )
