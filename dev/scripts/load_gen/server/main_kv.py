# server.py - 优化版（保持轮次顺序，请求间并发）
import os
import random
import asyncio
import time
import logging
import hashlib
import re
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Agent Simulator")

SPECIAL_TASK_KEYWORDS = ["iFlow CLI"]
MAIN_BRANCH = "main"

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
    prefix_pos_list: Optional[List[int]] = None


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
    """仅保留 semantic type 的首个非空字符串。"""
    if not value:
        return None
    first_item = value[0]
    if first_item is None:
        return None
    normalized = first_item.strip()
    return normalized or None


def slugify_branch_name(value: str) -> str:
    """将 semantic type 规范化为适合命令参数的 slug。"""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "branch"


def build_task_id(task_type: str) -> str:
    """基于 task_type 文本生成稳定短哈希。"""
    digest = hashlib.sha1(task_type.encode("utf-8")).hexdigest()
    return f"task-{digest[:12]}"


def is_loop_task_type(task_type: str) -> bool:
    return any(keyword in task_type for keyword in SPECIAL_TASK_KEYWORDS)


def build_gitkv_context(req: SimRequest) -> Optional[Dict[str, Any]]:
    """校验并预计算 gitkv 所需的 task 级元数据。"""
    presence = [
        req.task_type_list is not None,
        req.semantic_type_list is not None,
        req.prefix_pos_list is not None,
    ]
    if not any(presence):
        return None
    if not all(presence):
        raise HTTPException(
            status_code=400,
            detail="task_type_list, semantic_type_list, prefix_pos_list must all be provided together",
        )

    if len(req.task_type_list) != req.n:
        raise HTTPException(status_code=400, detail="task_type_list length must equal n")
    if len(req.semantic_type_list) != req.n:
        raise HTTPException(status_code=400, detail="semantic_type_list length must equal n")
    if len(req.prefix_pos_list) != req.n:
        raise HTTPException(status_code=400, detail="prefix_pos_list length must equal n")

    normalized_semantic_list: List[Optional[str]] = [
        normalize_semantic_type(value) for value in req.semantic_type_list
    ]
    branch_name_list: List[Optional[str]] = [
        slugify_branch_name(value) if value is not None else None
        for value in normalized_semantic_list
    ]

    task_states: Dict[str, Dict[str, Any]] = {}
    for round_idx, task_type_text in enumerate(req.task_type_list):
        branch_name = branch_name_list[round_idx]
        if task_type_text not in task_states:
            task_states[task_type_text] = {
                "task_id": build_task_id(task_type_text),
                "first_round": round_idx,
                "last_round": round_idx,
                "is_loop_task": is_loop_task_type(task_type_text),
                "branches": [],
                "branch_set": set(),
                "current_branch": MAIN_BRANCH,
                "init_emitted": False,
            }
        task_state = task_states[task_type_text]
        task_state["last_round"] = round_idx
        if branch_name and branch_name not in task_state["branch_set"]:
            task_state["branch_set"].add(branch_name)
            task_state["branches"].append(branch_name)

    return {
        "task_states": task_states,
        "normalized_semantic_list": normalized_semantic_list,
        "branch_name_list": branch_name_list,
        "prefix_pos_list": req.prefix_pos_list,
    }


def find_rebase_index(a_row: List[int], b_row: List[int], m: int) -> int:
    """找到首个缓存未命中的 message 索引。"""
    for idx in range(m):
        if b_row[idx] < a_row[idx]:
            return idx
    return m


def append_gitkv_commands(
    commands: List[str],
    gitkv_context: Optional[Dict[str, Any]],
    req: SimRequest,
    round_idx: int,
    agent_id: str,
    m: int,
    a_row: List[int],
    b_row: List[int],
) -> None:
    if gitkv_context is None:
        return

    task_type_text = req.task_type_list[round_idx]
    task_state = gitkv_context["task_states"][task_type_text]
    gitkv_task_id = task_state["task_id"]
    branch_name = gitkv_context["branch_name_list"][round_idx]

    if round_idx == task_state["first_round"] and not task_state["init_emitted"]:
        commands.append(f"gitkv init {agent_id} {gitkv_task_id}")
        task_state["init_emitted"] = True
        for branch in task_state["branches"]:
            commands.append(f"gitkv branch {agent_id} {gitkv_task_id} {branch}")

    if not task_state["is_loop_task"]:
        stash_list = list(range(0, m + 1))
        commands.append(f"gitkv stash {agent_id} {gitkv_task_id} {stash_list}")
    elif branch_name is not None:
        if task_state["current_branch"] != branch_name:
            commands.append(f"gitkv checkout {agent_id} {gitkv_task_id} {branch_name}")
            task_state["current_branch"] = branch_name
        stash_list = list(range(1, m + 1))
        commands.append(f"gitkv stash {agent_id} {gitkv_task_id} {stash_list}")
    else:
        if task_state["current_branch"] != MAIN_BRANCH:
            commands.append(f"gitkv checkout {agent_id} {gitkv_task_id} {MAIN_BRANCH}")
            task_state["current_branch"] = MAIN_BRANCH

        rebase_index = find_rebase_index(a_row, b_row, m)
        prefix_round = gitkv_context["prefix_pos_list"][round_idx]
        skip_rebase = (
            prefix_round >= 0
            and prefix_round < len(req.a_list)
            and rebase_index == len(req.a_list[prefix_round])
        )

        if not skip_rebase:
            commands.append(f"gitkv rebase {agent_id} {gitkv_task_id} {rebase_index}")

        commit_list = list(range(rebase_index, m + 1))
        commands.append(f"gitkv commit {agent_id} {gitkv_task_id} {commit_list}")

    if round_idx == task_state["last_round"]:
        commands.append(f"gitkv delete {agent_id} {gitkv_task_id}")


def simulate_sync(req_dict: Dict) -> Dict:
    """
    同步模拟函数 - 保持原有逻辑不变

    关键：这个函数会在线程池中执行，所以不会阻塞事件循环
    不同请求可以并行执行，但每个请求内部保持顺序
    """
    rid = str(uuid.uuid4())

    file = open(f'../result/{rid}.csv', 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['task_type', 'num_prefill_tokens', 'num_decode_tokens', 'theoretical_cached_tokens', 'num_cached_tokens', 'prefill_time', 'sum_decode_time', 'tpot', 'p50_tpot', 'p95_tpot', 'num_local_cache_tokens', 'num_global_cached_tokens'])

    start_time = time.perf_counter()

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

    gitkv_context = build_gitkv_context(req)
    gitkv_commands: List[str] = []

    # ========== 创建 OpenAI 客户端 ==========
    client_list: List[OpenAI] = []
    for port in range(12347, 12347 + req.n_task):
        client_list.append(
            OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{12347}/v1",
                timeout=1800,  # 增加超时
                max_retries=0
            )
        )

    # ========== 历史记录 ==========
    history_rounds: List[List[List[Dict[str, str]]]] = [[] for _ in range(req.n_task)]
    prev_msg_token_ids_per_round: List[List[List[List[int]]]] = [[] for _ in range(req.n_task)]

    # ========== 主循环 - 保持顺序执行 ==========
    for i in range(n):
        m = m_list[i]
        task_type = req.task_list[i]

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
        client_type = 0 if task_type == 0 else 1
        # 构建 m 条 message（保持顺序）
        for j in range(m):
            a_ij = int(a_row[j])
            b_ij = int(b_row[j])
            role_j = roles_row[j]

            # 获取上一轮的 token ids（保持依赖关系）
            reused_ids = []
            if i > 0 and prev_msg_token_ids_per_round[task_type]:
                prev_round_ids = prev_msg_token_ids_per_round[task_type][-1]
                if j < len(prev_round_ids):
                    prev_ids = prev_round_ids[j]
                    b_ij = min(b_ij, len(prev_ids), a_ij)
                    reused_ids = prev_ids[:b_ij]
                else:
                    b_ij = 0
                    reused_ids = []
            else:
                b_ij = 0
                reused_ids = []

            # 生成新 token ids
            new_count = a_ij - b_ij
            new_ids = random_id_sampler(tokenizer, new_count, forbidden_ids) if new_count > 0 else []
            token_ids = reused_ids + new_ids

            # decode -> prompt text
            prompt_text = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                #clean_up_tokenization_spaces=True
            )

            this_round_prompt_token_ids.append(token_ids)
            round_messages.append({role_j: prompt_text})

        # flatten history_rounds -> messages
        call_messages = []
        for item in round_messages:
            for key in item:
                role = key
                content = item[key]
                call_messages.append({"role": role, "content": content})

        # ========== LLM 调用（同步，但在线程池中执行） ==========
        try:
            completion = client_list[client_type].chat.completions.create(
                model=openai_model,
                messages=call_messages,
                max_tokens=s_list[i],
                logprobs=True,
                temperature=req.temperature,
                top_logprobs=1,
                extra_body={
                    "ignore_eos": True  # 将参数放在这里
                },
                agent_id=rid,
                task_id=task_type,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {e}")

        # 解析 assistant 文本
        #assistant_text = ""
        #try:
        #    assistant_text = completion.choices[0].message.content
        #except Exception:
        #    print(completion)
        #    try:
        #        assistant_text = completion.choices[0].get("message", {}).get("content", "")
        #    except Exception:
        #        print(completion)
        #        assistant_text = ""

        #try:
        #    tokenizer.encode(assistant_text, add_special_tokens=False)
        #except Exception:
        #    rid = completion.id
        #    with open(f'test1-{rid}-prompt.json', 'w', encoding='utf-8') as file:
        #        json.dump(call_messages, file, indent=4)

        if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
            assistant_text = ""
            for token_info in completion.choices[0].logprobs.content:
                assistant_text = assistant_text + token_info.token

        else:
            assistant_text = completion.choices[0].message.content

        assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)

        #try:
        #    assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        #except Exception:
        #    print(completion)
        #    assistant_token_ids = []

        # 存储历史记录
        round_messages.append({"assistant": assistant_text})
        history_rounds[task_type].append(round_messages)

        this_round_prompt_token_ids.append(assistant_token_ids)
        prev_msg_token_ids_per_round[task_type].append(this_round_prompt_token_ids)

        append_gitkv_commands(
            commands=gitkv_commands,
            gitkv_context=gitkv_context,
            req=req,
            round_idx=i,
            agent_id=rid,
            m=m,
            a_row=a_row,
            b_row=b_row,
        )

        cached_tokens = (
            completion.usage.prompt_tokens_details.cached_tokens
            if completion.usage.prompt_tokens_details
            else 0
        )

        decode_time = completion.decode_time
        sum_decode_time = np.sum(decode_time)
        avg_decode_time = np.average(decode_time)
        p50_decode_time = np.percentile(decode_time, 50)
        p95_decode_time = np.percentile(decode_time, 95)

        calculate_cached_tokens = np.sum(b_row)

        write_result = [task_type, completion.usage.prompt_tokens, completion.usage.completion_tokens, calculate_cached_tokens, cached_tokens, completion.prefill_time, sum_decode_time, avg_decode_time, p50_decode_time, p95_decode_time, completion.num_local_cache, completion.num_global_cache]
        csv_writer.writerow(write_result)

        # 等待指定时间（模拟 tool 调用延迟）
        if req.wait_time and req.wait_time[i] > 0:
            time.sleep(req.wait_time[i])

    end_time = time.perf_counter()
    processing_time = end_time - start_time

    logger.info(f"Request completed in {processing_time:.3f}s")

    file.close()

    return {
        "history_rounds": history_rounds,
        "prev_msg_token_ids_per_round": prev_msg_token_ids_per_round,
        "processing_time_s": processing_time,
        "gitkv_commands": gitkv_commands,
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


def run_server(host: str = "0.0.0.0", port: int = 12306, workers: int = 1):
    """启动服务器"""
    uvicorn.run(
        "main_kv:app",
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
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, workers=args.workers)
