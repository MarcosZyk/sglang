# server.py - 优化版（保持轮次顺序，请求间并发）
import os
import random
import asyncio
import time
import logging
from typing import List, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from openai import OpenAI  # 保持同步客户端
from transformers import AutoTokenizer
import uvicorn
import argparse

# 配置日志
logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Agent Simulator")

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


def simulate_sync(req_dict: Dict) -> Dict:
    """
    同步模拟函数 - 保持原有逻辑不变
    
    关键：这个函数会在线程池中执行，所以不会阻塞事件循环
    不同请求可以并行执行，但每个请求内部保持顺序
    """
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
    
    # ========== 创建 OpenAI 客户端 ==========
    client_list: List[OpenAI] = []
    for port in range(12347, 12347 + req.n_task):
        client_list.append(
            OpenAI(
                api_key="EMPTY", 
                base_url=f"http://localhost:{port}/v1",
                timeout=300.0,  # 增加超时
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
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
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
            completion = client_list[task_type].chat.completions.create(
                model=openai_model,
                messages=call_messages,
                max_tokens=s_list[i],
                temperature=req.temperature,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {e}")
        
        # 解析 assistant 文本
        assistant_text = ""
        try:
            assistant_text = completion.choices[0].message.content
        except Exception:
            try:
                assistant_text = completion.choices[0].get("message", {}).get("content", "")
            except Exception:
                assistant_text = ""
        
        assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        
        # 存储历史记录
        round_messages.append({"assistant": assistant_text})
        history_rounds[task_type].append(round_messages)
        
        this_round_prompt_token_ids.append(assistant_token_ids)
        prev_msg_token_ids_per_round[task_type].append(this_round_prompt_token_ids)
        
        # 等待指定时间（模拟 tool 调用延迟）
        if req.wait_time and req.wait_time[i] > 0:
            time.sleep(req.wait_time[i])
    
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    
    logger.info(f"Request completed in {processing_time:.3f}s")
    
    return {
        "history_rounds": history_rounds,
        "prev_msg_token_ids_per_round": prev_msg_token_ids_per_round,
        "processing_time_s": processing_time
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


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """启动服务器"""
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        loop="asyncio",
        http="httptools",
        log_level="warning",
        access_log=False,
        limit_concurrency=500,
        backlog=2048,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, workers=args.workers)