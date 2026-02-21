# server.py
import os
import random
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from transformers import AutoTokenizer
import uvicorn
import argparse
import time

app = FastAPI(title="LLM Agent Simulator (roles provided by client)")


# ---------- Pydantic 请求模型 ----------
class SimRequest(BaseModel):
    n: int  # 总共调用次数
    n_task: int  # 总共任务类型数
    s_list: Optional[List[int]] = None  # 每轮生成的 token 数 s[i], 长度 n (可选)
    m_list: Optional[List[int]] = None  # 每轮消息数 m[i], 长度 n (可选)
    a_list: Optional[List[List[int]]] = None  # a[i][j] 每条 message 的 token 数（可选）
    b_list: Optional[List[List[int]]] = None  # b[i][j] 与上一轮相同的 token 数（可选）
    roles_list: Optional[List[List[str]]] = (
        None  # roles_list[i][j] 指定每条 message 的 role（可选）
    )
    wait_time: Optional[List[float]] = (
        None  # 每次任务调用时，其距离上一轮调用的等待时间 (可用于模拟tools调用)
    )
    task_list: Optional[List[int]] = None
    tokenizer_name: Optional[str] = None
    openai_model: Optional[str] = None
    temperature: Optional[float] = 0.0


# ---------- Helper ----------
def random_id_sampler(tokenizer, k: int, forbidden: set):
    """从 tokenizer vocab 范围内随机生成 k 个 token ids，跳过 forbidden（允许重复采样）"""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer.get_vocab())
        except Exception:
            raise RuntimeError("无法获取 tokenizer 的 vocab 大小")
    out = []
    while len(out) < k:
        cand = random.randrange(0, vocab_size)
        if cand in forbidden:
            continue
        out.append(cand)
    return out

def simulate(req: SimRequest):
    if req.n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")

    if req.n_task <= 0:
        raise HTTPException(status_code=400, detail="n_task must be > 0")

    tokenizer_name = req.tokenizer_name
    openai_model = req.openai_model

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    forbidden_ids = set(getattr(tokenizer, "all_special_ids", []))

    n = req.n
    # m_list default：1,2,3,...
    if req.m_list:
        if len(req.m_list) != n:
            raise HTTPException(status_code=400, detail="m_list length must equal n")
        m_list = req.m_list
    else:
        raise HTTPException(status_code=400, detail="m_list length must equal n")

    # s_list default：小幅增长
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
        raise HTTPException(
            status_code=400, detail="a_list length must equal n if provided"
        )
    if b_list and len(b_list) != n:
        raise HTTPException(
            status_code=400, detail="b_list length must equal n if provided"
        )

    client_list: List[OpenAI] = []
    for port in range(12347, 12347 + req.n_task):
        client_list.append(
            OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        )

    # history_rounds: list of rounds; each round is {"round_index": i, "messages": [...], "assistant": {...}}
    history_rounds: List[List[List[Dict[str, str]]]] = [
        [] for _ in range(0, req.n_task)
    ]

    # prev_msg_token_ids_per_round: list where each item is a list of token_ids for that round's prompts
    prev_msg_token_ids_per_round: List[List[List[List[int]]]] = [
        [] for _ in range(0, req.n_task)
    ]
    start_time = time.perf_counter()
    for i in range(n):
        m = m_list[i]

        task_type = req.task_list[i]

        # a_row: token 长度每条 message（由客户端提供或默认生成）
        a_row = a_list[i]
        if len(a_row) != m:
            raise HTTPException(
                status_code=400, detail=f"a_list[{i}] length must be {m}"
            )

        # b_row: prefix reuse 数量（依赖 prev_msg_token_ids_per_round）
        b_row = b_list[i]
        if len(b_row) != m:
            raise HTTPException(
                status_code=400, detail=f"b_list[{i}] length must be {m}"
            )

        # roles_row: if provided use it, otherwise default to "user" for every message
        roles_row = roles_list[i]
        if len(roles_row) != m:
            raise HTTPException(
                status_code=400, detail=f"roles_list[{i}] length must be {m}"
            )

        round_messages = []
        this_round_prompt_token_ids: List[List[int]] = []

        for j in range(m):
            a_ij = int(a_row[j])
            b_ij = int(b_row[j])
            role_j = roles_row[j]

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

            new_count = a_ij - b_ij
            new_ids = (
                random_id_sampler(tokenizer, new_count, forbidden_ids)
                if new_count > 0
                else []
            )

            token_ids: List[int] = reused_ids + new_ids

            # decode -> prompt text
            prompt_text = tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            this_round_prompt_token_ids.append(token_ids)
            round_messages.append({role_j: prompt_text})

        # flatten history_rounds -> messages 用于发送给模型
        call_messages = []
        for item in round_messages:
            for key in item:
                role = key
                content = item[key]
                call_messages.append({"role": role, "content": content})

        try:
            completion = client_list[task_type].chat.completions.create(
                model=openai_model,
                messages=call_messages,
                max_tokens=s_list[i],
                temperature=req.temperature,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {e}")

        cached_tokens = (
            completion.usage.prompt_tokens_details.cached_tokens
            if completion.usage.prompt_tokens_details
            else 0
        )
        #print(task_type, cached_tokens)
        # 解析 assistant 文本
        assistant_text = ""
        try:
            assistant_text = completion.choices[0].message.content
        except Exception:
            try:
                assistant_text = (
                    completion.choices[0].get("message", {}).get("content", "")
                )
            except Exception:
                assistant_text = ""

        assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)

        # 将本轮的 prompt messages 和 assistant 作为一个 round 存入 history_rounds

        round_messages.append({"assistant": assistant_text})
        history_rounds[task_type].append(round_messages)

        # 记录本轮 prompt 的 token_ids 列表（二维： [msg_idx][token_ids...] ）
        this_round_prompt_token_ids.append(assistant_token_ids)
        prev_msg_token_ids_per_round[task_type].append(this_round_prompt_token_ids)

        time.sleep(req.wait_time[i])
    
    end_time = time.perf_counter()
    print(end_time - start_time)

    # 最终返回本次请求内完整的 rounds 结果、按轮组织的历史，以及每一轮的 prompt token ids
    return {
        "history_rounds": history_rounds,
        "prev_msg_token_ids_per_round": prev_msg_token_ids_per_round,
    }