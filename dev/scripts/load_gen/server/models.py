
from typing import Optional
from pydantic import BaseModel

class SimResponse(BaseModel):
    """模拟响应数据模型"""
    request_id: str
    status: str
    process_time_ms: float
    wait_time_ms: float
    was_queued: bool
    total_tokens: int
    total_messages: int
    n: int
    n_task: int
    model_name: Optional[str] = None
    tokenizer_name: Optional[str] = None


class ServerStats(BaseModel):
    """服务器统计信息"""
    total_received: int
    total_completed: int
    total_queued: int
    current_concurrent: int
    max_concurrent_seen: int
    queue_size: int
    avg_wait_time_ms: float
    avg_process_time_ms: float
    throughput_rps: float
    elapsed_seconds: float
    max_concurrent_allowed: int
    total_tokens_processed: int
    total_messages_processed: int