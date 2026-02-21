"""
FastAPI 服务端 - 性能优化版
"""
import time
import uuid
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from .simulator import SimRequest, simulate

# 使用 orjson 加速 JSON 序列化
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    print("Warning: orjson not installed, using standard json. Install with: pip install orjson")

from .limiter import ConcurrentLimiter

# 配置日志 - 降低级别减少开销
logging.basicConfig(
    level=logging.WARNING,  # 改为 WARNING 减少日志开销
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerConfig:
    """服务端配置"""
    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 max_concurrent: int = 32, work_delay: float = 0.001):  # 默认更小的延迟
        self.host = host
        self.port = port
        self.max_concurrent = max_concurrent
        self.work_delay = work_delay


# 全局变量
server_config = ServerConfig()
global_limiter: Optional[ConcurrentLimiter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global global_limiter
    global_limiter = ConcurrentLimiter(max_concurrent=server_config.max_concurrent)
    logger.info(f"Server started with max_concurrent={server_config.max_concurrent}")
    yield
    if global_limiter:
        stats = global_limiter.get_stats()
        logger.info(f"Server shutdown. Final stats: {stats}")


app = FastAPI(
    title="Load Gen Server",
    description="负载测试服务端 - 性能优化版",
    version="1.0.0",
    lifespan=lifespan
)


# 自定义 JSON 响应类（使用 orjson）
class ORJSONResponse(JSONResponse):
    media_type = "application/json"
    
    def render(self, content: Any) -> bytes:
        if USE_ORJSON:
            return orjson.dumps(content)
        return super().render(content)


def sync_business_logic(sim_request: SimRequest, request_id: str) -> Dict[str, Any]:
    """
    同步业务逻辑处理 - 性能优化版
    
    优化点：
    1. 减少不必要的计算
    2. 延迟验证
    3. 最小化 sleep 时间
    """
    process_start = time.perf_counter()
    
    # 快速计算 token 数（不验证完整列表）
    total_tokens = 0
    total_messages = sim_request.n  # 默认值
    
    if sim_request.s_list and isinstance(sim_request.s_list, list):
        try:
            total_tokens = sum(sim_request.s_list)
        except (TypeError, ValueError):
            total_tokens = 0
    
    if sim_request.m_list and isinstance(sim_request.m_list, list):
        try:
            total_messages = sum(sim_request.m_list)
        except (TypeError, ValueError):
            total_messages = sim_request.n
    
    # 最小化处理延迟
    if server_config.work_delay > 0:
        time.sleep(server_config.work_delay)
    
    process_time = time.perf_counter() - process_start
    
    return {
        "request_id": request_id,
        "status": "processed",
        "process_time_ms": process_time * 1000,
        "total_tokens": total_tokens,
        "total_messages": total_messages,
        "n": sim_request.n,
        "n_task": sim_request.n_task,
        "model_name": sim_request.openai_model,
        "tokenizer_name": sim_request.tokenizer_name,
        "temperature": sim_request.temperature
    }


@app.post("/api/process")
async def process_request(sim_request: SimRequest):
    """
    主处理接口 - 性能优化版
    
    优化点：
    1. 跳过不必要的验证
    2. 使用 ORJSONResponse 加速序列化
    3. 最小化临界区代码
    """
    request_id = str(uuid.uuid4())[:8]
    
    # 跳过验证（仅在需要时调用）
    # try:
    #     sim_request.validate_lists()
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    
    # 获取执行许可
    logger.info(f'Recv Req at {time.time()}')
    wait_time = await global_limiter.acquire(request_id)
    
    process_start = time.perf_counter()
    try:
        result = sync_business_logic(sim_request, request_id)
        result["wait_time_ms"] = wait_time * 1000
        result["was_queued"] = wait_time > 0.001
        
        return ORJSONResponse(content=result)
    
    finally:
        process_time = time.perf_counter() - process_start
        global_limiter.release(process_time)


@app.get("/api/stats")
async def get_stats():
    """获取服务器统计信息"""
    if not global_limiter:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    stats = global_limiter.get_stats()
    stats["max_concurrent_allowed"] = server_config.max_concurrent
    return ORJSONResponse(content=stats)


@app.post("/api/stats/reset")
async def reset_stats():
    """重置统计信息"""
    if not global_limiter:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    global_limiter.reset_stats()
    return ORJSONResponse(content={"status": "stats_reset"})


@app.get("/api/health")
async def health_check():
    """健康检查接口 - 最简化"""
    if not global_limiter:
        return ORJSONResponse(content={
            "status": "starting",
            "current_concurrent": 0,
            "queue_size": 0
        })
    
    stats = global_limiter.get_stats()
    return ORJSONResponse(content={
        "status": "healthy",
        "current_concurrent": stats["current_concurrent"],
        "queue_size": stats["queue_size"]
    })


@app.get("/")
async def root():
    """根路径"""
    return ORJSONResponse(content={
        "message": "Load Gen Server - Optimized",
        "docs": "/docs",
        "stats": "/api/stats",
        "process": "/api/process (POST)"
    })


def run_server(host: str = "0.0.0.0", port: int = 8000, 
               max_concurrent: int = 32, work_delay: float = 0.001,
               http_threads: int = 4):  # 新增 HTTP 线程数参数
    """启动服务器"""
    global server_config
    server_config = ServerConfig(
        host=host,
        port=port,
        max_concurrent=max_concurrent,
        work_delay=work_delay
    )
    
    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        workers=1,  # 保持单 worker 保证并发控制准确
        http="httptools",
        #http_threads=http_threads,  # 增加 HTTP 处理线程
        loop="asyncio",
        log_level="info",  # 降低日志级别
        access_log=False,  # 禁用访问日志减少开销
    )


if __name__ == "__main__":
    run_server()