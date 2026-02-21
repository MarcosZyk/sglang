"""
FastAPI 服务端 - 负载测试目标服务器
"""
import time
import uuid
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .limiter import ConcurrentLimiter
from .simulator import simulate, SimRequest
from .models import SimResponse, ServerStats

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerConfig:
    """服务端配置"""
    def __init__(self, host: str = "0.0.0.0", port: int = 12306,
                 max_concurrent: int = 32, work_delay: float = 0.01):
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
    description="负载测试服务端 - 支持 SimRequest 模拟请求",
    version="1.0.0",
    lifespan=lifespan
)


def sync_business_logic(sim_request: SimRequest, request_id: str) -> Dict[str, Any]:
    """
    同步业务逻辑处理 - 模拟 LLM 调用
    
    根据 SimRequest 中的参数模拟实际工作负载
    """
    process_start = time.perf_counter()
    
    # 计算总 token 数和消息数
    _ = simulate(sim_request)
    
      
    process_time = time.perf_counter() - process_start
    
    return {
        "request_id": request_id,
        "status": "processed",
        "process_time_ms": process_time * 1000,
    }


@app.post("/api/process", response_model=SimResponse)
async def process_request(sim_request: SimRequest):
    """
    主处理接口 - 接收 SimRequest 模拟请求
    
    流程：
    1. 验证请求数据
    2. 获取信号量许可（超出并发数时自动排队）
    3. 执行同步业务逻辑
    4. 释放信号量
    5. 返回结果
    """
    request_id = str(uuid.uuid4())[:8]
    
    # 验证请求数据
    #try:
    #    sim_request.validate_lists()
    #except ValueError as e:
    #    raise HTTPException(status_code=400, detail=str(e))
    
    # 获取执行许可（可能排队等待）
    wait_time = await global_limiter.acquire(request_id)
    
    process_start = time.perf_counter()
    try:
        # 执行同步业务逻辑
        result = sync_business_logic(sim_request, request_id)
        
        # 添加排队信息
        result["wait_time_ms"] = wait_time * 1000
        result["was_queued"] = wait_time > 0.001
        
        return JSONResponse(content=result)
    
    finally:
        process_time = time.perf_counter() - process_start
        global_limiter.release(process_time)


@app.get("/api/stats", response_model=ServerStats)
async def get_stats():
    """获取服务器统计信息"""
    if not global_limiter:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    stats = global_limiter.get_stats()
    stats["max_concurrent_allowed"] = server_config.max_concurrent
    return JSONResponse(content=stats)


@app.post("/api/stats/reset")
async def reset_stats():
    """重置统计信息"""
    if not global_limiter:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    global_limiter.reset_stats()
    return JSONResponse(content={"status": "stats_reset"})


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    if not global_limiter:
        return JSONResponse(content={
            "status": "starting",
            "current_concurrent": 0,
            "queue_size": 0
        })
    
    stats = global_limiter.get_stats()
    return JSONResponse(content={
        "status": "healthy",
        "current_concurrent": stats["current_concurrent"],
        "queue_size": stats["queue_size"]
    })


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Load Gen Server with SimRequest",
        "docs": "/docs",
        "stats": "/api/stats",
        "process": "/api/process (POST)",
        "sim_request_schema": "/docs#/SimRequest"
    }


def run_server(host: str = "0.0.0.0", port: int = 12306, 
               max_concurrent: int = 32, work_delay: float = 0.01):
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
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()