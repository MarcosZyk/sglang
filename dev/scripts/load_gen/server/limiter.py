"""
并发控制模块 - 使用信号量实现请求排队
"""
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestStats:
    """请求统计信息"""
    total_received: int = 0
    total_completed: int = 0
    total_queued: int = 0
    current_concurrent: int = 0
    max_concurrent_seen: int = 0
    queue_size: int = 0
    start_time: float = field(default_factory=lambda: time.time())
    total_wait_time: float = 0.0
    total_process_time: float = 0.0
    
    def to_dict(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            "total_received": self.total_received,
            "total_completed": self.total_completed,
            "total_queued": self.total_queued,
            "current_concurrent": self.current_concurrent,
            "max_concurrent_seen": self.max_concurrent_seen,
            "queue_size": self.queue_size,
            "avg_wait_time_ms": (self.total_wait_time / max(self.total_completed, 1)) * 1000,
            "avg_process_time_ms": (self.total_process_time / max(self.total_completed, 1)) * 1000,
            "throughput_rps": self.total_completed / max(elapsed, 0.001),
            "elapsed_seconds": elapsed
        }


class ConcurrentLimiter:
    """
    并发限制器 - 使用信号量控制并发数
    
    修复说明：
    - 不再依赖 semaphore._waiters 内部属性（不同 Python 版本实现不同）
    - 使用原子计数器自己跟踪等待队列大小
    """
    
    def __init__(self, max_concurrent: int = 32):
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats = RequestStats()
        self._stats_lock = threading.Lock()
        
        # 自己维护等待计数器（避免依赖 asyncio 内部实现）
        self._waiting_count = 0
        self._waiting_lock = threading.Lock()
        
    def _get_semaphore(self) -> asyncio.Semaphore:
        """获取或创建信号量"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    def _update_stats(self, **kwargs):
        """线程安全地更新统计信息"""
        with self._stats_lock:
            for key, value in kwargs.items():
                if hasattr(self._stats, key):
                    setattr(self._stats, key, value)
    
    def _increment_stats(self, **kwargs):
        """线程安全地递增统计信息"""
        with self._stats_lock:
            for key, value in kwargs.items():
                if hasattr(self._stats, key):
                    current = getattr(self._stats, key)
                    setattr(self._stats, key, current + value)
    
    async def acquire(self, request_id: str = "") -> float:
        """
        获取执行许可，如果需要则排队等待
        
        Returns:
            等待时间（秒）
        """
        semaphore = self._get_semaphore()
        
        wait_start = time.perf_counter()
        self._increment_stats(total_received=1)
        
        # 增加等待计数（在 acquire 之前）
        with self._waiting_lock:
            self._waiting_count += 1
            current_queue = max(0, self._waiting_count - self.max_concurrent)
        
        # 更新统计中的队列大小
        self._update_stats(queue_size=current_queue)
        
        # 获取信号量（如果达到最大并发，这里会阻塞等待）
        await semaphore.acquire()
        
        # 获取到信号量后，减少等待计数
        with self._waiting_lock:
            self._waiting_count -= 1
        
        wait_time = time.perf_counter() - wait_start
        
        # 更新统计
        with self._stats_lock:
            self._stats.current_concurrent += 1
            if self._stats.current_concurrent > self._stats.max_concurrent_seen:
                self._stats.max_concurrent_seen = self._stats.current_concurrent
            if wait_time > 0.001:  # 超过 1ms 才算排队
                self._stats.total_queued += 1
                self._stats.total_wait_time += wait_time
        
        if wait_time > 0.01:  # 记录显著等待
            logger.debug(f"Request {request_id} waited {wait_time*1000:.2f}ms")
        
        return wait_time
    
    def release(self, process_time: float = 0.0):
        """释放执行许可"""
        semaphore = self._get_semaphore()
        semaphore.release()
        
        with self._stats_lock:
            self._stats.current_concurrent -= 1
            self._stats.total_completed += 1
            self._stats.total_process_time += process_time
        
        # 更新队列大小统计
        with self._waiting_lock:
            current_queue = max(0, self._waiting_count - self.max_concurrent)
        self._update_stats(queue_size=current_queue)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._stats_lock:
            return self._stats.to_dict()
    
    def reset_stats(self):
        """重置统计信息"""
        with self._stats_lock:
            self._stats = RequestStats()
        with self._waiting_lock:
            self._waiting_count = 0


# 全局并发限制器实例（将在 lifespan 中初始化）
limiter: Optional[ConcurrentLimiter] = None