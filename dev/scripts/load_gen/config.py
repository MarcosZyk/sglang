"""
配置模块 - 统一管理服务端和客户端参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """服务端配置"""
    host: str = "0.0.0.0"
    port: int = 12306
    max_concurrent: int = 32  # 最大并发请求数
    work_delay: float = 0.1  # 模拟工作延迟（秒）
    enable_stats: bool = True  # 启用统计


@dataclass
class ClientConfig:
    """客户端配置"""
    server_url: str = "http://localhost:12306"
    rps: float = 100.0  # 每秒请求数
    total_requests: int = 1000  # 总请求数
    timeout: float = 1800.0  # 请求超时时间
    warmup_requests: int = 10  # 预热请求数


@dataclass
class LoadGenConfig:
    """整体配置"""
    # 修复：使用 default_factory 避免 mutable default 错误
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)



# 全局配置实例
config = LoadGenConfig()