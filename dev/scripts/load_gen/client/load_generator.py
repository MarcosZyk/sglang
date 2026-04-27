"""
客户端负载生成器 - 按固定 RPS 发送 SimRequest 请求
"""
import time
import threading
import statistics
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from generate_payload import read_replay_data
from pydantic import BaseModel
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAYLOAD_JSON = PROJECT_ROOT / "output_json_flatten" / "test1.json"

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
    task_type_list: Optional[List[str]] = None
    semantic_type_list: Optional[List[Optional[List[str]]]] = None


@dataclass
class RequestResult:
    """单次请求结果"""
    request_id: int
    status_code: int
    response_time_ms: float
    wait_time_ms: float
    process_time_ms: float
    was_queued: bool
    total_tokens: int
    total_messages: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class LoadTestStats:
    """负载测试统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    total_wait_time_ms: float = 0.0
    total_tokens: int = 0
    total_messages: int = 0
    response_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    start_time: float = field(default_factory=lambda: time.time())
    end_time: float = 0.0
    target_rps: float = 0.0
    actual_rps: float = 0.0
    server_accepted_rps: float = 0.0
    p50_response: float = 0.0
    p95_response: float = 0.0
    p99_response: float = 0.0
    
    def add_result(self, result: RequestResult):
        """添加请求结果"""
        self.total_requests += 1
        if result.status_code == 200:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_response_time_ms += result.response_time_ms
        self.total_wait_time_ms += result.wait_time_ms
        self.total_tokens += result.total_tokens
        self.total_messages += result.total_messages
        self.response_times.append(result.response_time_ms)
        self.wait_times.append(result.wait_time_ms)
        
        self.status_codes[result.status_code] = \
            self.status_codes.get(result.status_code, 0) + 1
    
    def calculate_metrics(self):
        """计算最终指标"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.actual_rps = self.total_requests / max(elapsed, 0.001)
        
        if self.response_times:
            self.p50_response = statistics.median(self.response_times)
            if len(self.response_times) >= 20:
                self.p95_response = statistics.quantiles(self.response_times, n=20)[18]
            else:
                self.p95_response = max(self.response_times)
            
            if len(self.response_times) >= 100:
                self.p99_response = statistics.quantiles(self.response_times, n=100)[98]
            else:
                self.p99_response = max(self.response_times)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        elapsed = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1) * 100,
            "duration_seconds": elapsed,
            "target_rps": self.target_rps,
            "actual_rps": self.actual_rps,
            "server_accepted_rps": self.server_accepted_rps,
            "avg_response_time_ms": self.total_response_time_ms / max(self.total_requests, 1),
            "p50_response_time_ms": self.p50_response,
            "p95_response_time_ms": self.p95_response,
            "p99_response_time_ms": self.p99_response,
            "avg_wait_time_ms": self.total_wait_time_ms / max(self.total_requests, 1),
            "total_tokens": self.total_tokens,
            "total_messages": self.total_messages,
            "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
            "avg_messages_per_request": self.total_messages / max(self.total_requests, 1),
            "tokens_per_second": self.total_tokens / max(elapsed, 0.001),
            "status_codes": self.status_codes
        }


class TokenBucket:
    """令牌桶 - 精确控制请求速率"""
    
    def __init__(self, rate: float, capacity: Optional[int] = 32):
        self.rate = rate
        self.capacity = capacity if capacity else int(rate)
        self.tokens = float(self.capacity)
        self.last_update = time.perf_counter()
        self._lock = threading.Lock()
    
    def acquire(self) -> float:
        """获取一个令牌，如果需要则等待"""
        wait_time = 0.0
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                wait_needed = (1 - self.tokens) / self.rate
                self.last_update += wait_needed
                self.tokens = 0
                wait_time = wait_needed
       
        if wait_time > 0:
            time.sleep(wait_time)
        
        return wait_time


class LoadGenerator:
    """负载生成器 - 按固定 RPS 发送 SimRequest 请求"""
    
    def __init__(self, server_url: str = "http://localhost:12306",
                 rps: float = 100.0, total_requests: int = 1000,
                 timeout: float = 36000.0, warmup_requests: int = 0,
                 sim_config: Optional[SimRequest] = None):
        self.server_url = server_url
        self.rps = rps
        self.total_requests = total_requests
        self.timeout = timeout
        self.warmup_requests = warmup_requests
        self.sim_config = sim_config
        
        self.stats = LoadTestStats()
        self.stats.target_rps = rps
        self._stop_flag = threading.Event()
        self._results: List[RequestResult] = []
        self._results_lock = threading.Lock()
        
        self.session = self._create_session()
        self.token_bucket = TokenBucket(rate=rps)
    
    def _create_session(self) -> requests.Session:
        """创建优化的 HTTP 会话"""
        session = requests.Session()
        retry = Retry(total=0, backoff_factor=0, status_forcelist=[])
        adapter = HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=retry,
            pool_block=False
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _send_request(self, request_id: int) -> RequestResult:
        """发送单个 SimRequest（同步）"""
        start_time = time.perf_counter()
        
        # 生成 SimRequest 数据
        sim_data = self.sim_config
        
        try:
            response = self.session.post(
                f"{self.server_url}/api/process",
                json=sim_data,
                timeout=self.timeout
            )
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return RequestResult(
                    request_id=request_id,
                    status_code=response.status_code,
                    response_time_ms=response_time,
                    wait_time_ms=data.get("wait_time_ms", 0),
                    process_time_ms=data.get("process_time_ms", 0),
                    was_queued=data.get("was_queued", False),
                    total_tokens=data.get("total_tokens", 0),
                    total_messages=data.get("total_messages", 0)
                )
            else:
                return RequestResult(
                    request_id=request_id,
                    status_code=response.status_code,
                    response_time_ms=response_time,
                    wait_time_ms=0,
                    process_time_ms=0,
                    was_queued=False,
                    total_tokens=0,
                    total_messages=0,
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                request_id=request_id,
                status_code=0,
                response_time_ms=response_time,
                wait_time_ms=0,
                process_time_ms=0,
                was_queued=False,
                total_tokens=0,
                total_messages=0,
                error=str(e)
            )
    
    def _worker(self, request_id: int):
        """工作线程"""
        #self.token_bucket.acquire()
        result = self._send_request(request_id)
        
        with self._results_lock:
            self.stats.add_result(result)
            self._results.append(result)
        
        if request_id % max(1, self.total_requests // 10) == 0:
            elapsed = time.time() - self.stats.start_time
            current_rps = request_id / max(elapsed, 0.001)
            logger.info(f"Progress: {request_id}/{self.total_requests} ({current_rps:.1f} RPS)")
    
    def run(self) -> LoadTestStats:
        """运行负载测试"""
        logger.info(f"Starting load test: {self.total_requests} requests at {self.rps} RPS")
        logger.info(f"Target server: {self.server_url}")
        logger.info(f"SimRequest config: n={self.sim_config['n']}, n_task={self.sim_config['n_task']}")
        
        self.stats = LoadTestStats()
        self.stats.target_rps = self.rps
        self.stats.start_time = time.time()
        self._results = []
        
        # 预热
        if self.warmup_requests > 0:
            logger.info(f"Warming up with {self.warmup_requests} requests...")
            for i in range(self.warmup_requests):
                try:
                    sim_data = self.sim_config
                    self.session.post(
                        f"{self.server_url}/api/process",
                        json=sim_data,
                        timeout=self.timeout
                    )
                except:
                    pass
            time.sleep(0.5)
        
        # 重置服务器统计
        try:
            self.session.post(f"{self.server_url}/api/stats/reset", timeout=5)
        except:
            pass
        
        max_workers = min(100, max(38, int(self.rps * 2)))
        logger.info(f"Using {max_workers} worker threads")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(self.total_requests):
                if self._stop_flag.is_set():
                    break
                futures.append(executor.submit(self._worker, i))
                time.sleep(1/self.rps)
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker error: {e}")

        
        self.stats.calculate_metrics()
        
        # 获取服务器端统计
        try:
            resp = self.session.get(f"{self.server_url}/api/stats", timeout=5)
            server_stats = resp.json()
            self.stats.server_accepted_rps = server_stats.get("throughput_rps", 0)
            logger.info(f"Server reported throughput: {self.stats.server_accepted_rps:.2f} RPS")
            logger.info(f"Server total tokens: {server_stats.get('total_tokens_processed', 0)}")
        except Exception as e:
            logger.warning(f"Could not get server stats: {e}")
        
        elapsed = time.time() - self.stats.start_time
        logger.info(f"Load test completed in {elapsed:.2f} seconds")
        
        return self.stats
    
    def stop(self):
        """停止测试"""
        self._stop_flag.set()
    
    def print_report(self):
        """打印测试报告"""
        metrics = self.stats.to_dict()
        
        print("\n" + "=" * 70)
        print("LOAD TEST REPORT - SimRequest")
        print("=" * 70)
        print(f"Target RPS:              {metrics['target_rps']:.2f}")
        print(f"Actual RPS:              {metrics['actual_rps']:.2f}")
        print(f"Server Accepted RPS:     {metrics['server_accepted_rps']:.2f}")
        print(f"Duration:                {metrics['duration_seconds']:.2f} seconds")
        print(f"Total Requests:          {metrics['total_requests']}")
        print(f"Successful:              {metrics['successful_requests']} ({metrics['success_rate']:.1f}%)")
        print(f"Failed:                  {metrics['failed_requests']}")
        print("-" * 70)
        print(f"Total Tokens:            {metrics['total_tokens']:,}")
        print(f"Total Messages:          {metrics['total_messages']:,}")
        print(f"Tokens/Second:           {metrics['tokens_per_second']:,.0f}")
        print(f"Avg Tokens/Request:      {metrics['avg_tokens_per_request']:.1f}")
        print(f"Avg Messages/Request:    {metrics['avg_messages_per_request']:.1f}")
        print("-" * 70)
        print(f"Avg Response Time:       {metrics['avg_response_time_ms']:.2f} ms")
        print(f"P50 Response Time:       {metrics['p50_response_time_ms']:.2f} ms")
        print(f"P95 Response Time:       {metrics['p95_response_time_ms']:.2f} ms")
        print(f"P99 Response Time:       {metrics['p99_response_time_ms']:.2f} ms")
        print(f"Avg Wait Time:           {metrics['avg_wait_time_ms']:.2f} ms")
        print("-" * 70)
        print(f"Status Codes:            {metrics['status_codes']}")
        print("=" * 70 + "\n")


def resolve_payload_json(json_file: Optional[str] = None) -> Path:
    """Resolve replay JSON path from CLI input or fall back to the default file."""
    if json_file is None:
        resolved_path = DEFAULT_PAYLOAD_JSON
    else:
        input_path = Path(json_file).expanduser()
        if input_path.is_absolute():
            resolved_path = input_path
        else:
            cwd_path = (Path.cwd() / input_path).resolve()
            if cwd_path.exists():
                resolved_path = cwd_path
            else:
                resolved_path = (PROJECT_ROOT / input_path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Replay JSON file not found: {resolved_path}")

    return resolved_path


def run_client(server_url: str = "http://localhost:12306",
               rps: float = 100.0, total_requests: int = 1000,
               model_name: str = None, json_file: Optional[str] = None):
    """运行客户端负载测试"""
    payload_json = resolve_payload_json(json_file)
    logger.info("Using replay JSON: %s", payload_json)
    sim_config = read_replay_data(str(payload_json), model_name)
    generator = LoadGenerator(
        server_url=server_url,
        rps=rps,
        total_requests=total_requests,
        sim_config=sim_config
    )
    stats = generator.run()
    generator.print_report()
    return stats


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Load Generator Client with SimRequest")
    parser.add_argument("--url", default="http://localhost:12309", help="Server URL")
    parser.add_argument("--rps", type=float, default=100.0, help="Requests per second")
    parser.add_argument("--requests", type=int, default=1000, help="Total requests")
    parser.add_argument("--sim-n", type=int, default=10, help="SimRequest n value")
    parser.add_argument("--sim-n-task", type=int, default=3, help="SimRequest n_task value")
    parser.add_argument('--model', default='Qwen/Qwen3-8B', type=str)
    parser.add_argument(
        "--json-file",
        default=None,
        help=f"Replay JSON file path (default: {DEFAULT_PAYLOAD_JSON})",
    )
    args = parser.parse_args()
    run_client(
        server_url=args.url,
        rps=args.rps,
        total_requests=args.requests,
        model_name=args.model,
        json_file=args.json_file,
    )
