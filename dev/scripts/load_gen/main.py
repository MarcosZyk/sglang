"""
Load Gen 项目主入口
支持单独启动服务端、客户端，或同时启动进行完整测试
"""
import argparse
import threading
import time
import sys


def start_server(host: str = "0.0.0.0", port: int = 12306,
                 max_concurrent: int = 32, work_delay: float = 0.01):
    """启动服务端"""
    from server.main import run_server
    run_server(host=host, port=port, max_concurrent=max_concurrent, work_delay=work_delay)


def start_client(server_url: str, rps: float, total_requests: int):
    """启动客户端"""
    from client.load_generator import run_client
    return run_client(server_url=server_url, rps=rps, total_requests=total_requests)


def run_full_test(host: str = "localhost", port: int = 12306,
                  max_concurrent: int = 32, rps: float = 100.0,
                  total_requests: int = 1000, work_delay: float = 0.01):
    """运行完整测试（服务端+客户端）"""
    server_url = f"http://{host}:{port}"
    
    # 在后台线程启动服务器
    server_thread = threading.Thread(
        target=start_server,
        kwargs={
            "host": host,
            "port": port,
            "max_concurrent": max_concurrent,
            "work_delay": work_delay
        },
        daemon=True
    )
    
    print(f"Starting server on {server_url} (max_concurrent={max_concurrent})...")
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(2)
    
    # 检查服务器是否就绪
    import requests
    for i in range(10):
        try:
            resp = requests.get(f"{server_url}/api/health", timeout=2)
            if resp.status_code == 200:
                print("Server is ready!")
                break
        except:
            pass
        time.sleep(0.5)
    else:
        print("ERROR: Server failed to start")
        sys.exit(1)
    
    # 运行客户端测试
    print(f"\nStarting load test: {total_requests} requests at {rps} RPS")
    print("-" * 60)
    
    stats = start_client(server_url=server_url, rps=rps, total_requests=total_requests)
    
    # 保持服务器运行一会儿以便查看最终统计
    print("\nTest completed. Server will shutdown in 3 seconds...")
    time.sleep(3)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Gen - 负载测试工具")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 服务端命令
    server_parser = subparsers.add_parser("server", help="启动服务端")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=12306)
    server_parser.add_argument("--max-concurrent", type=int, default=32)
    server_parser.add_argument("--work-delay", type=float, default=0.01)
    
    # 客户端命令
    client_parser = subparsers.add_parser("client", help="启动客户端")
    client_parser.add_argument("--url", default="http://localhost:12306")
    client_parser.add_argument("--rps", type=float, default=100.0)
    client_parser.add_argument("--requests", type=int, default=1000)
    
    # 完整测试命令
    test_parser = subparsers.add_parser("test", help="运行完整测试")
    test_parser.add_argument("--host", default="localhost")
    test_parser.add_argument("--port", type=int, default=12306)
    test_parser.add_argument("--max-concurrent", type=int, default=32)
    test_parser.add_argument("--rps", type=float, default=100.0)
    test_parser.add_argument("--requests", type=int, default=1000)
    test_parser.add_argument("--work-delay", type=float, default=0.01)
    
    args = parser.parse_args()
    
    if args.command == "server":
        start_server(
            host=args.host,
            port=args.port,
            max_concurrent=args.max_concurrent,
            work_delay=args.work_delay
        )
    elif args.command == "client":
        start_client(
            server_url=args.url,
            rps=args.rps,
            total_requests=args.requests
        )
    elif args.command == "test":
        run_full_test(
            host=args.host,
            port=args.port,
            max_concurrent=args.max_concurrent,
            rps=args.rps,
            total_requests=args.requests,
            work_delay=args.work_delay
        )
    else:
        parser.print_help()