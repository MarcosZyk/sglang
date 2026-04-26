from __future__ import annotations

import argparse
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:
    from models import (
        ChatCompletionRequest,
        CreateContextRequest,
        ContextIndexRequest,
        ContextOnlyRequest,
    )
    from runtime import ArtesiaError, ArtesiaSimulator, SimulatorConfig
except ImportError:
    from .models import (
        ChatCompletionRequest,
        CreateContextRequest,
        ContextIndexRequest,
        ContextOnlyRequest,
    )
    from .runtime import ArtesiaError, ArtesiaSimulator, SimulatorConfig


def create_app(simulator: Optional[ArtesiaSimulator] = None) -> FastAPI:
    app = FastAPI(title="Artesia Simulator")
    runtime = simulator or ArtesiaSimulator(SimulatorConfig())

    def translate_error(error: ArtesiaError) -> HTTPException:
        return HTTPException(status_code=error.status_code, detail=error.detail)

    @app.put("/context/{context_id}")
    async def create_context(context_id: str, request: CreateContextRequest) -> JSONResponse:
        try:
            result = await runtime.create_context(context_id, request.context_type)
        except ArtesiaError as error:
            raise translate_error(error) from error
        return JSONResponse(content=result)

    @app.delete("/context/{context_id}")
    async def delete_context(context_id: str) -> JSONResponse:
        try:
            result = await runtime.delete_context(context_id)
        except ArtesiaError as error:
            raise translate_error(error) from error
        return JSONResponse(content=result)

    @app.post("/truncate")
    async def truncate(request: ContextIndexRequest) -> JSONResponse:
        try:
            result = await runtime.truncate(request.context_id, request.msg_index)
        except ArtesiaError as error:
            raise translate_error(error) from error
        return JSONResponse(content=result)

    @app.post("/one-off")
    async def one_off(request: ContextIndexRequest) -> JSONResponse:
        try:
            result = await runtime.one_off(request.context_id, request.msg_index)
        except ArtesiaError as error:
            raise translate_error(error) from error
        return JSONResponse(content=result)

    @app.post("/suspend")
    async def suspend(request: ContextOnlyRequest) -> JSONResponse:
        try:
            result = await runtime.suspend(request.context_id)
        except ArtesiaError as error:
            raise translate_error(error) from error
        return JSONResponse(content=result)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
        try:
            completion = await runtime.generate(request)
        except ArtesiaError as error:
            raise translate_error(error) from error
        return JSONResponse(content=completion.model_dump())

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            content={
                "status": "healthy",
                "contexts": len(runtime.contexts),
                "gpu_bytes_used": runtime.gpu_bytes_used,
                "cpu_bytes_used": runtime.cpu_bytes_used,
                "decode_max_concurrency": runtime.config.decode_max_concurrency,
                "enable_artesia": runtime.config.enable_artesia,
                "tokenizer": runtime.config.tokenizer_name,
            }
        )

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": "Artesia Simulator",
                "docs": "/docs",
            }
        )

    return app


def build_config_from_args(args: argparse.Namespace) -> SimulatorConfig:
    return SimulatorConfig(
        cpu_gpu_bandwidth_gbps=args.cpu_gpu_bandwidth_gbps,
        gpu_sglang_bandwidth_gbps=args.gpu_sglang_bandwidth_gbps,
        prefill_throughput_tps=args.prefill_throughput_tps,
        decode_throughput_tps=args.decode_throughput_tps,
        decode_max_concurrency=args.decode_max_concurrency,
        kv_cache_kb_per_token=args.kv_cache_kb_per_token,
        gpu_capacity_gb=args.gpu_capacity_gb,
        cpu_capacity_gb=args.cpu_capacity_gb,
        eviction_policy=args.eviction_policy,
        enable_artesia=args.enable_artesia,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
    )


def run_server(
    *,
    host: str = "0.0.0.0",
    port: int = 50350,
    workers: int = 1,
    cpu_gpu_bandwidth_gbps: float = 12.0,
    gpu_sglang_bandwidth_gbps: float = 24.0,
    prefill_throughput_tps: float = 8000.0,
    decode_throughput_tps: float = 2000.0,
    decode_max_concurrency: int = 1,
    kv_cache_kb_per_token: float = 144.0,
    gpu_capacity_gb: float = 24.0,
    cpu_capacity_gb: float = 384.0,
    eviction_policy: str = "lru",
    enable_artesia: bool = False,
    tokenizer_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    simulator = ArtesiaSimulator(
        SimulatorConfig(
            cpu_gpu_bandwidth_gbps=cpu_gpu_bandwidth_gbps,
            gpu_sglang_bandwidth_gbps=gpu_sglang_bandwidth_gbps,
            prefill_throughput_tps=prefill_throughput_tps,
            decode_throughput_tps=decode_throughput_tps,
            decode_max_concurrency=decode_max_concurrency,
            kv_cache_kb_per_token=kv_cache_kb_per_token,
            gpu_capacity_gb=gpu_capacity_gb,
            cpu_capacity_gb=cpu_capacity_gb,
            eviction_policy=eviction_policy,
            enable_artesia=enable_artesia,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
    )

    uvicorn.run(
        create_app(simulator),
        host=host,
        port=port,
        workers=workers,
        loop="asyncio",
        log_level="info",
        access_log=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50350)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-gpu-bandwidth-gbps", type=float, default=12.0)
    parser.add_argument("--gpu-sglang-bandwidth-gbps", type=float, default=24.0)
    parser.add_argument("--prefill-throughput-tps", type=float, default=8000.0)
    parser.add_argument("--decode-throughput-tps", type=float, default=2000.0)
    parser.add_argument("--decode-max-concurrency", type=int, default=1)
    parser.add_argument("--kv-cache-kb-per-token", type=float, default=144.0)
    parser.add_argument("--gpu-capacity-gb", type=float, default=24.0)
    parser.add_argument("--cpu-capacity-gb", type=float, default=384.0)
    parser.add_argument("--eviction-policy", choices=["lru", "mru"], default="lru")
    parser.add_argument("--enable-artesia", action="store_true")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Optional tokenizer name/path. If omitted, use request.model as the tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config_from_args(args)
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        cpu_gpu_bandwidth_gbps=config.cpu_gpu_bandwidth_gbps,
        gpu_sglang_bandwidth_gbps=config.gpu_sglang_bandwidth_gbps,
        prefill_throughput_tps=config.prefill_throughput_tps,
        decode_throughput_tps=config.decode_throughput_tps,
        decode_max_concurrency=config.decode_max_concurrency,
        kv_cache_kb_per_token=config.kv_cache_kb_per_token,
        gpu_capacity_gb=config.gpu_capacity_gb,
        cpu_capacity_gb=config.cpu_capacity_gb,
        eviction_policy=config.eviction_policy,
        enable_artesia=config.enable_artesia,
        tokenizer_name=config.tokenizer_name,
        seed=config.seed,
    )
