# Load Gen 运行说明

本文档说明如何启动：

- `server/main_art.py`：接收 agent 请求的服务端
- `client/load_generator.py`：负载发生器

同时说明如何修改 `main_art.py` 使用的 ContextCake 地址。

## 目录说明

- `server/main_art.py`：服务端入口
- `server/contextcake_client.py`：ContextCake HTTP 客户端与默认 URL 配置
- `client/load_generator.py`：负载发生器入口
- `client/generate_payload.py`：把 replay JSON 转成 `main_art.py` 所需的请求格式
- `output_aone_flattened/test1.json`：当前 `load_generator.py` 默认读取的 replay 文件

## 前置条件

- Python 环境已安装 `fastapi`、`uvicorn`、`requests`、`transformers`、`numpy`
- 已安装带 `context_id` 扩展的 `openai-python`
- ContextCake 服务已启动，并且 `main_art.py` 可以访问到它

## 启动 `main_art.py`

推荐在 `server/` 目录下直接启动：

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/server
python main_art.py --host 0.0.0.0 --port 12306 --workers 1
```

如果要显式指定 ContextCake 地址：

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/server
python main_art.py \
  --host 0.0.0.0 \
  --port 12306 \
  --workers 1 \
  --contextcake-base-url http://127.0.0.1:50350
```

启动后可检查健康状态：

```bash
curl http://127.0.0.1:12306/api/health
```

## 启动 Load Generator

推荐在 `client/` 目录下直接启动。当前脚本默认读取 `../output_aone_flattened/test1.json`，因此工作目录应保持在 `client/`：

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/client
python load_generator.py \
  --url http://127.0.0.1:12306 \
  --rps 1 \
  --requests 1 \
  --model Qwen/Qwen3-8B
```

常用参数：

- `--url`：`main_art.py` 服务地址
- `--rps`：发送速率
- `--requests`：总请求数
- `--model`：会同时写入 `tokenizer_name` 和 `openai_model`

## 修改 ContextCake URL

### 方法 1：启动参数传入

这是最推荐的方式，只影响当前进程：

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/server
python main_art.py --contextcake-base-url http://10.0.0.8:50350
```

### 方法 2：环境变量

`main_art.py` 会读取环境变量 `CONTEXTCAKE_BASE_URL`：

```bash
export CONTEXTCAKE_BASE_URL=http://10.0.0.8:50350
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/server
python main_art.py --host 0.0.0.0 --port 12306
```

### 方法 3：修改默认值

如果你希望长期改默认地址，可以直接修改：

- `sglang/dev/scripts/load_gen/server/contextcake_client.py`

把这行里的默认值改掉：

```python
os.getenv("CONTEXTCAKE_BASE_URL", "http://localhost:50350")
```

## 推荐启动顺序

1. 启动 ContextCake
2. 启动 `main_art.py`
3. 用 `curl /api/health` 确认服务正常
4. 启动 `load_generator.py`

## 常见问题

### 1. `load_generator.py` 找不到 `test1.json`

请确认当前目录是：

```bash
/home/khfu/gitkv/sglang/dev/scripts/load_gen/client
```

因为默认路径是相对路径：

```python
../output_aone_flattened/test1.json
```

### 2. `main_art.py` 连不上 ContextCake

优先检查：

- `--contextcake-base-url` 是否正确
- `CONTEXTCAKE_BASE_URL` 是否被旧值覆盖
- ContextCake 是否真的监听了对应的 `ip:port`

### 3. 只想验证链路，不想真正跑 GPU 推理

可以保留 `main_art.py + load_generator.py` 的启动方式不变，只把 ContextCake 或 OpenAI completion 调用替换成测试桩。当前目录下的运行说明不限制你必须接真实 LLM serving。
