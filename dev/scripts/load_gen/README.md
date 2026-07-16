# 负载生成实验运行说明

本文档说明如何启动完整的 Artesia + SGLang PD 分离测试链路，以及测试结束后如何清理进程。

完整请求链路如下，其中括号内为程序或组件的原始名称：

```text
负载生成器（load_generator.py）
  -> 测试服务（main_art.py）
  -> 上下文管理服务（ContextCake）
  -> SGLang 预填充/解码路由器
  -> SGLang 预填充实例和解码实例
  -> Artesia 控制面和数据面
```

以下命令默认：

- SGLang 仓库位于 `/sgl-workspace/sglang`
- Artesia 仓库位于 `/artesia-workspace/artesia`
- 模型位于 `/artesia-workspace/models/GLM-4.7`
- P、D 和负载均衡器可以运行在不同节点

ContextCake、测试 server 和 load generator 可以部署在同一节点；跨节点地址通过启动脚本环境变量配置。

GLM-4.7 模型目录约为 668GB，单卡通常无法容纳。本文中的 GPU 编号仅作为命令格式示例；实际启动时需要根据机器数量、单卡显存和模型并行方案配置 `CUDA_VISIBLE_DEVICES`、`--tp-size` 以及 P/D 实例所在节点。多机部署可以参考 `dev/distributed/` 下的启动脚本。

## 1. 目录与默认端口

| 服务 | 默认端口 | 说明 |
|---|---:|---|
| Artesia 控制面 | 50250 | Artesia 元数据和控制服务 |
| SGLang 预填充实例 | 30000 | 预填充/解码分离的 P 实例 |
| SGLang 解码实例 | 每个 D 节点 30000 | 预填充/解码分离的 D 实例 |
| SGLang 预填充/解码路由器 | 12347 | ContextCake 访问的推理入口 |
| ContextCake 上下文管理服务 | 50350 | 上下文管理和 OpenAI 兼容接口 |
| `main_art.py` | 12306 | Load generator 访问的测试服务 |

负载生成工具目录：

```text
dev/scripts/load_gen/
├── client/load_generator.py
├── server/main_art.py
├── output_json_flatten/
├── result/
└── scripts/
```

## 2. 启动前准备

进入 SGLang 工作目录：

```bash
cd /sgl-workspace/sglang
```

建议为本次实验创建日志和进程号目录：

```bash
mkdir -p /tmp/load_gen_logs /tmp/load_gen_pids
```

如果上一次 Artesia 运行留下共享内存或套接字，可先执行仓库中已有的清理脚本；具体脚本位置取决于 Artesia 的安装目录。

检查实验端口是否已经被占用：

```bash
ss -ltnp | rg ':(50250|30000|12347|50350|12306)\b'
```

## 3. 启动 Artesia

Artesia 包含控制面（ControlPlane）和数据面（DataPlane）。先启动控制面，再启动数据面。

### 3.1 启动控制面

```bash
cd /sgl-workspace/sglang/dev/scripts

nohup bash launch_control_plane.sh \
  > /tmp/load_gen_logs/artesia_control_plane.log 2>&1 &

echo $! > /tmp/load_gen_pids/artesia_control_plane.pid
```

对应脚本实际运行：

```bash
python -m artesia.control_plane.launch_server \
  --page-bytes-size 32M \
  --enable-trace False
```

控制面默认监听 `127.0.0.1:50250`。

查看日志：

```bash
tail -f /tmp/load_gen_logs/artesia_control_plane.log
```

### 3.2 启动数据面

```bash
cd /sgl-workspace/sglang/dev/scripts

nohup bash launch_data_plane.sh \
  > /tmp/load_gen_logs/artesia_data_plane.log 2>&1 &

echo $! > /tmp/load_gen_pids/artesia_data_plane.pid
```

当前脚本使用 `mpirun -n 2` 启动两个 GPU 数据面进程，并配置：

```text
mem-size:   60G
chunk-size: 256M
```

查看日志：

```bash
tail -f /tmp/load_gen_logs/artesia_data_plane.log
```

确认所有数据面进程均已成功启动后，再启动 SGLang。

## 4. 启动 SGLang PD 分离实例

当前本机测试脚本位于：

```text
dev/artesia_glm47_v057/launch_server_p.sh
dev/artesia_glm47_v057/launch_server_d.sh
dev/artesia_glm47_v057/launch_lb.sh
```

P 实例启用了 `--enable-artesia`，D 实例不与 Artesia 交互。

### 4.1 启动预填充实例

在 P 节点启动，默认使用本机 GPU `0,1,2,3` 和端口 `30000`：

```bash
cd /sgl-workspace/sglang

nohup bash dev/artesia_glm47_v057/launch_server_p.sh \
  > /tmp/load_gen_logs/sglang_prefill.log 2>&1 &

echo $! > /tmp/load_gen_pids/sglang_prefill.pid
```

检查健康状态：

```bash
curl http://127.0.0.1:30000/health
```

### 4.2 启动解码实例

分别在两个 D 节点启动。每个节点默认使用本机 GPU `0,1,2,3`
和端口 `30000`：

```bash
cd /sgl-workspace/sglang

nohup bash dev/artesia_glm47_v057/launch_server_d.sh \
  > /tmp/load_gen_logs/sglang_decode.log 2>&1 &
echo $! > /tmp/load_gen_pids/sglang_decode.pid
```

在每个 D 节点检查健康状态：

```bash
curl http://127.0.0.1:30000/health
```

如果 D 实例数量或地址不同，通过 `DECODE_URL_1`、`DECODE_URL_2`
环境变量配置负载均衡器。

### 4.3 启动预填充/解码路由器

所有 P/D 实例健康后启动路由器：

```bash
cd /sgl-workspace/sglang

PREFILL_URL=http://10.87.79.111:30000 \
DECODE_URL_1=http://10.87.79.112:30000 \
DECODE_URL_2=http://10.87.79.113:30000 \
nohup bash dev/artesia_glm47_v057/launch_lb.sh \
  > /tmp/load_gen_logs/sglang_pd_router.log 2>&1 &

echo $! > /tmp/load_gen_pids/sglang_pd_router.pid
```

路由器默认监听：

```text
http://127.0.0.1:12347
```

检查：

```bash
curl http://127.0.0.1:12347/health
```

## 5. 启动 ContextCake

ContextCake 连接：

- Artesia 控制面：`127.0.0.1:50250`
- SGLang 预填充/解码路由器：`127.0.0.1:12347`

启动命令：

```bash
cd /artesia-workspace/artesia/context-cake

nohup python -m context_cake.launch \
  --host 0.0.0.0 \
  --port 50350 \
  --artesia-address 127.0.0.1:50250 \
  --sglang-address 127.0.0.1:12347 \
  --model-name /artesia-workspace/models/GLM-4.7 \
  --tokenizer-model-path /artesia-workspace/models/GLM-4.7 \
  --sglang-timeout 3600 \
  --artesia-timeout 3600 \
  > /tmp/load_gen_logs/context_cake.log 2>&1 &

echo $! > /tmp/load_gen_pids/context_cake.pid
```

检查健康状态：

```bash
curl http://127.0.0.1:50350/health
```

查看日志：

```bash
tail -f /tmp/load_gen_logs/context_cake.log
```

## 6. 启动测试服务

`main_art.py` 接收负载生成器发出的智能体请求，并调用 ContextCake。

推荐直接使用一键脚本。该脚本只启动 `main_art.py`，不会启动 Artesia、SGLang 或 ContextCake：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/server

bash run_server.sh
```

可以通过环境变量调整配置：

```bash
CONTEXTCAKE_BASE_URL=http://127.0.0.1:50350 \
PORT=12306 \
MAX_CONCURRENT=32 \
HTTP_LIMIT_CONCURRENCY=500 \
RESULT_DIR=/sgl-workspace/sglang/dev/scripts/load_gen/result \
bash run_server.sh
```

需要后台运行时：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/server

nohup bash run_server.sh \
  > /tmp/load_gen_logs/main_art.log 2>&1 &

echo $! > /tmp/load_gen_pids/main_art.pid
```

等价的完整命令如下：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/server

nohup python main_art.py \
  --host 0.0.0.0 \
  --port 12306 \
  --workers 1 \
  --max-concurrent 32 \
  --http-limit-concurrency 500 \
  --contextcake-base-url http://127.0.0.1:50350 \
  --result-dir /sgl-workspace/sglang/dev/scripts/load_gen/result \
  > /tmp/load_gen_logs/main_art.log 2>&1 &

echo $! > /tmp/load_gen_pids/main_art.pid
```

注意：

- `--workers` 固定为 `1`，不能设置为其他值。
- `--max-concurrent` 控制同时执行的智能体请求数。
- `--http-limit-concurrency` 控制 Uvicorn 的 HTTP 请求并发上限。
- 每个记录请求会生成一个 CSV 文件。

检查服务：

```bash
curl http://127.0.0.1:12306/api/health
```

## 7. 启动负载生成器

### 7.1 使用单个回放 JSON 文件

推荐使用客户端一键脚本：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/client

bash run_client.sh
```

可以通过环境变量调整实验参数：

```bash
SERVER_URL=http://127.0.0.1:12306 \
MODEL_PATH=/artesia-workspace/models/GLM-4.7 \
RPS=0.1 \
REQUESTS=96 \
RECORDED_REQUESTS=60 \
MAX_CONCURRENT=100 \
SUPPORT_SUBMIT_WORKERS=8 \
JSON_FILE=/sgl-workspace/sglang/dev/scripts/load_gen/output_json_flatten/test1.json \
bash run_client.sh
```

等价的完整命令如下：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/client

nohup python load_generator.py \
  --url http://127.0.0.1:12306 \
  --rps 0.1 \
  --requests 96 \
  --recorded-requests 60 \
  --max-concurrent 100 \
  --support-submit-workers 8 \
  --model /artesia-workspace/models/GLM-4.7 \
  --json-file ../output_json_flatten/test1.json \
  > /tmp/load_gen_logs/load_generator.log 2>&1 &

echo $! > /tmp/load_gen_pids/load_generator.pid
```

参数说明：

- `--requests`：最多发送的请求数。
- `--recorded-requests`：从开头起记录并等待完成的请求数，默认 `60`。
- 前 `recorded-requests` 个请求写 CSV，并纳入客户端统计。
- 后续请求只维持 Artesia/KV Cache 压力，不写 CSV，也不纳入统计。
- 记录请求全部完成后，负载生成器停止发送，并通知 `main_art.py` 取消剩余压力请求。
- `--max-concurrent`：客户端记录请求的最大并发数。
- `--support-submit-workers`：提交压力请求的线程数。

查看运行进度：

```bash
tail -f /tmp/load_gen_logs/load_generator.log
```

### 7.2 混合两个回放 JSON 文件

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/client

python load_generator.py \
  --url http://127.0.0.1:12306 \
  --rps 0.1 \
  --requests 96 \
  --recorded-requests 60 \
  --max-concurrent 100 \
  --support-submit-workers 8 \
  --model /artesia-workspace/models/GLM-4.7 \
  --json-file-a ../output_json_flatten/test8.json \
  --json-file-b ../output_json_flatten/test10.json \
  --json-ratio 1:1 \
  --trace-seed 0
```

## 8. 查看测试结果

默认结果目录：

```bash
ls -lah /sgl-workspace/sglang/dev/scripts/load_gen/result
```

每个被记录的智能体请求对应一个 CSV 文件：

```text
1.csv
2.csv
3.csv
...
```

CSV 文件中包括：

- prefill token 数
- decode token 数
- `prefill_time`
- `artesia_time`
- decode 总时间
- 本地/全局缓存 token 数
- 等待时间

运行汇总分析：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/scripts

python analyse.py \
  --input-root /path/to/experiment/result_root
```

## 9. 停止和清理测试进程

建议按请求链路的反方向停止：

```text
负载生成器
-> 测试服务
-> ContextCake
-> 预填充/解码路由器
-> SGLang 解码/预填充实例
-> Artesia 数据面
-> Artesia 控制面
```

### 9.1 使用进程号文件停止

```bash
for name in \
  load_generator \
  main_art \
  context_cake \
  sglang_pd_router \
  sglang_decode_1 \
  sglang_decode_2 \
  sglang_decode_3 \
  sglang_prefill \
  artesia_data_plane \
  artesia_control_plane
do
  pid_file="/tmp/load_gen_pids/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    kill "${pid}" 2>/dev/null || true
  fi
done
```

等待几秒后检查：

```bash
ps -ef | rg 'load_generator.py|main_art.py|context_cake.launch|sglang.launch_server|launch_lb|mini_lb|artesia.control_plane|artesia.data_plane'
```

如果进程没有正常退出，再对明确的进程号使用：

```bash
kill -KILL <pid>
```

### 9.2 按命令名称清理

仅在确认机器上没有其他同类实验时使用：

```bash
pkill -f 'load_generator.py' || true
pkill -f 'main_art.py' || true
pkill -f 'context_cake.launch' || true
pkill -f 'sglang.srt.disaggregation.launch_lb' || true
pkill -f 'sglang.srt.disaggregation.mini_lb' || true
pkill -f 'sglang.launch_server' || true
pkill -f 'artesia.data_plane.launch_server' || true
pkill -f 'artesia.control_plane.launch_server' || true
```

数据面由 `mpirun` 启动时，还需要确认 `mpirun` 子进程是否已经退出：

```bash
ps -ef | rg 'mpirun|artesia.data_plane'
```

### 9.3 按端口检查和清理

检查残留监听：

```bash
ss -ltnp | rg ':(50250|30000|30001|30002|30003|12347|50350|12306)\b'
```

确认端口只属于本次实验后，可以执行：

```bash
fuser -k 50250/tcp
fuser -k 30000/tcp
fuser -k 30001/tcp
fuser -k 30002/tcp
fuser -k 30003/tcp
fuser -k 12347/tcp
fuser -k 50350/tcp
fuser -k 12306/tcp
```

最后根据 Artesia 环境需要清理残留共享内存或套接字。

## 10. 推荐启动顺序

完整顺序：

1. Artesia 控制面
2. Artesia 数据面
3. SGLang 预填充实例
4. SGLang 解码实例
5. SGLang 预填充/解码路由器
6. ContextCake 上下文管理服务
7. 测试服务 `main_art.py`
8. 负载生成器 `load_generator.py`

每一步都确认健康状态或日志正常后，再启动下一步。

## 11. Artesia 仿真模式

如果只测试负载生成逻辑，不运行真实 SGLang、ContextCake 和 Artesia，可以使用：

```text
dev/scripts/load_gen/artesia_sim/
```

启动示例：

```bash
cd /sgl-workspace/sglang/dev/scripts/load_gen/artesia_sim

python main.py \
  --host 0.0.0.0 \
  --port 50353 \
  --workers 1 \
  --cpu-gpu-bandwidth-gbps 3.5 \
  --gpu-sglang-bandwidth-gbps 490 \
  --prefill-throughput-tps 31000 \
  --decode-throughput-tps 65 \
  --decode-max-concurrency 9 \
  --kv-cache-kb-per-token 144 \
  --gpu-capacity-gb 196 \
  --tokenizer /artesia-workspace/models/GLM-4.7 \
  --enable-artesia \
  --eviction-policy lru
```

此模式下 `main_art.py` 的 `--contextcake-base-url` 应指向：

```text
http://127.0.0.1:50353
```
