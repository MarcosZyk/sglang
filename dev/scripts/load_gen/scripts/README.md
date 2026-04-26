# 对比实验启动说明

这个目录下按 replay JSON 分成两组启动脚本：

1. `1_json_script/`：4 个脚本，统一使用 `output_json_flatten/test1.json`
2. `8_json_script/`：4 个脚本，统一使用 `output_json_flatten/test8.json`

每个脚本都会用 `nohup` 在后台启动 3 个进程：

- `artesia_sim/main.py`
- `server/main_art.py`
- `client/load_generator.py`

## 启动前准备

建议先确认当前环境已经安装并可用：

- `python`
- `uvicorn`
- `fastapi`
- `openai`
- `transformers`
- `requests`
- `numpy`

建议在项目根目录执行：

```bash
cd /home/khfu/gitkv
```

## 8 个启动脚本

### 1-json：启用 Artesia，offload = LRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/1_json_script/start_compare_enable_artesia_lru.sh
```

### 1-json：启用 Artesia，offload = MRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/1_json_script/start_compare_enable_artesia_mru.sh
```

### 1-json：不启用 Artesia，offload = LRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/1_json_script/start_compare_disable_artesia_lru.sh
```

### 1-json：不启用 Artesia，offload = MRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/1_json_script/start_compare_disable_artesia_mru.sh
```

### 8-json：启用 Artesia，offload = LRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/8_json_script/start_compare_enable_artesia_lru.sh
```

### 8-json：启用 Artesia，offload = MRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/8_json_script/start_compare_enable_artesia_mru.sh
```

### 8-json：不启用 Artesia，offload = LRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/8_json_script/start_compare_disable_artesia_lru.sh
```

### 8-json：不启用 Artesia，offload = MRU

```bash
bash /home/khfu/gitkv/sglang/dev/scripts/load_gen/scripts/8_json_script/start_compare_disable_artesia_mru.sh
```

## 每个脚本会做什么

每个脚本都会：

- 创建对应的结果目录 `result-*`
- 创建对应的日志目录 `logs/<method>/<result_bucket>/`
- 启动独立端口的 `artesia_sim`
- 启动独立端口的 `main_art.py`
- 启动一个带 `--json-file` 参数的 `load_generator.py`

脚本启动后会在终端打印 3 个后台进程的 PID、对应日志文件路径，以及本次使用的 replay JSON。

## 端口、结果目录、日志目录对应关系

| 方法 | artesia_sim 端口 | main_art 端口 | 结果目录 | 日志目录 |
|------|------------------|---------------|----------|----------|
| `enable-artesia-lru` | `50360` | `12310` | `result-enable-artesia-lru/` | `logs/enable-artesia-lru/` |
| `enable-artesia-mru` | `50361` | `12311` | `result-enable-artesia-mru/` | `logs/enable-artesia-mru/` |
| `disable-artesia-lru` | `50362` | `12312` | `result-disable-artesia-lru/` | `logs/disable-artesia-lru/` |
| `disable-artesia-mru` | `50363` | `12313` | `result-disable-artesia-mru/` | `logs/disable-artesia-mru/` |

目录都位于：

```bash
/home/khfu/gitkv/sglang/dev/scripts/load_gen/
```

## 查看日志

例如查看 `enable-artesia-lru` 的 3 个日志：

```bash
tail -f /home/khfu/gitkv/sglang/dev/scripts/load_gen/logs/enable-artesia-lru/result_128G_8/artesia_sim.log
tail -f /home/khfu/gitkv/sglang/dev/scripts/load_gen/logs/enable-artesia-lru/result_128G_8/main_art.log
tail -f /home/khfu/gitkv/sglang/dev/scripts/load_gen/logs/enable-artesia-lru/result_128G_8/load_generator.log
```

## 查看结果

例如查看 `disable-artesia-mru` 的结果目录：

```bash
ls -la /home/khfu/gitkv/sglang/dev/scripts/load_gen/result_128G_8/result-disable-artesia-mru
```

`main_art.py` 会把每个 agent 请求的结果写成单独的 CSV 文件。

## 停止进程

脚本启动时会打印 PID。可以直接按 PID 停止，例如：

```bash
kill <artesia_sim_pid> <main_art_pid> <load_generator_pid>
```

如果只想按端口停止，也可以用：

```bash
fuser -k 50360/tcp
fuser -k 12310/tcp
```

不同方法请替换成对应端口。

## 说明

- 8 个脚本可以分别单独运行。
- `1_json_script/` 固定传入 `test1.json`，`8_json_script/` 固定传入 `test8.json`。
- 如果要同时运行多组对比，请先检查端口是否冲突。
- 每组脚本里的其它超参数仍然写在对应的 `.sh` 文件中，可以直接编辑。
