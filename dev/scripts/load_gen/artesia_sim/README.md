# Artesia 仿真服务端

这个目录提供一个单进程、单服务的 Artesia 仿真后端，同时承接：

- 控制面接口：`create_context / delete_context / truncate / one_off / suspend`
- OpenAI 兼容接口：`POST /v1/chat/completions`

它只模拟时间，不做真实 LLM 推理，也不真实保存 KV。所有状态都只按 message 粒度记录：

- 一条 message 要么完整在 GPU
- 要么完整在 CPU
- 不支持部分 offload / 部分 fetch

默认运行在 non-Artesia 保留模式下；只有显式传入 `--enable-artesia` 时，才启用完整 Artesia 语义。

## 启动

建议直接在本目录启动，并保持 `--workers 1`，因为状态完全在单进程内存里：

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/artesia_sim
python main.py \
  --host 0.0.0.0 \
  --port 50350 \
  --workers 1 \
  --cpu-gpu-bandwidth-gbps 3.5 \
  --gpu-sglang-bandwidth-gbps 490 \
  --prefill-throughput-tps 31000 \
  --decode-throughput-tps 65 \
  --kv-cache-kb-per-token 144 \
  --gpu-capacity-gb 100 \
  --tokenizer /artesia-workspace/models/GLM-4.7 \
  --enable-artesia \
  --eviction-policy lru
```

## 与 `main_art.py` 联调

`main_art.py` 不需要改逻辑，只要把 `--contextcake-base-url` 指向这个服务：

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/server
python main_art.py \
  --host 0.0.0.0 \
  --port 12306 \
  --workers 1 \
  --contextcake-base-url http://127.0.0.1:50350
```

`main_art.py` 通过：

- `PUT /context/{context_id}`
- `DELETE /context/{context_id}`
- `POST /truncate`
- `POST /one-off`
- `POST /suspend`
- `POST /v1/chat/completions`

访问本服务。

## OpenAI 接口说明

- 如果启动时传了 `--tokenizer`，则统一使用该 tokenizer
- 如果没有传 `--tokenizer`，则回退到请求里的 `model` 作为 tokenizer 名称
- `context_id` 必填，且必须先由 `create_context` 创建
- `stream` 不支持
- 返回字段兼容 `openai-python` 的 chat completion，并额外包含：
  - `prefill_time`
  - `decode_time`
  - `num_local_cache`
  - `num_global_cache`

## 控制面语义

- `create_context`：创建空 context，后续新增 message 默认 `durable`
- `delete_context`：
  - `--enable-artesia` 时删除整个 context 并释放 GPU/CPU 占用
  - 默认模式下返回成功但不删除内容
- `truncate`：
  - `--enable-artesia` 时保留 `[0, msg_index)` 并删除后缀
  - 默认模式下保留 `[0, msg_index)`，把后缀移入内部 archive context
- `one_off`：
  - `--enable-artesia` 时仅保留下次 generate 完成后的 `[0, msg_index)`
  - 默认模式下把后缀移入内部 archive context，active context 仅保留 `[0, msg_index)`
- `suspend`：
  - `--enable-artesia` 时把当前 message 状态设为 `suspend`
  - 默认模式下为 no-op

## 运行测试

```bash
cd /home/khfu/gitkv/sglang/dev/scripts/load_gen/artesia_sim
pytest -q test_artesia_sim.py
```
