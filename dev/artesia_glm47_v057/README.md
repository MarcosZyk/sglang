# Artesia + GLM-4.7（SGLang v0.5.7）

本目录对应分支 `artesia-v0.5.7`。该分支基于 SGLang
`v0.5.7`，保留 Artesia KV Cache、PD Prefill 路径及
`prefill_time`、`decode_time`、`artesia_time` trace。

## 依赖

不要复用旧 `v0.4.9` 环境。`v0.5.7` 的关键依赖包括：

- PyTorch `2.9.1`
- Transformers `4.57.1`
- sgl-kernel `0.3.20`
- FlashInfer `0.5.3`

当前旧环境中的 Transformers `4.53.2` 无法识别 GLM-4.7 的
`glm4_moe` 配置。不要直接升级旧环境。

执行下面的命令创建独立环境：

```bash
./dev/artesia_glm47_v057/setup_env.sh
```

默认环境路径是 `/sgl-workspace/sglang/.venv`。如需放到其他位置：

```bash
SGLANG_VENV=/path/to/venv \
./dev/artesia_glm47_v057/setup_env.sh
```

安装脚本会：

- 安装 SGLang v0.5.7 固定版本的 Torch、Transformers、
  sgl-kernel 和 FlashInfer；
- 安装 PD 传输需要的 Mooncake Transfer Engine `0.3.8`；
- 安装 Artesia Python 客户端；
- 针对新 Torch 和当前 GPU 重新编译 Artesia C++/CUDA kernel；
- 自动执行 GPU、版本和 `glm4_moe` 支持检查。

以后可以单独检查环境：

```bash
./dev/artesia_glm47_v057/check_env.sh
```

启动脚本默认使用该虚拟环境。也可以通过 `SGLANG_VENV` 指定其他环境。

## 启动

先启动 Artesia，再分别在 P、D 节点执行：

```bash
./dev/artesia_glm47_v057/launch_server_p.sh
./dev/artesia_glm47_v057/launch_server_d.sh
```

P、D 实例健康后启动测试负载均衡器：

```bash
./dev/artesia_glm47_v057/launch_lb.sh
```

只有 P 实例启用 `--enable-artesia`。D 实例不会连接或读写 Artesia。
