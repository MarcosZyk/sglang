# Artesia + GLM-4.7（SGLang v0.5.7）

本目录对应分支 `artesia-glm47-v0.5.7`。该分支基于 SGLang
`v0.5.7`，保留 Artesia KV Cache、PD Prefill 路径及
`prefill_time`、`decode_time`、`artesia_time` trace。

## 依赖

不要复用旧 `v0.4.9` 环境。`v0.5.7` 的关键依赖包括：

- PyTorch `2.9.1`
- Transformers `4.57.1`
- sgl-kernel `0.3.20`
- FlashInfer `0.5.3`

当前旧环境中的 Transformers `4.53.2` 无法识别 GLM-4.7 的
`glm4_moe` 配置。建议使用独立虚拟环境或 SGLang v0.5.7 官方镜像。

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
