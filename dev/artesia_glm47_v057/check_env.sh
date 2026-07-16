#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

"${SGLANG_PYTHON}" - <<'PY'
from importlib.metadata import version

import torch
from transformers import AutoConfig

import artesia
import artesia_kernel.gpu_ops
import flashinfer
import sgl_kernel
import sglang

expected = {
    "torch": "2.9.1",
    "transformers": "4.57.1",
    "sgl-kernel": "0.3.20",
    "flashinfer-python": "0.5.3",
    "mooncake-transfer-engine": "0.3.8",
    "context-cake": "0.2.0",
}

for package, expected_version in expected.items():
    actual = version(package)
    if not actual.startswith(expected_version):
        raise RuntimeError(
            f"{package} 版本错误：期望 {expected_version}，实际 {actual}"
        )
    print(f"{package}: {actual}")

if not torch.cuda.is_available():
    raise RuntimeError("Torch 无法访问 CUDA GPU")

print(f"cuda: {torch.version.cuda}")
print(f"gpu_count: {torch.cuda.device_count()}")
for rank in range(torch.cuda.device_count()):
    print(
        f"gpu[{rank}]: {torch.cuda.get_device_name(rank)}, "
        f"capability={torch.cuda.get_device_capability(rank)}"
    )

# Verify that the Transformers version understands the GLM-4 MoE config type
# without downloading a model.
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

if "glm4_moe" not in CONFIG_MAPPING:
    raise RuntimeError("Transformers 无法识别 glm4_moe")

print("glm4_moe: supported")
print("SGLang v0.5.7 + Artesia 环境检查通过")
PY
