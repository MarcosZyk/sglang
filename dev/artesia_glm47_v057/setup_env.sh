#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SGLANG_VENV="${SGLANG_VENV:-/sgl-workspace/sglang/.venv}"
ARTESIA_ROOT="${ARTESIA_ROOT:-/artesia-workspace/artesia}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CUDA_INDEX="${CUDA_INDEX:-cu128}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
MAX_JOBS="${MAX_JOBS:-8}"

if ! command -v uv >/dev/null 2>&1; then
    echo "未找到 uv，请先安装 uv。" >&2
    exit 1
fi

if [[ ! -f "${ARTESIA_ROOT}/python/pyproject.toml" ]]; then
    echo "未找到 Artesia Python 源码：${ARTESIA_ROOT}/python" >&2
    exit 1
fi

if [[ ! -f "${ARTESIA_ROOT}/kernel/pyproject.toml" ]]; then
    echo "未找到 Artesia kernel 源码：${ARTESIA_ROOT}/kernel" >&2
    exit 1
fi

if [[ ! -x "${SGLANG_VENV}/bin/python" ]]; then
    uv venv --python "${PYTHON_BIN}" "${SGLANG_VENV}"
fi

VENV_PYTHON="${SGLANG_VENV}/bin/python"
UV_INSTALL=(
    uv pip install
    --python "${VENV_PYTHON}"
    --index-strategy unsafe-best-match
    --prerelease allow
)

"${UV_INSTALL[@]}" --upgrade pip setuptools wheel
"${UV_INSTALL[@]}" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_INDEX}" \
    -e "${REPO_ROOT}/python"

# Mooncake is required by the PD transfer backend but is not a core SGLang
# dependency. v0.3.8 is the version used by the v0.5.7 CI environment.
"${UV_INSTALL[@]}" \
    mooncake-transfer-engine==0.3.8 \
    matplotlib \
    pytest \
    pytest-asyncio

# Install the Python client normally and rebuild the C++/CUDA extension against
# the Torch version in this venv. Do not use editable mode for the kernel:
# editable builds would overwrite the .so files used by the legacy environment.
"${UV_INSTALL[@]}" "${ARTESIA_ROOT}/python"
"${UV_INSTALL[@]}" -e "${ARTESIA_ROOT}/context-cake"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
    "${UV_INSTALL[@]}" \
    --no-build-isolation \
    --force-reinstall \
    --no-deps \
    "${ARTESIA_ROOT}/kernel"

SGLANG_VENV="${SGLANG_VENV}" "${SCRIPT_DIR}/check_env.sh"
