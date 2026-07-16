#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SGLANG_VENV="${SGLANG_VENV:-/sgl-workspace/sglang/.venv}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${SGLANG_VENV}/bin/python}"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
    echo "SGLang v0.5.7 环境不存在：${SGLANG_PYTHON}" >&2
    echo "请先执行 ${SCRIPT_DIR}/setup_env.sh" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
