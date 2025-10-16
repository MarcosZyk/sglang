#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Small helpers
log() { printf '\n[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }
err() { printf '\n[ERROR %s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*" >&2; }
run_with_retries() {
    local cmd="$1"; shift
    local tries=${1:-3}; shift || true
    local backoff=${1:-2}; shift || true

    local i=0
    until [ "$i" -ge "$tries" ]; do
        if eval "$cmd"; then
            return 0
        fi
        i=$((i + 1))
        log "Command failed, retry $i/$tries: $cmd"
        sleep $((backoff * i))
    done
    return 1
}

# On error restore backed-up files if needed and show helpful info
BACKUP_PYPROJECT=""
cleanup() {
    local exit_code=$?
    if [ -n "$BACKUP_PYPROJECT" ] && [ -f "$BACKUP_PYPROJECT" ]; then
        log "Restoring backed up pyproject.toml"
        mv -f "$BACKUP_PYPROJECT" "$(dirname "$BACKUP_PYPROJECT")/pyproject.toml" || true
    fi
    if [ $exit_code -ne 0 ]; then
        err "Script failed (exit code $exit_code). See messages above."
    else
        log "Script completed successfully."
    fi
    exit $exit_code
}
trap cleanup EXIT

# Ensure conda is available and initialize it for this shell
if ! command -v conda >/dev/null 2>&1; then
    err "conda not found on PATH. Please install Miniconda/Anaconda or ensure conda is available."
    exit 2
fi

# Source conda.sh so `conda activate` works in non-interactive shells
# Source conda and activate the environment in one step so the activation is
# visible to the rest of this script. Using `conda info --base` avoids relying
# on the CONDA_BASE variable being correct earlier.
# shellcheck disable=SC2086
if ! source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" >/dev/null 2>&1 || ! command -v conda >/dev/null 2>&1; then
    err "Unable to source conda base or `conda` not available. Ensure conda is installed and on PATH."
    exit 2
fi

ENV_NAME="sglamx-dong1"
PYTHON_VER="3.12"

log "Creating or reusing conda env: $ENV_NAME (python=$PYTHON_VER)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda env $ENV_NAME already exists."
else
    run_with_retries "conda create -n $ENV_NAME -y python=$PYTHON_VER" || { err "Failed to create env with conda"; exit 3; }
fi

log "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Common pip args for your private mirrors
PIP_INDEX="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://mirrors.aliyun.com/pypi/simple"

log "Installing dependencies via conda (libsqlite gperftools tbb libnuma numactl)"
run_with_retries "conda install -y libsqlite==3.48.0 gperftools tbb libnuma numactl" || { err "Conda install failed"; exit 4; }

log "Installing intel-openmp (with retries)"
run_with_retries "pip install intel-openmp $PIP_INDEX" 4 3 || { err "pip install intel-openmp failed"; exit 5; }

#
# Install sglang (python) with CPU extras
#
PY_DIR="python"
SRC_PYPROJECT="$PY_DIR/pyproject_other.toml"
TARGET_PYPROJECT="$PY_DIR/pyproject.toml"

if [ ! -d "$PY_DIR" ]; then
    err "Directory $PY_DIR not found."
    exit 6
fi
if [ ! -f "$SRC_PYPROJECT" ]; then
    err "$SRC_PYPROJECT not found."
    exit 6
fi

# Backup existing pyproject.toml if present
if [ -f "$TARGET_PYPROJECT" ]; then
    BACKUP_PYPROJECT="${TARGET_PYPROJECT}.bak.$(date +%s)"
    log "Backing up existing $TARGET_PYPROJECT -> $BACKUP_PYPROJECT"
    mv "$TARGET_PYPROJECT" "$BACKUP_PYPROJECT"
else
    BACKUP_PYPROJECT=""
fi

log "Preparing pyproject for CPU install and running pip install"
cp "$SRC_PYPROJECT" "$TARGET_PYPROJECT"
run_with_retries "pip install -v -e \"${PY_DIR}[all_cpu]\" $PIP_INDEX" 4 3 || { err "pip install sglang failed"; exit 7; }

# If installation succeeded, remove backup
if [ -n "$BACKUP_PYPROJECT" ] && [ -f "$BACKUP_PYPROJECT" ]; then
    log "Removing backup $BACKUP_PYPROJECT"
    rm -f "$BACKUP_PYPROJECT" || true
    BACKUP_PYPROJECT=""
fi

#
# Install sgl-kernel with CPU support
#
KERNEL_DIR="sgl-kernel"
if [ -d "$KERNEL_DIR" ]; then
    pushd "$KERNEL_DIR" >/dev/null
    if [ ! -f "pyproject_cpu.toml" ]; then
        err "pyproject_cpu.toml not found in $KERNEL_DIR"
        popd >/dev/null
        exit 8
    fi

    # Backup/replace pyproject.toml safely
    if [ -f "pyproject.toml" ]; then
        KBACKUP="pyproject.toml.bak.$(date +%s)"
        mv pyproject.toml "$KBACKUP"
    else
        KBACKUP=""
    fi
    cp pyproject_cpu.toml pyproject.toml

    run_with_retries "pip install -v . $PIP_INDEX" 4 3 || { err "pip install sgl-kernel failed"; popd >/dev/null; exit 9; }

    # cleanup kernel backup
    if [ -n "$KBACKUP" ]; then
        rm -f "$KBACKUP" || true
    fi
    popd >/dev/null
else
    log "Directory $KERNEL_DIR not found, skipping sgl-kernel install."
fi

pip list | grep "torch\|sglang\|sgl-kernel"

echo "To activate the env in your current shell, run:"
echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate $ENV_NAME"

# success: disable trap restoring backups
trap - EXIT
cleanup 0