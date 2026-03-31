#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/artesia/python:${REPO_ROOT}/artesia/kernel/python:${REPO_ROOT}/sglang/python"

: "${ARTESIA_PYTHON_BIN:=python}"
: "${ARTESIA_DEVICE:=cuda}"
: "${ARTESIA_GPUS_PER_NODE:=4}"
: "${ARTESIA_WORLD_SIZE:=4}"
: "${ARTESIA_NODE_INDEX:=0}"
: "${ARTESIA_NODE_IP:=127.0.0.1}"
: "${ARTESIA_CONTROL_PLANE_IP:=127.0.0.1}"
: "${ARTESIA_CONTROL_PLANE_PORT:=50250}"
: "${ARTESIA_PAGE_BYTES_SIZE:=32M}"
: "${ARTESIA_MEM_SIZE:=60G}"
: "${ARTESIA_CHUNK_SIZE:=256M}"
: "${ARTESIA_BASE_C2P_PORT:=50150}"
: "${ARTESIA_BASE_P2P_PORT:=50050}"
: "${ARTESIA_ENABLE_TRACE:=False}"
: "${ARTESIA_LOG_DIR:=${SCRIPT_DIR}/artesia-logs}"
: "${PYTHONPATH:=${DEFAULT_PYTHONPATH}}"

mkdir -p "${ARTESIA_LOG_DIR}"

artesia_repo_python() {
    (
        cd "${REPO_ROOT}"
        export PYTHONPATH
        "${ARTESIA_PYTHON_BIN}" "$@"
    )
}

assert_port_bindable() {
    local host="$1"
    local port="$2"
    local label="$3"

    if ! "${ARTESIA_PYTHON_BIN}" - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except OSError as exc:
    print(exc)
    raise SystemExit(1)
finally:
    sock.close()
PY
    then
        echo "Port preflight failed for ${label} on ${host}:${port}" >&2
        if command -v lsof >/dev/null 2>&1; then
            echo "Current listeners on TCP port ${port}:" >&2
            lsof -nP -iTCP:"${port}" -sTCP:LISTEN >&2 || true
        fi
        exit 1
    fi
}

preflight_control_plane_port() {
    assert_port_bindable \
        "${ARTESIA_CONTROL_PLANE_IP}" \
        "${ARTESIA_CONTROL_PLANE_PORT}" \
        "ControlPlane"
}

preflight_local_data_plane_ports() {
    local local_rank
    local global_rank
    local c2p_port
    local p2p_port
    for ((local_rank = 0; local_rank < ARTESIA_GPUS_PER_NODE; local_rank++)); do
        global_rank=$((ARTESIA_NODE_INDEX * ARTESIA_GPUS_PER_NODE + local_rank))
        c2p_port=$((ARTESIA_BASE_C2P_PORT + global_rank))
        p2p_port=$((ARTESIA_BASE_P2P_PORT + global_rank))
        assert_port_bindable "${ARTESIA_NODE_IP}" "${c2p_port}" "DataPlane C2P rank ${global_rank}"
        assert_port_bindable "${ARTESIA_NODE_IP}" "${p2p_port}" "DataPlane P2P rank ${global_rank}"
    done
}

wait_for_tcp_ready() {
    local host="$1"
    local port="$2"
    local label="$3"
    local timeout_secs="${4:-30}"
    local elapsed=0

    while ((elapsed < timeout_secs)); do
        if "${ARTESIA_PYTHON_BIN}" - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
        then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo "${label} did not become reachable on ${host}:${port} within ${timeout_secs}s." >&2
    return 1
}

wait_for_control_plane_ready() {
    wait_for_tcp_ready \
        "${ARTESIA_CONTROL_PLANE_IP}" \
        "${ARTESIA_CONTROL_PLANE_PORT}" \
        "ControlPlane"
}

launch_control_plane() {
    local log_file="${1:-${ARTESIA_LOG_DIR}/control-plane.log}"
    (
        cd "${REPO_ROOT}"
        export PYTHONPATH
        "${ARTESIA_PYTHON_BIN}" -m artesia.control_plane.launch_server \
            --ip "${ARTESIA_CONTROL_PLANE_IP}" \
            --port "${ARTESIA_CONTROL_PLANE_PORT}" \
            --world-size "${ARTESIA_WORLD_SIZE}" \
            --page-bytes-size "${ARTESIA_PAGE_BYTES_SIZE}" \
            --enable-trace "${ARTESIA_ENABLE_TRACE}" \
            >"${log_file}" 2>&1
    ) &
    echo $!
}

launch_data_plane_rank() {
    local local_rank="$1"
    local global_rank=$((ARTESIA_NODE_INDEX * ARTESIA_GPUS_PER_NODE + local_rank))
    local c2p_port=$((ARTESIA_BASE_C2P_PORT + global_rank))
    local p2p_port=$((ARTESIA_BASE_P2P_PORT + global_rank))
    local log_file="${ARTESIA_LOG_DIR}/data-plane-rank-${global_rank}.log"

    (
        cd "${REPO_ROOT}"
        export PYTHONPATH
        "${ARTESIA_PYTHON_BIN}" -m artesia.data_plane.launch_server \
            --device "${ARTESIA_DEVICE}" \
            --ip "${ARTESIA_NODE_IP}" \
            --c2p-port "${c2p_port}" \
            --p2p-port "${p2p_port}" \
            --control-plane-ip "${ARTESIA_CONTROL_PLANE_IP}" \
            --control-plane-port "${ARTESIA_CONTROL_PLANE_PORT}" \
            --global-rank "${global_rank}" \
            --local-rank "${local_rank}" \
            --world-size "${ARTESIA_WORLD_SIZE}" \
            --mem-size "${ARTESIA_MEM_SIZE}" \
            --chunk-size "${ARTESIA_CHUNK_SIZE}" \
            >"${log_file}" 2>&1
    ) &
    echo $!
}

launch_local_data_planes() {
    local pids=()
    local local_rank
    for ((local_rank = 0; local_rank < ARTESIA_GPUS_PER_NODE; local_rank++)); do
        pids+=("$(launch_data_plane_rank "${local_rank}")")
    done
    printf '%s\n' "${pids[@]}"
}
