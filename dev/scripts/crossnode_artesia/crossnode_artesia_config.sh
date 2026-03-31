#!/usr/bin/env bash

# Edit only this file for cross-node Artesia deployment.

: "${ARTESIA_PYTHON_BIN:=python}"
: "${ARTESIA_DEVICE:=cuda}"
: "${ARTESIA_WORLD_SIZE:=8}"
: "${ARTESIA_GPUS_PER_NODE:=4}"
: "${ARTESIA_CONTROL_PLANE_IP:=10.0.0.1}"
: "${ARTESIA_CONTROL_PLANE_PORT:=50250}"
: "${ARTESIA_PAGE_BYTES_SIZE:=32M}"
: "${ARTESIA_MEM_SIZE:=60G}"
: "${ARTESIA_CHUNK_SIZE:=256M}"
: "${ARTESIA_BASE_C2P_PORT:=50150}"
: "${ARTESIA_BASE_P2P_PORT:=50050}"
: "${ARTESIA_ENABLE_TRACE:=False}"

# Head node config
: "${ARTESIA_HEAD_NODE_INDEX:=0}"
: "${ARTESIA_HEAD_NODE_IP:=10.0.0.1}"
: "${ARTESIA_HEAD_LOG_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artesia-logs-head}"

# Worker node config
: "${ARTESIA_WORKER_NODE_INDEX:=1}"
: "${ARTESIA_WORKER_NODE_IP:=10.0.0.2}"
: "${ARTESIA_WORKER_LOG_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artesia-logs-worker}"
