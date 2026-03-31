#!/usr/bin/env bash

# Edit only this file for standalone Artesia deployment.

: "${ARTESIA_PYTHON_BIN:=python}"
: "${ARTESIA_DEVICE:=cuda}"
: "${ARTESIA_WORLD_SIZE:=4}"
: "${ARTESIA_GPUS_PER_NODE:=4}"
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
: "${ARTESIA_LOG_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artesia-logs}"
