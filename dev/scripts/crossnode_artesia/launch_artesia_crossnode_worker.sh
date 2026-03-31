#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/crossnode_artesia_config.sh"

ARTESIA_NODE_INDEX="${ARTESIA_WORKER_NODE_INDEX}"
ARTESIA_NODE_IP="${ARTESIA_WORKER_NODE_IP}"
ARTESIA_LOG_DIR="${ARTESIA_WORKER_LOG_DIR}"

source "${SCRIPT_DIR}/../_artesia_common.sh"

echo "Launching cross-node Artesia worker node."
echo "ControlPlane on ${ARTESIA_CONTROL_PLANE_IP}:${ARTESIA_CONTROL_PLANE_PORT}"
echo "Worker node ip=${ARTESIA_NODE_IP}, node_index=${ARTESIA_NODE_INDEX}"
echo "Logs: ${ARTESIA_LOG_DIR}"
preflight_local_data_plane_ports
launch_local_data_planes
wait
