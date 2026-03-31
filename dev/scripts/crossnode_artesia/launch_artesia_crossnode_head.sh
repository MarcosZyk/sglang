#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/crossnode_artesia_config.sh"

ARTESIA_NODE_INDEX="${ARTESIA_HEAD_NODE_INDEX}"
ARTESIA_NODE_IP="${ARTESIA_HEAD_NODE_IP}"
ARTESIA_LOG_DIR="${ARTESIA_HEAD_LOG_DIR}"

source "${SCRIPT_DIR}/../_artesia_common.sh"

echo "Launching cross-node Artesia head node."
echo "ControlPlane on ${ARTESIA_CONTROL_PLANE_IP}:${ARTESIA_CONTROL_PLANE_PORT}"
echo "Head node ip=${ARTESIA_NODE_IP}, node_index=${ARTESIA_NODE_INDEX}"
echo "Logs: ${ARTESIA_LOG_DIR}"
preflight_control_plane_port
preflight_local_data_plane_ports
launch_control_plane "${ARTESIA_LOG_DIR}/control-plane.log"
wait_for_control_plane_ready
launch_local_data_planes
wait
