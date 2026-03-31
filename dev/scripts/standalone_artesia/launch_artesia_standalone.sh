#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/standalone_artesia_config.sh"
source "${SCRIPT_DIR}/../_artesia_common.sh"

echo "Launching standalone Artesia."
echo "ControlPlane on ${ARTESIA_CONTROL_PLANE_IP}:${ARTESIA_CONTROL_PLANE_PORT}"
echo "Local node ip=${ARTESIA_NODE_IP}, world_size=${ARTESIA_WORLD_SIZE}"
echo "Logs: ${ARTESIA_LOG_DIR}"
preflight_control_plane_port
preflight_local_data_plane_ports
launch_control_plane "${ARTESIA_LOG_DIR}/control-plane.log"
wait_for_control_plane_ready
launch_local_data_planes
wait
