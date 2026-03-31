#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/standalone_artesia_config.sh"
source "${SCRIPT_DIR}/../_artesia_common.sh"

echo "Launching Artesia DataPlanes on node_index=${ARTESIA_NODE_INDEX}, node_ip=${ARTESIA_NODE_IP}"
echo "ControlPlane=${ARTESIA_CONTROL_PLANE_IP}:${ARTESIA_CONTROL_PLANE_PORT}, world_size=${ARTESIA_WORLD_SIZE}"
preflight_local_data_plane_ports
launch_local_data_planes
wait
