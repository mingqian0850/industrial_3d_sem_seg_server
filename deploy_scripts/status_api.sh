#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-$(id -un)}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-volt-api-${OWNER}}"
HOST_PORT="${HOST_PORT:-8012}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"

docker ps --filter "name=^/${CONTAINER_NAME}$" \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl --fail --silent --show-error "http://${HEALTH_HOST}:${HOST_PORT}/health"
echo
