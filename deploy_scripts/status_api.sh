#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-$(id -un)}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-ditr-api-${OWNER}}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
HOST_PORT="${HOST_PORT:-8011}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[OFFLINE] ${CONTAINER_NAME} does not exist."
  exit 1
fi

docker ps \
  --filter "name=^/${CONTAINER_NAME}$" \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl --fail --silent --show-error "http://${HEALTH_HOST}:${HOST_PORT}/health"
echo
