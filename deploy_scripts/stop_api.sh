#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-$(id -un)}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-volt-api-${OWNER}}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[INFO] ${CONTAINER_NAME} does not exist."
  exit 0
fi
owner="$(docker inspect --format '{{index .Config.Labels "com.in3d.owner"}}' "${CONTAINER_NAME}")"
service="$(docker inspect --format '{{index .Config.Labels "com.in3d.service"}}' "${CONTAINER_NAME}")"
if [[ "${owner}" != "${OWNER}" || "${service}" != "volt-api" ]]; then
  echo "[ERROR] Refusing to stop unrecognized container ${CONTAINER_NAME}."
  exit 2
fi
docker stop "${CONTAINER_NAME}" >/dev/null
echo "[OK] Stopped ${CONTAINER_NAME}"
