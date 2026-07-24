#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-$(id -un)}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-ditr-api-${OWNER}}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[OK] ${CONTAINER_NAME} is already absent."
  exit 0
fi

owner="$(docker inspect --format '{{index .Config.Labels "com.in3d.owner"}}' "${CONTAINER_NAME}")"
service="$(docker inspect --format '{{index .Config.Labels "com.in3d.service"}}' "${CONTAINER_NAME}")"
if [[ "${owner}" != "${OWNER}" || "${service}" != "ditr-api" ]]; then
  echo "[ERROR] Refusing to stop container not owned by this deployment."
  exit 2
fi

docker stop "${CONTAINER_NAME}"
