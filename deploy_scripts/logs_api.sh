#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-$(id -un)}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-ditr-api-${OWNER}}"
TAIL="${TAIL:-200}"

docker logs --follow --tail "${TAIL}" "${CONTAINER_NAME}"
