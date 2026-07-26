#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-$(id -un)}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-volt-api-${OWNER}}"

docker logs --tail 200 --follow "${CONTAINER_NAME}"
