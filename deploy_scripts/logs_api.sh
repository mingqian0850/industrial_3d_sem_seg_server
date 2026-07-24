#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-in3d-ditr-api-mchen}"
TAIL="${TAIL:-200}"

docker logs --follow --tail "${TAIL}" "${CONTAINER_NAME}"
