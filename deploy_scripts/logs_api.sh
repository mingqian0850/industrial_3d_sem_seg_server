#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-in3d-ptv3-api-mchen}"
docker logs --tail 200 --follow "${CONTAINER_NAME}"
