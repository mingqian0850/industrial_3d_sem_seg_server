#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${OWNER:-$(id -un)}"
IMAGE="${IMAGE:-in3d-ptv3-api:${OWNER}}"

docker build \
  --pull \
  --tag "${IMAGE}" \
  "${ROOT_DIR}"
echo "[OK] Built ${IMAGE}"
