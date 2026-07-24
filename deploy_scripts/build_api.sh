#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${OWNER:-$(id -un)}"
IMAGE="${IMAGE:-in3d-ditr-api:${OWNER}}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

docker build \
  --pull \
  --file "${ROOT_DIR}/${DOCKERFILE}" \
  --tag "${IMAGE}" \
  "${ROOT_DIR}"

echo "[OK] Built ${IMAGE}"
