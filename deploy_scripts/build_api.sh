#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${OWNER:-$(id -un)}"
IMAGE="${IMAGE:-in3d-volt-api:${OWNER}}"

REQUIRE_GPU=0 "${ROOT_DIR}/deploy_scripts/check_host.sh"

docker build \
  --pull \
  --tag "${IMAGE}" \
  "${ROOT_DIR}"
echo "[OK] Built ${IMAGE}"
