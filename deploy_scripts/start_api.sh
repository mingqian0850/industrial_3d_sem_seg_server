#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${OWNER:-$(id -un)}"
IMAGE="${IMAGE:-in3d-ditr-api:${OWNER}}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-ditr-api-${OWNER}}"
BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
HOST_PORT="${HOST_PORT:-8011}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/ditr-industrial-aligned-23cls}"
HF_MODEL_REPO="${HF_MODEL_REPO:-min99ian/ditr-industrial-aligned-23cls}"
MAX_VALID_POINTS="${MAX_VALID_POINTS:-350000}"

if ! [[ "${GPU_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] GPU_DEVICE must be a non-negative GPU index."
  exit 2
fi
if ! nvidia-smi --id="${GPU_DEVICE}" --query-gpu=name --format=csv,noheader \
  >/dev/null 2>&1; then
  echo "[ERROR] NVIDIA GPU ${GPU_DEVICE} is not available."
  exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[ERROR] Image not found: ${IMAGE}. Run deploy_scripts/build_api.sh first."
  exit 2
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  owner="$(docker inspect --format '{{index .Config.Labels "com.in3d.owner"}}' "${CONTAINER_NAME}")"
  service="$(docker inspect --format '{{index .Config.Labels "com.in3d.service"}}' "${CONTAINER_NAME}")"
  if [[ "${owner}" != "${OWNER}" || "${service}" != "ditr-api" ]]; then
    echo "[ERROR] Refusing to replace container not owned by this deployment."
    exit 2
  fi
  docker rm --force "${CONTAINER_NAME}" >/dev/null
fi

if ss -ltnH | awk -v target=":${HOST_PORT}" '$4 ~ target"$" {found=1} END {exit !found}'; then
  echo "[ERROR] Host port ${HOST_PORT} is already in use."
  exit 2
fi

mkdir -p "${MODEL_DIR}"
hf_env=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf_env+=(--env HF_TOKEN)
fi

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --label "com.in3d.owner=${OWNER}" \
  --label com.in3d.service=ditr-api \
  --gpus "device=${GPU_DEVICE}" \
  --restart unless-stopped \
  --shm-size=8g \
  --cpus=8 \
  --pids-limit=1024 \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=4g \
  --user "$(id -u):$(id -g)" \
  --publish "${BIND_ADDRESS}:${HOST_PORT}:8000" \
  --volume "${MODEL_DIR}:/models/ditr-industrial-aligned-23cls" \
  "${hf_env[@]}" \
  --env HOME=/tmp \
  --env USER="${OWNER}" \
  --env LOGNAME="${OWNER}" \
  --env HOST_GPU_INDEX="${GPU_DEVICE}" \
  --env MODEL_DEVICE=cuda \
  --env TORCHDYNAMO_DISABLE=1 \
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  --env TRITON_CACHE_DIR=/tmp/triton \
  --env HF_MODEL_REPO="${HF_MODEL_REPO}" \
  --env MODEL_DIR=/models/ditr-industrial-aligned-23cls \
  --env MODEL_CONFIG=/models/ditr-industrial-aligned-23cls/config.py \
  --env MODEL_WEIGHT=/models/ditr-industrial-aligned-23cls/ditr-industrial-aligned-23cls.pth \
  --env MAX_UPLOAD_MB=256 \
  --env MAX_UNCOMPRESSED_MB=512 \
  --env MAX_VALID_POINTS="${MAX_VALID_POINTS}" \
  "${IMAGE}" >/dev/null

for _ in $(seq 1 90); do
  state="$(docker inspect --format '{{.State.Status}}' "${CONTAINER_NAME}")"
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${CONTAINER_NAME}")"
  if [[ "${health}" == "healthy" ]]; then
    echo "[OK] ${CONTAINER_NAME} is healthy on http://${BIND_ADDRESS}:${HOST_PORT}"
    exit 0
  fi
  if [[ "${state}" == "exited" || "${state}" == "dead" ]]; then
    echo "[ERROR] ${CONTAINER_NAME} stopped during startup."
    docker logs --tail 160 "${CONTAINER_NAME}"
    exit 1
  fi
  sleep 2
done

echo "[ERROR] Timed out waiting for ${CONTAINER_NAME} health."
docker logs --tail 160 "${CONTAINER_NAME}"
exit 1
