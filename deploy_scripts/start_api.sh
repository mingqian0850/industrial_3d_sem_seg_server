#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${OWNER:-$(id -un)}"
IMAGE="${IMAGE:-in3d-volt-api:${OWNER}}"
CONTAINER_NAME="${CONTAINER_NAME:-in3d-volt-api-${OWNER}}"
HOST_PORT="${HOST_PORT:-8012}"
BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/volt-industrial-23cls}"
HF_MODEL_REPO="${HF_MODEL_REPO:-min99ian/volt-industrial-23cls}"
HF_REVISION="${HF_REVISION:-1020052f8b4968235c727414e70103d43e78169b}"
MODEL_CONFIG_SHA256="${MODEL_CONFIG_SHA256:-6d5385d38c5d6060eacd3d226124c86804b8cee7e129664b4c365f3bd6e9ed36}"
MODEL_WEIGHT_SHA256="${MODEL_WEIGHT_SHA256:-65ba22f299847f2c4356a59f1f110160f865c7e9d82816d9e980161fe515c198}"
MAX_VALID_POINTS="${MAX_VALID_POINTS:-250000}"
MAX_VOXEL_POINTS="${MAX_VOXEL_POINTS:-60000}"
MAX_VOLT_TOKENS="${MAX_VOLT_TOKENS:-6000}"
VOXEL_SAMPLE_SEED="${VOXEL_SAMPLE_SEED:-0}"
REQUEST_QUEUE_TIMEOUT_SECONDS="${REQUEST_QUEUE_TIMEOUT_SECONDS:-30}"

for value_name in GPU_DEVICE HOST_PORT MAX_VALID_POINTS MAX_VOXEL_POINTS MAX_VOLT_TOKENS REQUEST_QUEUE_TIMEOUT_SECONDS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" == "0" ]]; then
    if [[ "${value_name}" == "GPU_DEVICE" && "${value}" == "0" ]]; then
      continue
    fi
    echo "[ERROR] ${value_name} must be a positive integer."
    exit 2
  fi
done
if ! [[ "${VOXEL_SAMPLE_SEED}" =~ ^[0-9]+$ ]] \
  || (( VOXEL_SAMPLE_SEED > 4294967295 )); then
  echo "[ERROR] VOXEL_SAMPLE_SEED must be between 0 and 4294967295."
  exit 2
fi

REQUIRE_GPU=1 "${ROOT_DIR}/deploy_scripts/check_host.sh"

if ! nvidia-smi --id="${GPU_DEVICE}" --query-gpu=name --format=csv,noheader \
  >/dev/null 2>&1; then
  echo "[ERROR] NVIDIA GPU ${GPU_DEVICE} is not available."
  exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[ERROR] Image ${IMAGE} does not exist. Run deploy_scripts/build_api.sh first."
  exit 2
fi

existing_id="$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"
if [[ -n "${existing_id}" ]]; then
  owner="$(docker inspect --format '{{index .Config.Labels "com.in3d.owner"}}' "${CONTAINER_NAME}")"
  service="$(docker inspect --format '{{index .Config.Labels "com.in3d.service"}}' "${CONTAINER_NAME}")"
  if [[ "${owner}" != "${OWNER}" || "${service}" != "volt-api" ]]; then
    echo "[ERROR] Refusing to replace unrecognized container ${CONTAINER_NAME}."
    exit 2
  fi
fi

if ss -ltnH | awk -v target=":${HOST_PORT}" '$4 ~ target"$" {found=1} END {exit !found}'; then
  owned_port=0
  if [[ -n "${existing_id}" ]] && docker ps -q \
    --filter "id=${existing_id}" | grep -q .; then
    if docker port "${CONTAINER_NAME}" 8000/tcp 2>/dev/null \
      | grep -Eq ":${HOST_PORT}$"; then
      owned_port=1
    fi
  fi
  if [[ "${owned_port}" -ne 1 ]]; then
    echo "[ERROR] Host port ${HOST_PORT} is already in use."
    exit 2
  fi
fi

if [[ -n "${existing_id}" ]]; then
  docker rm --force "${CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${MODEL_DIR}"
hf_env=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf_env+=(--env HF_TOKEN)
fi

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --label "com.in3d.owner=${OWNER}" \
  --label com.in3d.service=volt-api \
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
  --volume "${MODEL_DIR}:/models/volt-industrial-23cls" \
  "${hf_env[@]}" \
  --env HOME=/tmp \
  --env HOST_GPU_INDEX="${GPU_DEVICE}" \
  --env MODEL_DEVICE=cuda \
  --env HF_MODEL_REPO="${HF_MODEL_REPO}" \
  --env HF_REVISION="${HF_REVISION}" \
  --env MODEL_DIR=/models/volt-industrial-23cls \
  --env MODEL_CONFIG=/models/volt-industrial-23cls/config.py \
  --env MODEL_WEIGHT=/models/volt-industrial-23cls/volt-industrial-23cls.pth \
  --env MODEL_CONFIG_SHA256="${MODEL_CONFIG_SHA256}" \
  --env MODEL_WEIGHT_SHA256="${MODEL_WEIGHT_SHA256}" \
  --env MAX_UPLOAD_MB=256 \
  --env MAX_UNCOMPRESSED_MB=512 \
  --env MAX_VALID_POINTS="${MAX_VALID_POINTS}" \
  --env MAX_VOXEL_POINTS="${MAX_VOXEL_POINTS}" \
  --env MAX_VOLT_TOKENS="${MAX_VOLT_TOKENS}" \
  --env VOXEL_SAMPLE_SEED="${VOXEL_SAMPLE_SEED}" \
  --env REQUEST_QUEUE_TIMEOUT_SECONDS="${REQUEST_QUEUE_TIMEOUT_SECONDS}" \
  "${IMAGE}" >/dev/null

for _ in $(seq 1 150); do
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
