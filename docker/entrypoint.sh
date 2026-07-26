#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_CONFIG}" || ! -f "${MODEL_WEIGHT}" ]]; then
  echo "[INFO] Downloading ${HF_MODEL_REPO} into ${MODEL_DIR}"
  python -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_MODEL_REPO"],
    revision=os.environ.get("HF_REVISION", "main"),
    local_dir=os.environ["MODEL_DIR"],
    allow_patterns=[
        os.path.basename(os.environ["MODEL_CONFIG"]),
        os.path.basename(os.environ["MODEL_WEIGHT"]),
        "README.md",
    ],
    token=os.environ.get("HF_TOKEN"),
)
'
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "[ERROR] Missing model config after download: ${MODEL_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_WEIGHT}" ]]; then
  echo "[ERROR] Missing model checkpoint after download: ${MODEL_WEIGHT}" >&2
  exit 2
fi

verify_sha256() {
  local expected="$1"
  local path="$2"
  local label="$3"
  if [[ -z "${expected}" ]]; then
    return 0
  fi
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[ERROR] ${label} SHA-256 mismatch: expected ${expected}, got ${actual}" >&2
    exit 2
  fi
  echo "[OK] ${label} SHA-256 verified"
}

verify_sha256 "${MODEL_CONFIG_SHA256:-}" "${MODEL_CONFIG}" "model config"
verify_sha256 "${MODEL_WEIGHT_SHA256:-}" "${MODEL_WEIGHT}" "model checkpoint"

exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  "$@"
