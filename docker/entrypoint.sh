#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_CONFIG}" || ! -f "${MODEL_WEIGHT}" ]]; then
  echo "[INFO] Downloading ${HF_MODEL_REPO} into ${MODEL_DIR}"
  hf download "${HF_MODEL_REPO}" --local-dir "${MODEL_DIR}"
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "[ERROR] Missing model config after download: ${MODEL_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_WEIGHT}" ]]; then
  echo "[ERROR] Missing model checkpoint after download: ${MODEL_WEIGHT}" >&2
  exit 2
fi

exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  "$@"
