#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_CONFIG}" || ! -f "${MODEL_WEIGHT}" ]]; then
  echo "[INFO] Downloading ${HF_MODEL_REPO} into ${MODEL_DIR}"
  # Prefer the Python API: the image may ship huggingface-cli (hub<=0.33)
  # without the newer `hf` console script.
  python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["HF_MODEL_REPO"], local_dir=os.environ["MODEL_DIR"])'
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
