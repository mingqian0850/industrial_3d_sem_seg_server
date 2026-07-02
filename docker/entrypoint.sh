#!/usr/bin/env bash
set -euo pipefail

# Fetch model weights + config from Hugging Face unless already present
# (e.g. mounted as a volume).
if ! ls "${MODEL_DIR}"/*.pth >/dev/null 2>&1; then
    echo "No .pth found in ${MODEL_DIR}, downloading ${HF_MODEL_REPO} from Hugging Face..."
    hf download "${HF_MODEL_REPO}" --local-dir "${MODEL_DIR}"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
