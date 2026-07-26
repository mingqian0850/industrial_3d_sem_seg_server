#!/usr/bin/env bash
# Check-only host validation. It never installs packages or invokes sudo.
set -euo pipefail

REQUIRE_GPU="${REQUIRE_GPU:-1}"
GPU_SMOKE_IMAGE="${GPU_SMOKE_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"
errors=0

ok() { echo "[OK] $*"; }
info() { echo "[INFO] $*"; }
fail() { echo "[ERROR] $*"; errors=$((errors + 1)); }

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  fail "deployment requires Linux x86_64"
else
  ok "host is Linux x86_64"
fi

for command_name in docker curl ss; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    fail "${command_name} is not installed or not on PATH"
  fi
done

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    docker_version="$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo unknown)"
    ok "Docker daemon reachable (client ${docker_version})"
  else
    fail "Docker daemon is not reachable by user $(id -un)"
    info "ask the host administrator to grant Docker access; this script does not use sudo"
  fi
fi

if [[ "${REQUIRE_GPU}" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi is unavailable; an NVIDIA driver is required"
  elif ! nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi failed"
  else
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    ok "NVIDIA driver ${driver}; ${gpu_count} GPU(s) visible"
  fi

  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if docker run --rm --gpus all "${GPU_SMOKE_IMAGE}" nvidia-smi >/dev/null 2>&1; then
      ok "Docker NVIDIA runtime works"
    else
      fail "Docker cannot run an NVIDIA GPU container"
      info "ask the host administrator to install/configure NVIDIA Container Toolkit"
    fi
  fi
else
  info "skipping GPU runtime check (REQUIRE_GPU=0)"
fi

if [[ "${errors}" -gt 0 ]]; then
  echo "[ERROR] host dependency check failed (${errors} issue(s))."
  exit 1
fi

echo "[OK] host dependencies are ready"
