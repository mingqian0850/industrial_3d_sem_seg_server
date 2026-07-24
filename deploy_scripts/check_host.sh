#!/usr/bin/env bash
# Verify (and optionally repair) host dependencies needed to build/run the API.
#
# Usage:
#   ./deploy_scripts/check_host.sh              # check only
#   FIX=1 ./deploy_scripts/check_host.sh        # attempt sudo installs/config
#   REQUIRE_GPU=0 ./deploy_scripts/check_host.sh  # skip GPU / --gpus checks
set -euo pipefail

FIX="${FIX:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
GPU_SMOKE_IMAGE="${GPU_SMOKE_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"

errors=0
fixes=0

info() { echo "[INFO] $*"; }
ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[ERROR] $*"; errors=$((errors + 1)); }

need_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    return 0
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo is required to install/configure host packages."
    return 1
  fi
  return 0
}

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

detect_distro() {
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "${ID:-unknown}"
  else
    echo "unknown"
  fi
}

ensure_docker_group() {
  if id -nG | tr ' ' '\n' | grep -qx docker; then
    ok "user $(id -un) is in the docker group"
    return 0
  fi
  if [[ ! -e /var/run/docker.sock ]]; then
    return 0
  fi
  if getent group docker >/dev/null 2>&1; then
    if [[ "${FIX}" == "1" ]]; then
      if need_sudo; then
        info "adding $(id -un) to the docker group"
        run_root usermod -aG docker "$(id -un)"
        fixes=$((fixes + 1))
        warn "docker group membership updated; run: newgrp docker"
        warn "or log out and back in, then re-run this script"
      fi
    else
      fail "current user is not in the docker group (permission denied on docker.sock)"
      info "fix with: FIX=1 $0"
      info "or: sudo usermod -aG docker \$USER && newgrp docker"
    fi
  fi
}

install_nvidia_container_toolkit_deb() {
  need_sudo || return 1
  local arch
  arch="$(dpkg --print-architecture)"
  if [[ "${arch}" != "amd64" ]]; then
    fail "automatic nvidia-container-toolkit install supports amd64 only (got ${arch})"
    return 1
  fi

  info "installing NVIDIA Container Toolkit apt repository"
  run_root mkdir -p /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | run_root gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | run_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  run_root apt-get update
  run_root apt-get install -y nvidia-container-toolkit
  run_root nvidia-ctk runtime configure --runtime=docker
  run_root systemctl restart docker
  fixes=$((fixes + 1))
  ok "installed and configured nvidia-container-toolkit"
}

configure_nvidia_runtime() {
  need_sudo || return 1
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    return 1
  fi
  info "configuring Docker NVIDIA runtime"
  run_root nvidia-ctk runtime configure --runtime=docker
  run_root systemctl restart docker
  fixes=$((fixes + 1))
  ok "Docker NVIDIA runtime configured"
}

# --- checks ---

if [[ "${REQUIRE_GPU}" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found; install an NVIDIA driver >= 550"
  elif ! nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi failed; check the NVIDIA driver installation"
  else
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    ok "NVIDIA driver ${driver}"
  fi
else
  info "skipping NVIDIA driver check (REQUIRE_GPU=0)"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found; install Docker Engine 24+ (https://docs.docker.com/engine/install/)"
else
  ok "docker client $(docker version --format '{{.Client.Version}}' 2>/dev/null || echo present)"
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker daemon reachable"
  else
    err="$(docker info 2>&1 || true)"
    if grep -qi 'permission denied' <<<"${err}"; then
      ensure_docker_group
      if ! docker info >/dev/null 2>&1; then
        fail "cannot reach Docker daemon (permission denied on /var/run/docker.sock)"
      fi
    elif grep -qiE 'cannot connect|no such file|is the docker daemon running' <<<"${err}"; then
      fail "Docker daemon is not running; start it with: sudo systemctl start docker"
    else
      fail "Docker daemon not reachable: ${err}"
    fi
  fi
fi

if [[ "${REQUIRE_GPU}" == "1" ]] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  toolkit_ok=0
  if command -v nvidia-ctk >/dev/null 2>&1 \
    || dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
    toolkit_ok=1
  fi

  if [[ "${toolkit_ok}" -eq 0 ]]; then
    if [[ "${FIX}" == "1" ]]; then
      distro="$(detect_distro)"
      case "${distro}" in
        ubuntu|debian)
          if ! command -v curl >/dev/null 2>&1 || ! command -v gpg >/dev/null 2>&1; then
            need_sudo && run_root apt-get update \
              && run_root apt-get install -y curl gnupg ca-certificates
          fi
          install_nvidia_container_toolkit_deb || true
          ;;
        *)
          fail "nvidia-container-toolkit missing; install it for ${distro} manually"
          info "see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
          ;;
      esac
    else
      fail "NVIDIA Container Toolkit is not installed"
      info "fix with: FIX=1 $0"
      info "or follow NVIDIA's install guide, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
  else
    ok "NVIDIA Container Toolkit present"
  fi

  if docker info >/dev/null 2>&1; then
    if docker run --rm --gpus all "${GPU_SMOKE_IMAGE}" nvidia-smi >/dev/null 2>&1; then
      ok "Docker GPU runtime (--gpus) works"
    else
      smoke_err="$(docker run --rm --gpus all "${GPU_SMOKE_IMAGE}" nvidia-smi 2>&1 || true)"
      if grep -qi 'could not select device driver' <<<"${smoke_err}"; then
        if [[ "${FIX}" == "1" ]] && command -v nvidia-ctk >/dev/null 2>&1; then
          configure_nvidia_runtime || true
          if docker run --rm --gpus all "${GPU_SMOKE_IMAGE}" nvidia-smi >/dev/null 2>&1; then
            ok "Docker GPU runtime (--gpus) works after reconfigure"
          else
            fail "Docker GPU runtime still broken after reconfigure"
            info "${smoke_err}"
          fi
        else
          fail "Docker cannot use GPUs (nvidia runtime not configured)"
          info "fix with: FIX=1 $0"
          info "or: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        fi
      elif grep -qi 'permission denied' <<<"${smoke_err}"; then
        ensure_docker_group
        fail "GPU smoke test failed due to Docker permissions"
      else
        fail "GPU smoke test failed"
        info "${smoke_err}"
      fi
    fi
  fi
fi

if [[ "${fixes}" -gt 0 ]]; then
  info "applied ${fixes} host fix(es); re-run without FIX=1 to confirm a clean check"
fi

if [[ "${errors}" -gt 0 ]]; then
  echo "[ERROR] host dependency check failed (${errors} issue(s))."
  exit 1
fi

echo "[OK] host dependencies are ready"
exit 0
