# IN3D DiTR FastAPI server

DiTR combines PTv3 with frozen DINOv2-small image features for industrial
RGB-D semantic segmentation. This repository provides a reproducible Docker
deployment for the Isaac client `isaac-capture.v1` contract.

The image pins the DiTR implementation used for training and caches the
DINOv2-small backbone during the build. The aligned 23-class config and task
checkpoint are downloaded at first startup from
[`min99ian/ditr-industrial-aligned-23cls`](https://huggingface.co/min99ian/ditr-industrial-aligned-23cls).

## Target hardware

The container targets Linux x86_64 systems with NVIDIA Ada GPUs:

| GPU | VRAM | Starting `MAX_VALID_POINTS` | Recommended capture |
|---|---:|---:|---|
| RTX 4070 / 4070 Super | 12 GB | `350000` | up to 640 × 480 |
| RTX 4070 Ti Super / RTX 4080 | 16 GB | `450000` | 640 × 480; increase gradually |

These are conservative starting profiles, not fixed model limits. Actual
memory use depends on valid depth count and voxel density. Lower the limit or
capture resolution if CUDA reports out-of-memory.

The service and model have been functionally validated on an NVIDIA A40 with
CUDA 12.4. RTX 4070/4080 use the same Ada CUDA architecture, but throughput and
maximum scene size should be benchmarked on the target device.

Prerequisites:

- NVIDIA driver 550 or newer
- Docker 24 or newer
- NVIDIA Container Toolkit configured for Docker
- about 25 GB of free disk space
- network access to GitHub, Docker Hub, timm model storage, and Hugging Face
  during the initial build/start

Verify the GPU runtime:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Build

```bash
git clone https://github.com/mingqian0850/industrial_3d_sem_seg_server.git
cd industrial_3d_sem_seg_server
git switch ditr
./deploy_scripts/build_api.sh
```

The portable build uses:

- `pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel`
- DiTR commit `f66d1dadb82e97ded7750567a043e4783305b01c`
- `vit_small_patch14_reg4_dinov2` from timm

On the original training server only, an existing
`industrial-seg-server:ditr-v0.1` image can be reused:

```bash
DOCKERFILE=Dockerfile.runtime ./deploy_scripts/build_api.sh
```

Other machines should use the default portable `Dockerfile`.

## Start on an RTX 4070

The default service is bound to localhost:

```bash
GPU_DEVICE=0 \
MAX_VALID_POINTS=350000 \
./deploy_scripts/start_api.sh
```

## Start on an RTX 4080

```bash
GPU_DEVICE=0 \
MAX_VALID_POINTS=450000 \
./deploy_scripts/start_api.sh
```

To accept requests from another machine, bind to a private LAN or VPN address:

```bash
BIND_ADDRESS=192.168.1.50 \
GPU_DEVICE=0 \
HOST_PORT=8011 \
MAX_VALID_POINTS=350000 \
./deploy_scripts/start_api.sh
```

For a multi-GPU host, select the intended physical GPU explicitly, for example
`GPU_DEVICE=1`. Only that GPU is visible inside the container.

At first startup the container downloads:

- `config.py`
- `ditr-industrial-aligned-23cls.pth`

into `models/ditr-industrial-aligned-23cls/`. The checkpoint contains only the
trained task `state_dict`; optimizer and scheduler state were removed. Frozen
DINOv2 weights are supplied by the image.

If the model repository is private, export a read token:

```bash
export HF_TOKEN=hf_your_read_token
./deploy_scripts/start_api.sh
```

Never commit tokens to Git, Dockerfiles, or shell scripts.

## Client settings

- scheme: `http`
- host: target machine's private address
- port: `8011`
- health path: `/health`
- inference path: `/v1/infer`

PTv3 conventionally uses port `8010`. Both services use the same client
contract, so changing the port switches the inference model.

## Operations

```bash
./deploy_scripts/status_api.sh
./deploy_scripts/logs_api.sh
./deploy_scripts/stop_api.sh
```

The container uses `--restart unless-stopped`; no tmux session is needed.

Useful overrides:

| Variable | Default | Purpose |
|---|---|---|
| `IMAGE` | `in3d-ditr-api:<user>` | Docker image |
| `CONTAINER_NAME` | `in3d-ditr-api-<user>` | Container name |
| `GPU_DEVICE` | `0` | Physical host GPU index |
| `BIND_ADDRESS` | `127.0.0.1` | Host interface |
| `HOST_PORT` | `8011` | Host port |
| `MODEL_DIR` | `./models/ditr-industrial-aligned-23cls` | Model cache |
| `HF_MODEL_REPO` | `min99ian/ditr-industrial-aligned-23cls` | HF model ID |
| `MAX_VALID_POINTS` | `350000` | Request safety limit |

## Model integrity

Expected SHA-256 values:

```text
0f29a5ea17f8ff06beace23628f55afce9d73fe85f99fa6d8afc9cd08388a61e  ditr-industrial-aligned-23cls.pth
36cb0d2f6bf5c45aa2861453b21f30b14b77ada65ef44d9d505c4520c59922c7  config.py
```

Verify after download:

```bash
sha256sum models/ditr-industrial-aligned-23cls/ditr-industrial-aligned-23cls.pth
sha256sum models/ditr-industrial-aligned-23cls/config.py
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health, model, GPU and class metadata |
| `GET` | `/metadata` | Request/response contract metadata |
| `POST` | `/v1/infer` | Isaac multipart capture request; NPZ response |
| `POST` | `/segment` | Legacy DiTR binary request |
| `WS` | `/ws/segment` | Legacy DiTR WebSocket |

`/v1/infer` returns:

- `prediction`: client-registry class ID for each valid depth point
- `model_prediction`: native DiTR class ID
- `confidence`: maximum softmax probability

All arrays align with valid depth pixels in row-major order.

## Troubleshooting

- `could not select device driver`: install or configure NVIDIA Container
  Toolkit, then restart Docker.
- CUDA out of memory: lower `MAX_VALID_POINTS` and capture resolution.
- model download fails: check `HF_MODEL_REPO`, network access, and `HF_TOKEN`
  for a private repository.
- build fails while downloading DINOv2: verify outbound network access and
  rebuild; the completed image does not need that download at startup.
- health remains `starting`: inspect `./deploy_scripts/logs_api.sh`; initial
  model loading can take several minutes.

No host Python/Conda installation and no sudo access are required after Docker
and the NVIDIA runtime are configured.
