# IN3D PTv3 FastAPI deployment

This repository contains a Dockerized PTv3 semantic-segmentation service for
the Isaac client `isaac-capture.v1` contract. The response is a compressed NPZ
whose predictions use the client's 23-class registry.

The image pins the Pointcept model implementation used during training. The
trained config and checkpoint are downloaded at first startup from
[`min99ian/ptv3-industrial-23cls`](https://huggingface.co/min99ian/ptv3-industrial-23cls)
and cached in the local `models/` directory.

## Supported deployment hardware

The published container targets Linux x86_64 and NVIDIA Ada GPUs:

| GPU | VRAM | Recommended `MAX_VALID_POINTS` | Recommended capture |
|---|---:|---:|---|
| RTX 4070 / 4070 Super | 12 GB | `350000` | up to 640 × 480 |
| RTX 4070 Ti Super / RTX 4080 | 16 GB | `500000` | 640 × 480; increase gradually |

The limit is checked before voxelization. Actual memory use also depends on
scene density, so start with the 12 GB profile when the exact card is unknown.
Inference is serialized inside one Uvicorn worker.

Prerequisites:

- NVIDIA driver 550 or newer
- Docker 24 or newer
- NVIDIA Container Toolkit configured for Docker
- about 25 GB of free disk space for the base image, build layers, and model
- network access to GitHub, Docker Hub, and Hugging Face during the first build

Verify the GPU runtime before building:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Build

```bash
git clone https://github.com/mingqian0850/industrial_3d_sem_seg_server.git
cd industrial_3d_sem_seg_server
git switch ptv3
./deploy_scripts/build_api.sh
```

The build uses
`pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel` and pins
Pointcept commit `d7272257058337969292865e0892b91d2f362b6a`.

## Start on an RTX 4070

The default service is bound to localhost for safety:

```bash
GPU_DEVICE=0 \
MAX_VALID_POINTS=350000 \
./deploy_scripts/start_api.sh
```

## Start on an RTX 4080

```bash
GPU_DEVICE=0 \
MAX_VALID_POINTS=500000 \
./deploy_scripts/start_api.sh
```

To accept requests from another machine, bind the service to a private LAN or
VPN address rather than a public interface:

```bash
BIND_ADDRESS=192.168.1.50 \
GPU_DEVICE=0 \
HOST_PORT=8010 \
MAX_VALID_POINTS=350000 \
./deploy_scripts/start_api.sh
```

For a multi-GPU host, select the intended physical GPU explicitly, for example
`GPU_DEVICE=1`. The container only sees the selected GPU.

At first startup the container downloads:

- `config.py`
- `ptv3-industrial-23cls.pth`

into `models/ptv3-industrial-23cls/`. If the Hugging Face repository is
private, export a read token before starting:

```bash
export HF_TOKEN=hf_your_read_token
./deploy_scripts/start_api.sh
```

Do not put tokens in this repository, a Dockerfile, or shell history.

## Client settings

Use the target machine's private address:

- scheme: `http`
- host: target machine address
- port: `8010`
- health path: `/health`
- inference path: `/v1/infer`

## Operations

```bash
./deploy_scripts/status_api.sh
./deploy_scripts/logs_api.sh
./deploy_scripts/stop_api.sh
```

The container uses `--restart unless-stopped`; no tmux session is required.

Useful overrides:

| Variable | Default | Purpose |
|---|---|---|
| `IMAGE` | `in3d-ptv3-api:<user>` | Docker image |
| `CONTAINER_NAME` | `in3d-ptv3-api-<user>` | Container name |
| `GPU_DEVICE` | `0` | Physical host GPU index |
| `BIND_ADDRESS` | `127.0.0.1` | Host interface |
| `HOST_PORT` | `8010` | Host port |
| `MODEL_DIR` | `./models/ptv3-industrial-23cls` | Persistent model cache |
| `HF_MODEL_REPO` | `min99ian/ptv3-industrial-23cls` | Hugging Face model ID |
| `MAX_VALID_POINTS` | `350000` | Request safety limit |

## Model integrity

Expected SHA-256 values:

```text
56b09f5065cf75e476f8b6873014a33008ca9b65c7a0440f39b28b779a6728c1  ptv3-industrial-23cls.pth
0f76f93bd447500d3aea0646447839d0b0c92f3b2e9f76b55525e588fc198fbb  config.py
```

Verify after download:

```bash
sha256sum models/ptv3-industrial-23cls/ptv3-industrial-23cls.pth
sha256sum models/ptv3-industrial-23cls/config.py
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model, GPU, checkpoint and class metadata |
| `GET` | `/metadata` | Request/response contract metadata |
| `POST` | `/v1/infer` | Isaac multipart capture request; NPZ response |

`/v1/infer` returns:

- `prediction`: client-registry class ID for each valid depth point
- `model_prediction`: native PTv3 class ID
- `confidence`: maximum softmax probability

All arrays follow valid depth pixels in row-major order.

## Troubleshooting

- `could not select device driver`: install or configure NVIDIA Container
  Toolkit, then restart Docker.
- CUDA out of memory: lower `MAX_VALID_POINTS` and the client capture
  resolution.
- model download fails: check `HF_MODEL_REPO`, network access, and `HF_TOKEN`
  for a private repository.
- health check stays `starting`: run `./deploy_scripts/logs_api.sh`; first model
  initialization can take several minutes.

No host Python/Conda installation and no sudo access are required after Docker
and the NVIDIA runtime are configured.
