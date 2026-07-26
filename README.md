# IN3D Volt FastAPI deployment

This branch provides a portable Docker deployment for the Volt-S model
fine-tuned on the 23-class industrial RGB-D point-cloud dataset. It implements
the same `isaac-capture.v1` API contract as the repository's `ptv3` and `ditr`
branches.

The service downloads the frozen model config and inference-only EMA
checkpoint from
[`min99ian/volt-industrial-23cls`](https://huggingface.co/min99ian/volt-industrial-23cls)
on first startup and keeps them in the local `models/` directory.

## Runtime and model reproducibility

The image pins:

- `pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel`
- Volt repository `https://github.com/mingqian0850/Volt.git`
- Volt commit `089cc38d8b32e7c695dd939057f787f3c60dd35e`
- Hugging Face model revision `1020052f8b4968235c727414e70103d43e78169b`
- the epoch-98 EMA inference checkpoint

The model was trained with Python 3.12 and PyTorch 2.8/CUDA 12.6. Its
checkpoint was also strictly loaded and forward-tested with the deployment
image's PyTorch 2.5/CUDA 12.4 runtime on an NVIDIA A40. Using the lighter
deployment runtime keeps the image aligned with the PTv3 service and does not
require Conda, uv, Python, or CUDA toolkits on the host.

## Supported deployment hardware

The image targets Linux x86_64 and NVIDIA GPUs. The following values are
conservative starting profiles; benchmark the actual sensor density and scene
extent on the target device.

| GPU | VRAM | `MAX_VALID_POINTS` | `MAX_VOXEL_POINTS` | `MAX_VOLT_TOKENS` |
|---|---:|---:|---:|---:|
| RTX 4070 / 4070 Super | 12 GB | `180000` | `45000` | `5000` |
| RTX 4070 Ti Super / RTX 4080 | 16 GB | `250000` | `60000` | `6000` |

Volt uses global attention after 10 cm tokenization. Token count affects
runtime more strongly than raw point count, so do not remove the token safety
limit merely because GPU memory is available.

Prerequisites:

- Linux x86_64
- NVIDIA driver 550 or newer
- Docker Engine 24 or newer
- NVIDIA Container Toolkit configured for Docker
- `curl`, `ss`, and access to the Docker daemon as the current user
- about 25 GB of free disk space for the base image and build layers
- network access to GitHub, Docker Hub, and Hugging Face for the initial build
  and model download

The host check is read-only and never invokes `sudo`:

```bash
./deploy_scripts/check_host.sh
```

## Build

```bash
git clone https://github.com/mingqian0850/industrial_3d_sem_seg_server.git
cd industrial_3d_sem_seg_server
git switch volt
./deploy_scripts/build_api.sh
```

The build runs the protocol and checkpoint-selection unit tests inside the
image.

## Start

RTX 4070 / 4070 Super:

```bash
GPU_DEVICE=0 \
MAX_VALID_POINTS=180000 \
MAX_VOXEL_POINTS=45000 \
MAX_VOLT_TOKENS=5000 \
./deploy_scripts/start_api.sh
```

RTX 4070 Ti Super / RTX 4080:

```bash
GPU_DEVICE=0 \
MAX_VALID_POINTS=250000 \
MAX_VOXEL_POINTS=60000 \
MAX_VOLT_TOKENS=6000 \
./deploy_scripts/start_api.sh
```

The service binds to localhost by default. To accept requests from another
device, bind it to a private LAN or VPN address:

```bash
BIND_ADDRESS=192.168.1.50 \
GPU_DEVICE=0 \
HOST_PORT=8012 \
./deploy_scripts/start_api.sh
```

Do not expose the unauthenticated API directly to the public internet. For a
multi-GPU host, set `GPU_DEVICE` to the intended physical GPU index.

At first startup the container downloads:

- `config.py`
- `volt-industrial-23cls.pth`

into `models/volt-industrial-23cls/`. The checkpoint contains only the trained
EMA `state_dict`; optimizer, scheduler, scaler, and non-EMA weights were
removed. If `HF_MODEL_REPO` points to a private repository, export a read token
before starting:

```bash
export HF_TOKEN=hf_your_read_token
./deploy_scripts/start_api.sh
```

Never commit tokens to Git, Dockerfiles, or shell scripts.

## Operations

```bash
./deploy_scripts/status_api.sh
./deploy_scripts/logs_api.sh
./deploy_scripts/stop_api.sh
```

The container uses `--restart unless-stopped`, one Uvicorn worker, and a GPU
lock that serializes inference requests.

Useful overrides:

| Variable | Default | Purpose |
|---|---|---|
| `IMAGE` | `in3d-volt-api:<user>` | Docker image |
| `CONTAINER_NAME` | `in3d-volt-api-<user>` | Container name |
| `GPU_DEVICE` | `0` | Physical host GPU index |
| `BIND_ADDRESS` | `127.0.0.1` | Host interface |
| `HOST_PORT` | `8012` | Host port |
| `MODEL_DIR` | `./models/volt-industrial-23cls` | Persistent model cache |
| `HF_MODEL_REPO` | `min99ian/volt-industrial-23cls` | Hugging Face model ID |
| `HF_REVISION` | `1020052…` | Frozen Hugging Face model revision |
| `MAX_VALID_POINTS` | `250000` | Raw valid-depth point limit |
| `MAX_VOXEL_POINTS` | `60000` | 2 cm voxel limit |
| `MAX_VOLT_TOKENS` | `6000` | 10 cm Volt token limit |
| `VOXEL_SAMPLE_SEED` | `0` | Deterministic voxel representative seed |
| `REQUEST_QUEUE_TIMEOUT_SECONDS` | `30` | Busy-request wait before HTTP 503 |

When overriding the model files, also override
`MODEL_CONFIG_SHA256`/`MODEL_WEIGHT_SHA256`, or set them to an empty string to
disable verification intentionally.

## Model integrity

Expected SHA-256 values:

```text
6d5385d38c5d6060eacd3d226124c86804b8cee7e129664b4c365f3bd6e9ed36  config.py
65ba22f299847f2c4356a59f1f110160f865c7e9d82816d9e980161fe515c198  volt-industrial-23cls.pth
```

The entrypoint verifies both files before loading the model.

## API

Client settings:

- scheme: `http`
- host: deployment machine's private address
- port: `8012`
- health path: `/health`
- metadata path: `/metadata`
- inference path: `/v1/infer`

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service status and endpoint discovery |
| `GET` | `/health` | Model, GPU, checkpoint, class, and limit metadata |
| `GET` | `/metadata` | Stable request/response contract metadata |
| `POST` | `/v1/infer` | Isaac multipart capture request; NPZ response |

`POST /v1/infer` accepts a multipart request containing:

- `capture`: an `isaac-capture.v1` NPZ
- `contract_version`: exactly `isaac-capture.v1`
- `metadata`: the same JSON object stored in the NPZ's `metadata_json`

The response media type is `application/x-npz` and contains:

- `prediction int32`: class IDs remapped by class name to the client registry
- `model_prediction int32`: native zero-based Volt class IDs
- `confidence float32`: maximum averaged softmax probability

All arrays align with valid positive finite depth pixels in row-major order.
The model consumes RGB divided by 255 and surface normals as its six features;
XYZ determines the 2 cm voxel grid and spatial tokens.

## Troubleshooting

- `Docker daemon is not reachable`: ask the host administrator to grant your
  account Docker access. The deployment scripts do not use `sudo`.
- `Docker cannot run an NVIDIA GPU container`: the administrator must install
  and configure NVIDIA Container Toolkit.
- CUDA out of memory: lower capture resolution, `MAX_VALID_POINTS`,
  `MAX_VOXEL_POINTS`, and `MAX_VOLT_TOKENS`.
- token/voxel limit error: reduce scene density or resolution; the request is
  rejected before unsafe global-attention inference.
- model download fails: verify network access, `HF_MODEL_REPO`, and `HF_TOKEN`
  if using a private repository.
- checksum mismatch: remove the corrupt file from the model directory and
  restart so it is downloaded again.
- health remains `starting`: run `./deploy_scripts/logs_api.sh`; first model
  download and initialization can take several minutes.

No host Python installation, Conda environment, or root privileges are
required once Docker and the NVIDIA runtime are configured.
