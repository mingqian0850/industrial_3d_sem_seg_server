# industrial_3d_sem_seg_server

Inference server for industrial 3D semantic segmentation (23 classes).
This branch (`ditr`) serves the DITR model (PT-v3 + frozen DINOv2-small
feature injection), weights hosted at
[min99ian/ditr-industrial](https://huggingface.co/min99ian/ditr-industrial).

The server exposes a FastAPI HTTP endpoint and a WebSocket endpoint with a
simple binary protocol, so clients can be written in any language (Python,
C++, ROS 2 node, ...) without depending on the model environment.

```
app/
├── main.py        FastAPI app: /health, /metadata, POST /segment, WS /ws/segment
├── inference.py   model loading + preprocessing + prediction
└── protocol.py    binary request/response encoding
clients/
└── python_client.py   example client (numpy + requests only)
docker/
└── entrypoint.sh  downloads weights from HF if missing, starts uvicorn
Dockerfile
```

## 1. Build the image

```bash
docker build -t industrial-seg-server:ditr-v0.1 .
```

The base image (`pointcept/pointcept:v1.6.0-...`, public on Docker Hub) is
pulled automatically on first build; the DITR-specific packages are
installed on top by the Dockerfile itself.

The build pins the DITR code to the training commit and bakes the frozen
DINOv2-small weights into the image, so the container needs no internet at
runtime (except for the one-time model download, see below).

## 2. Distribute / pull the image

Push to a registry (recommended):

```bash
docker tag industrial-seg-server:ditr-v0.1 <registry>/<user>/industrial-seg-server:ditr-v0.1
docker push <registry>/<user>/industrial-seg-server:ditr-v0.1

# on the target machine
docker pull <registry>/<user>/industrial-seg-server:ditr-v0.1
```

Or transfer as a file (no registry needed):

```bash
docker save industrial-seg-server:ditr-v0.1 | gzip > industrial-seg-server.tar.gz
# on the target machine
docker load < industrial-seg-server.tar.gz
```

## 3. Run the server

The target machine needs an NVIDIA GPU, driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Model weights are downloaded from Hugging Face on first start and cached in
the `seg-models` volume:

```bash
docker run -d --name seg-server --gpus '"device=0"' \
  -p 8000:8000 \
  -v seg-models:/models \
  industrial-seg-server:ditr-v0.1
```

If the HF repo is private, add `-e HF_TOKEN=hf_xxx`. To use local weights
instead of downloading, mount a directory containing `config.py` and the
`.pth` file:

```bash
docker run -d --name seg-server --gpus '"device=0"' \
  -p 8000:8000 \
  -v /path/to/ditr-industrial-23cls:/models/ditr-industrial:ro \
  industrial-seg-server:ditr-v0.1
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `/models/ditr-industrial` | directory with `config.py` + one `.pth` |
| `HF_MODEL_REPO` | `min99ian/ditr-industrial` | HF repo to download if `MODEL_DIR` is empty |
| `HF_TOKEN` | – | HF access token for private repos |
| `PORT` | `8000` | server port |

Check it is up:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
curl http://localhost:8000/metadata   # class names + protocol description
```

## 4. API

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/metadata` | model name, class names, array layout (JSON) |
| POST | `/segment` | one frame per request, binary body, binary response |
| WS | `/ws/segment` | persistent connection; one binary message per frame |

HTTP is the simplest integration; use the WebSocket for continuous streams
(sensor feeds) to avoid per-request connection overhead. Both use the same
frame format.

### Binary frame format

Point clouds are far too large for JSON, so requests/responses are raw
little-endian binary:

```
[4 bytes]  uint32 LE: length L of the JSON header
[L bytes]  UTF-8 JSON header
[...]      raw arrays, concatenated in fixed order (C order, little-endian)
```

Request header:

```json
{"version": 1, "num_points": N, "image_height": H, "image_width": W}
```

Request arrays, in this exact order:

| Array | dtype | shape | Notes |
|---|---|---|---|
| `coord` | float32 | (N, 3) | point coordinates, meters |
| `color` | uint8 | (N, 3) | RGB 0–255 |
| `normal` | float32 | (N, 3) | unit normals |
| `image` | uint8 | (H, W, 3) | RGB camera image of the same frame |
| `image_coord` | float32 | (N, 2) | per-point pixel coordinate (x, y) in `image` |
| `image_mask` | uint8 | (N,) | 1 if the point projects into the image |

Note: DITR injects DINOv2 image features into the point backbone, so the
camera image and the point-to-pixel projection are required inputs, not
optional extras. Points outside the camera frustum are allowed —
set their `image_mask` to 0.

Response header: `{"version": 1, "num_points": N}`, followed by:

| Array | dtype | shape | Notes |
|---|---|---|---|
| `labels` | uint8 | (N,) | class index per input point (see `/metadata`) |
| `confidence` | float32 | (N,) | softmax probability of the predicted class |

### Example client

`clients/python_client.py` implements the protocol with plain numpy
(no torch) and works against a folder-per-sample frame directory:

```bash
pip install numpy requests websockets

python clients/python_client.py http http://localhost:8000 /path/to/frame_dir
python clients/python_client.py ws   http://localhost:8000 /path/to/frame_dir
```

Porting to other languages only requires: build the JSON header, memcpy the
arrays in order, HTTP POST (or WS send), parse the response the same way.

## Notes

- Preprocessing (image resize/normalize, 0.02 m grid sampling, color
  normalization) runs server-side and mirrors the training config; clients
  send raw per-point data only. Voxel predictions are mapped back to all N
  input points, so the response aligns 1:1 with the request arrays.
- Requests are serialized through a single GPU lock; concurrent clients are
  handled but share one GPU queue.
- Model/image versioning: tag images as `ditr-vX.Y` and record the HF
  weight revision here when releasing.
