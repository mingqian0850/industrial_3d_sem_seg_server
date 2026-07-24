from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .inference import PTV3Segmenter
from .protocol import (
    CONTRACT_VERSION,
    ContractError,
    encode_prediction,
    parse_capture,
    remap_model_predictions,
)


logger = logging.getLogger("uvicorn.error")

MODEL_CONFIG = os.environ.get(
    "MODEL_CONFIG",
    "/models/ptv3-industrial-23cls/config.py",
)
MODEL_WEIGHT = os.environ.get(
    "MODEL_WEIGHT",
    "/models/ptv3-industrial-23cls/ptv3-industrial-23cls.pth",
)
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda")
HOST_GPU_INDEX = os.environ.get("HOST_GPU_INDEX", "0")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "256")) * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = (
    int(os.environ.get("MAX_UNCOMPRESSED_MB", "512")) * 1024 * 1024
)
MAX_VALID_POINTS = int(os.environ.get("MAX_VALID_POINTS", "1000000"))

segmenter: PTV3Segmenter | None = None
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global segmenter
    segmenter = PTV3Segmenter(
        config_path=MODEL_CONFIG,
        weight_path=MODEL_WEIGHT,
        device=MODEL_DEVICE,
    )
    yield
    segmenter = None


app = FastAPI(
    title="IN3D PTv3 inference server",
    version="1.0.0",
    lifespan=lifespan,
)


async def _read_upload_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise ContractError(
                f"capture upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "in3d-ptv3-api",
        "status": "ok" if segmenter is not None else "starting",
        "health": "/health",
        "inference": "/v1/infer",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
        "host_gpu_index": HOST_GPU_INDEX,
        **segmenter.health(),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "contract_version": CONTRACT_VERSION,
        "endpoint": "/v1/infer",
        "response_media_type": "application/x-npz",
        "prediction_alignment": "valid depth points in row-major pixel order",
        "model_class_names": segmenter.class_names,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_valid_points": MAX_VALID_POINTS,
    }


@app.post("/v1/infer")
async def infer(
    capture: UploadFile = File(...),
    contract_version: str = Form(...),
    metadata: str = Form(...),
) -> Response:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    started = time.perf_counter()
    try:
        raw = await _read_upload_limited(capture)
        sample = parse_capture(
            raw,
            contract_version,
            metadata,
            max_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES,
            max_valid_points=MAX_VALID_POINTS,
        )
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        async with gpu_lock:
            model_prediction, confidence = await asyncio.to_thread(
                segmenter.predict,
                sample.coord,
                sample.color,
                sample.normal,
            )
        prediction, missing_client_classes = remap_model_predictions(
            model_prediction,
            segmenter.class_names,
            sample.client_class_names,
        )
        result = encode_prediction(prediction, confidence, model_prediction)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.exception("CUDA out of memory during inference")
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory; retry with a smaller capture",
            ) from exc
        logger.exception("PTv3 inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"PTv3 inference failed: {exc}",
        ) from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "inference complete: dataset=%s frame=%s points=%d elapsed=%.3fs",
        sample.metadata.get("dataset", "-"),
        sample.metadata.get("frame", "-"),
        sample.valid_points,
        elapsed,
    )
    return Response(
        content=result,
        media_type="application/x-npz",
        headers={
            "X-IN3D-Contract-Version": CONTRACT_VERSION,
            "X-IN3D-Point-Count": str(sample.valid_points),
            "X-IN3D-Inference-Ms": str(int(elapsed * 1000)),
            "X-IN3D-Unmapped-Model-Classes": str(len(missing_client_classes)),
        },
    )
