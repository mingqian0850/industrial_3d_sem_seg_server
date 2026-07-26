from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .inference import ModelInputError, VoltSegmenter
from .protocol import (
    CONTRACT_VERSION,
    ContractError,
    encode_prediction,
    parse_capture,
    remap_model_predictions,
)


logger = logging.getLogger("uvicorn.error")

MODEL_NAME = "volt-industrial-23cls"
MODEL_CONFIG = os.environ.get(
    "MODEL_CONFIG",
    f"/models/{MODEL_NAME}/config.py",
)
MODEL_WEIGHT = os.environ.get(
    "MODEL_WEIGHT",
    f"/models/{MODEL_NAME}/{MODEL_NAME}.pth",
)
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda")
HOST_GPU_INDEX = os.environ.get("HOST_GPU_INDEX", "0")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "256")) * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = int(os.environ.get("MAX_UNCOMPRESSED_MB", "512")) * 1024 * 1024
MAX_VALID_POINTS = int(os.environ.get("MAX_VALID_POINTS", "250000"))
MAX_VOXEL_POINTS = int(os.environ.get("MAX_VOXEL_POINTS", "60000"))
MAX_VOLT_TOKENS = int(os.environ.get("MAX_VOLT_TOKENS", "6000"))
VOXEL_SAMPLE_SEED = int(os.environ.get("VOXEL_SAMPLE_SEED", "0"))
REQUEST_QUEUE_TIMEOUT_SECONDS = float(
    os.environ.get("REQUEST_QUEUE_TIMEOUT_SECONDS", "30")
)
if REQUEST_QUEUE_TIMEOUT_SECONDS <= 0:
    raise ValueError("REQUEST_QUEUE_TIMEOUT_SECONDS must be positive")

segmenter: VoltSegmenter | None = None
gpu_lock = asyncio.Lock()


class InferenceAdmissionMiddleware:
    """Admit one inference request before multipart parsing begins."""

    def __init__(self, app: ASGIApp, *, timeout_seconds: float):
        self.app = app
        self.timeout_seconds = timeout_seconds
        self.slot = asyncio.Semaphore(1)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        is_inference = (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/v1/infer"
        )
        if not is_inference:
            await self.app(scope, receive, send)
            return

        try:
            await asyncio.wait_for(
                self.slot.acquire(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            body = json.dumps(
                {"detail": "inference queue is busy; retry later"}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"retry-after", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        try:
            await self.app(scope, receive, send)
        finally:
            self.slot.release()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global segmenter
    segmenter = VoltSegmenter(
        config_path=MODEL_CONFIG,
        weight_path=MODEL_WEIGHT,
        device=MODEL_DEVICE,
        max_voxel_points=MAX_VOXEL_POINTS,
        max_volt_tokens=MAX_VOLT_TOKENS,
        voxel_sample_seed=VOXEL_SAMPLE_SEED,
    )
    yield
    segmenter = None


app = FastAPI(
    title="IN3D Volt inference server",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    InferenceAdmissionMiddleware,
    timeout_seconds=REQUEST_QUEUE_TIMEOUT_SECONDS,
)


def _require_segmenter() -> VoltSegmenter:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return segmenter


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


def _run_inference_pipeline(
    model: VoltSegmenter,
    raw: bytes,
    contract_version: str,
    metadata_json: str,
) -> tuple[bytes, dict[str, Any], int, int]:
    sample = parse_capture(
        raw,
        contract_version,
        metadata_json,
        max_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES,
        max_valid_points=MAX_VALID_POINTS,
    )
    del raw
    model_prediction, confidence = model.predict(
        sample.coord,
        sample.color,
        sample.normal,
    )
    prediction, missing_client_classes = remap_model_predictions(
        model_prediction,
        model.class_names,
        sample.client_class_names,
    )
    result = encode_prediction(prediction, confidence, model_prediction)
    return (
        result,
        sample.metadata,
        sample.valid_points,
        len(missing_client_classes),
    )


async def _run_inference_non_cancellable(
    model: VoltSegmenter,
    raw: bytes,
    contract_version: str,
    metadata_json: str,
) -> tuple[bytes, dict[str, Any], int, int]:
    """Keep the GPU lock until a worker thread ends, even after disconnect."""
    async with gpu_lock:
        worker = asyncio.create_task(
            asyncio.to_thread(
                _run_inference_pipeline,
                model,
                raw,
                contract_version,
                metadata_json,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            try:
                worker.result()
            except Exception:
                logger.exception("inference failed after the client disconnected")
            raise


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "in3d-volt-api",
        "status": "ok" if segmenter is not None else "starting",
        "health": "/health",
        "inference": "/v1/infer",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model = _require_segmenter()
    return {
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
        "host_gpu_index": HOST_GPU_INDEX,
        **model.health(),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    model = _require_segmenter()
    return {
        "model": MODEL_NAME,
        "num_classes": model.num_classes,
        # Keep both names because the existing DiTR and PTv3 branches expose
        # different keys. New clients should prefer model_class_names.
        "class_names": model.class_names,
        "model_class_names": model.class_names,
        "contract_version": CONTRACT_VERSION,
        "endpoint": "/v1/infer",
        "response_media_type": "application/x-npz",
        "prediction_alignment": "valid depth points in row-major pixel order",
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_valid_points": MAX_VALID_POINTS,
        "max_voxel_points": model.max_voxel_points,
        "max_volt_tokens": model.max_volt_tokens,
        "request_queue_timeout_seconds": REQUEST_QUEUE_TIMEOUT_SECONDS,
    }


@app.post("/v1/infer")
async def infer(
    capture: UploadFile = File(...),
    contract_version: str = Form(...),
    metadata: str = Form(...),
) -> Response:
    model = _require_segmenter()
    started = time.perf_counter()
    try:
        try:
            raw = await _read_upload_limited(capture)
        finally:
            await capture.close()
        (
            result,
            request_metadata,
            valid_points,
            missing_client_class_count,
        ) = await _run_inference_non_cancellable(
            model,
            raw,
            contract_version,
            metadata,
        )
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.exception("CUDA out of memory during Volt inference")
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory; retry with a smaller capture",
            ) from exc
        logger.exception("Volt inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Volt inference failed: {exc}",
        ) from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "inference complete: dataset=%s frame=%s points=%d elapsed=%.3fs",
        request_metadata.get("dataset", "-"),
        request_metadata.get("frame", "-"),
        valid_points,
        elapsed,
    )
    return Response(
        content=result,
        media_type="application/x-npz",
        headers={
            "X-IN3D-Contract-Version": CONTRACT_VERSION,
            "X-IN3D-Model": MODEL_NAME,
            "X-IN3D-Point-Count": str(valid_points),
            "X-IN3D-Inference-Ms": str(int(elapsed * 1000)),
            "X-IN3D-Unmapped-Model-Classes": str(missing_client_class_count),
        },
    )
