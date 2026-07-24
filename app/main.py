from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.websockets import WebSocket, WebSocketDisconnect

from . import capture_protocol, protocol as legacy_protocol
from .inference import DITRSegmenter


logger = logging.getLogger("uvicorn.error")

MODEL_CONFIG = os.environ.get(
    "MODEL_CONFIG",
    "/models/ditr-industrial-aligned-23cls/config.py",
)
MODEL_WEIGHT = os.environ.get(
    "MODEL_WEIGHT",
    "/models/ditr-industrial-aligned-23cls/"
    "ditr-industrial-aligned-23cls.pth",
)
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda")
HOST_GPU_INDEX = os.environ.get("HOST_GPU_INDEX", "0")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "256")) * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = (
    int(os.environ.get("MAX_UNCOMPRESSED_MB", "512")) * 1024 * 1024
)
MAX_VALID_POINTS = int(os.environ.get("MAX_VALID_POINTS", "1000000"))

segmenter: DITRSegmenter | None = None
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global segmenter
    segmenter = DITRSegmenter(
        config_path=MODEL_CONFIG,
        weight_path=MODEL_WEIGHT,
        device=MODEL_DEVICE,
    )
    yield
    segmenter = None


app = FastAPI(
    title="IN3D DITR inference server",
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
            raise capture_protocol.ContractError(
                f"capture upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _require_segmenter() -> DITRSegmenter:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return segmenter


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "in3d-ditr-api",
        "status": "ok" if segmenter is not None else "starting",
        "health": "/health",
        "inference": "/v1/infer",
        "legacy_inference": "/segment",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model = _require_segmenter()
    return {
        "status": "ok",
        "contract_version": capture_protocol.CONTRACT_VERSION,
        "host_gpu_index": HOST_GPU_INDEX,
        **model.health(),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    model = _require_segmenter()
    return {
        "model": "ditr-industrial-aligned-23cls",
        "num_classes": model.num_classes,
        "class_names": model.class_names,
        "contract_version": capture_protocol.CONTRACT_VERSION,
        "endpoint": "/v1/infer",
        "response_media_type": "application/x-npz",
        "prediction_alignment": "valid depth points in row-major pixel order",
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_valid_points": MAX_VALID_POINTS,
        "legacy_protocol_version": legacy_protocol.PROTOCOL_VERSION,
        "legacy_endpoint": "/segment",
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
        raw = await _read_upload_limited(capture)
        sample = capture_protocol.parse_capture(
            raw,
            contract_version,
            metadata,
            max_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES,
            max_valid_points=MAX_VALID_POINTS,
        )
    except capture_protocol.ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        async with gpu_lock:
            model_prediction, confidence = await asyncio.to_thread(
                model.predict,
                {
                    "coord": sample.coord,
                    "color": sample.color,
                    "normal": sample.normal,
                    "image": sample.image,
                    "image_coord": sample.image_coord,
                    "image_mask": sample.image_mask,
                },
            )
        prediction, missing_client_classes = (
            capture_protocol.remap_model_predictions(
                model_prediction,
                model.class_names,
                sample.client_class_names,
            )
        )
        result = capture_protocol.encode_prediction(
            prediction,
            confidence,
            model_prediction,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.exception("CUDA out of memory during DITR inference")
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory; retry with a smaller capture",
            ) from exc
        logger.exception("DITR inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"DITR inference failed: {exc}",
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
            "X-IN3D-Contract-Version": capture_protocol.CONTRACT_VERSION,
            "X-IN3D-Model": "ditr-industrial-aligned-23cls",
            "X-IN3D-Point-Count": str(sample.valid_points),
            "X-IN3D-Inference-Ms": str(int(elapsed * 1000)),
            "X-IN3D-Unmapped-Model-Classes": str(
                len(missing_client_classes)
            ),
        },
    )


async def _run_legacy_inference(frame: bytes) -> bytes:
    model = _require_segmenter()
    raw = legacy_protocol.decode_request(frame)
    async with gpu_lock:
        labels, confidence = await asyncio.to_thread(model.predict, raw)
    return legacy_protocol.encode_response(labels, confidence)


@app.post("/segment")
async def segment(request: Request) -> Response:
    frame = await request.body()
    if len(frame) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "legacy request exceeds the upload size limit"},
        )
    try:
        result = await _run_legacy_inference(frame)
    except legacy_protocol.ProtocolError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return Response(content=result, media_type="application/octet-stream")


@app.websocket("/ws/segment")
async def ws_segment(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame = await ws.receive_bytes()
            try:
                result = await _run_legacy_inference(frame)
            except legacy_protocol.ProtocolError as exc:
                await ws.send_json({"error": str(exc)})
                continue
            await ws.send_bytes(result)
    except WebSocketDisconnect:
        pass
