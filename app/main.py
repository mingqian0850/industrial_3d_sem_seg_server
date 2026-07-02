"""
FastAPI server exposing the DITR industrial semantic segmentation model.

Endpoints:
    GET  /health            liveness + model status
    GET  /metadata          class names, protocol description
    POST /segment           one frame per request (binary body, see protocol.py)
    WS   /ws/segment        persistent connection, one frame per message

The GPU model is not concurrency-safe, so all predictions are serialized
through a single asyncio lock and executed in a worker thread to keep the
event loop responsive.
"""

import asyncio
import logging
import os
import time

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import protocol
from .inference import Segmenter

logger = logging.getLogger("uvicorn.error")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/ditr-industrial")

app = FastAPI(title="industrial-3d-sem-seg-server")
segmenter: Segmenter | None = None
gpu_lock = asyncio.Lock()


@app.on_event("startup")
def load_model():
    global segmenter
    segmenter = Segmenter(MODEL_DIR)


async def run_inference(frame: bytes) -> bytes:
    raw = protocol.decode_request(frame)
    async with gpu_lock:
        start = time.perf_counter()
        labels, confidence = await asyncio.to_thread(segmenter.predict, raw)
        elapsed = time.perf_counter() - start
    logger.info("inference: %d points in %.3fs", labels.shape[0], elapsed)
    return protocol.encode_response(labels, confidence)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": segmenter is not None}


@app.get("/metadata")
def metadata():
    return {
        "model": "ditr-industrial-23cls",
        "num_classes": segmenter.num_classes,
        "class_names": segmenter.class_names,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "request_arrays": [
            {"name": "coord", "dtype": "float32", "shape": ["N", 3]},
            {"name": "color", "dtype": "uint8", "shape": ["N", 3]},
            {"name": "normal", "dtype": "float32", "shape": ["N", 3]},
            {"name": "image", "dtype": "uint8", "shape": ["H", "W", 3]},
            {"name": "image_coord", "dtype": "float32", "shape": ["N", 2]},
            {"name": "image_mask", "dtype": "uint8", "shape": ["N"]},
        ],
        "response_arrays": [
            {"name": "labels", "dtype": "uint8", "shape": ["N"]},
            {"name": "confidence", "dtype": "float32", "shape": ["N"]},
        ],
    }


@app.post("/segment")
async def segment(request: Request):
    frame = await request.body()
    try:
        result = await run_inference(frame)
    except protocol.ProtocolError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return Response(content=result, media_type="application/octet-stream")


@app.websocket("/ws/segment")
async def ws_segment(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame = await ws.receive_bytes()
            try:
                result = await run_inference(frame)
            except protocol.ProtocolError as e:
                await ws.send_json({"error": str(e)})
                continue
            await ws.send_bytes(result)
    except WebSocketDisconnect:
        pass
