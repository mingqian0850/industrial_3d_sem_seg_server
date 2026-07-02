"""
Binary wire protocol shared by the HTTP and WebSocket endpoints.

A frame is:

    [4 bytes]  uint32 little-endian: length L of the JSON header
    [L bytes]  UTF-8 JSON header
    [...]      raw little-endian arrays, concatenated in a fixed order

Request header:
    {"version": 1, "num_points": N, "image_height": H, "image_width": W}

Request payload arrays (in this exact order):
    coord        float32  (N, 3)
    color        uint8    (N, 3)   RGB 0-255
    normal       float32  (N, 3)
    image        uint8    (H, W, 3) RGB
    image_coord  float32  (N, 2)   per-point pixel coord (x, y) in the image
    image_mask   uint8    (N,)     1 if the point is visible in the image

Response header:
    {"version": 1, "num_points": N}

Response payload arrays:
    labels       uint8    (N,)   class index per point
    confidence   float32  (N,)   softmax probability of the predicted class
"""

import json

import numpy as np

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


def _split_frame(frame: bytes) -> tuple[dict, memoryview]:
    if len(frame) < 4:
        raise ProtocolError("frame too short")
    header_len = int.from_bytes(frame[:4], "little")
    if len(frame) < 4 + header_len:
        raise ProtocolError("frame shorter than declared header length")
    try:
        header = json.loads(frame[4 : 4 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"invalid JSON header: {e}") from e
    return header, memoryview(frame)[4 + header_len :]


def _build_frame(header: dict, arrays: list[np.ndarray]) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    parts = [len(header_bytes).to_bytes(4, "little"), header_bytes]
    parts += [np.ascontiguousarray(a).tobytes() for a in arrays]
    return b"".join(parts)


def _take(buf: memoryview, offset: int, dtype, shape) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape))
    nbytes = count * np.dtype(dtype).itemsize
    if offset + nbytes > len(buf):
        raise ProtocolError("payload shorter than expected")
    # copy: downstream transforms mutate arrays in place
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
    return arr.reshape(shape).copy(), offset + nbytes


def decode_request(frame: bytes) -> dict:
    """Decode a request frame into the raw input dict for Segmenter.predict."""
    header, buf = _split_frame(frame)
    if header.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {header.get('version')}")
    try:
        n = int(header["num_points"])
        h = int(header["image_height"])
        w = int(header["image_width"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProtocolError(f"invalid header fields: {e}") from e
    if n <= 0 or h <= 0 or w <= 0:
        raise ProtocolError("num_points / image size must be positive")

    offset = 0
    coord, offset = _take(buf, offset, np.float32, (n, 3))
    color, offset = _take(buf, offset, np.uint8, (n, 3))
    normal, offset = _take(buf, offset, np.float32, (n, 3))
    image, offset = _take(buf, offset, np.uint8, (h, w, 3))
    image_coord, offset = _take(buf, offset, np.float32, (n, 2))
    image_mask, offset = _take(buf, offset, np.uint8, (n,))
    if offset != len(buf):
        raise ProtocolError("payload longer than expected")

    return dict(
        coord=coord,
        color=color,
        normal=normal,
        image=image,
        image_coord=image_coord,
        image_mask=image_mask.astype(bool),
    )


def encode_response(labels: np.ndarray, confidence: np.ndarray) -> bytes:
    header = {"version": PROTOCOL_VERSION, "num_points": int(labels.shape[0])}
    return _build_frame(
        header, [labels.astype(np.uint8), confidence.astype(np.float32)]
    )


def decode_response(frame: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Client-side helper: decode a response frame into (labels, confidence)."""
    header, buf = _split_frame(frame)
    n = int(header["num_points"])
    offset = 0
    labels, offset = _take(buf, offset, np.uint8, (n,))
    confidence, _ = _take(buf, offset, np.float32, (n,))
    return labels, confidence
