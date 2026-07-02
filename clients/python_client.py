"""
Example client for the industrial 3D semantic segmentation server.

Only needs numpy + requests (and websockets for the WS variant); no torch.

Usage:
    python python_client.py http  <server_url> <sample_dir>
    python python_client.py ws    <server_url> <sample_dir>

<sample_dir> is a Pointcept folder-per-sample directory containing
coord.npy, color.npy, normal.npy, image.npy, image_coord.npy, image_mask.npy
(e.g. one frame of the ditr_injection_dataset).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from protocol import decode_response  # noqa: E402

PROTOCOL_VERSION = 1


def load_sample(sample_dir: str) -> dict:
    data = {
        name: np.load(os.path.join(sample_dir, f"{name}.npy"))
        for name in ["coord", "color", "normal", "image", "image_coord", "image_mask"]
    }
    # dataset stores image as (1, H, W, 3) and per-point fields with a
    # camera axis; the wire protocol is single-view without that axis
    data["image"] = data["image"].reshape(data["image"].shape[-3:])
    n = data["coord"].shape[0]
    data["image_coord"] = data["image_coord"].reshape(n, 2)
    data["image_mask"] = data["image_mask"].reshape(n)
    return data


def encode_request(data: dict) -> bytes:
    n = data["coord"].shape[0]
    h, w = data["image"].shape[:2]
    header = json.dumps(
        {"version": PROTOCOL_VERSION, "num_points": n, "image_height": h, "image_width": w}
    ).encode("utf-8")
    return b"".join(
        [
            len(header).to_bytes(4, "little"),
            header,
            data["coord"].astype(np.float32).tobytes(),
            data["color"].astype(np.uint8).tobytes(),
            data["normal"].astype(np.float32).tobytes(),
            data["image"].astype(np.uint8).tobytes(),
            data["image_coord"].astype(np.float32).tobytes(),
            data["image_mask"].astype(np.uint8).tobytes(),
        ]
    )


def run_http(server_url: str, sample_dir: str):
    import requests

    meta = requests.get(f"{server_url}/metadata").json()
    class_names = meta["class_names"]
    print(f"server model: {meta['model']}, {meta['num_classes']} classes")

    frame = encode_request(load_sample(sample_dir))
    start = time.perf_counter()
    resp = requests.post(
        f"{server_url}/segment",
        data=frame,
        headers={"Content-Type": "application/octet-stream"},
    )
    resp.raise_for_status()
    labels, confidence = decode_response(resp.content)
    print(f"round-trip: {time.perf_counter() - start:.3f}s, {len(labels)} points")
    report(labels, confidence, class_names)


def run_ws(server_url: str, sample_dir: str):
    import asyncio

    import websockets

    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")

    async def _run():
        import requests

        class_names = requests.get(f"{server_url}/metadata").json()["class_names"]
        frame = encode_request(load_sample(sample_dir))
        # compression=None: permessage-deflate on ~10MB binary frames costs
        # far more time than it saves
        async with websockets.connect(
            f"{ws_url}/ws/segment", max_size=None, compression=None
        ) as ws:
            for i in range(3):  # simulate a small stream
                start = time.perf_counter()
                await ws.send(frame)
                result = await ws.recv()
                labels, confidence = decode_response(result)
                print(f"frame {i}: {time.perf_counter() - start:.3f}s")
        report(labels, confidence, class_names)

    asyncio.run(_run())


def report(labels: np.ndarray, confidence: np.ndarray, class_names: list):
    print(f"mean confidence: {confidence.mean():.3f}")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {class_names[cls]:>20s}: {cnt} points")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("http", "ws"):
        print(__doc__)
        sys.exit(1)
    mode, server_url, sample_dir = sys.argv[1:]
    (run_http if mode == "http" else run_ws)(server_url.rstrip("/"), sample_dir)
