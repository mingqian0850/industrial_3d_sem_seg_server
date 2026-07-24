from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import requests


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8011/v1/infer")
    args = parser.parse_args()

    raw = args.capture.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        metadata = scalar_text(archive["metadata_json"])
        depth = np.asarray(archive["depth"])
    valid_points = int((np.isfinite(depth) & (depth > 0)).sum())

    response = requests.post(
        args.url,
        files={"capture": (args.capture.name, raw, "application/x-npz")},
        data={
            "contract_version": "isaac-capture.v1",
            "metadata": metadata,
        },
        timeout=300,
    )
    response.raise_for_status()
    with np.load(io.BytesIO(response.content), allow_pickle=False) as archive:
        prediction = np.asarray(archive["prediction"])
        confidence = np.asarray(archive["confidence"])
        model_prediction = np.asarray(archive["model_prediction"])
    if prediction.shape != (valid_points,):
        raise RuntimeError(
            f"prediction length mismatch: {prediction.shape} != {(valid_points,)}"
        )
    print(
        json.dumps(
            {
                "status": response.status_code,
                "model": response.headers.get("X-IN3D-Model"),
                "valid_points": valid_points,
                "prediction_shape": list(prediction.shape),
                "model_prediction_shape": list(model_prediction.shape),
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "unique_client_labels": np.unique(prediction).tolist(),
                "inference_ms": response.headers.get("X-IN3D-Inference-Ms"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
