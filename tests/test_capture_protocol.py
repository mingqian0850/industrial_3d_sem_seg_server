from __future__ import annotations

import io
import json
import unittest

import numpy as np

from app.capture_protocol import (
    CONTRACT_VERSION,
    encode_prediction,
    parse_capture,
    remap_model_predictions,
)


CLASS_NAMES = ["floor", "wall", "robot"]


def make_capture() -> tuple[bytes, str]:
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "dataset": "test",
        "frame": "frame_0000",
        "class_names": CLASS_NAMES,
    }
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.array([[1.0, 0.0, 2.0], [np.nan, 3.0, -1.0]], dtype=np.float32)
    normals = np.zeros((2, 3, 3), dtype=np.float32)
    intrinsics = np.array(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        rgb=rgb,
        depth=depth,
        normals=normals,
        semantic_gt=np.zeros((2, 3), dtype=np.int32),
        semantic_train_gt=np.zeros((2, 3), dtype=np.int32),
        intrinsics=intrinsics,
        camera_pose=np.eye(4, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return buffer.getvalue(), json.dumps(metadata)


class CaptureProtocolTests(unittest.TestCase):
    def test_parse_builds_ditr_projection_fields(self):
        raw, metadata = make_capture()
        sample = parse_capture(
            raw,
            CONTRACT_VERSION,
            metadata,
            max_uncompressed_bytes=1_000_000,
            max_valid_points=100,
        )
        self.assertEqual(sample.valid_points, 3)
        np.testing.assert_array_equal(
            sample.image_coord,
            np.array([[0, 0], [2, 0], [1, 1]], dtype=np.float32),
        )
        np.testing.assert_array_equal(sample.image_mask, np.ones(3, dtype=bool))
        self.assertEqual(sample.image.shape, (2, 3, 3))
        self.assertEqual(sample.coord.shape, (3, 3))

    def test_prediction_remap_and_response(self):
        remapped, missing = remap_model_predictions(
            np.array([0, 2, 1], dtype=np.int32),
            ["wall", "robot", "unknown"],
            CLASS_NAMES,
        )
        np.testing.assert_array_equal(remapped, np.array([1, -1, 2], dtype=np.int32))
        self.assertEqual(missing, ["unknown"])
        payload = encode_prediction(
            remapped,
            np.array([0.9, 0.8, 0.7], dtype=np.float32),
            np.array([0, 2, 1], dtype=np.int32),
        )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["prediction"], remapped)


if __name__ == "__main__":
    unittest.main()
