from __future__ import annotations

import io
import json
import unittest

import numpy as np

from app.protocol import (
    CONTRACT_VERSION,
    ContractError,
    encode_prediction,
    parse_capture,
    remap_model_predictions,
)


def build_capture(metadata: dict) -> bytes:
    depth = np.asarray([[2.0, np.inf], [1.0, 3.0]], dtype=np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=depth,
        normals=np.ones((2, 2, 3), dtype=np.float32),
        semantic_gt=np.zeros((2, 2), dtype=np.int32),
        semantic_train_gt=np.zeros((2, 2), dtype=np.int32),
        intrinsics=np.asarray(
            [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        camera_pose=np.eye(4, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return buffer.getvalue()


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "contract_version": CONTRACT_VERSION,
            "dataset": "test",
            "frame": "frame_0000",
            "class_names": ["workpiece", "floor"],
        }

    def test_capture_points_follow_usd_camera_convention(self) -> None:
        capture = parse_capture(
            build_capture(self.metadata),
            CONTRACT_VERSION,
            json.dumps(self.metadata),
            max_uncompressed_bytes=10_000_000,
            max_valid_points=10,
        )
        self.assertEqual(capture.coord.shape, (3, 3))
        np.testing.assert_allclose(capture.coord[0], [-0.5, 0.5, -2.0])
        np.testing.assert_allclose(capture.coord[1], [-0.25, -0.25, -1.0])

    def test_metadata_mismatch_is_rejected(self) -> None:
        changed = {**self.metadata, "frame": "frame_0001"}
        with self.assertRaises(ContractError):
            parse_capture(
                build_capture(self.metadata),
                CONTRACT_VERSION,
                json.dumps(changed),
                max_uncompressed_bytes=10_000_000,
                max_valid_points=10,
            )

    def test_model_ids_are_remapped_by_class_name(self) -> None:
        prediction, missing = remap_model_predictions(
            np.asarray([0, 1, 2]),
            ["floor", "wall", "workpiece"],
            ["workpiece", "floor"],
        )
        np.testing.assert_array_equal(prediction, np.asarray([1, -1, 0]))
        self.assertEqual(missing, ["wall"])

    def test_out_of_range_model_id_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            remap_model_predictions(
                np.asarray([0, 2]),
                ["floor", "workpiece"],
                ["workpiece", "floor"],
            )

    def test_prediction_archive_has_stable_dtypes(self) -> None:
        raw = encode_prediction(
            np.asarray([3, -1]),
            np.asarray([0.75, 0.25]),
            np.asarray([1, 2]),
        )
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            self.assertEqual(
                archive.files,
                [
                    "prediction",
                    "confidence",
                    "model_prediction",
                ],
            )
            self.assertEqual(archive["prediction"].dtype, np.int32)
            self.assertEqual(archive["confidence"].dtype, np.float32)
            self.assertEqual(archive["model_prediction"].dtype, np.int32)

    def test_valid_point_limit_is_enforced(self) -> None:
        with self.assertRaises(ContractError):
            parse_capture(
                build_capture(self.metadata),
                CONTRACT_VERSION,
                json.dumps(self.metadata),
                max_uncompressed_bytes=10_000_000,
                max_valid_points=2,
            )


if __name__ == "__main__":
    unittest.main()
