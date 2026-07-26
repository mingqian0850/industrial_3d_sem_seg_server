from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from app.inference import _select_state_dict


class InferenceHelpersTests(unittest.TestCase):
    def test_ema_is_preferred_for_training_checkpoint(self) -> None:
        checkpoint = {
            "state_dict": OrderedDict(
                {"module.seg_heads.Default.bias": torch.zeros(23)}
            ),
            "ema_state_dict": OrderedDict(
                {"module.seg_heads.Default.bias": torch.ones(23)}
            ),
        }
        state_dict, source = _select_state_dict(checkpoint, use_ema=True)
        self.assertEqual(source, "ema_state_dict")
        self.assertEqual(list(state_dict), ["seg_heads.Default.bias"])
        torch.testing.assert_close(
            state_dict["seg_heads.Default.bias"],
            torch.ones(23),
        )

    def test_inference_artifact_state_dict_is_supported(self) -> None:
        checkpoint = {
            "state_dict": OrderedDict({"seg_heads.Default.bias": torch.ones(23)}),
            "artifact_type": "volt-ema-inference",
        }
        state_dict, source = _select_state_dict(checkpoint, use_ema=True)
        self.assertEqual(source, "state_dict")
        self.assertEqual(list(state_dict), ["seg_heads.Default.bias"])

    def test_checkpoint_without_weights_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            _select_state_dict({"epoch": 98}, use_ema=True)

    def test_non_ema_training_state_is_not_silently_used(self) -> None:
        with self.assertRaises(RuntimeError):
            _select_state_dict(
                {
                    "state_dict": OrderedDict(
                        {"seg_heads.Default.bias": torch.zeros(23)}
                    )
                },
                use_ema=True,
            )


if __name__ == "__main__":
    unittest.main()
