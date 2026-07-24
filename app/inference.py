from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointcept.datasets.transform import TRANSFORMS, Compose
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config


logger = logging.getLogger("uvicorn.error")


class DITRSegmenter:
    """Load one DITR checkpoint and predict one aligned RGB-D frame."""

    def __init__(
        self,
        config_path: str,
        weight_path: str,
        device: str = "cuda",
    ):
        self.config_path = str(Path(config_path).resolve())
        self.weight_path = str(Path(weight_path).resolve())
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(f"model config not found: {self.config_path}")
        if not Path(self.weight_path).is_file():
            raise FileNotFoundError(f"model checkpoint not found: {self.weight_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the container")

        self.cfg = Config.fromfile(self.config_path)
        self.class_names = [
            str(name)
            for name in getattr(self.cfg, "class_names", self.cfg.data.names)
        ]
        self.num_classes = int(
            getattr(self.cfg, "num_classes", self.cfg.data.num_classes)
        )
        if self.num_classes != len(self.class_names):
            raise RuntimeError("config num_classes does not match class_names")
        self.device = torch.device(device)

        logger.info("Loading DITR config: %s", self.config_path)
        logger.info("Loading DITR checkpoint: %s", self.weight_path)
        model = build_model(self.cfg.model)
        checkpoint = torch.load(
            self.weight_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Frozen DINOv2 weights are provided by the runtime image and are not
        # stored in the task checkpoint.
        real_missing = [key for key in missing if "img_enc" not in key]
        if real_missing or unexpected:
            raise RuntimeError(
                "state_dict mismatch: "
                f"missing={real_missing}, unexpected={unexpected}"
            )
        self.model = model.to(self.device).eval()

        image_size = self._image_resize_from_cfg()
        self.transform = Compose(
            [
                dict(type="ImageResize", size=image_size),
                dict(type="ImageNormalize"),
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    keys=(
                        "coord",
                        "color",
                        "normal",
                        "image_coord",
                        "image_mask",
                    ),
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
            ]
        )
        self.to_tensor = TRANSFORMS.build(dict(type="ToTensor"))
        self.collect = TRANSFORMS.build(
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "image",
                    "image_coord",
                    "image_mask",
                ),
                feat_keys=("color", "normal"),
            )
        )

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.total_vram_bytes = int(props.total_memory)
        else:
            self.device_name = str(self.device)
            self.total_vram_bytes = 0
        logger.info(
            "DITR ready: classes=%d device=%s (%s) image_size=%s",
            self.num_classes,
            self.device,
            self.device_name,
            image_size,
        )

    def _image_resize_from_cfg(self) -> list[int]:
        for raw_spec in self.cfg.data.test.transform:
            spec = dict(raw_spec)
            if spec.get("type") == "ImageResize":
                return [int(value) for value in spec["size"]]
        raise ValueError("no ImageResize transform found in config data.test")

    @torch.inference_mode()
    def predict(self, raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        point_count = int(raw["coord"].shape[0])
        data: dict[str, Any] = {
            "coord": np.asarray(raw["coord"], dtype=np.float32),
            "color": np.asarray(raw["color"], dtype=np.float32),
            "normal": np.asarray(raw["normal"], dtype=np.float32),
            "image": np.asarray(raw["image"], dtype=np.uint8)[np.newaxis, ...],
            "image_coord": np.asarray(
                raw["image_coord"], dtype=np.float32
            ).reshape(point_count, 1, 2),
            "image_mask": np.asarray(
                raw["image_mask"], dtype=bool
            ).reshape(point_count, 1),
        }
        data = self.transform(data)
        if "inverse" not in data:
            raise RuntimeError("GridSample did not return the inverse point map")
        inverse = np.asarray(data.pop("inverse"), dtype=np.int64)
        data = self.collect(self.to_tensor(data))
        batch = collate_fn([data])
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)

        output = self.model(batch)
        logits = output["seg_logits"]
        probability = torch.softmax(logits, dim=-1)
        voxel_confidence, voxel_prediction = probability.max(dim=-1)
        voxel_prediction_np = voxel_prediction.cpu().numpy().astype(np.int32)
        voxel_confidence_np = voxel_confidence.cpu().numpy().astype(np.float32)
        prediction = voxel_prediction_np[inverse]
        confidence = voxel_confidence_np[inverse]
        if prediction.shape != (point_count,):
            raise RuntimeError(
                f"prediction alignment failed: expected {point_count}, "
                f"got {prediction.shape}"
            )
        return prediction, confidence

    def health(self) -> dict[str, Any]:
        return {
            "model_loaded": True,
            "model": "ditr-industrial-aligned-23cls",
            "device": str(self.device),
            "device_name": self.device_name,
            "visible_cuda_devices": torch.cuda.device_count(),
            "total_vram_bytes": self.total_vram_bytes,
            "config": self.config_path,
            "checkpoint": self.weight_path,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }


# Backward-compatible name for the original binary /segment implementation.
Segmenter = DITRSegmenter
