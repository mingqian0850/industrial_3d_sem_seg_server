from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config


logger = logging.getLogger("uvicorn.error")


def _plain_transform_config(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value))


class PTV3Segmenter:
    """Loads the trained PTv3 checkpoint once and predicts one RGB-D frame."""

    def __init__(self, config_path: str, weight_path: str, device: str = "cuda"):
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

        logger.info("Loading PTv3 config: %s", self.config_path)
        logger.info("Loading PTv3 checkpoint: %s", self.weight_path)
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
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()

        self.pre_transform, self.to_tensor, self.collect = self._build_val_pipeline()
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.total_vram_bytes = int(props.total_memory)
        else:
            self.device_name = str(self.device)
            self.total_vram_bytes = 0
        logger.info(
            "PTv3 ready: classes=%d device=%s (%s)",
            self.num_classes,
            self.device,
            self.device_name,
        )

    def _build_val_pipeline(self) -> tuple[Any, Any, Any]:
        pre_specs: list[dict[str, Any]] = []
        collect_spec: dict[str, Any] | None = None
        for raw_spec in self.cfg.data.val.transform:
            spec = _plain_transform_config(raw_spec)
            transform_type = spec.get("type")
            if transform_type == "GridSample":
                spec["return_inverse"] = True
                pre_specs.append(spec)
            elif transform_type == "ToTensor":
                continue
            elif transform_type == "Collect":
                keys = tuple(
                    key for key in spec.get("keys", ()) if key != "segment"
                )
                spec["keys"] = keys
                collect_spec = spec
            else:
                pre_specs.append(spec)
        if collect_spec is None:
            raise RuntimeError("validation pipeline does not contain Collect")
        if not any(spec.get("type") == "GridSample" for spec in pre_specs):
            raise RuntimeError("validation pipeline does not contain GridSample")
        return (
            Compose(pre_specs),
            TRANSFORMS.build(dict(type="ToTensor")),
            TRANSFORMS.build(collect_spec),
        )

    @torch.inference_mode()
    def predict(
        self,
        coord: np.ndarray,
        color: np.ndarray,
        normal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_count = int(coord.shape[0])
        data = {
            "coord": np.asarray(coord, dtype=np.float32),
            "color": np.asarray(color, dtype=np.float32),
            "normal": np.asarray(normal, dtype=np.float32),
        }
        data = self.pre_transform(data)
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
            "model": "ptv3-industrial-23cls",
            "device": str(self.device),
            "device_name": self.device_name,
            "visible_cuda_devices": torch.cuda.device_count(),
            "total_vram_bytes": self.total_vram_bytes,
            "config": self.config_path,
            "checkpoint": self.weight_path,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
