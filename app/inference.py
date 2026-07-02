"""
DITR (PT-v3 + DINOv2 injection) inference wrapper.

Loads the merged Pointcept config + slimmed checkpoint once, then serves
single-frame predictions. Preprocessing mirrors the validation pipeline of
the training config (ImageResize -> ImageNormalize -> CenterShift ->
GridSample -> NormalizeColor), with GridSample additionally returning the
inverse map so voxel-level predictions are propagated back to every
original input point.
"""

import logging
import os

import numpy as np
import torch

from pointcept.datasets.transform import TRANSFORMS, Compose
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config

logger = logging.getLogger("uvicorn.error")


class Segmenter:
    def __init__(self, model_dir: str, device: str = "cuda"):
        config_path = os.path.join(model_dir, "config.py")
        weight_path = self._find_weight(model_dir)
        logger.info("Loading config: %s", config_path)
        logger.info("Loading weights: %s", weight_path)

        self.cfg = Config.fromfile(config_path)
        self.class_names: list[str] = list(self.cfg.class_names)
        self.num_classes: int = int(self.cfg.num_classes)
        self.device = torch.device(device)

        # FrozenDINOv2 inside the backbone calls .cuda() during construction,
        # so a CUDA device is required.
        assert torch.cuda.is_available(), "CUDA is required for the DITR backbone"
        model = build_model(self.cfg.model)
        checkpoint = torch.load(
            weight_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint["state_dict"]
        # Checkpoints saved from DDP training carry a "module." prefix.
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # DINOv2 weights are frozen and not part of the checkpoint for
        # FrozenDINOv2; anything else missing is a real problem.
        real_missing = [k for k in missing if "img_enc" not in k]
        if real_missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch: missing={real_missing}, unexpected={unexpected}"
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
                    keys=("coord", "color", "normal", "image_coord", "image_mask"),
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
                keys=("coord", "grid_coord", "image", "image_coord", "image_mask"),
                feat_keys=("color", "normal"),
            )
        )
        logger.info(
            "Model ready: %d classes, device=%s, image_size=%s",
            self.num_classes,
            self.device,
            image_size,
        )

    @staticmethod
    def _find_weight(model_dir: str) -> str:
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected exactly one .pth in {model_dir}, found: {candidates}"
            )
        return os.path.join(model_dir, candidates[0])

    def _image_resize_from_cfg(self) -> list[int]:
        for t in self.cfg.data.test.transform:
            if t["type"] == "ImageResize":
                return list(t["size"])
        raise ValueError("no ImageResize transform found in config data.test")

    @torch.inference_mode()
    def predict(self, raw: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        raw: dict with numpy arrays
            coord (N,3) float32, color (N,3) uint8/float, normal (N,3) float32,
            image (H,W,3) uint8, image_coord (N,2) float32, image_mask (N,) bool
        Returns (labels (N,) uint8, confidence (N,) float32) per input point.
        """
        n_points = raw["coord"].shape[0]
        data = dict(
            coord=raw["coord"].astype(np.float32),
            color=raw["color"].astype(np.float32),
            normal=raw["normal"].astype(np.float32),
            # single camera view -> CAM axis of size 1
            image=raw["image"][np.newaxis, ...].astype(np.uint8),
            image_coord=raw["image_coord"].reshape(n_points, 1, 2).astype(np.float32),
            image_mask=raw["image_mask"].reshape(n_points, 1).astype(bool),
        )
        data = self.transform(data)
        inverse = data.pop("inverse")  # (N,) original point -> voxel index
        data = self.collect(self.to_tensor(data))
        batch = collate_fn([data])
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)

        seg_logits = self.model(batch)["seg_logits"]  # (n_voxels, num_classes)
        prob = torch.softmax(seg_logits, dim=-1)
        conf, pred = prob.max(dim=-1)

        pred = pred.cpu().numpy().astype(np.uint8)
        conf = conf.cpu().numpy().astype(np.float32)
        return pred[inverse], conf[inverse]
