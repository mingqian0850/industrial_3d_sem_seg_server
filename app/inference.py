from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pointcept
import torch

from pointcept.models import build_model
from pointcept.utils.config import Config


logger = logging.getLogger("uvicorn.error")


def _load_dataset_module(module_name: str, filename: str) -> ModuleType:
    """Load one dataset helper without executing pointcept.datasets.__init__."""
    pointcept_root = Path(pointcept.__file__).resolve().parent
    path = pointcept_root / "datasets" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Pointcept dataset helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_transform_module = _load_dataset_module(
    "_in3d_volt_transform",
    "transform.py",
)
_dataset_utils_module = _load_dataset_module(
    "_in3d_volt_dataset_utils",
    "utils.py",
)
Compose = _transform_module.Compose
TRANSFORMS = _transform_module.TRANSFORMS
collate_fn = _dataset_utils_module.collate_fn


class ModelInputError(ValueError):
    """Raised when a valid capture is unsafe or unsupported by Volt."""


def _plain_transform_config(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value))


def _select_state_dict(
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    """Select the inference weights and normalize a possible DDP prefix."""
    source = ""
    raw_state: Mapping[str, Any] | None = None
    if use_ema:
        if isinstance(checkpoint.get("ema_state_dict"), Mapping):
            raw_state = checkpoint["ema_state_dict"]
            source = "ema_state_dict"
        elif checkpoint.get("artifact_type") == "volt-ema-inference" and isinstance(
            checkpoint.get("state_dict"), Mapping
        ):
            raw_state = checkpoint["state_dict"]
            source = "state_dict"
        else:
            raise RuntimeError(
                "config requires EMA weights, but checkpoint is not an EMA "
                "training checkpoint or a Volt EMA inference artifact"
            )
    elif isinstance(checkpoint.get("state_dict"), Mapping):
        raw_state = checkpoint["state_dict"]
        source = "state_dict"
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        raw_state = checkpoint
        source = "checkpoint"

    if raw_state is None:
        raise RuntimeError("checkpoint contains neither a usable EMA nor a state_dict")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise RuntimeError("checkpoint state_dict contains an invalid entry")
        state_dict[key.removeprefix("module.")] = value
    return state_dict, source


def _config_class_names(cfg: Config) -> list[str]:
    if hasattr(cfg, "class_names"):
        values = cfg.class_names
    elif hasattr(cfg, "data") and hasattr(cfg.data, "names"):
        values = cfg.data.names
    else:
        raise RuntimeError("model config does not define class_names or data.names")
    names = [str(value) for value in values]
    if not names or len(set(names)) != len(names):
        raise RuntimeError("model config class names must be non-empty and unique")
    return names


def _config_num_classes(cfg: Config) -> int:
    if hasattr(cfg, "num_classes"):
        return int(cfg.num_classes)
    if hasattr(cfg, "data") and hasattr(cfg.data, "num_classes"):
        return int(cfg.data.num_classes)
    raise RuntimeError("model config does not define num_classes")


class VoltSegmenter:
    """Load the industrial Volt checkpoint and segment one RGB-D frame."""

    def __init__(
        self,
        config_path: str,
        weight_path: str,
        device: str = "cuda",
        *,
        max_voxel_points: int = 60_000,
        max_volt_tokens: int = 6_000,
        voxel_sample_seed: int = 0,
    ):
        self.config_path = str(Path(config_path).resolve())
        self.weight_path = str(Path(weight_path).resolve())
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(f"model config not found: {self.config_path}")
        if not Path(self.weight_path).is_file():
            raise FileNotFoundError(f"model checkpoint not found: {self.weight_path}")
        if not device.startswith("cuda"):
            raise ValueError("Volt deployment requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in the container"
            )
        if max_voxel_points <= 0 or max_volt_tokens <= 0:
            raise ValueError("model safety limits must be positive")

        self.cfg = Config.fromfile(self.config_path)
        self.class_names = _config_class_names(self.cfg)
        self.num_classes = _config_num_classes(self.cfg)
        if self.num_classes != len(self.class_names):
            raise RuntimeError("config num_classes does not match class_names")
        self.device = torch.device(device)
        self.max_voxel_points = int(max_voxel_points)
        self.max_volt_tokens = int(max_volt_tokens)
        self.voxel_sample_seed = int(voxel_sample_seed)
        if not 0 <= self.voxel_sample_seed <= np.iinfo(np.uint32).max:
            raise ValueError("voxel_sample_seed must be between 0 and 2^32 - 1")

        logger.info("Loading Volt config: %s", self.config_path)
        logger.info("Loading Volt checkpoint: %s", self.weight_path)
        model = build_model(self.cfg.model)
        checkpoint = torch.load(
            self.weight_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("Volt checkpoint must contain a mapping")
        state_dict, self.weight_source = _select_state_dict(
            checkpoint,
            use_ema=bool(getattr(self.cfg, "use_ema", False)),
        )
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()

        self._build_test_pipeline()
        backbone = self.model.backbone
        self.patch_size = int(backbone.tokenizer.kernel_size)
        self.rope_grid_limit = (
            int(backbone.pos_enc.cis_cache_x.shape[0]),
            int(backbone.pos_enc.cis_cache_y.shape[0]),
            int(backbone.pos_enc.cis_cache_z.shape[0]),
        )

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.total_vram_bytes = int(props.total_memory)
        else:
            self.device_name = str(self.device)
            self.total_vram_bytes = 0
        logger.info(
            "Volt ready: classes=%d device=%s (%s) weight_source=%s",
            self.num_classes,
            self.device,
            self.device_name,
            self.weight_source,
        )

    def _build_test_pipeline(self) -> None:
        if not hasattr(self.cfg.data, "test"):
            raise RuntimeError("model config does not define data.test")
        test_data_cfg = self.cfg.data.test
        if not hasattr(test_data_cfg, "test_cfg"):
            raise RuntimeError("model config data.test does not define test_cfg")

        pre_specs = [_plain_transform_config(spec) for spec in test_data_cfg.transform]
        if [spec.get("type") for spec in pre_specs] != ["Copy", "GridSample"]:
            raise RuntimeError(
                "Volt test preprocessing must be Copy followed by GridSample"
            )
        if dict(pre_specs[0].get("keys_dict", {})) != {"segment": "origin_segment"}:
            raise RuntimeError("Volt Copy transform does not preserve origin_segment")
        grid_specs = [spec for spec in pre_specs if spec.get("type") == "GridSample"]
        if len(grid_specs) != 1:
            raise RuntimeError(
                "Volt test preprocessing must contain exactly one GridSample"
            )
        if (
            float(grid_specs[0].get("grid_size", -1)) != 0.02
            or grid_specs[0].get("hash_type") != "fnv"
            or grid_specs[0].get("mode") != "train"
        ):
            raise RuntimeError(
                "Volt pre-voxelization must be 2 cm FNV GridSample in train mode"
            )
        grid_specs[0]["return_inverse"] = True
        self.voxel_size = float(grid_specs[0]["grid_size"])
        self.pre_transform = Compose(pre_specs)

        test_cfg = test_data_cfg.test_cfg
        voxelize_spec = _plain_transform_config(test_cfg.voxelize)
        if (
            voxelize_spec.get("type") != "GridSample"
            or float(voxelize_spec.get("grid_size", -1)) != self.voxel_size
            or voxelize_spec.get("hash_type") != "fnv"
            or voxelize_spec.get("mode") != "test"
        ):
            raise RuntimeError(
                "Volt fragment voxelization must be 2 cm FNV GridSample in test mode"
            )
        voxelize_spec["return_grid_coord"] = True
        self.test_voxelize = TRANSFORMS.build(voxelize_spec)
        if test_cfg.crop is not None:
            raise RuntimeError("Volt deployment does not support a test crop")
        self.test_crop = None

        post_specs = [_plain_transform_config(spec) for spec in test_cfg.post_transform]
        if [spec.get("type") for spec in post_specs] != [
            "CenterShift",
            "NormalizeColor",
            "ToTensor",
            "Collect",
        ]:
            raise RuntimeError("Volt post-transform pipeline does not match training")
        if bool(post_specs[0].get("apply_z", True)):
            raise RuntimeError("Volt post CenterShift must keep the z origin")
        if tuple(post_specs[-1].get("keys", ())) != (
            "coord",
            "grid_coord",
            "index",
        ) or tuple(post_specs[-1].get("feat_keys", ())) != ("color", "normal"):
            raise RuntimeError("Volt Collect keys do not match the six-channel model")
        self.post_transform = Compose(post_specs)

        augmentation_specs = [
            [_plain_transform_config(spec) for spec in augmentation]
            for augmentation in test_cfg.aug_transform
        ]
        if (
            len(augmentation_specs) != 1
            or len(augmentation_specs[0]) != 1
            or augmentation_specs[0][0].get("type") != "RandomRotateTargetAngle"
            or list(augmentation_specs[0][0].get("angle", ())) != [0]
            or augmentation_specs[0][0].get("axis") != "z"
            or float(augmentation_specs[0][0].get("p", 0)) != 1.0
        ):
            raise RuntimeError("Volt deployment requires the frozen identity TTA")
        self.aug_transforms = [
            Compose(augmentation) for augmentation in augmentation_specs
        ]

    def _prepare_fragments(
        self,
        coord: np.ndarray,
        color: np.ndarray,
        normal: np.ndarray,
    ) -> tuple[list[dict[str, Any]], np.ndarray, int]:
        point_count = int(coord.shape[0])
        data: dict[str, Any] = {
            "coord": np.asarray(coord, dtype=np.float32),
            "color": np.asarray(color, dtype=np.float32),
            "normal": np.asarray(normal, dtype=np.float32),
            # The frozen test config begins with Copy(segment -> origin_segment).
            "segment": np.full(point_count, -1, dtype=np.int32),
        }

        random_state = np.random.get_state()
        np.random.seed(self.voxel_sample_seed)
        try:
            data = self.pre_transform(data)
            if "inverse" not in data:
                raise RuntimeError("GridSample did not return the inverse point map")
            inverse = np.asarray(data.pop("inverse"), dtype=np.int64)
            data.pop("segment", None)
            data.pop("origin_segment", None)

            voxel_points = int(data["coord"].shape[0])
            if voxel_points > self.max_voxel_points:
                raise ModelInputError(
                    f"capture produces {voxel_points} 2 cm voxels; "
                    f"limit is {self.max_voxel_points}"
                )

            fragments: list[dict[str, Any]] = []
            for augmentation in self.aug_transforms:
                augmented = augmentation(deepcopy(data))
                parts = self.test_voxelize(augmented)
                for part in parts:
                    cropped_parts = self.test_crop(part) if self.test_crop else [part]
                    fragments.extend(
                        self.post_transform(cropped) for cropped in cropped_parts
                    )
        finally:
            np.random.set_state(random_state)

        if inverse.shape != (point_count,):
            raise RuntimeError(
                f"inverse point map has shape {inverse.shape}, expected {(point_count,)}"
            )
        if np.any((inverse < 0) | (inverse >= voxel_points)):
            raise RuntimeError("inverse point map contains an out-of-range index")
        if not fragments:
            raise RuntimeError("Volt preprocessing produced no inference fragments")
        return fragments, inverse, voxel_points

    def _validate_fragment(self, batch: dict[str, Any]) -> int:
        grid_coord = batch.get("grid_coord")
        if not isinstance(grid_coord, torch.Tensor) or grid_coord.ndim != 2:
            raise RuntimeError("Volt fragment does not contain Nx3 grid_coord")
        grid_np = grid_coord.detach().cpu().numpy().astype(np.int64, copy=False)
        if grid_np.shape[1] != 3 or np.any(grid_np < 0):
            raise RuntimeError("Volt fragment grid coordinates are invalid")

        coarse = grid_np // self.patch_size
        token_count = int(np.unique(coarse, axis=0).shape[0])
        if token_count > self.max_volt_tokens:
            raise ModelInputError(
                f"capture produces {token_count} Volt tokens; "
                f"limit is {self.max_volt_tokens}"
            )
        coarse_max = coarse.max(axis=0)
        for axis, (value, limit) in enumerate(
            zip(coarse_max.tolist(), self.rope_grid_limit, strict=True)
        ):
            if value >= limit:
                raise ModelInputError(
                    f"Volt token grid exceeds RoPE axis {axis} limit: "
                    f"index {value}, limit {limit - 1}"
                )
        return token_count

    @torch.inference_mode()
    def predict(
        self,
        coord: np.ndarray,
        color: np.ndarray,
        normal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_count = int(coord.shape[0])
        fragments, inverse, voxel_points = self._prepare_fragments(
            coord,
            color,
            normal,
        )

        probability_sum = torch.zeros(
            (voxel_points, self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        coverage = torch.zeros(
            voxel_points,
            dtype=torch.float32,
            device=self.device,
        )
        max_fragment_tokens = 0

        for fragment in fragments:
            batch = collate_fn([fragment])
            max_fragment_tokens = max(
                max_fragment_tokens,
                self._validate_fragment(batch),
            )
            index = batch["index"].to(
                self.device,
                dtype=torch.long,
                non_blocking=True,
            )
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device, non_blocking=True)

            output = self.model(batch)
            logits = output["seg_logits"]
            if logits.shape != (index.numel(), self.num_classes):
                raise RuntimeError(
                    f"Volt returned logits with shape {tuple(logits.shape)}"
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError("Volt returned non-finite logits")
            probability = torch.softmax(logits.float(), dim=-1)
            probability_sum.index_add_(0, index, probability)
            coverage.index_add_(
                0,
                index,
                torch.ones(index.numel(), dtype=torch.float32, device=self.device),
            )

        if torch.any(coverage == 0):
            raise RuntimeError("Volt fragment aggregation left uncovered voxels")
        probability = probability_sum / coverage.unsqueeze(-1)
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
        logger.debug(
            "Volt preprocessing: raw=%d voxels=%d fragments=%d max_tokens=%d",
            point_count,
            voxel_points,
            len(fragments),
            max_fragment_tokens,
        )
        return prediction, confidence

    def health(self) -> dict[str, Any]:
        return {
            "model_loaded": True,
            "model": "volt-industrial-23cls",
            "device": str(self.device),
            "device_name": self.device_name,
            "visible_cuda_devices": torch.cuda.device_count(),
            "total_vram_bytes": self.total_vram_bytes,
            "config": self.config_path,
            "checkpoint": self.weight_path,
            "weight_source": self.weight_source,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "voxel_size": self.voxel_size,
            "max_voxel_points": self.max_voxel_points,
            "max_volt_tokens": self.max_volt_tokens,
            "voxel_sample_seed": self.voxel_sample_seed,
            "runtime_torch": torch.__version__,
            "runtime_cuda": torch.version.cuda,
        }
