from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np


CONTRACT_VERSION = "isaac-capture.v1"
REQUIRED_ARRAYS = {
    "rgb",
    "depth",
    "normals",
    "semantic_gt",
    "semantic_train_gt",
    "intrinsics",
    "camera_pose",
    "metadata_json",
}


class ContractError(ValueError):
    """Raised when an uploaded capture does not satisfy isaac-capture.v1."""


@dataclass(frozen=True)
class CaptureInput:
    coord: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    client_class_names: list[str]
    metadata: dict[str, Any]
    image_shape: tuple[int, int]
    valid_points: int


def _metadata_from_scalar(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise ContractError("metadata_json must be a scalar string")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    try:
        payload = json.loads(str(item))
    except json.JSONDecodeError as exc:
        raise ContractError(f"metadata_json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("metadata_json must contain a JSON object")
    return payload


def _parse_form_metadata(raw: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > 1_000_000:
        raise ContractError("metadata form field is too large")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"metadata form field is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("metadata form field must contain a JSON object")
    return payload


def _check_npz_container(raw: bytes, max_uncompressed_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) > 32:
                raise ContractError("capture NPZ contains too many members")
            expanded_size = sum(member.file_size for member in members)
            if expanded_size > max_uncompressed_bytes:
                raise ContractError(
                    "capture NPZ expands beyond the configured uncompressed size limit"
                )
    except zipfile.BadZipFile as exc:
        raise ContractError("capture is not a valid NPZ/ZIP archive") from exc


def _require_dtype(name: str, array: np.ndarray, dtype: np.dtype[Any]) -> None:
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise ContractError(f"{name} must have dtype {expected}, got {array.dtype}")


def _validate_class_names(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("class_names")
    if not isinstance(values, list) or not values:
        raise ContractError("metadata.class_names must be a non-empty list")
    names = [str(value) for value in values]
    if any(not name for name in names):
        raise ContractError("metadata.class_names may not contain empty names")
    if len(set(names)) != len(names):
        raise ContractError("metadata.class_names may not contain duplicates")
    return names


def parse_capture(
    raw: bytes,
    form_contract_version: str,
    form_metadata_json: str,
    *,
    max_uncompressed_bytes: int,
    max_valid_points: int,
) -> CaptureInput:
    if form_contract_version != CONTRACT_VERSION:
        raise ContractError(
            f"unsupported contract_version {form_contract_version!r}; "
            f"expected {CONTRACT_VERSION!r}"
        )
    _check_npz_container(raw, max_uncompressed_bytes)
    form_metadata = _parse_form_metadata(form_metadata_json)

    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            missing = sorted(REQUIRED_ARRAYS.difference(archive.files))
            if missing:
                raise ContractError(f"capture NPZ is missing fields: {missing}")
            rgb = np.asarray(archive["rgb"]).copy()
            depth = np.asarray(archive["depth"]).copy()
            normals = np.asarray(archive["normals"]).copy()
            semantic_gt = np.asarray(archive["semantic_gt"])
            semantic_train_gt = np.asarray(archive["semantic_train_gt"])
            intrinsics = np.asarray(archive["intrinsics"]).copy()
            camera_pose = np.asarray(archive["camera_pose"])
            archive_metadata = _metadata_from_scalar(archive["metadata_json"])
    except ContractError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ContractError(f"unable to read capture NPZ: {exc}") from exc

    _require_dtype("rgb", rgb, np.uint8)
    _require_dtype("depth", depth, np.float32)
    _require_dtype("normals", normals, np.float32)
    _require_dtype("semantic_gt", semantic_gt, np.int32)
    _require_dtype("semantic_train_gt", semantic_train_gt, np.int32)
    _require_dtype("intrinsics", intrinsics, np.float32)
    _require_dtype("camera_pose", camera_pose, np.float32)

    if depth.ndim != 2:
        raise ContractError(f"depth must have shape HxW, got {depth.shape}")
    height, width = depth.shape
    if rgb.shape != (height, width, 3):
        raise ContractError(f"rgb must have shape HxWx3, got {rgb.shape}")
    if normals.shape != (height, width, 3):
        raise ContractError(f"normals must have shape HxWx3, got {normals.shape}")
    if semantic_gt.shape != depth.shape or semantic_train_gt.shape != depth.shape:
        raise ContractError("semantic arrays must have the same HxW shape as depth")
    if intrinsics.shape != (3, 3):
        raise ContractError(f"intrinsics must have shape 3x3, got {intrinsics.shape}")
    if camera_pose.shape != (4, 4):
        raise ContractError(f"camera_pose must have shape 4x4, got {camera_pose.shape}")
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_pose).all():
        raise ContractError("intrinsics and camera_pose must contain finite values")

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ContractError("intrinsics fx and fy must be positive")

    for source, name in (
        (form_metadata, "metadata form field"),
        (archive_metadata, "metadata_json"),
    ):
        if source.get("contract_version") != CONTRACT_VERSION:
            raise ContractError(f"{name} has an invalid contract_version")
    if form_metadata != archive_metadata:
        raise ContractError("metadata form field does not match NPZ metadata_json")
    class_names = _validate_class_names(archive_metadata)

    valid = np.isfinite(depth) & (depth > 0.0)
    valid_points = int(valid.sum())
    if valid_points == 0:
        raise ContractError("capture contains no valid positive finite depth points")
    if valid_points > max_valid_points:
        raise ContractError(
            f"capture contains {valid_points} valid points; limit is {max_valid_points}"
        )

    rows, cols = np.nonzero(valid)
    z_depth = depth[valid]
    x = (cols.astype(np.float32) - cx) * z_depth / fx
    y = -(rows.astype(np.float32) - cy) * z_depth / fy
    coord = np.stack((x, y, -z_depth), axis=1).astype(np.float32, copy=False)
    color = rgb[valid].astype(np.uint8, copy=False)
    normal = normals[valid].astype(np.float32, copy=True)
    np.nan_to_num(normal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return CaptureInput(
        coord=coord,
        color=color,
        normal=normal,
        client_class_names=class_names,
        metadata=archive_metadata,
        image_shape=(height, width),
        valid_points=valid_points,
    )


def remap_model_predictions(
    model_prediction: np.ndarray,
    model_class_names: list[str],
    client_class_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    model_prediction = np.asarray(model_prediction, dtype=np.int64).reshape(-1)
    client_id = {name: index for index, name in enumerate(client_class_names)}
    lookup = np.full(len(model_class_names), -1, dtype=np.int32)
    missing = []
    for model_id, name in enumerate(model_class_names):
        if name in client_id:
            lookup[model_id] = client_id[name]
        else:
            missing.append(name)
    if np.any((model_prediction < 0) | (model_prediction >= len(model_class_names))):
        raise RuntimeError("model returned an out-of-range class id")
    return lookup[model_prediction], missing


def encode_prediction(
    prediction: np.ndarray,
    confidence: np.ndarray,
    model_prediction: np.ndarray,
) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prediction=np.asarray(prediction, dtype=np.int32),
        confidence=np.asarray(confidence, dtype=np.float32),
        model_prediction=np.asarray(model_prediction, dtype=np.int32),
    )
    return buffer.getvalue()
