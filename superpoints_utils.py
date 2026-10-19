"""Shared ABC point-cloud loading, superpoint segmentation, and output helpers.

Imported by diff_regularizations.py; experiment configuration lives in the runner.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from geon.algorithms.superpoints import segment_superpoints
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData


def load_abc_as_geon(source: Path) -> PointCloudData:
    """Convert ABC root datasets to geon's typed point-cloud fields."""
    with h5py.File(source, "r") as handle:
        required = {"points", "normals", "labels", "prim", "T_param"}
        missing = required.difference(handle.keys())
        if missing:
            raise ValueError(f"{source.name}: missing ABC datasets {sorted(missing)}")
        arrays = {name: handle[name][()] for name in required}

    # The native algorithm requires contiguous float32 coordinates of shape N x 3.
    points = np.ascontiguousarray(arrays["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Expected at least two points with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Coordinates contain NaN or infinity")
    cloud = PointCloudData(points)

    normals = np.ascontiguousarray(arrays["normals"], dtype=np.float32)
    if normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError("Normals must be finite and have the same (N, 3) shape as points")
    lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    if np.any(lengths == 0):
        raise ValueError("Zero-length normals cannot be used as segmentation features")
    normals = np.ascontiguousarray(normals / lengths[:, None], dtype=np.float32)
    cloud.add_field(name="normals", data=normals, field_type=FieldType.NORMAL)

    for name, field_type in (("labels", FieldType.INSTANCE), ("prim", FieldType.SCALAR)):
        values = arrays[name]
        if values.shape not in {(len(points),), (len(points), 1)}:
            raise ValueError(f"{name} must contain one value per point, got {values.shape}")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain integer IDs, got {values.dtype}")
        # Generic geon fields require N x F; instance fields also accept flat IDs.
        cloud.add_field(name=name, data=values.reshape(-1, 1), field_type=field_type)

    parameters = arrays["T_param"]
    if parameters.ndim != 2 or parameters.shape[0] != len(points) or parameters.shape[1] == 0:
        raise ValueError(f"T_param must have shape (N, F), got {parameters.shape}")
    cloud.add_field(name="T_param", data=parameters, field_type=FieldType.VECTOR)
    return cloud


def save_outputs(
    source: Path,
    geon_output_dir: Path,
    abc_output_dir: Path,
    cloud: PointCloudData,
    superpoints: np.ndarray,
    settings: dict,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save a geon document and an ABC-compatible copy without changing source."""
    geon_output_dir.mkdir(parents=True, exist_ok=True)
    abc_output_dir.mkdir(parents=True, exist_ok=True)
    geon_path = geon_output_dir / f"{source.stem}_geon_superpoints.h5"
    abc_path = abc_output_dir / f"{source.stem}_abc_superpoints.h5"
    for destination, option in (
        (geon_path, "--geon-output-dir"), (abc_path, "--abc-output-dir")
    ):
        if destination.resolve() == source.resolve():
            raise ValueError(f"Output would overwrite the input: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists; choose another {option}: {destination}")

    cloud.add_field(name="superpoints", data=superpoints, field_type=FieldType.INSTANCE)
    document = Document(name=f"ABC_{source.stem}_superpoints")
    document.add_data(cloud)
    metadata = json.dumps(settings, sort_keys=True)

    # Prepare complete files before moving them into their final locations.
    with (
        tempfile.TemporaryDirectory(prefix="geon_superpoints_", dir=geon_output_dir) as temporary_geon_dir,
        tempfile.TemporaryDirectory(prefix="abc_superpoints_", dir=abc_output_dir) as temporary_abc_dir,
    ):
        temporary_geon = Path(temporary_geon_dir) / geon_path.name
        temporary_abc = Path(temporary_abc_dir) / abc_path.name
        document.save_hdf5(temporary_geon)
        with h5py.File(temporary_geon, "a") as handle:
            handle.attrs["source_abc_file"] = str(source.resolve())
            handle.attrs["superpoint_settings"] = metadata

        with h5py.File(source, "r") as original, h5py.File(temporary_abc, "w") as target:
            for name in original:
                original.copy(name, target)
            for key, value in original.attrs.items():
                target.attrs[key] = value
            if "superpoints" in target:
                raise ValueError("Input already contains a superpoints dataset")
            target.create_dataset("superpoints", data=superpoints, compression="gzip")
            target.attrs["source_abc_file"] = str(source.resolve())
            target.attrs["superpoint_settings"] = metadata
        temporary_geon.replace(geon_path)
        temporary_abc.replace(abc_path)
    return geon_path, abc_path
