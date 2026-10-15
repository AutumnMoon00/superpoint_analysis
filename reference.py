from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import h5py
import numpy as np

from geon.algorithms.superpoints import segment_superpoints
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData


DEFAULT_INPUT_DIR = Path(r"D:\GIA\ABC\ABC_final")
DEFAULT_GEON_OUTPUT_DIR = Path(r"D:\GIA\ABC\ABC_final_geon_superpoints")
DEFAULT_ABC_OUTPUT_DIR = Path(r"D:\GIA\ABC\ABC_final_abc_superpoints")


def select_file(input_dir: Path, seed: int | None) -> Path:
    """Choose uniformly from the HDF5 files directly inside input_dir."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    candidates = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not candidates:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
    return random.Random(seed).choice(candidates)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-file", type=Path, help="Process a specific file instead of choosing randomly")
    parser.add_argument("--geon-output-dir", type=Path, default=DEFAULT_GEON_OUTPUT_DIR,
                        help="Directory for native geon documents")
    parser.add_argument("--abc-output-dir", type=Path, default=DEFAULT_ABC_OUTPUT_DIR,
                        help="Directory for ABC data with superpoint IDs")
    parser.add_argument("--seed", type=int, help="Reproduce the random file choice for an unchanged directory")
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--coordinates-only", action="store_true", help="Segment without normal features")
    parser.add_argument("--serial", action="store_true", help="Disable parallel execution in geon")
    parser.add_argument("--verbose", action="store_true", help="Print geon's algorithm progress")
    args = parser.parse_args()
    if args.k_neighbors < 1 or args.cutoff < 1 or args.iterations < 1:
        parser.error("k-neighbors, cutoff, and iterations must be positive")
    if not np.isfinite(args.regularization) or args.regularization <= 0:
        parser.error("regularization must be finite and positive")
    if not np.isfinite(args.spatial_weight) or args.spatial_weight < 0:
        parser.error("spatial-weight must be finite and nonnegative")
    return args


def main() -> None:
    args = parse_args()
    source = (args.input_file or select_file(args.input_dir, args.seed)).resolve()
    print(f"Selected ABC file: {source}", flush=True)
    cloud = load_abc_as_geon(source)
    if args.k_neighbors >= len(cloud.points):
        raise ValueError("k-neighbors must be smaller than the number of points")
    settings = {
        "k_neighbors": args.k_neighbors,
        "regularization": args.regularization,
        "spatial_weight": args.spatial_weight,
        "cutoff": args.cutoff,
        "iterations": args.iterations,
        "parallel": not args.serial,
    }
    feature_names = [] if args.coordinates_only else ["normals"]
    print(f"Points: {len(cloud.points)}; feature fields: {feature_names}", flush=True)
    superpoints, stats = segment_superpoints(
        cloud, feature_field_names=feature_names, verbose=args.verbose, **settings
    )
    if superpoints.shape != (len(cloud.points),) or np.any(superpoints < 0):
        raise RuntimeError("geon did not return a valid superpoint ID for every point")
    ids, counts = np.unique(superpoints, return_counts=True)
    geon_path, abc_path = save_outputs(
        source, args.geon_output_dir, args.abc_output_dir, cloud, superpoints,
        {**settings, "feature_field_names": feature_names, "selection_seed": args.seed},
    )
    print(f"Superpoints: {len(ids)}; group sizes: {counts.min()}..{counts.max()}")
    print(f"geon statistics: {stats}")
    print(f"geon document: {geon_path}")
    print(f"ABC data with superpoints: {abc_path}")


if __name__ == "__main__":
    main()

