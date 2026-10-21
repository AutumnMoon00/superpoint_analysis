"""Collapse each ABC superpoint to one point, independently in every HDF5 file.

Run in a Python environment containing numpy and h5py (no geon dependency):
    python ABC_superpoints_code/superpoints_merge.py --limit 10
    python ABC_superpoints_code/superpoints_merge.py --workers 8

Outputs have the same filenames as their inputs, in only_superpoints by
default. Rows follow sorted original superpoint IDs. Coordinates and normals
are arithmetic means; normals are not normalized afterwards. prim and labels
are voted independently, with ties choosing the smallest value. T_param is
copied from the first original point WITHIN the superpoint whose label won.
The original superpoint ID and source point count are stored in superpoints
and superpoint_sizes. All original field shapes (apart from row count), field
dtypes, and HDF5 attributes are retained; integer coordinates/normals are
promoted to float64 so fractional means are not truncated.

Existing outputs are skipped unless --overwrite is supplied. Complete files
are prepared in temporary directories before replacing their destinations.
Failed files are reported and make the script exit with a nonzero status.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT_DIR = Path("/data/users/wm-sharath/fib/ABC_final_abc_superpoints_reg_0.1")
DEFAULT_OUTPUT_DIR = Path("/data/users/wm-sharath/fib/only_superpoints")
FIELDS = ("points", "normals", "prim", "labels", "T_param", "superpoints")


def _majority(values: np.ndarray, groups: np.ndarray) -> tuple:
    """Return each group's mode and its first matching original row.

    Sorting by group, then value, makes ties choose the smallest value.
    lexsort is stable, so each value's first row retains source order.
    No array is allocated using the maximum label or superpoint ID.
    """
    order = np.lexsort((values, groups))
    sorted_groups = groups[order]
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[
        True,
        (sorted_groups[1:] != sorted_groups[:-1]) |
        (sorted_values[1:] != sorted_values[:-1]),
    ])
    counts = np.diff(np.r_[starts, len(order)])
    pair_groups = sorted_groups[starts]
    group_starts = np.flatnonzero(np.r_[True, pair_groups[1:] != pair_groups[:-1]])
    maxima = np.maximum.reduceat(counts, group_starts)
    candidates = np.flatnonzero(counts == maxima[pair_groups])
    winners = candidates[np.r_[True, pair_groups[candidates[1:]] != pair_groups[candidates[:-1]]]]
    rows = order[starts[winners]]
    return values[rows], rows


def merge_superpoints(arrays: dict) -> dict:
    """Merge ABC root arrays without modifying them or renumbering any IDs."""
    missing = set(FIELDS).difference(arrays)
    if missing:
        raise ValueError(f"Missing ABC datasets: {sorted(missing)}")
    points = arrays["points"]
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"points must have nonempty shape (N, 3), got {points.shape}")
    point_count = len(points)
    if arrays["normals"].shape != points.shape:
        raise ValueError("normals must have the same (N, 3) shape as points")
    parameters = arrays["T_param"]
    if parameters.ndim != 2 or parameters.shape[0] != point_count or parameters.shape[1] == 0:
        raise ValueError(f"T_param must have shape (N, F), got {parameters.shape}")
    for name in ("points", "normals", "T_param"):
        values = arrays[name]
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite real numbers")
    for name in ("prim", "labels", "superpoints"):
        values = arrays[name]
        if values.shape not in {(point_count,), (point_count, 1)} or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain one integer ID per point, got {values.shape}, {values.dtype}")
    if np.any(arrays["superpoints"] < 0):
        raise ValueError("superpoints contains negative (unassigned) IDs")

    ids, groups, counts = np.unique(
        arrays["superpoints"].reshape(-1), return_inverse=True, return_counts=True
    )
    merged = {}
    for name in ("points", "normals"):
        values = arrays[name]
        means = np.column_stack([
            np.bincount(groups, weights=values[:, axis], minlength=len(ids)) / counts
            for axis in range(3)
        ])
        dtype = values.dtype if values.dtype.kind == "f" else np.dtype("float64")
        merged[name] = means.astype(dtype)
    for name in ("prim", "labels"):
        values, rows = _majority(arrays[name].reshape(-1), groups)
        merged[name] = values.reshape((len(ids),) + arrays[name].shape[1:])
        if name == "labels":
            # Copy the whole parameter vector for the winning label; never average it.
            merged["T_param"] = parameters[rows].copy()
    merged["superpoints"] = ids.reshape((len(ids),) + arrays["superpoints"].shape[1:])
    merged["superpoint_sizes"] = counts.astype(np.int64)
    return merged


def merge_file(source: Path, output_dir: Path, overwrite: bool = False) -> dict:
    """Write one ABC file atomically; return a small progress record."""
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    destination = output_dir / source.name
    if destination.resolve() == source:
        raise ValueError(f"Output would overwrite the input: {source}")
    if destination.exists() and not overwrite:
        if not destination.is_file():
            raise ValueError(f"Output path is not a file: {destination}")
        return {"source": source.name, "status": "skipped", "points": 0, "superpoints": 0}

    with h5py.File(str(source), "r") as original:
        missing = set(FIELDS).difference(original.keys())
        if missing:
            raise ValueError(f"{source.name}: missing ABC datasets {sorted(missing)}")
        merged = merge_superpoints({name: original[name][()] for name in FIELDS})
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".merge_superpoints_", dir=str(output_dir)) as temporary_dir:
            temporary_path = Path(temporary_dir) / source.name
            with h5py.File(str(temporary_path), "w") as target:
                for key, value in original.attrs.items():
                    target.attrs[key] = value
                for name, values in merged.items():
                    dataset = target.create_dataset(name, data=values, compression="gzip")
                    if name in FIELDS:
                        for key, value in original[name].attrs.items():
                            dataset.attrs[key] = value
                target["superpoint_sizes"].attrs["description"] = "Number of original points in this superpoint"
                target.attrs["source_superpoints_file"] = str(source)
                target.attrs["original_point_count"] = len(original["points"])
                target.attrs["merged_point_count"] = len(merged["points"])
                target.attrs["merge_rules"] = (
                    "points/normals: arithmetic mean, without normal renormalization; "
                    "prim/labels: independent modes, smallest ID wins ties; "
                    "T_param: first source row within superpoint with winning label; "
                    "rows: ascending original superpoint ID"
                )
            temporary_path.replace(destination)
        return {"source": source.name, "status": "success", "points": len(original["points"]),
                "superpoints": len(merged["points"])}


def _process_file(source: Path, output_dir: Path, overwrite: bool) -> dict:
    try:
        return merge_file(source, output_dir, overwrite)
    except Exception as error:
        return {"source": source.name, "status": "failed", "points": 0, "superpoints": 0,
                "error": f"{type(error).__name__}: {error}"}


def _results(sources: list, output_dir: Path, overwrite: bool, workers: int):
    if workers == 1:
        for source in sources:
            yield _process_file(source, output_dir, overwrite)
        return
    # Keep only a small number of files queued, including when interrupted.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = set()
        remaining = iter(sources)

        def submit_next() -> None:
            source = next(remaining, None)
            if source is not None:
                pending.add(pool.submit(_process_file, source, output_dir, overwrite))

        for _ in range(workers * 2):
            submit_next()
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                submit_next()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    inputs.add_argument("--input-file", type=Path, help="Merge just this ABC HDF5 file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Number of file-processing workers (default: up to 8)")
    parser.add_argument("--limit", type=int, help="Merge only the first N sorted input files")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    for name in ("workers", "limit", "progress_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.input_file is not None:
        if not args.input_file.is_file():
            parser.error(f"Input file does not exist: {args.input_file}")
    elif not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    return args


def main() -> int:
    args = parse_args()
    if args.input_file is not None:
        sources = [args.input_file.resolve()]
    else:
        sources = sorted(path.resolve() for path in args.input_dir.iterdir()
                         if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"})
    if args.limit is not None:
        sources = sources[:args.limit]
    if not sources:
        print(f"No HDF5 files found in {args.input_dir}", file=sys.stderr)
        return 1
    output_dir = args.output_dir.resolve()
    if any((output_dir / source.name).resolve() == source for source in sources):
        print("Output directory must differ from the input directory.", file=sys.stderr)
        return 1
    workers = min(args.workers, len(sources))
    print(f"Files: {len(sources)}; workers: {workers}; output: {output_dir}", flush=True)
    counts = {"success": 0, "skipped": 0, "failed": 0}
    original_points = merged_points = 0
    started = time.perf_counter()
    try:
        for result in _results(sources, output_dir, args.overwrite, workers):
            counts[result["status"]] += 1
            original_points += result["points"]
            merged_points += result["superpoints"]
            if result["status"] == "failed":
                print(f"FAILED {result['source']}: {result['error']}", file=sys.stderr, flush=True)
            done = sum(counts.values())
            if done % args.progress_every == 0 or done == len(sources):
                print(f"[{done}/{len(sources)}] success={counts['success']} "
                      f"skipped={counts['skipped']} failed={counts['failed']} "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun to skip completed outputs and continue.", file=sys.stderr)
        return 130
    print(f"Finished: {counts['success']} written, {counts['skipped']} skipped, {counts['failed']} failed; "
          f"newly merged points: {original_points} -> {merged_points}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
