"""Combine regularization-specific superpoint IDs with bounded file workers.

Run in the hpnet_l40s environment (numpy, h5py, and geon are required)::

    python combine_superpoints.py --workers 4 --limit 20 --progress-every 1
    python combine_superpoints.py --workers 4

Inputs default to /mnt/gpu_data_storage/sharath/ABC/ and are named
ABC_final_{abc,geon}_superpoints_reg_{0.025,0.05,0.1}. Outputs go into
ABC_final_{abc,geon}_superpoints_combined under the same parent directory.
Filenames, original fields (including ``superpoints``), and metadata are kept
from the first regularization, 0.025 by default. Three fields are added:
``superpoints_reg_0.025``, ``superpoints_reg_0.05``, ``superpoints_reg_0.1``.

GEON inputs are loaded through Document.load_hdf5; new INSTANCE fields are
serialized through GEON. ABC inputs use h5py. Point order and all other point
fields must match exactly across regularizations. Missing, unreadable, and
mismatched files are logged in the CSV report and skipped, independently for
each format. Existing outputs are skipped on reruns. Inputs are never edited.
Each worker reads and writes one file at a time, using one native thread.
Only paths and small status records cross process boundaries. The parent
writes the CSV, and at most two tasks per worker are queued. The default is
one worker; the Slurm launcher sets workers to the number of allocated CPUs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import closing, contextmanager
import csv
from datetime import datetime
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


# Do this before importing numpy or GEON, even when this module is imported.
for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ[_variable] = "1"


DEFAULT_DATA_DIR = Path("/mnt/gpu_data_storage/sharath/ABC")
REPORT_COLUMNS = (
    "format", "filename", "status", "regularization", "source", "output",
    "points", "seconds", "error_type", "error",
)


class InputError(Exception):
    """Keep the precise source and regularization in the failure report."""

    def __init__(self, path: Path, regularization: str, error: Exception):
        super().__init__(str(error))
        self.path = path
        self.regularization = regularization
        self.error = error


@contextmanager
def source_context(path: Path, regularization: str):
    try:
        yield
    except InputError:
        raise
    except Exception as error:
        raise InputError(path, regularization, error) from error


def require_same_array(expected, actual, name: str) -> None:
    import numpy as np

    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(
            f"{name}: shape/dtype differs from the first regularization: "
            f"{expected.shape}/{expected.dtype} versus {actual.shape}/{actual.dtype}"
        )
    equal_nan = np.issubdtype(expected.dtype, np.inexact)
    if not np.array_equal(expected, actual, equal_nan=equal_nan):
        raise ValueError(f"{name}: values or point order differ from the first regularization")


def validate_points(points) -> int:
    import numpy as np

    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Expected nonempty points with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Points contain NaN or infinity")
    return len(points)


def validate_ids(values, point_count: int) -> None:
    import numpy as np

    if values.shape not in {(point_count,), (point_count, 1)}:
        raise ValueError(f"superpoints must have shape (N,) or (N, 1), got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("superpoints must contain nonnegative integer IDs")


def read_abc_fields(handle) -> dict:
    """Read each non-superpoint dataset once, including custom nested fields."""
    import h5py

    fields = {}

    def read(name, item):
        if name != "superpoints":
            # None records a group, so empty groups are checked as well.
            fields[name] = item[()] if isinstance(item, h5py.Dataset) else None

    handle.visititems(read)
    return fields


def compare_abc_fields(reference: dict, candidate: dict) -> None:
    if reference.keys() != candidate.keys():
        raise ValueError("ABC dataset/group names differ across regularizations")
    for name, expected in reference.items():
        actual = candidate[name]
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"{name}: dataset/group type differs")
        else:
            require_same_array(expected, actual, name)


def combine_abc(sources: list[tuple[str, Path]], destination: Path) -> int:
    import h5py

    _, first_path = sources[0]
    ids = []
    attributes = []
    for reg, path in sources:
        with source_context(path, reg):
            with h5py.File(path, "r") as candidate:
                for field_name, _ in sources:
                    if f"superpoints_reg_{field_name}" in candidate:
                        raise ValueError("Input already contains a combined superpoint field")
                fields = read_abc_fields(candidate)
                if path == first_path:
                    reference = fields
                    point_count = validate_points(reference["points"])
                else:
                    compare_abc_fields(reference, fields)
                values = candidate["superpoints"][()]
                validate_ids(values, point_count)
                ids.append(values)
                attributes.append(dict(candidate["superpoints"].attrs))

    with tempfile.TemporaryDirectory(prefix=".combine_", dir=destination.parent) as temporary:
        temporary_path = Path(temporary) / destination.name
        shutil.copyfile(first_path, temporary_path)
        with h5py.File(temporary_path, "a") as target:
            for (reg, path), values, attrs in zip(sources, ids, attributes):
                field = target.create_dataset(f"superpoints_reg_{reg}", data=values, compression="gzip")
                for key, value in attrs.items():
                    field.attrs[key] = value
                field.attrs["regularization"] = float(reg)
                field.attrs["source_file"] = str(path)
        temporary_path.replace(destination)
    return point_count


def load_geon_cloud(path: Path):
    from geon.data.document import Document
    from geon.data.pointcloud import PointCloudData

    document = Document.load_hdf5(path)
    clouds = [item for item in document.scene_items.values() if isinstance(item, PointCloudData)]
    if len(clouds) != 1:
        raise ValueError(f"Expected one point cloud from diff_regularizations.py, got {len(clouds)}")
    return clouds[0]


def geon_fields(cloud) -> dict:
    fields = cloud.get_fields()
    by_name = {field.name: field for field in fields}
    if len(fields) != len(by_name):
        raise ValueError("Point cloud contains duplicate field names")
    return by_name


def combine_geon(sources: list[tuple[str, Path]], destination: Path) -> int:
    import h5py
    from geon.data.pointcloud import FieldType

    first_reg, first_path = sources[0]
    with source_context(first_path, first_reg):
        reference = load_geon_cloud(first_path)
        point_count = validate_points(reference.points)
        fields = geon_fields(reference)
        validate_ids(fields["superpoints"].data, point_count)
        new_names = {f"superpoints_reg_{reg}" for reg, _ in sources}
        if new_names.intersection(fields):
            raise ValueError("Input already contains a combined superpoint field")

    for reg, path in sources:
        with source_context(path, reg):
            candidate = reference if path == first_path else load_geon_cloud(path)
            other_fields = geon_fields(candidate)
            if set(fields) != set(other_fields):
                raise ValueError("Point-cloud field names differ across regularizations")
            require_same_array(reference.points, candidate.points, "points")
            for name, field in fields.items():
                other = other_fields[name]
                if field.field_type != other.field_type:
                    raise ValueError(f"{name}: GEON field types differ")
                if name != "superpoints":
                    require_same_array(field.data, other.data, name)
            values = other_fields["superpoints"].data
            validate_ids(values, point_count)
            reference.add_field(name=f"superpoints_reg_{reg}", data=values,
                                field_type=FieldType.INSTANCE)

    # Copy the original instead of reserializing it: GEON does not round-trip
    # every custom HDF5 attribute. Only the added fields go through its writer.
    with tempfile.TemporaryDirectory(prefix=".combine_", dir=destination.parent) as temporary:
        temporary_path = Path(temporary) / destination.name
        shutil.copyfile(first_path, temporary_path)
        with h5py.File(temporary_path, "a") as target:
            fields_group = target["document"][reference.id].require_group("fields")
            for reg, path in sources:
                field = reference.get_fields(names=f"superpoints_reg_{reg}")[0]
                group = field.save_hdf5(fields_group)
                group.attrs["regularization"] = float(reg)
                group.attrs["source_file"] = str(path)
        temporary_path.replace(destination)
    return point_count


def overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--regularizations", nargs="+", type=float, default=[0.025, 0.05, 0.1],
                        help="Values in input directory suffixes; first value supplies original fields")
    parser.add_argument("--abc-base-dir", type=Path,
                        help="ABC input base path; _reg_<value> is appended")
    parser.add_argument("--geon-base-dir", type=Path,
                        help="GEON input base path; _reg_<value> is appended")
    parser.add_argument("--abc-output-dir", type=Path)
    parser.add_argument("--geon-output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("abc", "geon"), default=["abc", "geon"])
    parser.add_argument("--workers", type=int, default=1,
                        help="File worker processes (default: 1); each uses one native thread")
    parser.add_argument("--limit", type=int, help="Process the first N sorted filenames per format")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--log-file", type=Path, help="New CSV report path; existing reports are never overwritten")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus is not None and args.workers > int(allocated_cpus):
        parser.error(f"--workers exceeds the {allocated_cpus} CPUs allocated by Slurm")
    if any(not math.isfinite(reg) or reg <= 0 for reg in args.regularizations):
        parser.error("--regularizations must be finite and positive")
    args.regularizations = list(dict.fromkeys(str(reg) for reg in args.regularizations))
    args.formats = list(dict.fromkeys(args.formats))
    args.log_file = (args.log_file or (
        Path(__file__).resolve().parent / "batch_logs" /
        f"combine_superpoints_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    )).resolve()

    plans = []
    for format_name in args.formats:
        base = (getattr(args, f"{format_name}_base_dir") or
                args.data_dir / f"ABC_final_{format_name}_superpoints").resolve()
        output = (getattr(args, f"{format_name}_output_dir") or
                  base.with_name(f"{base.name}_combined")).resolve()
        directories = [(reg, base.with_name(f"{base.name}_reg_{reg}").resolve())
                       for reg in args.regularizations]
        for _, directory in directories:
            if not directory.is_dir():
                parser.error(f"Input directory does not exist: {directory}")
        plans.append((format_name, directories, output))

    inputs = [directory for _, directories, _ in plans for _, directory in directories]
    outputs = [output for _, _, output in plans]
    for output in outputs:
        if any(overlaps(output, directory) for directory in inputs):
            parser.error(f"Output directory overlaps an input directory: {output}")
        if output.exists() and not output.is_dir():
            parser.error(f"Output path is not a directory: {output}")
    if len(outputs) == 2 and overlaps(*outputs):
        parser.error("ABC and GEON output directories must be separate")
    if any(args.log_file == directory or directory in args.log_file.parents
           for directory in inputs + outputs):
        parser.error("Choose a CSV report path outside the input and output directories")
    if args.log_file.exists():
        parser.error(f"CSV report already exists: {args.log_file}")
    return args, plans


def process_file(format_name: str, directories: list, output: Path, filename: str) -> dict:
    """Run entirely in one worker; never return point arrays or HDF5 handles."""
    started = time.perf_counter()
    destination = output / filename
    sources = [(reg, directory / filename) for reg, directory in directories]
    row = dict.fromkeys(REPORT_COLUMNS, "")
    row.update(format=format_name, filename=filename, output=str(destination))
    try:
        if destination.is_symlink():
            raise ValueError(f"Output is a symbolic link: {destination}")
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Output exists and is not a file: {destination}")
            row["status"] = "skipped"
        else:
            combine = combine_abc if format_name == "abc" else combine_geon
            row.update(points=combine(sources, destination), status="success")
    except Exception as error:
        if isinstance(error, InputError):
            row.update(source=str(error.path), regularization=error.regularization)
            error = error.error
        row.update(status="failed", error_type=type(error).__name__, error=str(error))
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


def file_results(format_name, directories, output, filenames, workers):
    """Yield completed results promptly, with bounded queued work and memory."""
    if workers == 1:
        for filename in filenames:
            yield process_file(format_name, directories, output, filename)
        return

    # Fresh processes avoid sharing HDF5 state inherited from the parent.
    pool = ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"))
    remaining = iter(filenames)
    pending = set()

    def submit_next():
        filename = next(remaining, None)
        if filename is not None:
            pending.add(pool.submit(process_file, format_name, directories, output, filename))

    try:
        for _ in range(2 * workers):
            submit_next()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.remove(future)
                yield future.result()
                submit_next()
    finally:
        # On interruption, finish active files and cancel queued files. Their
        # completed outputs can be skipped safely when the job is resumed.
        pool.shutdown(wait=True, cancel_futures=True)


def input_filenames(directories) -> list[str]:
    filenames = set()
    for _, directory in directories:
        # DirEntry can use directory metadata instead of a separate stat call
        # for every file on the network filesystem. Only the parent scans.
        with os.scandir(directory) as entries:
            for entry in entries:
                if Path(entry.name).suffix.lower() in {".h5", ".hdf5"} and entry.is_file():
                    filenames.add(entry.name)
    return sorted(filenames)


def run(args, plans) -> int:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    totals = {"success": 0, "skipped": 0, "failed": 0}
    with args.log_file.open("x", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        report.flush()
        print(f"Requested workers: {args.workers}; native threads per worker: 1\n"
              f"CSV report: {args.log_file}", flush=True)
        for format_name, directories, output in plans:
            # Use the union so files missing from even the first directory are logged.
            filenames = input_filenames(directories)
            if args.limit is not None:
                filenames = filenames[:args.limit]
            if not filenames:
                print(f"No HDF5 files found for {format_name}", file=sys.stderr, flush=True)
                totals["failed"] += 1
                continue
            output.mkdir(parents=True, exist_ok=True)
            counts = {"success": 0, "skipped": 0, "failed": 0}
            workers = min(args.workers, len(filenames))
            started = time.perf_counter()
            print(f"\n{format_name.upper()}: {len(filenames)} files; workers: {workers} -> {output}", flush=True)
            with closing(file_results(format_name, directories, output, filenames, workers)) as results:
                for index, row in enumerate(results, 1):
                    if row["status"] == "failed":
                        print(f"FAILED {format_name}/{row['filename']} {row['source']}: "
                              f"{row['error_type']}: {row['error']}", flush=True)
                    writer.writerow(row)
                    report.flush()
                    counts[row["status"]] += 1
                    totals[row["status"]] += 1
                    if index % args.progress_every == 0 or index == len(filenames):
                        elapsed = time.perf_counter() - started
                        print(f"[{format_name} {index}/{len(filenames)}] success={counts['success']} "
                              f"skipped={counts['skipped']} failed={counts['failed']} "
                              f"elapsed={elapsed:.1f}s rate={index / max(elapsed, 0.001):.2f} files/s", flush=True)
    print(f"Finished: success={totals['success']}, skipped={totals['skipped']}, "
          f"failed={totals['failed']}\nCSV report: {args.log_file}", flush=True)
    return 1 if totals["failed"] else 0


def main(argv=None) -> int:
    args, plans = parse_args(argv)
    try:
        # Fail before creating outputs if the selected runtime lacks dependencies.
        import h5py  # noqa: F401
        import numpy  # noqa: F401
        if "geon" in args.formats:
            from geon.data.document import Document  # noqa: F401
            from geon.data.pointcloud import PointCloudData  # noqa: F401
        return run(args, plans)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Rerun to continue; completed outputs are skipped.\n"
              f"CSV report: {args.log_file}", file=sys.stderr)
        return 130
    except BrokenProcessPool as error:
        print(f"Worker process stopped unexpectedly: {error}\n"
              f"Rerun to continue; completed outputs are skipped.\nCSV report: {args.log_file}",
              file=sys.stderr)
        return 1
    except (ImportError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
