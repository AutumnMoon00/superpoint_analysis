from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path


_WORKER_CONFIG = None
_UTILS = None


def limit_native_threads() -> None:
    # Set before importing numpy/geon, including in Windows spawned workers.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"


def initialize_worker(config: dict) -> None:
    global _WORKER_CONFIG, _UTILS
    limit_native_threads()
    import superpoints_utils as utils
    _WORKER_CONFIG = config
    _UTILS = utils


def process_file(source: Path) -> dict:
    """Only small result records cross process boundaries, never point arrays."""
    started = time.perf_counter()
    config = _WORKER_CONFIG
    utils = _UTILS
    geon_path = config["geon_output_dir"] / f"{source.stem}_geon_superpoints.h5"
    abc_path = config["abc_output_dir"] / f"{source.stem}_abc_superpoints.h5"
    result = {"source": str(source), "status": "", "points": "",
              "superpoints": "", "seconds": "", "geon_output": str(geon_path),
              "abc_output": str(abc_path), "error": ""}
    try:
        if geon_path.is_file() and abc_path.is_file() and not config["overwrite"]:
            result["status"] = "skipped"
        else:
            cloud = utils.load_abc_as_geon(source)
            settings = config["settings"]
            if settings["k_neighbors"] >= len(cloud.points):
                raise ValueError("k-neighbors must be smaller than the number of points")
            labels, stats = utils.segment_superpoints(
                cloud, feature_field_names=config["feature_names"], **settings
            )
            np = utils.np
            if stats.get("cancelled") or labels.shape != (len(cloud.points),) or np.any(labels < 0):
                raise RuntimeError("geon did not return a valid superpoint ID for every point")
            # Replace partial pairs together so both files contain the same IDs.
            utils.save_outputs(
                source, config["geon_output_dir"], config["abc_output_dir"],
                cloud, labels,
                {**settings, "feature_field_names": config["feature_names"], "selection_seed": None},
                overwrite=True,
            )
            result.update(status="success", points=len(cloud.points),
                          superpoints=len(np.unique(labels)))
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("/data/users/wm-sharath/fib/ABC_final/"))
    parser.add_argument("--geon-output-dir", type=Path,
                        default=Path("/data/users/wm-sharath/fib/ABC_final_geon_superpoints/"),
                        help="GEON output base directory; _reg_<value> is appended")
    parser.add_argument("--abc-output-dir", type=Path,
                        default=Path("/data/users/wm-sharath/fib/ABC_final_abc_superpoints/"),
                        help="ABC output base directory; _reg_<value> is appended")
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)),
                        help="Number of worker processes (default: up to 8)")
    parser.add_argument("--limit", type=int, help="Process only the first N sorted input files for each value")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even when both outputs exist")
    parser.add_argument("--log-file", type=Path,
                        help="CSV report base path; each value adds _reg_<value> before the extension")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--regularizations", "--regularization", dest="regularizations",
                        type=float, nargs="+", default=[0.1],
                        help="Regularization values to run (default: 0.025 0.05 0.1)")
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--coordinates-only", action="store_true")
    args = parser.parse_args()
    for name in ("workers", "progress_every", "k_neighbors", "cutoff", "iterations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in args.regularizations):
        parser.error("--regularizations must all be finite and positive")
    # Run repeated values only once, preserving their requested order.
    args.regularizations = list(dict.fromkeys(args.regularizations))
    if not math.isfinite(args.spatial_weight) or args.spatial_weight < 0:
        parser.error("--spatial-weight must be finite and nonnegative")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    return args


def main() -> int:
    limit_native_threads()
    args = parse_args()
    files = sorted(path.resolve() for path in args.input_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"})
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No HDF5 files found in {args.input_dir}", file=sys.stderr)
        return 1
    # Different extensions with the same stem must not target the same output.
    stems = [path.stem.casefold() for path in files]
    if len(set(stems)) != len(stems):
        print("Input files have duplicate stems; output filenames would collide.", file=sys.stderr)
        return 1

    exit_code = 0
    for regularization in args.regularizations:
        result = run_experiment(args, files, regularization)
        if result == 130:
            return result
        if result:
            exit_code = 1
    return exit_code


def run_experiment(args: argparse.Namespace, files: list[Path], regularization: float) -> int:
    """Keep outputs, logs, and worker configuration separate for each value."""
    tag = f"reg_{regularization}"
    geon_base = args.geon_output_dir.resolve()
    abc_base = args.abc_output_dir.resolve()
    geon_output_dir = geon_base.with_name(f"{geon_base.name}_{tag}")
    abc_output_dir = abc_base.with_name(f"{abc_base.name}_{tag}")
    for directory in (geon_output_dir, abc_output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = args.log_file.with_name(
            f"{args.log_file.stem}_{tag}{args.log_file.suffix}"
        )
    else:
        log_path = (
            Path(__file__).resolve().parent / "batch_logs" /
            f"superpoints_{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        )
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "geon_output_dir": geon_output_dir, "abc_output_dir": abc_output_dir,
        "overwrite": args.overwrite,
        "feature_names": [] if args.coordinates_only else ["normals"],
        "settings": {"k_neighbors": args.k_neighbors, "regularization": regularization,
                     "spatial_weight": args.spatial_weight, "cutoff": args.cutoff,
                     "iterations": args.iterations, "parallel": False},
    }
    workers = min(args.workers, len(files))
    print(f"\nRegularization: {regularization}", flush=True)
    print(f"Files: {len(files)}; workers: {workers}; features: {config['feature_names']}", flush=True)
    print(f"Input: {args.input_dir.resolve()}\ngeon output: {geon_output_dir}\n"
          f"ABC output: {abc_output_dir}\nCSV report: {log_path}", flush=True)
    counts = {"success": 0, "skipped": 0, "failed": 0}
    started = time.perf_counter()
    exit_code = 0
    columns = ["source", "status", "points", "superpoints", "seconds",
               "geon_output", "abc_output", "error"]
    # Exclusive creation prevents an explicit log path from truncating an old report.
    with log_path.open("x", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=columns)
        writer.writeheader()
        report.flush()
        pool = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initialize_worker, initargs=(config,))
        pending = {}
        source_iterator = iter(files)

        def submit_next() -> bool:
            source = next(source_iterator, None)
            if source is None:
                return False
            pending[pool.submit(process_file, source)] = source
            return True

        try:
            # Bound the queue instead of submitting all 32,113 files at once.
            for _ in range(workers * 2):
                if not submit_next():
                    break
            while pending:
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    source = pending.pop(future)
                    result = future.result()
                    writer.writerow(result)
                    report.flush()
                    counts[result["status"]] += 1
                    done = sum(counts.values())
                    if result["status"] == "failed":
                        print(f"FAILED {source.name}: {result['error'].strip().splitlines()[-1]}", flush=True)
                    if done % args.progress_every == 0 or done == len(files):
                        print(f"[{done}/{len(files)}] success={counts['success']} "
                              f"skipped={counts['skipped']} failed={counts['failed']} "
                              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
                    submit_next()
        except KeyboardInterrupt:
            print("\nInterrupted. Waiting for active files; rerun to continue.", flush=True)
            exit_code = 130
        except BrokenProcessPool as error:
            print(f"Worker process stopped unexpectedly: {error}. Rerun to continue.", file=sys.stderr)
            exit_code = 1
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"Finished: success={counts['success']}, skipped={counts['skipped']}, "
          f"failed={counts['failed']}; elapsed={time.perf_counter() - started:.1f}s\n"
          f"CSV report: {log_path}", flush=True)
    return exit_code or (1 if counts["failed"] else 0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
