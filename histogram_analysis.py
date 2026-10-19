"""
Purity (%) = 100 * largest instance-label count / total points in a superpoint.
The normal histogram counts superpoints; the point-weighted histogram counts
all original points belonging to those superpoints. Both use the same purity
bins. A scatter plot shows superpoint size versus purity, with one dot per
superpoint. Superpoint IDs are grouped separately within each point cloud,
even when IDs repeat in other clouds or files. Every instance label is included.

Examples (run in the hpnet_l40s environment):
    python histogram_analysis.py /path/to/00000_geon_superpoints.h5
    python histogram_analysis.py /data/users/wm-sharath/fib/ABC_final_geon_superpoints_reg_0.05
    python histogram_analysis.py /path/to/geon_outputs --limit 10 --bins 20

"""
from __future__ import annotations

from geon.data.document import Document
from geon.data.pointcloud import PointCloudData

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _id_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Accept the (N,) and (N, 1) layouts used for geon instance fields."""
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array with shape (N,) or (N, 1)")
    return values


def calculate_purities_and_sizes(
    superpoints: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned purity percentages and point counts for each superpoint."""
    superpoints = _id_vector(superpoints, "Superpoint IDs")
    labels = _id_vector(labels, "Instance labels")
    if len(superpoints) != len(labels):
        raise ValueError("Superpoint IDs and instance labels must have the same length")
    if not len(superpoints):
        raise ValueError("Cannot calculate purity for an empty point cloud")

    # Use compact indices: IDs need not start at zero or be consecutive.
    _, sp_indices, totals = np.unique(superpoints, return_inverse=True, return_counts=True)
    _, label_indices = np.unique(labels, return_inverse=True)
    pairs, pair_counts = np.unique(
        np.column_stack((sp_indices, label_indices)), axis=0, return_counts=True
    )
    majority_counts = np.zeros(len(totals), dtype=np.int64)
    np.maximum.at(majority_counts, pairs[:, 0], pair_counts)
    return 100.0 * (majority_counts / totals), totals


def load_purities_and_sizes(
    source: Path, superpoint_field: str, label_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read each cloud separately and combine its aligned purities and sizes."""

    document = Document.load_hdf5(source)
    clouds = [item for item in document.scene_items.values() if isinstance(item, PointCloudData)]
    if not clouds:
        raise ValueError("Document contains no point clouds")

    purity_parts = []
    size_parts = []
    for cloud in clouds:
        arrays = []
        for name in (superpoint_field, label_field):
            fields = cloud.get_fields(names=name)
            if len(fields) != 1:
                raise ValueError(f"Point cloud {cloud.id}: expected one field named {name!r}")
            values = _id_vector(fields[0].data, name)
            if len(values) != len(cloud.points):
                raise ValueError(f"Point cloud {cloud.id}: field {name!r} must contain one ID per point")
            arrays.append(values)
        purities, sizes = calculate_purities_and_sizes(*arrays)
        purity_parts.append(purities)
        size_parts.append(sizes)
    return np.concatenate(purity_parts), np.concatenate(size_parts)


def _save_histogram(
    output: Path, counts: np.ndarray, edges: np.ndarray,
    title: str, ylabel: str, summary: str, count_column: str,
) -> None:
    """Share plot styling and CSV output between the two histogram types."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color="#3579b8", edgecolor="white", linewidth=0.6)
    ax.set(xlabel="Purity (%)", ylabel=ylabel, title=title, xlim=(0, 100))
    ax.set_xticks(np.arange(0, 101, 10))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.97, summary,
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["purity_min_percent", "purity_max_percent", count_column])
        writer.writerows(zip(edges[:-1], edges[1:], counts))


def save_superpoint_histogram(
    output: Path, counts: np.ndarray, edges: np.ndarray,
    title: str, total: int, pure: int, mean: float,
) -> None:
    """Plot how many superpoints fall in each purity bin."""
    summary = (
        f"Superpoints: {total:,}\n"
        f"Exactly 100% pure: {pure:,} ({100 * pure / total:.2f}%)\n"
        f"Mean purity: {mean:.2f}%"
    )
    _save_histogram(output, counts, edges, title, "Number of superpoints",
                    summary, "number_of_superpoints")


def save_point_weighted_histogram(
    output: Path, counts: np.ndarray, edges: np.ndarray,
    title: str, total: int, pure: int, mean: float,
) -> None:
    """Plot original point counts, assigning each superpoint's full size to its bin."""
    summary = (
        f"Points: {total:,}\n"
        f"Points in 100% pure superpoints: {pure:,} ({100 * pure / total:.2f}%)\n"
        f"Point-weighted mean purity: {mean:.2f}%"
    )
    _save_histogram(output, counts, edges, title, "Number of original points",
                    summary, "number_of_points")


def save_size_vs_purity_plot(
    output: Path, sizes: np.ndarray, purities: np.ndarray, title: str,
) -> None:
    """Plot one dot per superpoint and export its size and purity to CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sizes, purities, s=6, alpha=0.25, color="#3579b8",
               edgecolors="none", rasterized=True)
    ax.set(xlabel="Number of points per superpoint", ylabel="Purity (%)",
           title=title, ylim=(0, 105))
    ax.set_xlim(left=0)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number_of_points", "purity_percent"])
        writer.writerows(zip(sizes, purities))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=Path, help="One native geon HDF5 file or a directory of files")
    parser.add_argument("--output", type=Path, help="Normal plot path (default: purity_histograms/<input-name>_purity.png); other plots add _point_weighted and _size_vs_purity to the filename")
    parser.add_argument("--bins", type=int, default=20, help="Equal-width bins from 0 to 100 (default: 20, i.e. 5%% each)")
    parser.add_argument("--superpoint-field", default="superpoints")
    parser.add_argument("--label-field", default="labels", help="Ground-truth instance field (default: labels)")
    parser.add_argument("--limit", type=int, help="Read only the first N sorted files for a quick trial")
    parser.add_argument("--skip-errors", action="store_true", help="Continue past unreadable/invalid files and report failures")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.input_path.exists():
        parser.error(f"Input does not exist: {args.input_path}")
    if args.output and args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf, or .svg")
    return args


def main() -> int:
    args = parse_args()
    source = args.input_path.resolve()
    if source.is_file():
        files = [source] if source.suffix.lower() in {".h5", ".hdf5"} else []
    else:
        files = sorted(path for path in source.iterdir()
                       if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"})
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No HDF5 files found: {source}", file=sys.stderr)
        return 1

    edges = np.linspace(0, 100, args.bins + 1)
    counts = np.zeros(args.bins, dtype=np.int64)
    point_counts = np.zeros(args.bins, dtype=np.int64)
    total = pure = failed = 0
    total_points = pure_points = 0
    purity_sum = 0.0
    weighted_purity_sum = 0.0
    purity_parts = []
    size_parts = []
    for index, path in enumerate(files, 1):
        try:
            purities, sizes = load_purities_and_sizes(path, args.superpoint_field, args.label_field)
        except (OSError, ValueError, KeyError) as error:
            print(f"FAILED {path.name}: {error}", file=sys.stderr, flush=True)
            if not args.skip_errors:
                return 1
            failed += 1
        else:
            # Histograms use bin totals; the scatter plot needs each superpoint's pair.
            counts += np.histogram(purities, bins=edges)[0]
            point_counts += np.histogram(purities, bins=edges, weights=sizes)[0]
            total += len(purities)
            pure += int(np.count_nonzero(purities == 100.0))
            purity_sum += float(purities.sum())
            total_points += int(sizes.sum())
            pure_points += int(sizes[purities == 100.0].sum())
            weighted_purity_sum += float(np.dot(purities, sizes))
            purity_parts.append(purities)
            size_parts.append(sizes)
        if index % 100 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] superpoints={total:,}; failed files={failed}", flush=True)

    if not total:
        print("No superpoints available for the histogram.", file=sys.stderr)
        return 1
    output_name = source.stem if source.is_file() else source.name
    output = args.output or Path("purity_histograms") / f"{output_name}_purity.png"
    weighted_output = output.with_name(f"{output.stem}_point_weighted{output.suffix}")
    scatter_output = output.with_name(f"{output.stem}_size_vs_purity{output.suffix}")
    excluded = f" ({failed} files excluded)" if failed else ""
    title = f"Superpoint purity: {source.name}"
    save_superpoint_histogram(output, counts, edges, title + excluded,
                             total, pure, purity_sum / total)
    weighted_title = f"Point-weighted purity: {source.name}"
    save_point_weighted_histogram(weighted_output, point_counts, edges,
                                 weighted_title + excluded, total_points, pure_points,
                                 weighted_purity_sum / total_points)
    scatter_title = f"Superpoint size vs. purity: {source.name}"
    save_size_vs_purity_plot(scatter_output, np.concatenate(size_parts),
                            np.concatenate(purity_parts), scatter_title + excluded)
    print(f"Files analyzed: {len(files) - failed}; failed: {failed}\n"
          f"Superpoints: {total:,}\nExactly 100% pure: {pure:,} ({100 * pure / total:.2f}%)\n"
          f"Mean purity: {purity_sum / total:.2f}%\n"
          f"Original points: {total_points:,}\n"
          f"Points in 100% pure superpoints: {pure_points:,} ({100 * pure_points / total_points:.2f}%)\n"
          f"Point-weighted mean purity: {weighted_purity_sum / total_points:.2f}%\n"
          f"Histogram: {output.resolve()}\nBin counts: {output.with_suffix('.csv').resolve()}")
    print(f"Point-weighted histogram: {weighted_output.resolve()}\n"
          f"Point-weighted bin counts: {weighted_output.with_suffix('.csv').resolve()}")
    print(f"Size vs. purity plot: {scatter_output.resolve()}\n"
          f"Size vs. purity data: {scatter_output.with_suffix('.csv').resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
