import argparse
import os

import numpy as np
import rasterio

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _count_valid(src: rasterio.DatasetReader, nodata: float | None) -> int:
    total_valid = 0
    for _, window in src.block_windows(1):
        data = src.read(1, window=window)
        valid = np.isfinite(data)
        if nodata is not None and not np.isnan(nodata):
            valid &= data != nodata
        total_valid += int(valid.sum())
    return total_valid


def _sample_values(
    src: rasterio.DatasetReader,
    nodata: float | None,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values: list[np.ndarray] = []

    total_valid = _count_valid(src, nodata)
    if total_valid == 0:
        return np.array([], dtype=np.float32)

    prob = min(1.0, sample_size / total_valid)

    for _, window in src.block_windows(1):
        data = src.read(1, window=window)
        valid = np.isfinite(data)
        if nodata is not None and not np.isnan(nodata):
            valid &= data != nodata

        if not valid.any():
            continue

        vals = data[valid]
        if prob >= 1.0:
            values.append(vals)
        else:
            keep = rng.random(vals.size) < prob
            if keep.any():
                values.append(vals[keep])

    if not values:
        return np.array([], dtype=np.float32)

    sample = np.concatenate(values)
    if sample.size > sample_size:
        idx = rng.choice(sample.size, size=sample_size, replace=False)
        sample = sample[idx]

    return sample.astype(np.float32, copy=False)


def plot_histogram(
    input_path: str,
    output_path: str,
    sample_size: int = 1_500_000,
    bins: int = 80,
    seed: int = 42,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = np.random.default_rng(seed)

    with rasterio.open(input_path) as src:
        nodata = src.nodata
        sample = _sample_values(src, nodata, sample_size, rng)

    if sample.size == 0:
        raise RuntimeError("No valid pixels found for histogram.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sample, bins=bins, range=(0.0, 1.0), color="#2ecc71", edgecolor="#1b7f4b")

    ax.set_title("Suitability Probability Distribution", fontsize=14)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Pixel Count (sample)", fontsize=12)
    ax.grid(axis="y", alpha=0.25)

    stats = {
        "sample_size": int(sample.size),
        "min": float(sample.min()),
        "max": float(sample.max()),
        "mean": float(sample.mean()),
        "p10": float(np.percentile(sample, 10)),
        "p50": float(np.percentile(sample, 50)),
        "p90": float(np.percentile(sample, 90)),
    }

    stats_text = (
        f"n={stats['sample_size']:,}  "
        f"min={stats['min']:.3f}  max={stats['max']:.3f}  mean={stats['mean']:.3f}\n"
        f"p10={stats['p10']:.3f}  p50={stats['p50']:.3f}  p90={stats['p90']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Histogram saved: {output_path}")
    print(
        "Stats: "
        f"n={stats['sample_size']:,}, min={stats['min']:.4f}, max={stats['max']:.4f}, "
        f"mean={stats['mean']:.4f}, p10={stats['p10']:.4f}, p50={stats['p50']:.4f}, p90={stats['p90']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histogram of suitability probabilities from GeoTIFF."
    )
    parser.add_argument(
        "--input",
        default="outputs/data/suitability_full.tif",
        help="Input suitability raster.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/probability_histogram.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1_500_000,
        help="Sample size of valid pixels.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    plot_histogram(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample,
        bins=args.bins,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
