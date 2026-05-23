import argparse

import numpy as np
import rasterio


def audit(path: str, low: float = 0.1) -> None:
    with rasterio.open(path) as src:
        nodata = src.nodata

        total = 0
        nodata_count = 0
        nodata_value_count = 0
        valid_count = 0
        zero_count = 0
        low_count = 0
        nan_count = 0
        out_of_range_count = 0
        vmin = None
        vmax = None

        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            total += data.size

            is_nan = ~np.isfinite(data)
            nan_count += int(is_nan.sum())

            nodata_mask = is_nan.copy()
            if nodata is not None and not np.isnan(nodata):
                nodata_value_mask = data == nodata
                nodata_value_count += int(nodata_value_mask.sum())
                nodata_mask |= nodata_value_mask

            out_of_range_mask = (data < 0.0) | (data > 1.0)
            out_of_range_mask = out_of_range_mask & ~nodata_mask
            out_of_range_count += int(out_of_range_mask.sum())

            nodata_count += int(nodata_mask.sum())

            invalid = nodata_mask | out_of_range_mask
            valid = ~invalid
            valid_count += int(valid.sum())

            if valid.any():
                zero_count += int((valid & (data == 0.0)).sum())
                low_count += int((valid & (data < low)).sum())

                block_min = float(data[valid].min())
                block_max = float(data[valid].max())
                vmin = block_min if vmin is None else min(vmin, block_min)
                vmax = block_max if vmax is None else max(vmax, block_max)

        def pct(x: int, denom: int) -> float:
            return 0.0 if denom == 0 else (100.0 * x / denom)

        print(f"Raster: {path}")
        print(f"NoData value: {nodata}")
        print(f"Total pixels: {total:,}")
        print(f"NoData/NaN: {nodata_count:,} ({pct(nodata_count, total):.2f}%)")
        if nodata is not None and not np.isnan(nodata):
            print(f"  NoData value count: {nodata_value_count:,}")
        print(f"  NaN/Inf count: {nan_count:,}")
        print(
            "Out-of-range (<0 or >1): "
            f"{out_of_range_count:,} ({pct(out_of_range_count, total):.2f}%)"
        )
        print(f"Valid pixels: {valid_count:,} ({pct(valid_count, total):.2f}%)")
        print(f"Valid prob min/max: {vmin} / {vmax}")
        print(f"Prob == 0.0: {zero_count:,} ({pct(zero_count, valid_count):.2f}% of valid)")
        print(f"Prob < {low}: {low_count:,} ({pct(low_count, valid_count):.2f}% of valid)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit probability raster for NoData, valid coverage, and low-probability share."
        )
    )
    parser.add_argument("path", help="Path to suitability_full.tif")
    parser.add_argument("--low", type=float, default=0.1, help="Low-probability cutoff")
    args = parser.parse_args()

    audit(args.path, low=args.low)


if __name__ == "__main__":
    main()
