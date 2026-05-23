"""
Phase 4: Local tile-based inference on single GeoTIFF.
Processes a pre-computed Feature Stack GeoTIFF in chunks,
runs XGBoost inference, writes suitability probability raster.
"""

import os
import time
import json
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import config
from src.utils import logger


def run_phase4(
    input_path: str,
    output_path: str,
    model_path: str = None,
    scaler_path: str = None,
    tile_size_px: int = 1024,
    batch_size: int = 100000,
):
    if model_path is None:
        model_path = str(config.output_dir / "models" / "xgb_classifier.pkl")
    if scaler_path is None:
        scaler_path = str(config.output_dir / "models" / "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feature_names_path = str(config.output_dir / "models" / "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_order = json.load(f)
    else:
        feature_order = list(config.FEATURE_COLUMNS)

    if output_path is None:
        output_path = str(config.output_dir / "data" / "suitability_full.tif")

    logger.info("Phase 4 inference starting...")

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        n_bands = src.count
        logger.info("Raster: %d x %d x %d", n_bands, height, width)

        descriptions = list(src.descriptions) if src.descriptions else []
        if descriptions and any(d is not None for d in descriptions):
            name_to_index = {
                name.upper(): idx + 1
                for idx, name in enumerate(descriptions)
                if name is not None
            }
            missing = [name for name in feature_order if name.upper() not in name_to_index]
            if missing:
                raise ValueError(
                    "Missing expected bands in raster: "
                    f"{missing}. Available: {descriptions}"
                )
            band_indices = [name_to_index[name.upper()] for name in feature_order]
        else:
            if n_bands != len(feature_order):
                raise ValueError(
                    "Raster band count does not match feature order length "
                    f"({n_bands} vs {len(feature_order)})."
                )
            band_indices = list(range(1, len(feature_order) + 1))
            logger.warning("No band descriptions found; assuming raster band order.")

        logger.info("Feature order: %s", feature_order)
        logger.info("Band indices: %s", band_indices)

        # Build explicit GTiff profile (VRT profile can't be used for writing)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "compress": "lzw",
            "nodata": -1.0,
            "tiled": False,
            "blockxsize": 256,
            "blockysize": 256,
        }

        # Check memory: if raster is small enough, process in one shot
        total_pixels = height * width
        one_shot = total_pixels < 5_000_000  # < 5M pixels → direct

        if one_shot:
            bands = src.read(indexes=band_indices)
            pixels = bands.reshape(len(feature_order), -1).T.astype(np.float64)
            valid = np.all(np.isfinite(pixels), axis=1)
            nodata = src.nodata
            if nodata is not None and not np.isnan(nodata):
                valid &= ~np.any(pixels == nodata, axis=1)

            if valid.sum() == 0:
                logger.error("No valid pixels found!")
                return

            pixels_s = scaler.transform(pixels[valid])
            proba = model.predict_proba(pixels_s)[:, 1]

            suitability = np.full(total_pixels, -1.0, dtype=np.float32)
            suitability[valid] = proba
            suitability = suitability.reshape(height, width)

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(suitability, 1)

            n_high = (proba > 0.5).sum()
            logger.info(
                "Done: %d/%d valid, %d high-suit (>0.5), %.1f%%",
                valid.sum(), total_pixels, n_high,
                100 * n_high / valid.sum() if valid.sum() > 0 else 0,
            )
        else:
            # Tile-based processing for large rasters
            logger.info("Processing in tiles (%dx%d px)...", tile_size_px, tile_size_px)

            with rasterio.open(output_path, "w", **profile) as dst:
                for row_start in range(0, height, tile_size_px):
                    row_end = min(row_start + tile_size_px, height)
                    for col_start in range(0, width, tile_size_px):
                        col_end = min(col_start + tile_size_px, width)
                        tile_h = row_end - row_start
                        tile_w = col_end - col_start

                        window = Window(col_start, row_start, tile_w, tile_h)
                        bands = src.read(indexes=band_indices, window=window)

                        pixels = bands.reshape(len(feature_order), -1).T.astype(np.float64)
                        valid = np.all(np.isfinite(pixels), axis=1)
                        nodata = src.nodata
                        if nodata is not None and not np.isnan(nodata):
                            valid &= ~np.any(pixels == nodata, axis=1)

                        if valid.sum() == 0:
                            dst.write(
                                np.full((tile_h, tile_w), -1.0, dtype=np.float32),
                                1, window=window,
                            )
                            continue

                        # Process in batches to keep memory low
                        proba = np.zeros(valid.sum(), dtype=np.float32)
                        for i in range(0, valid.sum(), batch_size):
                            batch = pixels[valid][i : i + batch_size]
                            batch_s = scaler.transform(batch)
                            proba[i : i + batch_size] = model.predict_proba(batch_s)[:, 1]

                        tile_out = np.full(tile_h * tile_w, -1.0, dtype=np.float32)
                        tile_out[valid] = proba
                        tile_out = tile_out.reshape(tile_h, tile_w)

                        dst.write(tile_out, 1, window=window)

                    logger.info("  Row %d/%d", row_end, height)

            logger.info("Phase 4 complete: %s", output_path)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4: Local tile inference")
    parser.add_argument("--input", required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", default=None, help="Output suitability path")
    args = parser.parse_args()

    run_phase4(args.input, args.output)
