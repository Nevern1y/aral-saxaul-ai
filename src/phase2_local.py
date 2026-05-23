"""
Phase 2 (local): Export raw GEE data -> compute indices locally -> label -> CSV.

This avoids GEE's heavy batch exports by using fast getPixels downloads
of raw bands, then computing all indices in numpy.
"""

import os
import time
import numpy as np
import pandas as pd
import ee
import geemap
import rasterio
from scipy.ndimage import sobel

from src.config import config
from src.utils import initialize_gee, logger


def _deg_to_meters(lat):
    return 111320.0, 111320.0 * np.cos(np.deg2rad(lat))


def export_s2_mosaic(aoi, scale, out_dir):
    """Export S2 mosaic (B2, B3, B4, B8, B11, B12) at given scale."""
    path = os.path.join(out_dir, f"s2_raw_{scale}m.tif")
    if os.path.exists(path):
        logger.info("S2 mosaic already cached: %s", path)
        return path

    s2 = (
        ee.ImageCollection(config.S2_DATASET)
        .filterBounds(aoi)
        .filterDate("2025-08-15", "2025-08-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    mosaic = s2.mosaic().select(["B2", "B3", "B4", "B8", "B11", "B12"]).clip(aoi).toFloat()

    logger.info("Exporting S2 mosaic at %dm...", scale)
    geemap.ee_export_image(mosaic, filename=path, scale=scale, region=aoi, crs="EPSG:4326", file_per_band=False)
    logger.info("S2 mosaic downloaded: %.1f MB", os.path.getsize(path) / 1e6)
    return path


def export_dem(aoi, scale, out_dir):
    """Export Copernicus DEM at given scale."""
    path = os.path.join(out_dir, f"dem_{scale}m.tif")
    if os.path.exists(path):
        logger.info("DEM already cached: %s", path)
        return path

    dem = (
        ee.ImageCollection(config.DEM_DATASET)
        .filterBounds(aoi).mosaic().select("DEM").clip(aoi).toFloat()
    )
    logger.info("Exporting DEM at %dm...", scale)
    geemap.ee_export_image(dem, filename=path, scale=scale, region=aoi, crs="EPSG:4326", file_per_band=False)
    logger.info("DEM downloaded: %.1f MB", os.path.getsize(path) / 1e6)
    return path


def export_s1_vh(aoi, scale, out_dir):
    """Export Sentinel-1 VH median composite at given scale."""
    path = os.path.join(out_dir, f"s1_vh_{scale}m.tif")
    if os.path.exists(path):
        logger.info("S1 VH already cached: %s", path)
        return path

    s1 = (
        ee.ImageCollection(config.S1_DATASET)
        .filterBounds(aoi)
        .filterDate("2025-08-15", "2025-08-31")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    vh_db = s1.map(lambda img: img.select("VH").max(1e-6).log10().multiply(10.0)).median().clip(aoi).toFloat()
    logger.info("Exporting S1 VH at %dm...", scale)
    geemap.ee_export_image(vh_db, filename=path, scale=scale, region=aoi, crs="EPSG:4326", file_per_band=False)
    logger.info("S1 VH downloaded: %.1f MB", os.path.getsize(path) / 1e6)
    return path


def _compute_feature_stack(output_dir, scale_m, s2_path=None, dem_path=None, s1_path=None):
    """Load raw exports, compute all 7 indices, save as GeoTIFF, return path."""
    if s2_path is None:
        s2_path = os.path.join(output_dir, f"s2_raw_{scale_m}m.tif")
    if dem_path is None:
        dem_path = os.path.join(output_dir, f"dem_{scale_m}m.tif")
    if s1_path is None:
        s1_path = os.path.join(output_dir, f"s1_vh_{scale_m}m.tif")

    with rasterio.open(s2_path) as src:
        b2, b3, b4, b8, b11, b12 = [src.read(i).astype("float32") for i in range(1, 7)]
        profile = src.profile
        transform = src.transform

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")

    with rasterio.open(s1_path) as src:
        vh = src.read(1).astype("float32")

    eps = 1e-10
    ndmi = (b8 - b11) / (b8 + b11 + eps)
    msavi = (2 * b8 + 1 - np.sqrt(np.clip((2 * b8 + 1) ** 2 - 8 * (b8 - b4), 0, None))) / 2
    si = (b2 - b8) / (b2 + b8 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)

    height = b2.shape[0]
    lat_center = transform[5] + (height * transform[4]) / 2
    deg_to_m_y, deg_to_m_x = _deg_to_meters(lat_center)
    cellsize_x_m = abs(transform[0]) * deg_to_m_x
    cellsize_y_m = abs(transform[4]) * deg_to_m_y
    dzdx = sobel(dem, axis=1) / (8 * cellsize_x_m)
    dzdy = sobel(dem, axis=0) / (8 * cellsize_y_m)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    slope = np.where(np.isfinite(slope), slope, 0.0).astype("float32")

    slope_rad = slope * np.pi / 180.0
    tan_slope = np.maximum(np.tan(slope_rad), 1e-6)
    twi = np.log(1.0 / tan_slope)

    stack = np.stack([ndmi, msavi, si, slope, twi, vh, ndwi], axis=0)
    profile.update(count=7, dtype="float32", compress="lzw")
    fs_path = os.path.join(output_dir, f"feature_stack_{scale_m}m.tif")
    with rasterio.open(fs_path, "w", **profile) as dst:
        for i in range(7):
            dst.write(stack[i], i + 1)
    logger.info("Feature stack saved: %.1f MB", os.path.getsize(fs_path) / 1e6)
    return fs_path


def run_phase2_local(
    aoi,
    scale_m=100,
    n_samples=5000,
    output_dir=None,
):
    if output_dir is None:
        output_dir = str(config.output_dir / "data")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export raw data from GEE
    s2_path = export_s2_mosaic(aoi, scale_m, output_dir)
    dem_path = export_dem(aoi, scale_m, output_dir)
    s1_path = export_s1_vh(aoi, scale_m, output_dir)

    # 2. Load bands + build feature stack
    fs_path = _compute_feature_stack(output_dir, scale_m, s2_path, dem_path, s1_path)
    with rasterio.open(fs_path) as src:
        stack = src.read()
        height, width = src.height, src.width
        ndmi, msavi, si, slope, twi, vh, ndwi = [stack[i] for i in range(7)]

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")

    # 3. Sample random points
    valid = (dem > -1000) & np.isfinite(ndmi)
    valid_indices = np.where(valid)
    n_valid = len(valid_indices[0])

    np.random.seed(42)
    n_samples = min(n_samples, n_valid)
    sample_idx = np.random.choice(n_valid, n_samples, replace=False)
    rows, cols = valid_indices[0][sample_idx], valid_indices[1][sample_idx]

    data = []
    for r, c in zip(rows, cols):
        data.append({
            "NDMI": ndmi[r, c], "MSAVI": msavi[r, c], "SI": si[r, c],
            "Slope": slope[r, c], "TWI": twi[r, c], "VH": vh[r, c],
            "NDWI": ndwi[r, c],
        })
    df = pd.DataFrame(data)

    # 5. Label: SI >= P85 (bare saline soil) AND Slope <= 5 (flat terrain)
    si_p85 = df["SI"].quantile(0.85)
    df["label"] = ((df["SI"] >= si_p85) & (df["Slope"] <= 5)).astype(int)

    logger.info(
        "Labels: %d pos / %d neg (%.1f%% suitable)",
        df["label"].sum(), len(df) - df["label"].sum(),
        100 * df["label"].mean(),
    )

    csv_path = os.path.join(output_dir, "synthetic_labels.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Training data saved: %s (%d rows)", csv_path, len(df))
    return csv_path
