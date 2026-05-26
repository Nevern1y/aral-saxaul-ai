"""
V5.0 Rule-Based Classification with Multi-Index Salt Physics
Sentinel-2 10m native resolution, in-memory SWIR upsampling via WarpedVRT.

Classes
-------
0  WATER_NODATA   — SCL clouds/water/shadow, MNDWI>0, or BI<0.15 shadow
1  OPTIMAL        — Default for bare dry land
3  RISK_DRY_SALT  — High NDSI_Green_SWIR2 + low NDMI (dry salt crust)
4  DEAD_WET_TOXIC — High NDMI + high BR_NIR_SWIR2 (capillary brine)
5  OBSTACLE_TOPO  — Slope > 5°
10 VEGETATION     — NDVI > 0.08
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from v5_config import ZoneClass

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# FUNCTION 1: GLOBAL THRESHOLDS via Systematic Grid Subsampling
# ============================================================================


def compute_global_thresholds(paths: Dict[str, Path]) -> Dict[str, float]:
    """Compute P15/P85 thresholds from strided subsamples across the AOI.

    Strategy
    --------
    - 10m bands (B3, B4, B8, SCL):  step = 20  → 200 m effective spacing
    - 20m bands (B11, B12):          step = 10  → 200 m effective spacing
    - Mask: SCL == 5 (Bare Soils) AND MNDWI <= 0

    Returns
    -------
    dict with keys:
        NDMI_P15, NDMI_P85,
        NDSI_Green_SWIR2_P15, NDSI_Green_SWIR2_P85,
        BR_NIR_SWIR2_P15, BR_NIR_SWIR2_P85
    """
    log.info("=" * 60)
    log.info("COMPUTE GLOBAL THRESHOLDS  —  Strided Sampling")
    log.info("=" * 60)

    t0 = time.time()

    # --- Read 10 m bands (step = 20) ---
    step_10 = 20
    with rasterio.open(paths["B3"]) as src:
        b3 = src.read(1)[::step_10, ::step_10].ravel().astype(np.float32)
    with rasterio.open(paths["B4"]) as src:
        b4 = src.read(1)[::step_10, ::step_10].ravel().astype(np.float32)
    with rasterio.open(paths["B8"]) as src:
        b8 = src.read(1)[::step_10, ::step_10].ravel().astype(np.float32)
    with rasterio.open(paths["SCL"]) as src:
        scl = src.read(1)[::step_10, ::step_10].ravel().astype(np.float32)

    # --- Read 20 m bands at native resolution (step = 10) ---
    step_20 = 10
    with rasterio.open(paths["B11"]) as src:
        b11 = src.read(1)[::step_20, ::step_20].ravel().astype(np.float32)
    with rasterio.open(paths["B12"]) as src:
        b12 = src.read(1)[::step_20, ::step_20].ravel().astype(np.float32)

    # Truncate to matching pixel count (both sample grids are ~200 m)
    min_len = min(len(b3), len(b11))
    if min_len == 0:
        raise RuntimeError("No valid pixels after strided sampling — check input rasters")

    b3, b4, b8, scl = b3[:min_len], b4[:min_len], b8[:min_len], scl[:min_len]
    b11, b12 = b11[:min_len], b12[:min_len]

    # --- Convert to reflectance ---
    b3 /= 10000.0
    b4 /= 10000.0
    b8 /= 10000.0
    b11 /= 10000.0
    b12 /= 10000.0

    # --- Land mask (native 10m NDWI — no resampling artefact) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (b3 - b8) / (b3 + b8)

    valid = (
        ~np.isin(scl, [3, 6, 7, 8, 9, 10])
        & (ndwi <= 0)
        & np.isfinite(ndwi)
        & np.isfinite(b3)
        & np.isfinite(b8)
        & np.isfinite(b11)
        & np.isfinite(b12)
    )

    n_valid = int(valid.sum())
    log.info("  Samples: %d total, %d valid (SCL≠3/6/7/8/9/10 & NDWI<=0)", len(b3), n_valid)

    if n_valid < 1000:
        log.warning("  Very few valid samples (%d) — thresholds may be unreliable", n_valid)

    b3_v, b8_v = b3[valid], b8[valid]
    b11_v, b12_v = b11[valid], b12[valid]

    with np.errstate(divide="ignore", invalid="ignore"):
        ndmi = (b8_v - b11_v) / (b8_v + b11_v)
        ndsi_green_swir2 = (b3_v - b12_v) / (b3_v + b12_v)
        br_nir_swir2 = b8_v / b12_v

    thresholds = {
        "NDMI_P15": float(np.nanpercentile(ndmi, 15)),
        "NDMI_P85": float(np.nanpercentile(ndmi, 85)),
        "NDSI_Green_SWIR2_P15": float(np.nanpercentile(ndsi_green_swir2, 15)),
        "NDSI_Green_SWIR2_P85": float(np.nanpercentile(ndsi_green_swir2, 85)),
        "BR_NIR_SWIR2_P15": float(np.nanpercentile(br_nir_swir2, 15)),
        "BR_NIR_SWIR2_P85": float(np.nanpercentile(br_nir_swir2, 85)),
    }

    elapsed = time.time() - t0
    log.info("  Elapsed: %.1f s", elapsed)
    log.info("  Thresholds:")
    for k, v in thresholds.items():
        log.info("    %-25s =  %.6f", k, v)

    return thresholds


# ============================================================================
# FUNCTION 2: BLOCK-WISE CLASSIFICATION
# ============================================================================


def run_block_inference(
    paths: Dict[str, Path],
    thresholds: Dict[str, float],
    output_path: Path,
) -> None:
    """Block-wise rule-based classification with in-memory SWIR upsampling.

    Parameters
    ----------
    paths : dict
        Keys: B3, B4, B8, B11, B12, SCL, Slope
    thresholds : dict
        From ``compute_global_thresholds``.
    output_path : Path
        Path for the resulting ``uint8`` suitability GeoTIFF.
    """
    log.info("=" * 60)
    log.info("RUN BLOCK INFERENCE  V5.0")
    log.info("=" * 60)

    t_start = time.time()

    # ── Open all sources ──────────────────────────────────────────────
    b8_src = rasterio.open(paths["B8"])
    b3_src = rasterio.open(paths["B3"])
    b4_src = rasterio.open(paths["B4"])
    scl_src = rasterio.open(paths["SCL"])

    b11_src = rasterio.open(paths["B11"])
    b12_src = rasterio.open(paths["B12"])
    slope_src = rasterio.open(paths["Slope"])

    # ── WarpedVRT: 20 m / 30 m → 10 m grid (matches B8) ──────────────
    vrt_kw: Dict[str, Any] = dict(
        crs=b8_src.crs,
        transform=b8_src.transform,
        width=b8_src.width,
        height=b8_src.height,
    )
    b11_vrt = WarpedVRT(b11_src, resampling=Resampling.bilinear, **vrt_kw)
    b12_vrt = WarpedVRT(b12_src, resampling=Resampling.bilinear, **vrt_kw)
    slope_vrt = WarpedVRT(slope_src, resampling=Resampling.cubic, **vrt_kw)

    # ── Output profile ────────────────────────────────────────────────
    profile = b8_src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress="lzw",
        nodata=int(ZoneClass.WATER_NODATA),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_NEEDED",
    )

    windows = list(b8_src.block_windows(1))
    log.info(
        "  Reference grid: %d × %d px  (%d blocks, %s windowing)",
        b8_src.width,
        b8_src.height,
        len(windows),
        "B8 block_windows",
    )

    # ── Block loop ────────────────────────────────────────────────────
    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (_, window) in enumerate(windows):
            if (idx + 1) % 50 == 0 or idx == 0:
                elapsed = time.time() - t_start
                pct = (idx + 1) / len(windows) * 100
                log.info(
                    "  Block [%d/%d]  %.0f%%  (%ds)",
                    idx + 1,
                    len(windows),
                    pct,
                    int(elapsed),
                )

            # --- Read block ---
            try:
                b3 = b3_src.read(1, window=window).astype(np.float32)
                b4 = b4_src.read(1, window=window).astype(np.float32)
                b8 = b8_src.read(1, window=window).astype(np.float32)
                scl = scl_src.read(1, window=window).astype(np.float32)

                b11 = b11_vrt.read(1, window=window).astype(np.float32)
                b12 = b12_vrt.read(1, window=window).astype(np.float32)
                slope = slope_vrt.read(1, window=window).astype(np.float32)
            except Exception as exc:
                log.warning("  Block %d read error: %s — filling NODATA", idx, exc)
                dst.write(
                    np.full(
                        (window.height, window.width),
                        ZoneClass.WATER_NODATA,
                        dtype=np.uint8,
                    ),
                    1,
                    window=window,
                )
                continue

            # --- Reflectance (raw DN → [0, 1]) ---
            b3 /= 10000.0
            b4 /= 10000.0
            b8 /= 10000.0
            b11 /= 10000.0
            b12 /= 10000.0

            # --- Spectral indices ---
            with np.errstate(divide="ignore", invalid="ignore"):
                ndwi = (b3 - b8) / (b3 + b8)
                ndvi = (b8 - b4) / (b8 + b4)
                ndmi = (b8 - b11) / (b8 + b11)
                ndsi_green_swir2 = (b3 - b12) / (b3 + b12)
                br_nir_swir2 = b8 / b12
                bi = np.sqrt(b3 ** 2 + b4 ** 2)

            rows, cols = b8.shape

            # --- Classification cascade (priority order) ---
            cond_water = (
                np.isin(scl, [3, 6, 7, 8, 9, 10])
                | (ndwi > 0)
                | ~np.isfinite(ndwi)
            )

            cond_topo = (
                ~cond_water
                & np.isfinite(slope)
                & (slope > 5.0)
            )

            cond_veg = (
                ~cond_water
                & ~cond_topo
                & np.isfinite(ndvi)
                & (ndvi > 0.08)
            )

            cond_shadow = (
                ~cond_water
                & ~cond_topo
                & ~cond_veg
                & np.isfinite(bi)
                & (bi < 0.15)
            )

            cond_dead = (
                ~cond_water
                & ~cond_topo
                & ~cond_veg
                & ~cond_shadow
                & np.isfinite(ndmi)
                & np.isfinite(br_nir_swir2)
                & (ndmi > thresholds["NDMI_P85"])
                & (br_nir_swir2 > thresholds["BR_NIR_SWIR2_P85"])
            )

            cond_risk = (
                ~cond_water
                & ~cond_topo
                & ~cond_veg
                & ~cond_shadow
                & ~cond_dead
                & np.isfinite(ndsi_green_swir2)
                & np.isfinite(ndmi)
                & (ndsi_green_swir2 > thresholds["NDSI_Green_SWIR2_P85"])
                & (ndmi < thresholds["NDMI_P15"])
            )

            out = np.select(
                [
                    cond_water.ravel(),
                    cond_topo.ravel(),
                    cond_veg.ravel(),
                    cond_shadow.ravel(),
                    cond_dead.ravel(),
                    cond_risk.ravel(),
                ],
                [
                    ZoneClass.WATER_NODATA,
                    ZoneClass.OBSTACLE_TOPO,
                    ZoneClass.VEGETATION,
                    ZoneClass.WATER_NODATA,
                    ZoneClass.DEAD_WET_TOXIC,
                    ZoneClass.RISK_DRY_SALT,
                ],
                default=ZoneClass.OPTIMAL,
            ).reshape(rows, cols).astype(np.uint8)

            dst.write(out, 1, window=window)

    # ── Cleanup ────────────────────────────────────────────────────────
    b3_src.close()
    b4_src.close()
    b8_src.close()
    scl_src.close()
    b11_src.close()
    b12_src.close()
    slope_src.close()
    b11_vrt.close()
    b12_vrt.close()
    slope_vrt.close()

    elapsed = time.time() - t_start
    log.info("  Inference complete: %d blocks in %.1f s", len(windows), elapsed)
    log.info("  Output: %s", output_path)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    import sys

    BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
    DATA = BASE / "outputs" / "data"

    BAND_PATHS = {
        "B3": DATA / "B3_10m.tif",
        "B4": DATA / "B4_10m.tif",
        "B8": DATA / "B8_10m.tif",
        "B11": DATA / "B11_20m.tif",
        "B12": DATA / "B12_20m.tif",
        "SCL": DATA / "SCL_10m.tif",
        "Slope": DATA / "dem_slope_30m.tif",
    }

    OUTPUT = DATA / "suitability_map_v5.tif"

    for key, p in BAND_PATHS.items():
        if not p.exists():
            log.error("Missing input: %s -> %s", key, p)
            sys.exit(1)

    import json

    log.info("V5.0  Aral Saxaul AI  —  Multi-Index Rule-Based Pipeline")
    log.info("Input bands:")
    for k, v in BAND_PATHS.items():
        log.info("  %-6s  %s", k, v)

    thresholds = compute_global_thresholds(BAND_PATHS)

    THRESHOLDS_OUT = DATA / "thresholds_v5.json"
    with open(THRESHOLDS_OUT, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Thresholds saved -> %s", THRESHOLDS_OUT)

    run_block_inference(BAND_PATHS, thresholds, OUTPUT)
