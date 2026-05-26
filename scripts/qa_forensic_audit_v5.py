"""
qa_forensic_audit_v5.py — Reverse-Engineering Pixel Audit for V5.0.

Picks 100 random pixels per class from the final map, re-computes all
spectral indices from raw band values, and asserts every classification
decision matches the original ``np.select`` cascade.

Usage:
    python scripts/qa_forensic_audit_v5.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("forensic_audit")

BASE = Path(__file__).resolve().parent.parent
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
MAP_PATH = DATA / "suitability_map_v5.tif"
THRESH_PATH = DATA / "thresholds_v5.json"

# ── Class labels (must match v5_config.ZoneClass) ─────────────────────
CLASS_NAMES = {0: "WATER_NODATA", 1: "OPTIMAL", 3: "RISK_DRY_SALT",
               4: "DEAD_WET_TOXIC", 5: "OBSTACLE_TOPO", 10: "VEGETATION"}

N_SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42


# =========================================================================
# 1. LOAD MAP + SAMPLE COORDINATES
# =========================================================================


def _sample_pixels(map_array: np.ndarray, class_val: int, n: int, rng: np.random.Generator
                   ) -> List[Tuple[int, int, int, int]]:
    """Return up to ``n`` (row, col) tuples for a given class in the map.

    Returns list of ``(row, col, block_row, block_col)``.
    """
    rows, cols = np.where(map_array == class_val)
    n_avail = len(rows)
    if n_avail == 0:
        return []
    chosen = rng.choice(n_avail, size=min(n, n_avail), replace=False)
    return [(int(rows[i]), int(cols[i]), int(rows[i] // 256), int(cols[i] // 256))
            for i in chosen]


# =========================================================================
# 2. COMPUTE INDICES + CHECK CASCADE
# =========================================================================


def _recompute_and_verify(
    b3: float, b4: float, b8: float, b11: float, b12: float,
    scl: float, slope: float,
    map_class: int,
    thresholds: Dict[str, float],
    row: int, col: int,
) -> None:
    """Convert raw DN to reflectance, compute all indices, run cascade assert."""
    # Reflectance
    b3 /= 10000.0
    b4 /= 10000.0
    b8 /= 10000.0
    b11 /= 10000.0
    b12 /= 10000.0

    with np.errstate(divide="ignore", invalid="ignore"):
        mndwi = (b3 - b11) / (b3 + b11) if (b3 + b11) != 0 else np.nan
        ndvi = (b8 - b4) / (b8 + b4) if (b8 + b4) != 0 else np.nan
        ndmi = (b8 - b11) / (b8 + b11) if (b8 + b11) != 0 else np.nan
        ndsi_green_swir2 = (b3 - b12) / (b3 + b12) if (b3 + b12) != 0 else np.nan
        br_nir_swir2 = b8 / b12 if b12 != 0 else np.nan
        bi = np.sqrt(b3 ** 2 + b4 ** 2)

    t = thresholds

    # ── Cascade (must match run_inference_v5.py exactly) ──────────────
    cond_water = (
        scl in (3, 8, 9, 10)
        or (np.isfinite(mndwi) and mndwi > 0)
        or not np.isfinite(mndwi)
    )
    cond_topo = (
        not cond_water
        and np.isfinite(slope)
        and slope > 5.0
    )
    cond_veg = (
        not cond_water and not cond_topo
        and np.isfinite(ndvi) and ndvi > 0.08
    )
    cond_shadow = (
        not cond_water and not cond_topo and not cond_veg
        and np.isfinite(bi) and bi < 0.15
    )
    cond_dead = (
        not cond_water and not cond_topo and not cond_veg and not cond_shadow
        and np.isfinite(ndmi) and np.isfinite(br_nir_swir2)
        and ndmi > t["NDMI_P85"]
        and br_nir_swir2 > t["BR_NIR_SWIR2_P85"]
    )
    cond_risk = (
        not cond_water and not cond_topo and not cond_veg and not cond_shadow
        and not cond_dead
        and np.isfinite(ndsi_green_swir2) and np.isfinite(ndmi)
        and ndsi_green_swir2 > t["NDSI_Green_SWIR2_P85"]
        and ndmi < t["NDMI_P15"]
    )
    # Anything left = OPTIMAL

    # Determine expected class from cascade priority
    if cond_water:
        expected = 0
    elif cond_topo:
        expected = 5
    elif cond_veg:
        expected = 10
    elif cond_shadow:
        expected = 0  # shadow → WATER_NODATA
    elif cond_dead:
        expected = 4
    elif cond_risk:
        expected = 3
    else:
        expected = 1

    if int(map_class) != expected:
        raise AssertionError(
            f"\n  MISMATCH at (row={row}, col={col})\n"
            f"    Map class: {map_class} ({CLASS_NAMES.get(map_class, '?')})\n"
            f"    Re-computed expected: {expected} ({CLASS_NAMES.get(expected, '?')})\n"
            f"    Raw values: B3={b3*10000:.0f} B4={b4*10000:.0f} B8={b8*10000:.0f} "
            f"B11={b11*10000:.0f} B12={b12*10000:.0f} SCL={scl} Slope={slope:.1f}\n"
            f"    Indices: NDMI={ndmi:.4f} NDSI={ndsi_green_swir2:.4f} "
            f"BR={br_nir_swir2:.4f} NDVI={ndvi:.4f} MNDWI={mndwi:.4f} BI={bi:.4f}"
        )


# =========================================================================
# 3. MAIN
# =========================================================================


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    # Validate all inputs exist
    log.info("V5.0 FORENSIC AUDIT — Reverse-Engineering Cascade Verification")
    log.info("=" * 60)

    for key, p in BAND_PATHS.items():
        if not p.exists():
            log.error("Missing: %s -> %s", key, p)
            sys.exit(1)
    if not MAP_PATH.exists():
        log.error("Missing map: %s", MAP_PATH)
        sys.exit(1)
    if not THRESH_PATH.exists():
        log.error("Missing thresholds: %s", THRESH_PATH)
        sys.exit(1)

    with open(THRESH_PATH) as f:
        thresholds = json.load(f)
    log.info("Thresholds loaded: %s", list(thresholds.keys()))

    # ── Load classification map ───────────────────────────────────────
    log.info("Loading classification map ...")
    with rasterio.open(MAP_PATH) as src:
        map_array = src.read(1)

    # ── Sample coordinates per class ──────────────────────────────────
    all_samples: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for cls in sorted(CLASS_NAMES):
        samples = _sample_pixels(map_array, cls, N_SAMPLES_PER_CLASS, rng)
        all_samples[cls] = samples
        n = len(samples)
        log.info("  Class %2d (%-15s): %3d samples selected from map", cls, CLASS_NAMES[cls], n)

    total_pixels = sum(len(v) for v in all_samples.values())
    log.info("  Total pixels to verify: %d", total_pixels)

    # Free map memory
    del map_array

    # ── Open band sources ─────────────────────────────────────────────
    log.info("Opening band sources ...")
    b3_src = rasterio.open(BAND_PATHS["B3"])
    b4_src = rasterio.open(BAND_PATHS["B4"])
    b8_src = rasterio.open(BAND_PATHS["B8"])
    scl_src = rasterio.open(BAND_PATHS["SCL"])
    b11_src = rasterio.open(BAND_PATHS["B11"])
    b12_src = rasterio.open(BAND_PATHS["B12"])
    slope_src = rasterio.open(BAND_PATHS["Slope"])

    # WarpedVRT — match B8 grid (10m reference)
    vrt_kw: Dict[str, Any] = dict(
        crs=b8_src.crs,
        transform=b8_src.transform,
        width=b8_src.width,
        height=b8_src.height,
    )
    b11_vrt = WarpedVRT(b11_src, resampling=Resampling.bilinear, **vrt_kw)
    b12_vrt = WarpedVRT(b12_src, resampling=Resampling.bilinear, **vrt_kw)
    slope_vrt = WarpedVRT(slope_src, resampling=Resampling.cubic, **vrt_kw)

    # ── Verify each pixel ──────────────────────────────────────────────
    log.info("Verifying %d pixels ...", total_pixels)
    results: Dict[int, Dict[str, int]] = {}
    for cls in sorted(all_samples):
        samples = all_samples[cls]
        if not samples:
            results[cls] = {"passed": 0, "total": 0}
            continue
        passed = 0
        for row, col, *_ in samples:
            try:
                window = Window(col_off=col, row_off=row, width=1, height=1)
                b3 = float(b3_src.read(1, window=window)[0, 0])
                b4 = float(b4_src.read(1, window=window)[0, 0])
                b8 = float(b8_src.read(1, window=window)[0, 0])
                scl = float(scl_src.read(1, window=window)[0, 0])
                b11 = float(b11_vrt.read(1, window=window)[0, 0])
                b12 = float(b12_vrt.read(1, window=window)[0, 0])
                slope = float(slope_vrt.read(1, window=window)[0, 0])

                _recompute_and_verify(
                    b3, b4, b8, b11, b12, scl, slope,
                    cls, thresholds, row, col,
                )
                passed += 1
            except AssertionError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Read error at (row={row}, col={col}, class={cls}): {exc}"
                ) from exc

        results[cls] = {"passed": passed, "total": len(samples)}

    # ── Cleanup ────────────────────────────────────────────────────────
    for s in [b3_src, b4_src, b8_src, scl_src, b11_src, b12_src, slope_src,
              b11_vrt, b12_vrt, slope_vrt]:
        s.close()

    # ── Report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("FORENSIC AUDIT RESULTS")
    log.info("=" * 60)

    grand_passed = 0
    grand_total = 0
    for cls in sorted(results):
        r = results[cls]
        gpct = r["passed"] / r["total"] * 100 if r["total"] > 0 else 0
        marker = "PASS" if r["passed"] == r["total"] else "FAIL"
        log.info("  Class %2d (%-15s):  %3d/%d passed  %s  (%.0f%%)",
                 cls, CLASS_NAMES.get(cls, "?"),
                 r["passed"], r["total"], marker, gpct)
        grand_passed += r["passed"]
        grand_total += r["total"]

    log.info("─" * 60)
    log.info("  TOTAL:  %d / %d passed  (%.1f%%)", grand_passed, grand_total,
             grand_passed / grand_total * 100 if grand_total else 0)

    elapsed_tot = time.time() - t0
    log.info("  Elapsed: %.1f s", elapsed_tot)
    log.info("=" * 60)

    if grand_passed == grand_total:
        log.info("VERDICT: ALL CASCADE RULES VERIFIED — zero classification errors.")
    else:
        log.error("VERDICT: %d mismatches found — classification cascade has bugs!",
                  grand_total - grand_passed)
        sys.exit(1)


if __name__ == "__main__":
    main()
