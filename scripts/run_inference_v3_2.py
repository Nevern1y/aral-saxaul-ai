"""
run_inference_v3_2.py — Final V3.2 Rule-Based suitability map
Adds AOI mask from V1.0 suitability_full.tif to isolate historical Aral Sea bed.
Only NDMI + NDWI used (SI removed). 3 classes + water mask.
Output: suitability_map_v3_2.tif (uint8: 0=NoData/Water/Outside_AOI, 1=Optimal, 2=Risk, 3=Dead)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import rasterio
from pathlib import Path
import time

sys.stdout.reconfigure(encoding='utf-8')

# ── V3.2 Final thresholds ──────────────────────────────────────────────
NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
NDWI_WATER = 0.0

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
INPUT_CANDIDATES = [
    BASE / "outputs/data/feature_stack_30m.vrt",
    BASE / "outputs/data/feature_stack_30m_tile1.tif",
    BASE / "outputs/data/feature_stack_30m_tile0_redo.tif",
]
AOI_MASK_PATH = BASE / "outputs/data/suitability_full.tif"
OUT_PATH = BASE / "outputs/data/suitability_map_v3_2.tif"

print("=" * 60)
print("V3.2 RULE-BASED INFERENCE (AOI mask + NDMI-only)")
print("=" * 60, flush=True)

# ── 1. Open data raster ────────────────────────────────────────────────
src = None
for p in INPUT_CANDIDATES:
    if p.exists():
        try:
            src = rasterio.open(p)
            print(f"\nOpened data: {p.name}", flush=True)
            break
        except Exception as e:
            print(f"  Failed: {p.name} — {e}", flush=True)
            continue

if src is None:
    print("ERROR: No valid data raster found!", flush=True)
    sys.exit(1)

# Band indices (1-based): 1=NDMI, 4=NDWI
BAND_NDMI = 1
BAND_NDWI = 4

total_pixels = src.width * src.height
print(f"  Data size: {src.width}x{src.height} = {total_pixels:,} pixels", flush=True)

windows = list(src.block_windows(1))
print(f"  Data blocks: {len(windows)}", flush=True)

# ── 2. Open AOI mask raster ────────────────────────────────────────────
try:
    aoi_src = rasterio.open(AOI_MASK_PATH)
    print(f"Opened AOI mask: {AOI_MASK_PATH.name}", flush=True)
    print(f"  AOI size: {aoi_src.width}x{aoi_src.height}", flush=True)
    print(f"  AOI nodata: {aoi_src.nodata}", flush=True)
except Exception as e:
    print(f"ERROR: Cannot open AOI mask: {e}", flush=True)
    print("FALLBACK: Using data raster NoData mask only (no AOI filter)")
    aoi_src = None

# ── 3. Profile for output ──────────────────────────────────────────────
profile = src.profile.copy()
profile.update(
    driver="GTiff",
    dtype="uint8",
    count=1,
    compress="lzw",
    nodata=0,
)

# ── 4. Block-wise classification ────────────────────────────────────────
print(f"\nClassifying...", flush=True)
t0 = time.time()

with rasterio.open(OUT_PATH, "w", **profile) as dst:
    for idx, (_, window) in enumerate(windows):
        if (idx + 1) % 200 == 0 or idx == 0:
            elapsed = time.time() - t0
            pct = (idx + 1) / len(windows) * 100
            print(f"  [{idx+1:5d}/{len(windows)}] {pct:.0f}%  ({elapsed:.0f}s)", flush=True)

        try:
            ndmi_block = src.read(BAND_NDMI, window=window)
            ndwi_block = src.read(BAND_NDWI, window=window)
        except Exception:
            dst.write(np.zeros((window.height, window.width), dtype=np.uint8), 1, window=window)
            continue

        rows, cols = ndmi_block.shape
        ndmi_flat = ndmi_block.ravel().astype(np.float32)
        ndwi_flat = ndwi_block.ravel().astype(np.float32)

        # NoData mask from raster
        valid = ~(np.isnan(ndmi_flat) | np.isnan(ndwi_flat) |
                  np.isinf(ndmi_flat) | np.isinf(ndwi_flat))

        # AOI mask from V1.0 suitability_full.tif
        if aoi_src is not None:
            try:
                aoi_block = aoi_src.read(1, window=window)
                aoi_valid = aoi_block.ravel() != aoi_src.nodata
                valid = valid & aoi_valid
            except Exception:
                pass

        out = np.zeros(ndmi_flat.shape, dtype=np.uint8)

        if valid.sum() > 0:
            ndmi_v = ndmi_flat[valid]
            ndwi_v = ndwi_flat[valid]

            cond_water = ndwi_v > NDWI_WATER
            cond_dead = ~cond_water & (ndmi_v > NDMI_DEAD)
            cond_opt = ~cond_water & ~cond_dead & (ndmi_v < NDMI_OPTIMAL)

            out[valid] = np.select(
                [cond_water, cond_opt, cond_dead],
                [0, 1, 3],
                default=2
            )

        dst.write(out.reshape(rows, cols), 1, window=window)

if aoi_src is not None:
    aoi_src.close()
src.close()

elapsed = time.time() - t0
print(f"  [{len(windows):5d}/{len(windows)}] 100%  ({elapsed:.0f}s)", flush=True)

# ── 5. Statistics ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print("OUTPUT STATS", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"Saved to: {OUT_PATH}", flush=True)

with rasterio.open(OUT_PATH) as res:
    data = res.read(1)
    total = data.size
    classes, counts = np.unique(data, return_counts=True)
    class_labels = {0: "0 NoData/Water/Outside", 1: "1 Optimal", 2: "2 Risk", 3: "3 Dead"}

    valid_ct = sum(counts[i] for i, cls in enumerate(classes) if cls != 0)

    print(f"\n  {'Class':<27} {'Pixels':>12} {'% of valid':>12} {'% of total':>12}")
    print(f"  {'-'*63}")
    for cls, cnt in zip(classes, counts):
        pct_valid = cnt / valid_ct * 100 if valid_ct > 0 else 0
        pct_total = cnt / total * 100
        label = class_labels.get(cls, f"{cls} Unknown")
        print(f"  {label:<27} {cnt:>12,} {pct_valid:>11.1f}% {pct_total:>11.1f}%")

    print(f"  {'TOTAL':<27} {total:>12,} {'':>11} {'':>11}")

    opt_ct = int(counts[classes == 1].sum()) if 1 in classes else 0
    dead_ct = int(counts[classes == 3].sum()) if 3 in classes else 0
    risk_ct = int(counts[classes == 2].sum()) if 2 in classes else 0
    water_ct = int(counts[classes == 0].sum()) if 0 in classes else 0

    opt_area_km2 = opt_ct * 900 // 1_000_000
    opt_area_ha = opt_ct * 900 // 10_000
    dead_area_km2 = dead_ct * 900 // 1_000_000
    risk_area_km2 = risk_ct * 900 // 1_000_000

    print(f"\n  Valid (AOI) pixels:  {valid_ct:,} / {total:,} ({valid_ct/total*100:.1f}%)")
    print(f"  AOI land area:       ~{valid_ct*900//1_000_000:,} km²  ({valid_ct*900//10_000:,} ha)")
    print(f"  Area optimal:        {opt_area_km2:,} km²  ({opt_area_ha:,} ha)  ({opt_ct/valid_ct*100:.1f}% of AOI)" if valid_ct > 0 else "  Area optimal: 0")
    print(f"  Area risk:           {risk_area_km2:,} km²")
    print(f"  Area dead:           {dead_area_km2:,} km²")
    print(f"  Area water/outside:  ~{water_ct*900//1_000_000:,} km²")

    # V1.0 comparison
    print(f"\n  {'COMPARISON:':<27}")
    print(f"  {'V1.0 suitable:':<27} {'6,558 km²':>12} {'(within AOI)':>20}")
    print(f"  {'V3.2 optimal:':<27} {opt_area_km2:>12,} km²")
    print(f"  {'V3.2 optimal + risk:':<27} {opt_area_km2 + risk_area_km2:>12,} km²")

    print(f"  CRS: {res.crs}")
    print(f"  Bounds: {res.bounds}")

print(f"\nDone. V3.2 suitability map ready.", flush=True)
