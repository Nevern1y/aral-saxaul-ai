"""
run_inference_v4.py — V4 Rule-Based suitability with topographic filter
Adds Slope (Band 5) filter: areas steeper than 5° are Class 4 (Topo Obstacle).
Preserves V3.2 NDMI + NDWI + AOI mask logic.
Output: suitability_map_v4.tif (uint8: 0=NoData/Water/Outside, 1=Optimal,
        2=Risk, 3=Dead, 4=Topo Obstacle)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import rasterio
from pathlib import Path
import time

sys.stdout.reconfigure(encoding='utf-8')

NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
NDWI_WATER = 0.0
SLOPE_MAX = 5.0

BAND_NDMI = 1
BAND_NDWI = 4
BAND_SLOPE = 5

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
INPUT_CANDIDATES = [
    BASE / "outputs/data/feature_stack_30m.vrt",
    BASE / "outputs/data/feature_stack_30m_tile1.tif",
    BASE / "outputs/data/feature_stack_30m_tile0_redo.tif",
]
AOI_MASK_PATH = BASE / "outputs/data/aoi_mask_v5.tif"
SLOPE_FILE_PATH = BASE / "outputs/data/dem_slope_30m.tif"
OUT_PATH = BASE / "outputs/data/suitability_map_v4.tif"

print("=" * 60)
print("V4 RULE-BASED INFERENCE (AOI + NDMI + Slope filter)")
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

total_pixels = src.width * src.height
print(f"  Data size: {src.width}x{src.height} = {total_pixels:,} pixels", flush=True)

windows = list(src.block_windows(1))
print(f"  Data blocks: {len(windows)}", flush=True)

# ── 2. Open AOI mask (built by build_aoi_mask.py) ──────────────────────
try:
    aoi_src = rasterio.open(AOI_MASK_PATH)
    print(f"\nAOI mask: {AOI_MASK_PATH.name}", flush=True)
    print(f"  Size: {aoi_src.width}x{aoi_src.height}", flush=True)
    print(f"  Nodata: {aoi_src.nodata}", flush=True)
    print(f"  Land pixels: {(aoi_src.read(1) != 0).sum():,}", flush=True)
    aoi_src.close()

    # Reopen for block-wise use
    aoi_src = rasterio.open(AOI_MASK_PATH)
except Exception as e:
    print(f"\nWARNING: Cannot open AOI mask: {e}", flush=True)
    print(f"  Run: python scripts/build_aoi_mask.py", flush=True)
    print("  FALLBACK: Using data raster NoData mask only", flush=True)
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

# ── 4. Detect slope data source ─────────────────────────────────────────
slope_src = None
slope_valid = False
slope_source_name = ""

# Try VRT band 5 first
for _, window in windows[:20]:
    try:
        s = src.read(BAND_SLOPE, window=window)
        if np.any(~np.isnan(s)):
            slope_valid = True
            slope_source_name = f"{src.name} Band {BAND_SLOPE}"
            break
    except Exception:
        pass

# Fallback: external slope file
if not slope_valid and SLOPE_FILE_PATH.exists():
    try:
        slope_src_test = rasterio.open(SLOPE_FILE_PATH)
        s = slope_src_test.read(1, window=windows[0][1])
        if np.any(~np.isnan(s)):
            slope_src = slope_src_test
            slope_valid = True
            slope_source_name = SLOPE_FILE_PATH.name
        else:
            slope_src_test.close()
    except Exception:
        pass

if slope_valid:
    print(f"\n  Slope source: {slope_source_name}", flush=True)
else:
    print(f"\n  WARNING: No valid slope data found!", flush=True)
    print(f"  Tried: VRT Band {BAND_SLOPE}, external file {SLOPE_FILE_PATH.name}", flush=True)
    print("  Slope filter DISABLED (all terrain treated as passable).", flush=True)
    print(f"  Run `python scripts/prepare_slope_data.py` to generate slope data.", flush=True)

# ── 5. Block-wise classification ────────────────────────────────────────
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
            if slope_src is not None:
                slope_block = slope_src.read(1, window=window)
            elif slope_valid:
                slope_block = src.read(BAND_SLOPE, window=window)
            else:
                slope_block = None
        except Exception:
            dst.write(np.zeros((window.height, window.width), dtype=np.uint8), 1, window=window)
            continue

        rows, cols = ndmi_block.shape
        ndmi_flat = ndmi_block.ravel().astype(np.float32)
        ndwi_flat = ndwi_block.ravel().astype(np.float32)

        valid = ~(np.isnan(ndmi_flat) | np.isnan(ndwi_flat) |
                  np.isinf(ndmi_flat) | np.isinf(ndwi_flat))

        # AOI mask
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

            if slope_valid and slope_block is not None:
                slope_flat = slope_block.ravel().astype(np.float32)
                slope_v = slope_flat[valid]
                slope_v = np.nan_to_num(slope_v, nan=0.0)
                cond_topo = ~cond_water & ~cond_dead & (slope_v > SLOPE_MAX)
                cond_opt = ~cond_water & ~cond_dead & ~cond_topo & (ndmi_v < NDMI_OPTIMAL)

                out[valid] = np.select(
                    [cond_water, cond_opt, cond_dead, cond_topo],
                    [0, 1, 3, 4],
                    default=2
                )
            else:
                # No slope data — skip topo filter, same as V3.2
                cond_opt = ~cond_water & ~cond_dead & (ndmi_v < NDMI_OPTIMAL)

                out[valid] = np.select(
                    [cond_water, cond_opt, cond_dead],
                    [0, 1, 3],
                    default=2
                )

        dst.write(out.reshape(rows, cols), 1, window=window)

if aoi_src is not None:
    aoi_src.close()
if slope_src is not None:
    slope_src.close()
src.close()

elapsed = time.time() - t0
print(f"  [{len(windows):5d}/{len(windows)}] 100%  ({elapsed:.0f}s)", flush=True)

# ── 6. Statistics ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print("OUTPUT STATS", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"Saved to: {OUT_PATH}", flush=True)

with rasterio.open(OUT_PATH) as res:
    data = res.read(1)
    total = data.size
    classes, counts = np.unique(data, return_counts=True)
    class_labels = {
        0: "0 NoData/Water/Outside",
        1: "1 Optimal",
        2: "2 Risk",
        3: "3 Dead",
        4: "4 Topo Obstacle",
    }

    valid_ct = sum(counts[i] for i, cls in enumerate(classes) if cls != 0)

    print(f"\n  {'Class':<30} {'Pixels':>12} {'% of valid':>12} {'% of total':>12}")
    print(f"  {'-'*68}")
    for cls, cnt in zip(classes, counts):
        pct_valid = cnt / valid_ct * 100 if valid_ct > 0 else 0
        pct_total = cnt / total * 100
        label = class_labels.get(cls, f"{cls} Unknown")
        print(f"  {label:<30} {cnt:>12,} {pct_valid:>11.1f}% {pct_total:>11.1f}%")

    print(f"  {'TOTAL':<30} {total:>12,}")

    def get_ct(cls_val):
        return int(counts[classes == cls_val].sum()) if cls_val in classes else 0

    opt_ct = get_ct(1)
    dead_ct = get_ct(3)
    risk_ct = get_ct(2)
    water_ct = get_ct(0)
    topo_ct = get_ct(4)

    opt_area_km2 = opt_ct * 900 // 1_000_000
    opt_area_ha = opt_ct * 900 // 10_000
    dead_area_km2 = dead_ct * 900 // 1_000_000
    risk_area_km2 = risk_ct * 900 // 1_000_000
    topo_area_km2 = topo_ct * 900 // 1_000_000

    print(f"\n  Valid (AOI) pixels:  {valid_ct:,} / {total:,} ({valid_ct/total*100:.1f}%)")
    print(f"  AOI land area:       ~{valid_ct*900//1_000_000:,} km²  ({valid_ct*900//10_000:,} ha)")
    print(f"  Area optimal:        {opt_area_km2:,} km²  {opt_area_ha:,} ha  ({opt_ct/valid_ct*100:.1f}% of AOI)" if valid_ct > 0 else "  Area optimal: 0")
    print(f"  Area topo obstacle:  {topo_area_km2:,} km²  ({topo_ct/valid_ct*100:.1f}% of AOI)" if valid_ct > 0 else "  Area topo obstacle: 0")
    print(f"  Area risk:           {risk_area_km2:,} km²")
    print(f"  Area dead:           {dead_area_km2:,} km²")
    print(f"  Area water/outside:  ~{water_ct*900//1_000_000:,} km²")

    print(f"\n  {'COMPARISON:':<30}")
    print(f"  {'V1.0 suitable:':<30} {'6,558 km²':>12}")
    print(f"  {'V3.2 optimal (no topo):':<30} {'14,962 km²':>12}")
    print(f"  {'V4 optimal (with topo):':<30} {opt_area_km2:>12,} km²")
    if slope_valid:
        print(f"  {'Topo obstacle (lost):':<30} {topo_area_km2:>12,} km²")
    else:
        print(f"  {'WARNING:':<30} Slope data missing — topo filter NOT applied")

print(f"\nDone. V4 suitability map ready.", flush=True)
