"""
run_inference_v3_1.py — Rule-Based V3.1 suitability map (3 classes, uint8)
V3.1 changes: added NDWI water mask, tightened NDMI_OPTIMAL, removed SI from dead zone.
Input:  7-band feature stack (NDMI=Band1, SI=Band3, NDWI=Band4)
Output: suitability_map_v3_1.tif (uint8: 0=NoData/Water, 1=Optimal, 2=Risk, 3=Dead)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import rasterio
from pathlib import Path
import time

sys.stdout.reconfigure(encoding='utf-8')

# ── V3.1 Rule thresholds ──────────────────────────────────────────────
NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
SI_MAX = 3000
NDWI_WATER = 0.0

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
INPUT_CANDIDATES = [
    BASE / "outputs/data/feature_stack_30m.vrt",
    BASE / "outputs/data/feature_stack_30m_tile1.tif",
    BASE / "outputs/data/feature_stack_30m_tile0_redo.tif",
]
OUT_PATH = BASE / "outputs/data/suitability_map_v3_1.tif"

print("=" * 60)
print("V3.1 RULE-BASED INFERENCE (water mask + tightened thresholds)")
print("=" * 60, flush=True)

# ── 1. Open raster ────────────────────────────────────────────────────
src = None
for p in INPUT_CANDIDATES:
    if p.exists():
        try:
            src = rasterio.open(p)
            print(f"\nOpened: {p.name}", flush=True)
            break
        except Exception as e:
            print(f"  Failed: {p.name} — {e}", flush=True)
            continue

if src is None:
    print("ERROR: No valid raster found!", flush=True)
    sys.exit(1)

# Band indices (1-based): 1=NDMI, 3=SI, 4=NDWI
BAND_NDMI = 1
BAND_SI = 3
BAND_NDWI = 4

total_pixels = src.width * src.height
print(f"  Size: {src.width}x{src.height} = {total_pixels:,} pixels", flush=True)

windows = list(src.block_windows(1))
print(f"  Blocks: {len(windows)}", flush=True)

# ── 2. Profile for output ────────────────────────────────────────────
profile = src.profile.copy()
profile.update(
    driver="GTiff",
    dtype="uint8",
    count=1,
    compress="lzw",
    nodata=0,
)

# ── 3. Block-wise classification ──────────────────────────────────────
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
            si_block = src.read(BAND_SI, window=window)
            ndwi_block = src.read(BAND_NDWI, window=window)
        except Exception:
            dst.write(np.zeros((window.height, window.width), dtype=np.uint8), 1, window=window)
            continue

        rows, cols = ndmi_block.shape
        ndmi_flat = ndmi_block.ravel().astype(np.float32)
        si_flat = si_block.ravel().astype(np.float32)
        ndwi_flat = ndwi_block.ravel().astype(np.float32)

        valid = ~(np.isnan(ndmi_flat) | np.isnan(si_flat) | np.isnan(ndwi_flat) |
                  np.isinf(ndmi_flat) | np.isinf(si_flat) | np.isinf(ndwi_flat))

        out = np.zeros(ndmi_flat.shape, dtype=np.uint8)

        if valid.sum() > 0:
            ndmi_v = ndmi_flat[valid]
            si_v = si_flat[valid]
            ndwi_v = ndwi_flat[valid]

            cond_water = ndwi_v > NDWI_WATER
            cond_dead = ~cond_water & (ndmi_v > NDMI_DEAD)
            cond_opt = ~cond_water & ~cond_dead & (ndmi_v < NDMI_OPTIMAL) & (si_v < SI_MAX)

            out[valid] = np.select(
                [cond_water, cond_opt, cond_dead],
                [0, 1, 3],
                default=2
            )

        dst.write(out.reshape(rows, cols), 1, window=window)

src.close()
elapsed = time.time() - t0
print(f"  [{len(windows):5d}/{len(windows)}] 100%  ({elapsed:.0f}s)", flush=True)

# ── 4. Statistics ────────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print("OUTPUT STATS", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"Saved to: {OUT_PATH}", flush=True)

with rasterio.open(OUT_PATH) as res:
    data = res.read(1)
    total = data.size
    classes, counts = np.unique(data, return_counts=True)
    class_labels = {0: "0 NoData/Water", 1: "1 Optimal", 2: "2 Risk", 3: "3 Dead"}

    valid_ct = sum(counts[cls] for cls, cnt in zip(classes, counts) if cls != 0)

    print(f"\n  {'Class':<20} {'Pixels':>12} {'% of valid':>12} {'% of total':>12}")
    print(f"  {'-'*56}")
    for cls, cnt in zip(classes, counts):
        pct_valid = cnt / valid_ct * 100 if valid_ct > 0 else 0
        pct_total = cnt / total * 100
        label = class_labels.get(cls, f"{cls} Unknown")
        print(f"  {label:<20} {cnt:>12,} {pct_valid:>11.1f}% {pct_total:>11.1f}%")

    print(f"  {'TOTAL':<20} {total:>12,} {'':>11} {'':>11}")

    opt_ct = counts[classes == 1].sum() if 1 in classes else 0
    dead_ct = counts[classes == 3].sum() if 3 in classes else 0
    risk_ct = counts[classes == 2].sum() if 2 in classes else 0
    water_ct = counts[classes == 0].sum() if 0 in classes else 0

    print(f"\n  Valid (land) pixels: {valid_ct:,} / {total:,} ({valid_ct/total*100:.1f}%)")
    opt_area_km2 = int(opt_ct) * 900 // 1_000_000
    opt_area_ha = int(opt_ct) * 900 // 10_000
    print(f"  Area optimal (@30m): {opt_area_km2:,} km²  ({opt_area_ha:,} ha)")
    print(f"  Area water:         ~{int(water_ct)*900//1_000_000:,} km²")
    print(f"  CRS: {res.crs}")
    print(f"  Bounds: {res.bounds}")

print(f"\nDone. V3.1 suitability map ready.", flush=True)
