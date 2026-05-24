"""
run_inference_v3.py — Rule-Based suitability map (3 classes, uint8)
Uses NDMI as primary + SI as secondary discriminator.
Input:  7-band feature stack (NDMI=Band1, SI=Band3)
Output: suitability_map_v3.tif (uint8: 0=NoData, 1=Optimal, 2=Risk, 3=Dead)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import rasterio
from pathlib import Path
import time

sys.stdout.reconfigure(encoding='utf-8')

# ── V3.0 Rule thresholds (calibrated via temp_threshold_analysis.py) ──
NDMI_DEAD = -0.025
SI_DEAD = 2700
NDMI_OPTIMAL = -0.050
SI_OPTIMAL = 2800

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
INPUT_CANDIDATES = [
    BASE / "outputs/data/feature_stack_30m.vrt",
    BASE / "outputs/data/feature_stack_30m_tile1.tif",
    BASE / "outputs/data/feature_stack_30m_tile0_redo.tif",
]
OUT_PATH = BASE / "outputs/data/suitability_map_v3.tif"

print("=" * 60)
print("V3.0 RULE-BASED INFERENCE")
print("=" * 60, flush=True)

# ── 1. Open raster (try VRT first, fallback to tiles) ─────────────────
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

# Band 1 = NDMI (0-based: 0), Band 3 = SI (0-based: 2)
BAND_NDMI = 1
BAND_SI = 3

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
        except Exception:
            dst.write(np.zeros((window.height, window.width), dtype=np.uint8), 1, window=window)
            continue

        rows, cols = ndmi_block.shape
        ndmi_flat = ndmi_block.ravel().astype(np.float32)
        si_flat = si_block.ravel().astype(np.float32)

        valid = ~(np.isnan(ndmi_flat) | np.isnan(si_flat) |
                  np.isinf(ndmi_flat) | np.isinf(si_flat))

        out = np.zeros(ndmi_flat.shape, dtype=np.uint8)

        if valid.sum() > 0:
            ndmi_v = ndmi_flat[valid]
            si_v = si_flat[valid]

            cond_dead = (ndmi_v > NDMI_DEAD) & (si_v > SI_DEAD)
            cond_opt = (ndmi_v < NDMI_OPTIMAL) & (si_v < SI_OPTIMAL)

            out[valid] = np.select(
                [cond_opt, cond_dead],
                [1, 3],
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
    nodata_ct = (data == 0).sum()
    opt_ct = (data == 1).sum()
    risk_ct = (data == 2).sum()
    dead_ct = (data == 3).sum()
    valid_ct = opt_ct + risk_ct + dead_ct

    print(f"\n  {'Class':<15} {'Pixels':>12} {'% of valid':>12} {'% of total':>12}")
    print(f"  {'-'*51}")
    print(f"  {'1 Optimal':<15} {opt_ct:>12,} {opt_ct/valid_ct*100:>11.1f}% {opt_ct/total*100:>11.1f}%")
    print(f"  {'2 Risk':<15} {risk_ct:>12,} {risk_ct/valid_ct*100:>11.1f}% {risk_ct/total*100:>11.1f}%")
    print(f"  {'3 Dead':<15} {dead_ct:>12,} {dead_ct/valid_ct*100:>11.1f}% {dead_ct/total*100:>11.1f}%")
    print(f"  {'0 NoData/Water':<15} {nodata_ct:>12,} {'—':>11} {nodata_ct/total*100:>11.1f}%")
    print(f"  {'TOTAL':<15} {total:>12,} {'':>11} {'':>11}")
    print(f"\n  Valid pixels: {valid_ct:,} / {total:,} ({valid_ct/total*100:.1f}%)")
    opt_area_km2 = int(opt_ct) * 900 // 1_000_000
    opt_area_ha = int(opt_ct) * 900 // 10_000
    print(f"  Area optimal (@30m): {opt_area_km2:,} km²  ({opt_area_ha:,} ha)")
    print(f"  CRS: {res.crs}")
    print(f"  Bounds: {res.bounds}")

print(f"\nDone. V3.0 suitability map ready.", flush=True)
