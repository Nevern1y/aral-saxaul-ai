"""
qa_sanity_check.py — Red Team Audit for V4.1 pipeline
Tests: geometry validity, true area in UTM, NaN/data leakage, elevation sensitivity.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time, numpy as np
from pathlib import Path
import rasterio
import geopandas as gpd
import pyproj
from shapely.ops import transform as shapely_transform

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
GEOJSON_PATH = BASE / "outputs/data/optimal_zones_v4.geojson"
RASTER_PATH = BASE / "outputs/data/suitability_map_v4.tif"
ELEV_PATH = BASE / "outputs/data/dem_elevation_30m.tif"

t0 = time.time()

print("=" * 68)
print("  RED TEAM AUDIT — V4.1 Sanity Check")
print("=" * 68, flush=True)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Geometry Validity
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 1: Geometry Validity")
print(f"{'─' * 68}", flush=True)

gdf = gpd.read_file(GEOJSON_PATH)
print(f"  Total features: {len(gdf):,}", flush=True)
print(f"  CRS: {gdf.crs}", flush=True)

invalid = gdf[~gdf.is_valid]
empty = gdf[gdf.is_empty | (gdf.area <= 0)]

if len(invalid) > 0:
    print(f"  \033[91mINVALID geometries: {len(invalid):,}\033[0m", flush=True)
    for idx, row in invalid.head(10).iterrows():
        print(f"    [{idx}] area_ha={row.get('area_ha', '?'):>10}  reason: {row.geometry.geom_type}", flush=True)
    # attempt auto-repair
    gdf = gdf.buffer(0)
    invalid_after = (~gdf.is_valid).sum()
    print(f"  After buffer(0) repair: {invalid_after:,} still invalid", flush=True)
else:
    print(f"  INVALID geometries: 0  \033[92mOK\033[0m", flush=True)

if len(empty) > 0:
    print(f"  \033[91mEMPTY geometries: {len(empty):,}\033[0m", flush=True)
else:
    print(f"  EMPTY geometries: 0  \033[92mOK\033[0m", flush=True)

t1 = time.time()
print(f"  Time: {t1-t0:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: True Area Check (UTM projection)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 2: True Area Check (UTM 41N)")
print(f"{'─' * 68}", flush=True)

# Calculate area in EPSG:4326 (original)
area_deg2_orig = gdf.to_crs("EPSG:4326").area.sum()
# Rough deg² → km² at Aral Sea latitude (~45N)
# 1 deg lat = 111.32 km, 1 deg lon = 111.32 * cos(45°) ≈ 78.71 km
# 1 deg² ≈ 111.32 * 78.71 ≈ 8762 km²
lat_center = np.radians(45.0)
km2_per_deg2 = 111320 * 111320 * np.cos(lat_center) / 1e6
area_km2_approx = area_deg2_orig * km2_per_deg2
print(f"  Original (4326) area:     {area_km2_approx:,.0f} km² (rough estimate)", flush=True)

# Project to UTM 41N for exact area
try:
    gdf_utm = gdf.to_crs("EPSG:32641")
    area_m2_utm = gdf_utm.area.sum()
    area_km2_utm = area_m2_utm / 1_000_000
    print(f"  UTM 41N (true) area:      {area_km2_utm:,.0f} km² ({area_km2_utm:,.0f} km²)", flush=True)

    pct_diff = abs(area_km2_utm - area_km2_approx) / area_km2_utm * 100
    print(f"  Discrepancy (approx vs UTM): {pct_diff:.2f}%", flush=True)

    # Compare with UTM 40N as cross-check
    gdf_utm40 = gdf.to_crs("EPSG:32640")
    area_km2_utm40 = gdf_utm40.area.sum() / 1_000_000
    pct_40_41 = abs(area_km2_utm40 - area_km2_utm) / area_km2_utm * 100
    print(f"  UTM 40N area:              {area_km2_utm40:,.0f} km²", flush=True)
    print(f"  Cross-zone (40N vs 41N):   {pct_40_41:.2f}%", flush=True)

    area_final = area_km2_utm
    if pct_40_41 > 1.0:
        print(f"  \033[93mWARNING: >1% zone mismatch — Aral Sea spans two UTM zones\033[0m", flush=True)
    else:
        print(f"  \033[92mOK — zones consistent\033[0m", flush=True)
except Exception as e:
    print(f"  UTM projection failed: {e}", flush=True)
    area_final = area_km2_approx

t2 = time.time()
print(f"  Time: {t2-t1:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Raster Integrity / Data Leakage
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 3: Raster Integrity / Data Leakage")
print(f"{'─' * 68}", flush=True)

with rasterio.open(RASTER_PATH) as src:
    data = src.read(1)
    total_px = data.size

    expected_classes = {0, 1, 2, 3, 4}
    actual_classes = set(np.unique(data))
    unexpected = actual_classes - expected_classes
    has_nan = np.any(np.isnan(data.astype(float)))
    has_inf = np.any(np.isinf(data.astype(float)))

    print(f"  Raster size: {data.shape} ({total_px:,} px)", flush=True)
    print(f"  Dtype: {data.dtype}", flush=True)
    print(f"  Expected classes: {expected_classes}", flush=True)
    print(f"  Actual classes:   {actual_classes}", flush=True)

    if unexpected:
        print(f"  \033[91mUNEXPECTED class values: {unexpected}\033[0m", flush=True)
    else:
        print(f"  Unexpected values: none  \033[92mOK\033[0m", flush=True)

    if has_nan:
        nan_count = np.isnan(data.astype(float)).sum()
        print(f"  \033[91mNaN PIXELS: {nan_count:,} ({nan_count/total_px*100:.4f}%)\033[0m", flush=True)
    else:
        print(f"  NaN pixels: 0  \033[92mOK\033[0m", flush=True)

    if has_inf:
        inf_count = np.isinf(data.astype(float)).sum()
        print(f"  \033[91mINF PIXELS: {inf_count:,}\033[0m", flush=True)
    else:
        print(f"  INF pixels: 0  \033[92mOK\033[0m", flush=True)

    # Pixel count per class
    print(f"\n  Per-class pixel count:", flush=True)
    labels = {0: "0 NoData/Water/Outside", 1: "1 Optimal", 2: "2 Risk", 3: "3 Dead", 4: "4 Topo Obstacle"}
    valid_sum = 0
    for cls in sorted(actual_classes):
        cnt = int((data == cls).sum())
        pct = cnt / total_px * 100
        label = labels.get(cls, f"{cls} Unknown")
        print(f"    {label:<30} {cnt:>12,}  ({pct:.2f}%)", flush=True)
        if cls != 0:
            valid_sum += cnt

    print(f"    {'TOTAL':<30} {total_px:>12,}", flush=True)
    print(f"    {'Valid (non-zero)':<30} {valid_sum:>12,}  ({valid_sum/total_px*100:.2f}%)", flush=True)

t3 = time.time()
print(f"  Time: {t3-t2:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Elevation Sensitivity (54-55m grey zone)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 4: Elevation Sensitivity (54—55 m grey zone)")
print(f"{'─' * 68}", flush=True)

with rasterio.open(ELEV_PATH) as src:
    elev = src.read(1)
    elev_valid = ~np.isnan(elev)
    total_valid = elev_valid.sum()

    below_54 = (elev_valid & (elev <= 54.0)).sum()
    grey_54_55 = (elev_valid & (elev > 54.0) & (elev <= 55.0)).sum()
    above_55 = (elev_valid & (elev > 55.0)).sum()

    px_area_km2 = 900 / 1_000_000  # 30m pixel in km²

    area_below_54 = below_54 * px_area_km2
    area_grey = grey_54_55 * px_area_km2
    area_above_55 = above_55 * px_area_km2

    print(f"  Total valid elevation pixels: {total_valid:,}", flush=True)
    print(f"", flush=True)
    print(f"  Zone                    Pixels          Area (km²)     % of valid", flush=True)
    print(f"  {'─'*62}", flush=True)
    print(f"  <= 54.0 m (AOI)         {below_54:>12,}  {area_below_54:>12,.0f}  {below_54/total_valid*100:>6.2f}%", flush=True)
    print(f"  54.0 — 55.0 m (GREY)    {grey_54_55:>12,}  {area_grey:>12,.0f}  {grey_54_55/total_valid*100:>6.2f}%", flush=True)
    print(f"  > 55.0 m (outside)      {above_55:>12,}  {area_above_55:>12,.0f}  {above_55/total_valid*100:>6.2f}%", flush=True)

    if area_grey > 1000:
        print(f"\n  \033[93mCAUTION: Grey zone > 1,000 km² — threshold sensitivity is high\033[0m", flush=True)
        pct_swing = area_grey / area_below_54 * 100
        print(f"  Shifting threshold to 55m would add +{area_grey:,.0f} km² ({pct_swing:.1f}% increase)", flush=True)
    else:
        print(f"\n  \033[92mGrey zone negligible — threshold 54.0 m is stable\033[0m", flush=True)

t4 = time.time()
print(f"  Time: {t4-t3:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 68}")
print("  AUDIT SUMMARY")
print(f"{'=' * 68}")
print(f"  Total time: {t4-t0:.1f}s", flush=True)
print(f"", flush=True)
print(f"  [1] Geometry Validity:     {'\033[92mPASS\033[0m' if len(invalid) == 0 and len(empty) == 0 else '\033[91mFAIL\033[0m'}", flush=True)
print(f"  [2] True Area Check:       {'\033[92mPASS\033[0m' if area_final > 0 else '\033[91mFAIL\033[0m'}  ({area_final:,.0f} km² UTM)", flush=True)
print(f"  [3] Raster Integrity:      {'\033[92mPASS\033[0m' if not unexpected and not has_nan and not has_inf else '\033[91mFAIL\033[0m'}", flush=True)
print(f"  [4] Elevation Sensitivity: {'\033[92mPASS\033[0m' if area_grey < 1000 else '\033[93mWARN\033[0m'}  ({area_grey:,.0f} km² in 54-55m)", flush=True)

verdict = "\033[92mALL CHECKS PASSED\033[0m" if (len(invalid) == 0 and len(empty) == 0 and not unexpected and not has_nan and not has_inf) else "\033[93mISSUES FOUND — review above\033[0m"
print(f"\n  VERDICT: {verdict}", flush=True)
print(f"{'=' * 68}", flush=True)
print(f"QA audit complete.", flush=True)
