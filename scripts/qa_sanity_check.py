"""
qa_sanity_check.py — Red Team Audit for V5.0 pipeline
Tests: raster integrity, class set, thresholds plausibility, band geometry, no NaN/inf.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio

import sys
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"

RASTER_PATH = DATA / "suitability_map_v5.tif"
THRESH_PATH = DATA / "thresholds_v5.json"
BAND_PATHS = {
    "B3": DATA / "B3_10m.tif",
    "B4": DATA / "B4_10m.tif",
    "B8": DATA / "B8_10m.tif",
    "B11": DATA / "B11_20m.tif",
    "B12": DATA / "B12_20m.tif",
    "SCL": DATA / "SCL_10m.tif",
    "Slope": DATA / "dem_slope_30m.tif",
}

EXPECTED_CLASSES = {0, 1, 3, 4, 5, 10}
CLASS_NAMES = {
    0: "WATER_NODATA",
    1: "OPTIMAL",
    3: "RISK_DRY_SALT",
    4: "DEAD_WET_TOXIC",
    5: "OBSTACLE_TOPO",
    10: "VEGETATION",
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  {label:<45} {PASS}  {detail}")
    else:
        print(f"  {label:<45} {FAIL}  {detail}")
        errors.append(label)


t0 = time.time()

print("=" * 68)
print("  RED TEAM AUDIT — V5.0 Sanity Check")
print("=" * 68, flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: File Existence
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 1: Input file existence")
print(f"{'─' * 68}", flush=True)

check(RASTER_PATH.exists(), "suitability_map_v5.tif", f"({RASTER_PATH.stat().st_size / 1e6:.1f} MB)")
check(THRESH_PATH.exists(), "thresholds_v5.json")
for key, p in BAND_PATHS.items():
    check(p.exists(), f"Band: {key}", f"({p.name})")

t1 = time.time()
print(f"  Time: {t1 - t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Thresholds Sanity
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 2: Thresholds plausibility (thresholds_v5.json)")
print(f"{'─' * 68}", flush=True)

with open(THRESH_PATH) as f:
    thresholds = json.load(f)

for key, val in thresholds.items():
    check(np.isfinite(val), f"  {key}", f"= {val:.6f}")

# Physical plausibility checks
ndmi_p15 = thresholds["NDMI_P15"]
ndmi_p85 = thresholds["NDMI_P85"]
check(ndmi_p15 < ndmi_p85, "  NDMI_P15 < NDMI_P85", f"({ndmi_p15:.4f} < {ndmi_p85:.4f})")
check(-1 < ndmi_p15 < 1, "  NDMI_P15 in (-1, 1)", f"= {ndmi_p15:.4f}")
check(-1 < ndmi_p85 < 1, "  NDMI_P85 in (-1, 1)", f"= {ndmi_p85:.4f}")

ndsi_p85 = thresholds["NDSI_Green_SWIR2_P85"]
br_p85 = thresholds["BR_NIR_SWIR2_P85"]
check(br_p85 > 0, "  BR_NIR_SWIR2_P85 > 0", f"= {br_p85:.4f}")

t2 = time.time()
print(f"  Time: {t2 - t1:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Raster Integrity
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 3: Raster Integrity (suitability_map_v5.tif)")
print(f"{'─' * 68}", flush=True)

with rasterio.open(RASTER_PATH) as src:
    profile = src.profile
    width, height = src.width, src.height
    crs = src.crs
    data = src.read(1)

total_px = data.size
print(f"  Dimensions: {width} x {height} = {total_px:,} px", flush=True)
print(f"  CRS: {crs}", flush=True)
print(f"  Dtype: {data.dtype}", flush=True)
print(f"  Compression: {profile.get('compress', 'none')}", flush=True)
print(f"  Tiled: {profile.get('tiled', False)}", flush=True)

check(data.dtype == np.uint8, "  dtype == uint8")
check(crs is not None, "  CRS is set")

actual_classes = set(np.unique(data))
unexpected = actual_classes - EXPECTED_CLASSES
missing = EXPECTED_CLASSES - actual_classes

check(not unexpected, "  No unexpected classes", f"extra: {unexpected}" if unexpected else "")
check(not missing, "  All expected classes present", f"missing: {missing}" if missing else "")

has_nan = np.any(np.isnan(data.astype(float)))
has_inf = np.any(np.isinf(data.astype(float)))
check(not has_nan, "  No NaN pixels")
check(not has_inf, "  No Inf pixels")

# Per-class distribution
print(f"\n  Per-class pixel count:", flush=True)
valid_sum = 0
for cls in sorted(EXPECTED_CLASSES):
    cnt = int((data == cls).sum())
    pct = cnt / total_px * 100
    name = CLASS_NAMES.get(cls, f"Class {cls}")
    print(f"    {cls} ({name:<16}) {cnt:>12,}  ({pct:6.2f}%)", flush=True)
    if cls != 0:
        valid_sum += cnt

print(f"    {'Total':<30} {total_px:>12,}", flush=True)
print(f"    {'Valid (non-zero)':<30} {valid_sum:>12,}  ({valid_sum / total_px * 100:6.2f}%)", flush=True)

# Check that no class other than 0 is >99% (if so, cascade is broken)
for cls in [1, 3, 4, 5, 10]:
    pct = (data == cls).sum() / total_px * 100
    check(pct < 99, f"  Class {cls} not dominant", f"({pct:.2f}%)")

del data

t3 = time.time()
print(f"  Time: {t3 - t2:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Band Geometry Consistency
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 4: Band geometry consistency (B8 reference grid)")
print(f"{'─' * 68}", flush=True)

with rasterio.open(BAND_PATHS["B8"]) as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_width = ref.width
    ref_height = ref.height

for key in ["B3", "B4", "SCL"]:
    with rasterio.open(BAND_PATHS[key]) as src:
        if src.crs == ref_crs and src.transform == ref_transform and src.width == ref_width and src.height == ref_height:
            check(True, f"  {key} matches B8 grid")
        else:
            check(False, f"  {key} matches B8 grid",
                  f"wh=({src.width},{src.height}) vs ref=({ref_width},{ref_height})")

t4 = time.time()
print(f"  Time: {t4 - t3:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 68}")
print("  AUDIT SUMMARY")
print(f"{'=' * 68}")
print(f"  Total time: {t4 - t0:.1f}s", flush=True)
print(f"  Tests run:  4", flush=True)
print(f"  Checks:     {sum(1 for e in errors)} failed  |  {2 if not errors else 0} passed", flush=True)
print(f"", flush=True)

if not errors:
    print(f"  VERDICT: {PASS} ALL CHECKS PASSED", flush=True)
else:
    print(f"  VERDICT: {FAIL} {len(errors)} FAILURES", flush=True)
    for e in errors:
        print(f"    - {e}", flush=True)

print(f"{'=' * 68}", flush=True)
print(f"QA audit complete.", flush=True)
