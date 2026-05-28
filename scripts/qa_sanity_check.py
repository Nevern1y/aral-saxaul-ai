"""
qa_sanity_check.py — Red Team Audit for V5.1 screening pipeline
Tests: raster integrity, class set, thresholds plausibility, band geometry, no NaN/inf.
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio

from v5_rules import CLASS_NAMES

import sys
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"
SCIENCE = BASE / "outputs" / "science"

RASTER_PATH = DATA / "suitability_map_v5.tif"
FILTERED_RASTER_PATH = DATA / "suitability_map_v5_filtered.tif"
THRESH_PATH = DATA / "thresholds_v5.json"
STATS_PATH = DATA / "v5_stats.json"
OPERATIONAL_GEOJSON_PATH = DATA / "operational_zones_v5.geojson"
KZ_OPERATIONAL_GEOJSON_PATH = DATA / "operational_zones_v5_kazakhstan.geojson"
POINT_SAMPLES_PATH = SCIENCE / "v5_point_samples.csv"
VALIDATION_SUMMARY_PATH = SCIENCE / "v5_validation_summary.json"
LOGISTICS = BASE / "outputs" / "logistics"
ROADS_PATH = LOGISTICS / "aralkum_roads.geojson"
KZ_ROADS_PATH = LOGISTICS / "kazakhstan_access_roads.geojson"
KZ_BOUNDARY_PATH = LOGISTICS / "kazakhstan_boundary.geojson"
ROADS_METADATA_PATH = LOGISTICS / "v5_roads_metadata.json"
TASKS_INDEX_PATH = LOGISTICS / "tasks_index_v5_enriched.csv"
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
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = []
checks_run = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global checks_run
    checks_run += 1
    if cond:
        print(f"  {label:<45} {PASS}  {detail}")
    else:
        print(f"  {label:<45} {FAIL}  {detail}")
        errors.append(label)


t0 = time.time()

print("=" * 68)
print("  RED TEAM AUDIT — V5.1 Sanity Check")
print("=" * 68, flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: File Existence
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 1: Input file existence")
print(f"{'─' * 68}", flush=True)

check(RASTER_PATH.exists(), "suitability_map_v5.tif", f"({RASTER_PATH.stat().st_size / 1e6:.1f} MB)")
check(FILTERED_RASTER_PATH.exists(), "suitability_map_v5_filtered.tif")
check(THRESH_PATH.exists(), "thresholds_v5.json")
check(STATS_PATH.exists(), "v5_stats.json")
check(OPERATIONAL_GEOJSON_PATH.exists(), "operational_zones_v5.geojson")
check(KZ_OPERATIONAL_GEOJSON_PATH.exists(), "operational_zones_v5_kazakhstan.geojson")
check(POINT_SAMPLES_PATH.exists(), "v5_point_samples.csv")
check(VALIDATION_SUMMARY_PATH.exists(), "v5_validation_summary.json")
check(ROADS_PATH.exists(), "aralkum_roads.geojson")
check(KZ_ROADS_PATH.exists(), "kazakhstan_access_roads.geojson")
check(KZ_BOUNDARY_PATH.exists(), "kazakhstan_boundary.geojson")
check(TASKS_INDEX_PATH.exists(), "tasks_index_v5_enriched.csv")
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
# TEST 4: Filtered Raster + Derived Products
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 4: Filtered raster and derived products")
print(f"{'─' * 68}", flush=True)

with rasterio.open(RASTER_PATH) as raw_src, rasterio.open(FILTERED_RASTER_PATH) as filtered_src:
    check(filtered_src.width == raw_src.width and filtered_src.height == raw_src.height, "  filtered dimensions match raw")
    check(filtered_src.crs == raw_src.crs, "  filtered CRS matches raw")
    check(filtered_src.transform == raw_src.transform, "  filtered transform matches raw")
    check(filtered_src.nodata is None, "  filtered nodata metadata unset", f"= {filtered_src.nodata}")
    filtered = filtered_src.read(1)

filtered_classes = set(np.unique(filtered))
check(not (filtered_classes - EXPECTED_CLASSES), "  filtered has no unexpected classes", f"extra: {filtered_classes - EXPECTED_CLASSES}")
check(int((filtered == 1).sum()) > 0, "  filtered has candidate pixels")
check(FILTERED_RASTER_PATH.stat().st_mtime >= RASTER_PATH.stat().st_mtime, "  filtered is not older than raw")

with open(STATS_PATH, encoding="utf-8") as f:
    stats = json.load(f)
required_stats = {
    "area_ha",
    "candidate_10m_area_ha",
    "class_pixels_10m",
    "class_total_pixels_10m",
    "pixel_area_ha_10m",
    "candidate_100m_area_ha",
    "operational_area_ha",
    "clusters",
    "top10_ha",
}
check(required_stats.issubset(stats), "  stats has required keys", f"missing: {required_stats - set(stats)}")
class_pixels_10m = {int(cls): int(count) for cls, count in stats.get("class_pixels_10m", {}).items()}
check(sum(class_pixels_10m.values()) == stats.get("class_total_pixels_10m"), "  10m class pixel counts sum to total")
check(class_pixels_10m.get(1, 0) > 0, "  stats fallback has candidate pixels")
check(stats.get("candidate_100m_area_ha", 0) >= stats.get("operational_area_ha", 0), "  candidate area >= operational area")
check(stats.get("clusters", 0) > 0, "  operational clusters > 0", f"= {stats.get('clusters', 0):,}")

with open(OPERATIONAL_GEOJSON_PATH, encoding="utf-8") as f:
    operational = json.load(f)
features = operational.get("features", [])
geojson_area = sum(float(feature.get("properties", {}).get("area_ha", 0) or 0) for feature in features)
check(len(features) == stats.get("clusters"), "  GeoJSON feature count matches stats", f"{len(features):,} vs {stats.get('clusters', 0):,}")
check(abs(geojson_area - stats.get("operational_area_ha", 0)) <= max(1, len(features) * 0.1), "  GeoJSON area matches stats", f"diff={geojson_area - stats.get('operational_area_ha', 0):.1f} ha")

del filtered

t4 = time.time()
print(f"  Time: {t4 - t3:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Band Geometry Consistency
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 5: Band geometry consistency (B8 reference grid)")
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

t5 = time.time()
print(f"  Time: {t5 - t4:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Logistics Road Layers
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 6: Logistics road layers")
print(f"{'─' * 68}", flush=True)

if ROADS_PATH.exists() and KZ_ROADS_PATH.exists() and KZ_BOUNDARY_PATH.exists() and TASKS_INDEX_PATH.exists():
    with open(ROADS_PATH, encoding="utf-8") as f:
        roads_features = json.load(f).get("features", [])
    with open(KZ_ROADS_PATH, encoding="utf-8") as f:
        kz_road_features = json.load(f).get("features", [])
    with open(KZ_BOUNDARY_PATH, encoding="utf-8") as f:
        kz_boundary_features = json.load(f).get("features", [])
    with open(TASKS_INDEX_PATH, encoding="utf-8") as f:
        task_rows = list(csv.DictReader(f))

    check(len(kz_boundary_features) > 0, "  Kazakhstan boundary present", f"= {len(kz_boundary_features):,}")
    check(len(kz_road_features) > 0, "  Kazakhstan roads present", f"= {len(kz_road_features):,}")
    check(len(roads_features) >= len(kz_road_features), "  merged roads include KZ roads", f"{len(roads_features):,} >= {len(kz_road_features):,}")
    check(task_rows and "distance_to_kazakhstan_road_km" in task_rows[0], "  tasks include Kazakhstan distance")
    kz_distances = [row.get("distance_to_kazakhstan_road_km", "") for row in task_rows]
    check(all(str(value).strip() != "" for value in kz_distances), "  all tasks have KZ distance")
    scopes = {row.get("territory_scope", "") for row in task_rows}
    check(scopes == {"kazakhstan"}, "  all logistics tasks are Kazakhstan-scoped", f"= {sorted(scopes)}")
    if KZ_OPERATIONAL_GEOJSON_PATH.exists():
        with open(KZ_OPERATIONAL_GEOJSON_PATH, encoding="utf-8") as f:
            kz_zone_features = json.load(f).get("features", [])
        check(len(kz_zone_features) > 0, "  Kazakhstan clipped zones present", f"= {len(kz_zone_features):,}")
    if ROADS_METADATA_PATH.exists():
        with open(ROADS_METADATA_PATH, encoding="utf-8") as f:
            road_metadata = json.load(f)
        check(not road_metadata.get("failed_tiles"), "  road metadata has no failed tiles")
else:
    check(False, "  logistics road outputs available", "run `python scripts/v5_roads_prep.py` and `python scripts/v5_logistics_prep.py`")

t6 = time.time()
print(f"  Time: {t6 - t5:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Science Output Consistency
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 68}")
print("  TEST 7: Science output consistency")
print(f"{'─' * 68}", flush=True)

if POINT_SAMPLES_PATH.exists() and VALIDATION_SUMMARY_PATH.exists():
    with open(VALIDATION_SUMMARY_PATH, encoding="utf-8") as f:
        validation_summary = json.load(f)
    with open(POINT_SAMPLES_PATH, encoding="utf-8") as f:
        point_rows = list(csv.DictReader(f))

    sources = {row.get("coordinate_source", "") for row in point_rows}
    authoritative_rows = [row for row in point_rows if row.get("coordinate_source") == "authoritative_dd"]
    check(len(point_rows) == validation_summary.get("n_point_samples"), "  point sample count matches summary", f"{len(point_rows)} vs {validation_summary.get('n_point_samples')}")
    check(len(authoritative_rows) == validation_summary.get("n_authoritative_point_samples"), "  authoritative sample count matches summary")
    check("raw_aralfield_dd" in sources and "shifted_odt_derived_dd" in sources, "  raw and shifted sources retained")
    if validation_summary.get("coordinate_policy") == "dual_source_unresolved":
        check(not authoritative_rows, "  unresolved policy has no authoritative rows")
    check(validation_summary.get("authority_template_rows", 0) >= validation_summary.get("authority_template_resolved_rows", 0), "  template resolved rows <= total")
else:
    check(False, "  science outputs available", "run `python scripts/run_v5_science_suite.py --qa`")

t7 = time.time()
print(f"  Time: {t7 - t6:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 68}")
print("  AUDIT SUMMARY")
print(f"{'=' * 68}")
print(f"  Total time: {t7 - t0:.1f}s", flush=True)
print(f"  Tests run:  7", flush=True)
print(f"  Checks:     {len(errors)} failed  |  {checks_run - len(errors)} passed", flush=True)
print(f"", flush=True)

if not errors:
    print(f"  VERDICT: {PASS} ALL CHECKS PASSED", flush=True)
else:
    print(f"  VERDICT: {FAIL} {len(errors)} FAILURES", flush=True)
    for e in errors:
        print(f"    - {e}", flush=True)

print(f"{'=' * 68}", flush=True)
print(f"QA audit complete.", flush=True)
