"""
v6_logistics_prep.py — Split V6 salinity-risk zones (class 1 low-risk +
class 3 moderate-risk) into 0.1 deg x 0.1 deg tractor task grid cells within
Kazakhstan territory, calculate road distance, and export enriched index +
individual KML files.

This is the V6 analogue of scripts/v5_logistics_prep.py. It reuses the same
road layers and Kazakhstan boundary, but reads from the V6 vectorized zones
(outputs/data/suitability_zones_v6.shp, produced by
scripts/v6/build_v6_vectors.py) instead of the V5.1 operational zones, and
writes to separate v6-suffixed output paths so the V5.1 product is untouched.

Pipeline:
  1. Load suitability_zones_v6.shp, keep class in {1, 3} (candidate + moderate
     salinity risk -- see CLAUDE.md V6 scope: class 4/strong-risk and class 10
     /vegetation are not field-check targets).
  2. Clip to Kazakhstan territory (kazakhstan_boundary.geojson).
  3. Build 0.1 deg x 0.1 deg grid over the clipped zones' bounds.
  4. Per-cell intersection (R-tree index) -> area check (>= 10 ha).
  5. Centroid -> distance to nearest road and Kazakhstan access road.
  6. Export individual KML files + tasks_index_v6_enriched.csv.

Usage:
  python scripts/v6/v6_logistics_prep.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import unary_union
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent.parent
V6_ZONES_PATH = BASE / "outputs/data/suitability_zones_v6.shp"
KZ_BOUNDARY_PATH = BASE / "outputs/logistics/kazakhstan_boundary.geojson"
KZ_ZONES_OUT = BASE / "outputs/data/suitability_zones_v6_kazakhstan.geojson"
ROADS_PATH = BASE / "outputs/logistics/aralkum_roads.geojson"
KZ_ROADS_PATH = BASE / "outputs/logistics/kazakhstan_access_roads.geojson"
KML_DIR = BASE / "outputs/logistics/tractor_tasks_v6"
INDEX_OUT = BASE / "outputs/logistics/tasks_index_v6_enriched.csv"

GRID_STEP = 0.1
MIN_AREA_HA = 10.0
UTM_CRS = "EPSG:32641"
TERRITORY_SCOPE = "kazakhstan"
FIELD_CHECK_CLASSES = [1, 3]  # V6: candidate (low salinity) + moderate salinity risk

print("=" * 60)
print("V6 LOGISTICS PREP — Salinity-Risk Zones -> Tractor Tasks")
print("=" * 60, flush=True)
t0 = time.time()

# ── 1. Load V6 vectorized zones, keep field-check classes ────────────
if not V6_ZONES_PATH.exists():
    print(f"ERROR: {V6_ZONES_PATH} not found. Run scripts/v6/build_v6_vectors.py first.", flush=True)
    sys.exit(1)

print(f"\nLoading: {V6_ZONES_PATH.name} ...", flush=True)
zones_all = gpd.read_file(V6_ZONES_PATH)
zones = zones_all[zones_all["class"].isin(FIELD_CHECK_CLASSES)].copy()
if zones.empty:
    print("ERROR: No V6 zones in field-check classes {FIELD_CHECK_CLASSES}.", flush=True)
    sys.exit(1)

print(f"  Features (classes {FIELD_CHECK_CLASSES}): {len(zones):,} of {len(zones_all):,} total")
print(f"  CRS: {zones.crs}")

bounds = zones.total_bounds
print(f"  Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}] — [{bounds[2]:.4f}, {bounds[3]:.4f}]")

# ── 2. Clip to Kazakhstan territory ───────────────────────────────────
print(f"\nClipping V6 zones to Kazakhstan territory: {KZ_BOUNDARY_PATH.name} ...", flush=True)
if not KZ_BOUNDARY_PATH.exists():
    print(f"ERROR: {KZ_BOUNDARY_PATH} not found. Run scripts/v5_kazakhstan_boundary_prep.py first.", flush=True)
    sys.exit(1)

kz_boundary = gpd.read_file(KZ_BOUNDARY_PATH).to_crs("EPSG:4326")
kz_geom = unary_union(kz_boundary.geometry.values)
clipped_zones = zones.copy()
clipped_zones.geometry = clipped_zones.geometry.intersection(kz_geom)
clipped_zones = clipped_zones[
    clipped_zones.geometry.notna()
    & (~clipped_zones.is_empty)
    & (clipped_zones.geom_type.isin(["Polygon", "MultiPolygon"]))
].copy()

if clipped_zones.empty:
    print("ERROR: No V6 field-check zones intersect Kazakhstan territory.", flush=True)
    sys.exit(1)

clipped_utm = clipped_zones.to_crs(UTM_CRS)
clipped_zones["area_ha"] = (clipped_utm.geometry.area / 10000.0).round(1)
clipped_zones = clipped_zones[clipped_zones["area_ha"] >= MIN_AREA_HA].copy()
clipped_zones["territory_scope"] = TERRITORY_SCOPE

if clipped_zones.empty:
    print("ERROR: Kazakhstan-clipped V6 zones are all below 10 ha.", flush=True)
    sys.exit(1)

clipped_zones.to_file(KZ_ZONES_OUT, driver="GeoJSON")
zones = clipped_zones
bounds = zones.total_bounds
print(f"  Kazakhstan features: {len(zones):,}")
print(f"  Kazakhstan area: {zones['area_ha'].sum():,.1f} ha")
print(f"  Kazakhstan bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}] — [{bounds[2]:.4f}, {bounds[3]:.4f}]")
print(f"  Saved: {KZ_ZONES_OUT.name}")

# ── 3. Build 0.1 deg x 0.1 deg grid ───────────────────────────────────
print(f"\nBuilding {GRID_STEP}°×{GRID_STEP}° grid ...", flush=True)
x_steps = int((bounds[2] - bounds[0]) / GRID_STEP) + 1
y_steps = int((bounds[3] - bounds[1]) / GRID_STEP) + 1

grid_cells = []
for i in range(x_steps):
    for j in range(y_steps):
        x_min = bounds[0] + i * GRID_STEP
        y_min = bounds[1] + j * GRID_STEP
        grid_cells.append(box(x_min, y_min, x_min + GRID_STEP, y_min + GRID_STEP))

print(f"  Grid cells: {len(grid_cells):,} ({x_steps} × {y_steps})")

t1 = time.time()
print(f"  Grid built in {t1 - t0:.1f}s")

# ── 4. Load roads & project ───────────────────────────────────────────
print(f"\nLoading roads: {ROADS_PATH.name} ...", flush=True)
if ROADS_PATH.exists():
    roads = gpd.read_file(ROADS_PATH)
    print(f"  Road segments: {len(roads):,}")
    roads_utm = roads.to_crs(UTM_CRS)
else:
    print("  No roads found — distance will be NaN.")
    roads_utm = None

print(f"\nLoading Kazakhstan access roads: {KZ_ROADS_PATH.name} ...", flush=True)
if KZ_ROADS_PATH.exists():
    kz_roads = gpd.read_file(KZ_ROADS_PATH)
    print(f"  Kazakhstan road segments: {len(kz_roads):,}")
    kz_roads_utm = kz_roads.to_crs(UTM_CRS)
else:
    print("  No Kazakhstan access road layer found — Kazakhstan distance will be NaN.")
    kz_roads_utm = None


def nearest_distance_km(point_utm: gpd.GeoDataFrame, roads_layer_utm) -> float | None:
    if roads_layer_utm is None or roads_layer_utm.empty:
        return None
    try:
        point_geom = point_utm.geometry.iloc[0]
        distances = roads_layer_utm.geometry.distance(point_geom)
        return round(float(distances.min()) / 1000.0, 2)
    except Exception:
        return None


# ── 5. Per-cell intersection ──────────────────────────────────────────
print("\nIntersecting zones with grid cells (R-tree index)...", flush=True)
KML_DIR.mkdir(parents=True, exist_ok=True)
for stale_kml in KML_DIR.glob("*.kml"):
    stale_kml.unlink()

sindex = zones.sindex
task_records = []
task_count = 0
total_exported_ha = 0.0

for idx, cell in enumerate(grid_cells):
    candidates_idx = sindex.query(cell, predicate="intersects")
    if len(candidates_idx) == 0:
        continue

    candidates = zones.iloc[candidates_idx].copy()
    clipped = candidates
    clipped.geometry = clipped.geometry.intersection(cell)

    clipped = clipped[
        clipped.geometry.notna()
        & (~clipped.is_empty)
        & (clipped.geom_type.isin(["Polygon", "MultiPolygon"]))
    ]

    if clipped.empty:
        continue

    try:
        clipped_utm = clipped.to_crs(UTM_CRS)
        area_ha = clipped_utm.geometry.area.sum() / 10000.0
    except Exception:
        continue

    if area_ha < MIN_AREA_HA:
        continue

    # class-mix summary for this cell (helps prioritize low-risk-only cells)
    class_area_ha = (
        clipped.assign(_a=clipped_utm.geometry.area / 10000.0)
        .groupby("class")["_a"].sum()
        .round(1)
        .to_dict()
    )
    low_risk_ha = class_area_ha.get(1, 0.0)
    moderate_risk_ha = class_area_ha.get(3, 0.0)

    planting_geom = unary_union(clipped.geometry.values)
    if planting_geom.is_empty:
        continue

    centroid_lat = round(cell.centroid.y, 2)
    centroid_lon = round(cell.centroid.x, 2)
    kml_name = f"task_grid_v6_{centroid_lat:.2f}_{centroid_lon:.2f}.kml"
    kml_path = KML_DIR / kml_name

    export_gdf = gpd.GeoDataFrame(
        {
            "Name": [kml_name.replace(".kml", "")],
            "area_ha": [round(area_ha, 1)],
            "low_risk_ha": [round(low_risk_ha, 1)],
            "moderate_risk_ha": [round(moderate_risk_ha, 1)],
            "grid_cell_deg": [f"{GRID_STEP}°×{GRID_STEP}°"],
        },
        geometry=[planting_geom],
        crs="EPSG:4326",
    )
    try:
        export_gdf.to_file(str(kml_path), driver="KML")
    except Exception as e:
        print(f"  WARNING: KML export failed for {kml_name}: {e}", flush=True)
        continue

    cell_pt = gpd.GeoDataFrame(
        {"geometry": [Point(cell.centroid.x, cell.centroid.y)]},
        crs="EPSG:4326",
    ).to_crs(UTM_CRS)
    dist_km = nearest_distance_km(cell_pt, roads_utm)
    dist_kz_km = nearest_distance_km(cell_pt, kz_roads_utm)

    task_records.append({
        "filename": kml_name,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "area_ha": round(area_ha, 1),
        "low_risk_ha": round(low_risk_ha, 1),
        "moderate_risk_ha": round(moderate_risk_ha, 1),
        "distance_to_road_km": dist_km if dist_km is not None else "",
        "distance_to_kazakhstan_road_km": dist_kz_km if dist_kz_km is not None else "",
        "territory_scope": TERRITORY_SCOPE,
    })
    task_count += 1
    total_exported_ha += area_ha

    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(grid_cells)} cells ... ({task_count} tasks)", flush=True)

t2 = time.time()
print(f"  Intersection & export in {t2 - t1:.1f}s")

# ── 6. Save enriched CSV ───────────────────────────────────────────────
print(f"\nSaving enriched index: {INDEX_OUT.name} ...", flush=True)
INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
index_df = pd.DataFrame(task_records)
if not index_df.empty:
    index_df = index_df.sort_values("area_ha", ascending=False)
    index_df.to_csv(INDEX_OUT, index=False, encoding="utf-8")
    print(f"  Records saved: {len(index_df):,}")

t3 = time.time()

# ── 7. Summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("V6 LOGISTICS PREP SUMMARY")
print(f"{'=' * 60}")
print(f"  Total pipeline:             {t3 - t0:.1f}s")
print(f"  Source zones loaded:        {len(zones):,} (classes {FIELD_CHECK_CLASSES})")
print(f"  Grid cells evaluated:       {len(grid_cells):,}")
print(f"  Generated task KMLs:        {task_count:,}")
print(f"  Total exported area:        {total_exported_ha:,.1f} ha ({total_exported_ha / 100:,.1f} km²)")
print(f"  KML directory:              {KML_DIR}")
print(f"  Enriched index:             {INDEX_OUT}")

if task_count > 0 and "distance_to_road_km" in index_df.columns:
    accessible = index_df[index_df["distance_to_road_km"].notna() & (index_df["distance_to_road_km"] != "")]
    if not accessible.empty:
        acc_km = accessible["distance_to_road_km"].astype(float)
        print(f"\n  Tasks with road distance:   {len(accessible):,}")
        print(f"  Mean distance to road:     {acc_km.mean():.2f} km")
        print(f"  Median distance to road:   {acc_km.median():.2f} km")
        print(f"  Tasks within 2 km:         {(acc_km < 2).sum():,} ({(acc_km < 2).mean() * 100:.1f}%)")

if task_count > 0 and "distance_to_kazakhstan_road_km" in index_df.columns:
    accessible_kz = index_df[
        index_df["distance_to_kazakhstan_road_km"].notna() & (index_df["distance_to_kazakhstan_road_km"] != "")
    ]
    if not accessible_kz.empty:
        acc_kz_km = accessible_kz["distance_to_kazakhstan_road_km"].astype(float)
        print(f"\n  Tasks with Kazakhstan road distance: {len(accessible_kz):,}")
        print(f"  Mean distance to Kazakhstan road:   {acc_kz_km.mean():.2f} km")
        print(f"  Median distance to Kazakhstan road: {acc_kz_km.median():.2f} km")
        print(f"  Tasks within 25 km of KZ road:      {(acc_kz_km < 25).sum():,} ({(acc_kz_km < 25).mean() * 100:.1f}%)")

print(f"\n{'=' * 60}")
print("DONE. V6 candidate task grid prepared for field review.", flush=True)
