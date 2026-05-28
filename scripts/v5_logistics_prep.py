"""
v5_logistics_prep.py — Split V5 candidate zones into 0.1°×0.1° tractor task
grid cells, calculate road distance, and export enriched index + KML.

Pipeline:
  1. Load operational_zones_v5.geojson and clip to Kazakhstan territory
  2. Build 0.1°×0.1° grid over Kazakhstan candidate-zone bounds
  3. Per-cell intersection (R-tree index) → area check (≥10 ha)
  4. Centroid → distance to nearest road and Kazakhstan access road
  5. Export individual KML files + tasks_index_v5_enriched.csv

Usage:
  python scripts/v5_logistics_prep.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import unary_union
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
GEOJSON_PATH = BASE / "outputs/data/operational_zones_v5.geojson"
KZ_BOUNDARY_PATH = BASE / "outputs/logistics/kazakhstan_boundary.geojson"
KZ_ZONES_OUT = BASE / "outputs/data/operational_zones_v5_kazakhstan.geojson"
ROADS_PATH = BASE / "outputs/logistics/aralkum_roads.geojson"
KZ_ROADS_PATH = BASE / "outputs/logistics/kazakhstan_access_roads.geojson"
KML_DIR = BASE / "outputs/logistics/tractor_tasks_v5"
INDEX_OUT = BASE / "outputs/logistics/tasks_index_v5_enriched.csv"

GRID_STEP = 0.1
MIN_AREA_HA = 10.0
UTM_CRS = "EPSG:32641"
TERRITORY_SCOPE = "kazakhstan"

print("=" * 60)
print("V5.1 LOGISTICS PREP — Candidate Zones → Tractor Tasks")
print("=" * 60, flush=True)
t0 = time.time()

# ── 1. Load V5 operational zones ─────────────────────────────────────
if not GEOJSON_PATH.exists():
    print(f"ERROR: {GEOJSON_PATH} not found. Run v5_extract_stats.py first.", flush=True)
    sys.exit(1)

print(f"\nLoading: {GEOJSON_PATH.name} ...", flush=True)
zones = gpd.read_file(GEOJSON_PATH)
if zones.empty:
    print("ERROR: Empty GeoJSON.", flush=True)
    sys.exit(1)

print(f"  Features: {len(zones):,}")
print(f"  CRS: {zones.crs}")

bounds = zones.total_bounds
print(f"  Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}] — [{bounds[2]:.4f}, {bounds[3]:.4f}]")

if TERRITORY_SCOPE == "kazakhstan":
    print(f"\nClipping candidate zones to Kazakhstan territory: {KZ_BOUNDARY_PATH.name} ...", flush=True)
    if not KZ_BOUNDARY_PATH.exists():
        print(f"ERROR: {KZ_BOUNDARY_PATH} not found. Run v5_kazakhstan_boundary_prep.py first.", flush=True)
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
        print("ERROR: No candidate zones intersect Kazakhstan territory.", flush=True)
        sys.exit(1)

    clipped_utm = clipped_zones.to_crs(UTM_CRS)
    clipped_zones["area_ha"] = (clipped_utm.geometry.area / 10000.0).round(1)
    clipped_zones = clipped_zones[clipped_zones["area_ha"] >= MIN_AREA_HA].copy()
    clipped_zones["territory_scope"] = "kazakhstan"

    if clipped_zones.empty:
        print("ERROR: Kazakhstan-clipped candidate zones are all below 10 ha.", flush=True)
        sys.exit(1)

    clipped_zones.to_file(KZ_ZONES_OUT, driver="GeoJSON")
    zones = clipped_zones
    bounds = zones.total_bounds
    print(f"  Kazakhstan features: {len(zones):,}")
    print(f"  Kazakhstan area: {zones['area_ha'].sum():,.1f} ha")
    print(f"  Kazakhstan bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}] — [{bounds[2]:.4f}, {bounds[3]:.4f}]")
    print(f"  Saved: {KZ_ZONES_OUT.name}")

# ── 2. Build 0.1°×0.1° grid ──────────────────────────────────────────
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
print(f"  Grid built in {t1-t0:.1f}s")

# ── 3. Load roads & project ──────────────────────────────────────────
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


def nearest_distance_km(point_utm: gpd.GeoDataFrame, roads_layer_utm: gpd.GeoDataFrame | None) -> float | None:
    if roads_layer_utm is None or roads_layer_utm.empty:
        return None
    try:
        point_geom = point_utm.geometry.iloc[0]
        distances = roads_layer_utm.geometry.distance(point_geom)
        return round(float(distances.min()) / 1000.0, 2)
    except Exception:
        return None

# ── 4. Per-cell intersection ─────────────────────────────────────────
print(f"\nIntersecting zones with grid cells (R-tree index)...", flush=True)
KML_DIR.mkdir(parents=True, exist_ok=True)
for stale_kml in KML_DIR.glob("*.kml"):
    stale_kml.unlink()

sindex = zones.sindex
task_records = []
task_count = 0
total_exported_ha = 0.0

for idx, cell in enumerate(grid_cells):
    candidates_idx = sindex.query(cell, predicate='intersects')
    if len(candidates_idx) == 0:
        continue

    candidates = zones.iloc[candidates_idx].copy()
    clipped = candidates
    clipped.geometry = clipped.geometry.intersection(cell)

    clipped = clipped[
        clipped.geometry.notna()
        & (~clipped.is_empty)
        & (clipped.geom_type.isin(['Polygon', 'MultiPolygon']))
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

    planting_geom = unary_union(clipped.geometry.values)
    if planting_geom.is_empty:
        continue

    centroid_lat = round(cell.centroid.y, 2)
    centroid_lon = round(cell.centroid.x, 2)
    kml_name = f"task_grid_{centroid_lat:.2f}_{centroid_lon:.2f}.kml"
    kml_path = KML_DIR / kml_name

    # Export KML
    export_gdf = gpd.GeoDataFrame(
        {
            "Name": [kml_name.replace(".kml", "")],
            "area_ha": [round(area_ha, 1)],
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

    # Distance to nearest road
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
        "distance_to_road_km": dist_km if dist_km is not None else "",
        "distance_to_kazakhstan_road_km": dist_kz_km if dist_kz_km is not None else "",
        "territory_scope": TERRITORY_SCOPE,
    })
    task_count += 1
    total_exported_ha += area_ha

    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx+1}/{len(grid_cells)} cells ... ({task_count} tasks)", flush=True)

t2 = time.time()
print(f"  Intersection & export in {t2-t1:.1f}s")

# ── 5. Save enriched CSV ────────────────────────────────────────────
print(f"\nSaving enriched index: {INDEX_OUT.name} ...", flush=True)
INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
index_df = pd.DataFrame(task_records)
if not index_df.empty:
    index_df = index_df.sort_values("area_ha", ascending=False)
    index_df.to_csv(INDEX_OUT, index=False, encoding="utf-8")
    print(f"  Records saved: {len(index_df):,}")

t3 = time.time()

# ── 6. Summary ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("V5 LOGISTICS PREP SUMMARY")
print(f"{'=' * 60}")
print(f"  Total pipeline:             {t3-t0:.1f}s")
print(f"  Source zones loaded:        {len(zones):,}")
print(f"  Grid cells evaluated:       {len(grid_cells):,}")
print(f"  Generated task KMLs:        {task_count:,}")
print(f"  Total exported area:        {total_exported_ha:,.1f} ha ({total_exported_ha/100:,.1f} km²)")
print(f"  KML directory:              {KML_DIR}")
print(f"  Enriched index:             {INDEX_OUT}")

if task_count > 0 and "distance_to_road_km" in index_df.columns:
    accessible = index_df[index_df["distance_to_road_km"].notna() & (index_df["distance_to_road_km"] != "")]
    if not accessible.empty:
        acc_km = accessible["distance_to_road_km"].astype(float)
        print(f"\n  Tasks with road distance:   {len(accessible):,}")
        print(f"  Mean distance to road:     {acc_km.mean():.2f} km")
        print(f"  Median distance to road:   {acc_km.median():.2f} km")
        print(f"  Tasks within 2 km:         {(acc_km < 2).sum():,} ({(acc_km < 2).mean()*100:.1f}%)")

if task_count > 0 and "distance_to_kazakhstan_road_km" in index_df.columns:
    accessible_kz = index_df[index_df["distance_to_kazakhstan_road_km"].notna() & (index_df["distance_to_kazakhstan_road_km"] != "")]
    if not accessible_kz.empty:
        acc_kz_km = accessible_kz["distance_to_kazakhstan_road_km"].astype(float)
        print(f"\n  Tasks with Kazakhstan road distance: {len(accessible_kz):,}")
        print(f"  Mean distance to Kazakhstan road:   {acc_kz_km.mean():.2f} km")
        print(f"  Median distance to Kazakhstan road: {acc_kz_km.median():.2f} km")
        print(f"  Tasks within 25 km of KZ road:      {(acc_kz_km < 25).sum():,} ({(acc_kz_km < 25).mean()*100:.1f}%)")

print(f"\n{'=' * 60}")
print("DONE. Candidate task grid prepared for field review.", flush=True)
