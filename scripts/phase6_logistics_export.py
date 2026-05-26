"""
phase6_logistics_export.py — Split optimal zones into 0.1°×0.1° operational
grid cells and export each valid cell as KML for offline tractor navigation.

Pipeline:
  1. Load optimal_zones_v4.geojson
  2. Build regular 0.1°×0.1° grid over total_bounds
  3. Per-cell intersection via R-tree spatial index → clip → project (UTM 41N)
  4. Filter cells where planting area < 5 ha
  5. Export valid cells as individual KML files
  6. Save tasks_index.csv with centroid coords and area
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
GEOJSON_PATH = BASE / "outputs/data/optimal_zones_v4.geojson"
KML_DIR = BASE / "outputs/logistics/tractor_tasks"
INDEX_PATH = BASE / "outputs/logistics/tasks_index.csv"

GRID_STEP = 0.1
MIN_AREA_HA = 5.0
UTM_CRS = "EPSG:32641"

print("=" * 60)
print("PHASE 6 — Logistics Export: Optimal Zones → Tractor Tasks")
print("=" * 60, flush=True)
t0 = time.time()

# ── 1. Load optimal zones ──────────────────────────────────────────────
print(f"\nLoading: {GEOJSON_PATH.name} ...", flush=True)
zones = gpd.read_file(GEOJSON_PATH)
if zones.empty:
    print("ERROR: Empty GeoJSON. Nothing to export.", flush=True)
    sys.exit(1)

print(f"  Features: {len(zones):,}", flush=True)
print(f"  CRS: {zones.crs}", flush=True)

bounds = zones.total_bounds
print(f"  Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}] — [{bounds[2]:.4f}, {bounds[3]:.4f}]", flush=True)

t1 = time.time()
print(f"  Loaded in {t1-t0:.1f}s", flush=True)

# ── 2. Build 0.1°×0.1° grid over total_bounds ─────────────────────────
print(f"\nBuilding {GRID_STEP}°×{GRID_STEP}° grid...", flush=True)

x_steps = int((bounds[2] - bounds[0]) / GRID_STEP) + 1
y_steps = int((bounds[3] - bounds[1]) / GRID_STEP) + 1

grid_cells = []
for i in range(x_steps):
    for j in range(y_steps):
        x_min = bounds[0] + i * GRID_STEP
        y_min = bounds[1] + j * GRID_STEP
        grid_cells.append(box(x_min, y_min, x_min + GRID_STEP, y_min + GRID_STEP))

print(f"  Grid cells: {len(grid_cells):,} ({x_steps} × {y_steps})", flush=True)

t2 = time.time()
print(f"  Grid built in {t2-t1:.1f}s", flush=True)

# ── 3. Per-cell intersection (R-tree spatial index) ────────────────────
print(f"\nIntersecting zones with grid cells (R-tree index)...", flush=True)

KML_DIR.mkdir(parents=True, exist_ok=True)

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

    # Dissolve clipped zones into a single geometry for a clean KML
    planting_geom = unary_union(clipped.geometry.values)
    if planting_geom.is_empty:
        continue

    centroid_lat = round(cell.centroid.y, 2)
    centroid_lon = round(cell.centroid.x, 2)
    kml_name = f"task_grid_{centroid_lat:.2f}_{centroid_lon:.2f}.kml"
    kml_path = KML_DIR / kml_name

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
        task_count += 1
        total_exported_ha += area_ha
        task_records.append({
            "filename": kml_name,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "area_ha": round(area_ha, 1),
        })
    except Exception as e:
        print(f"  WARNING: Failed to export {kml_name}: {e}", flush=True)

    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx+1}/{len(grid_cells)} cells ... ({task_count} tasks)", flush=True)

t3 = time.time()
print(f"  Intersection & export completed in {t3-t2:.1f}s", flush=True)

# ── 4. Save index CSV ──────────────────────────────────────────────────
print(f"\nSaving task index: {INDEX_PATH} ...", flush=True)
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
index_df = pd.DataFrame(task_records)
if not index_df.empty:
    index_df = index_df.sort_values("area_ha", ascending=False)
    index_df.to_csv(INDEX_PATH, index=False, encoding="utf-8")
    print(f"  Records saved: {len(index_df):,}", flush=True)

t4 = time.time()

# ── 5. Summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("LOGISTICS EXPORT SUMMARY")
print(f"{'=' * 60}")
print(f"  Total pipeline:             {t4-t0:.1f}s")
print(f"  Source zones loaded:        {len(zones):,}")
print(f"  Grid cells evaluated:       {len(grid_cells):,}")
print(f"  Generated task KML files:   {task_count:,}")
print(f"  Total exported area:        {total_exported_ha:,.1f} ha ({total_exported_ha/100:,.1f} km²)")
print(f"  KML directory:              {KML_DIR}")
print(f"  Task index:                 {INDEX_PATH}")
print(f"\n{'=' * 60}")
print("DONE. Ready for field brigade deployment.", flush=True)
