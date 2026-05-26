"""
phase7_infrastructure.py — Download OSM road network via Overpass API
(1°×1° tiled queries), calculate distance from each tractor task centroid
to the nearest road, and enrich the task index.

Pipeline:
  1. Query Overpass API for highway ways (1° tiles over AOI BBOX)
  2. Save raw roads to GeoJSON
  3. Load tasks, project centroids to UTM 41N
  4. sjoin_nearest → distance_to_road_km per task
  5. Save enriched CSV + aralkum_roads.geojson
  6. Console: top-5 accessible, top-5 remote tasks
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import time
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
INDEX_PATH = BASE / "outputs/logistics/tasks_index.csv"
ROADS_OUT = BASE / "outputs/logistics/aralkum_roads.geojson"
INDEX_OUT = BASE / "outputs/logistics/tasks_index_enriched.csv"

MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 57.5, 43.3, 62.0, 46.7
UTM_CRS = "EPSG:32641"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Highway types relevant for tractor access in desert terrain
HIGHWAY_FILTER = "track|unclassified|tertiary|secondary|service|residential|path|road|primary|primary_link|secondary_link|tertiary_link"

print("=" * 60)
print("PHASE 7 — Infrastructure: OSM Roads & Task Accessibility")
print("=" * 60, flush=True)
t0 = time.time()

# ── 1. Download OSM roads via tiled Overpass queries ──────────────────
print(f"\nDownloading road network from OSM (1°×1° tiles) ...", flush=True)

STEP = 1.0
x_steps = int((MAX_LON - MIN_LON) / STEP) + 1
y_steps = int((MAX_LAT - MIN_LAT) / STEP) + 1
total_tiles = x_steps * y_steps

roads_list = []
failed_tiles = []

for i in range(x_steps):
    for j in range(y_steps):
        sub_min_lon = MIN_LON + i * STEP
        sub_max_lon = min(sub_min_lon + STEP, MAX_LON)
        sub_min_lat = MIN_LAT + j * STEP
        sub_max_lat = min(sub_min_lat + STEP, MAX_LAT)
        tile_idx = i * y_steps + j + 1

        query = f"""
        [out:json][timeout:180];
        (
          way["highway"~"{HIGHWAY_FILTER}"]
             ({sub_min_lat},{sub_min_lon},{sub_max_lat},{sub_max_lon});
        );
        out geom;
        """

        print(f"  Tile {tile_idx}/{total_tiles}: "
              f"[{sub_min_lon:.1f}, {sub_min_lat:.1f}] — "
              f"[{sub_max_lon:.1f}, {sub_max_lat:.1f}] ...", flush=True)

        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=300,
                headers={"User-Agent": "AralSaxaulAI/1.0"},
            )
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} — skip", flush=True)
                failed_tiles.append(tile_idx)
                continue

            data = resp.json()
            elements = data.get("elements", [])
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            failed_tiles.append(tile_idx)
            continue

        tile_count = 0
        for el in elements:
            if el.get("type") != "way":
                continue
            pts = el.get("geometry")
            if not pts or len(pts) < 2:
                continue
            coords = [(p["lon"], p["lat"]) for p in pts]
            line = LineString(coords)
            if line.is_empty:
                continue
            tags = el.get("tags", {})
            road_type = tags.get("highway", "unknown")
            roads_list.append({"fclass": road_type, "geometry": line})
            tile_count += 1

        print(f"    → {tile_count} road segments", flush=True)

if not roads_list:
    print("ERROR: No roads downloaded from any tile.", flush=True)
    sys.exit(1)

roads = gpd.GeoDataFrame(roads_list, crs="EPSG:4326")
print(f"\n  Total road segments: {len(roads):,}", flush=True)
if failed_tiles:
    print(f"  Failed tiles: {len(failed_tiles)}/{total_tiles}", flush=True)

fc = roads["fclass"].value_counts()
print(f"  Road type breakdown:")
for k, v in fc.items():
    print(f"    {k}: {v}")

# Export
ROADS_OUT.parent.mkdir(parents=True, exist_ok=True)
roads.to_file(ROADS_OUT, driver="GeoJSON")
print(f"  Saved: {ROADS_OUT.name} ({len(roads):,} features, "
      f"{os.path.getsize(ROADS_OUT)/1024:.0f} KB)", flush=True)

t1 = time.time()
print(f"  OSM download + export in {t1-t0:.1f}s", flush=True)

# ── 2. Load tasks, build centroids, distance matrix ───────────────────
print(f"\nLoading task index: {INDEX_PATH.name} ...", flush=True)
tasks_df = pd.read_csv(INDEX_PATH)
print(f"  Tasks: {len(tasks_df):,}", flush=True)

geometry = [Point(lon, lat) for lat, lon in zip(tasks_df.centroid_lat, tasks_df.centroid_lon)]
tasks_gdf = gpd.GeoDataFrame(tasks_df, geometry=geometry, crs="EPSG:4326")

print(f"Projecting to {UTM_CRS} ...", flush=True)
tasks_utm = tasks_gdf.to_crs(UTM_CRS)
roads_utm = roads.to_crs(UTM_CRS)

print(f"Running sjoin_nearest ...", flush=True)
joined = gpd.sjoin_nearest(
    tasks_utm,
    roads_utm[["geometry"]],
    how="left",
    distance_col="distance_m",
)

joined["distance_to_road_km"] = (joined["distance_m"] / 1000.0).round(2)

t2 = time.time()
print(f"  Distance matrix in {t2-t1:.1f}s", flush=True)

# ── 3. Enrich and save ────────────────────────────────────────────────
print(f"\nSaving enriched index ...", flush=True)
out_df = joined[["filename", "centroid_lat", "centroid_lon", "area_ha", "distance_to_road_km"]]
out_df.to_csv(INDEX_OUT, index=False, encoding="utf-8")
print(f"  Saved: {INDEX_OUT.name}", flush=True)

# ── 4. Analytics ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("ACCESSIBILITY ANALYSIS")
print(f"{'=' * 60}")

nearby = out_df[out_df.distance_to_road_km < 2.0]
print(f"\nTasks within 2 km of a road: {len(nearby):,} / {len(out_df):,} "
      f"({len(nearby)/len(out_df)*100:.1f}%)")
print(f"  Total area within 2 km:    {nearby.area_ha.sum():,.0f} ha / "
      f"{out_df.area_ha.sum():,.0f} ha "
      f"({nearby.area_ha.sum()/out_df.area_ha.sum()*100:.1f}%)")

print(f"\n{'─' * 50}")
print("TOP-5 MOST ACCESSIBLE (max area, road < 2 km):")
print(f"{'─' * 50}")
top_accessible = nearby.nlargest(5, "area_ha")
for _, row in top_accessible.iterrows():
    print(f"  {row['filename']:35s}  area={row['area_ha']:>8.1f} ha  "
          f"road_dist={row['distance_to_road_km']:>6.2f} km")

print(f"\n{'─' * 50}")
print("TOP-5 MOST REMOTE (farthest from any road):")
print(f"{'─' * 50}")
top_remote = out_df.nlargest(5, "distance_to_road_km")
for _, row in top_remote.iterrows():
    print(f"  {row['filename']:35s}  area={row['area_ha']:>8.1f} ha  "
          f"road_dist={row['distance_to_road_km']:>6.2f} km")

t3 = time.time()
print(f"\n{'=' * 60}")
print(f"Total pipeline: {t3-t0:.1f}s", flush=True)
print("DONE. Tasks enriched with distance_to_road_km.", flush=True)
