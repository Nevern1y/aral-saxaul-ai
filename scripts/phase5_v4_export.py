"""
phase5_v4_export.py — V4 vectorization, export & Folium map
AOI boundary is a strict BBOX rectangle. All geometries clipped to BBOX.
1. Sieve filter (remove clusters < 1 ha)
2. Vectorize Class 1 (Optimal) only
3. Simplify geometry
4. Clip to BBOX
5. Export to GeoJSON
6. Generate Folium HTML map with clean rectangular boundary
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import time
import numpy as np
import rasterio
from rasterio import features
from pathlib import Path
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shapely_transform
import pyproj
import folium
from folium import plugins

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
RASTER_PATH = BASE / "outputs/data/suitability_map_v4.tif"
GEOJSON_PATH = BASE / "outputs/data/optimal_zones_v4.geojson"
MAP_PATH = BASE / "outputs/reports/suitability_map_v4.html"

# AOI Bounding Box (from config.py: ARAL_BBOX)
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 57.5, 43.3, 62.0, 46.7
BBOX_POLYGON = box(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)

MMU_PX = 11       # ~1 ha at 30m
SIMPLIFY_TOL = 0.0005  # degrees (~50m)
TOP_N_CLUSTERS = 1000  # max polygons on map

print("=" * 60)
print("V4 EXPORT — Vectorization & Map")
print("=" * 60, flush=True)
t0 = time.time()

# ── 1. Read raster ────────────────────────────────────────────────────
print(f"\nReading: {RASTER_PATH.name}", flush=True)
with rasterio.open(RASTER_PATH) as src:
    data = src.read(1).astype("int16")
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs

t1 = time.time()
print(f"  Loaded: {data.shape} ({data.size:,} px) in {t1-t0:.1f}s", flush=True)

# Print class distribution
classes, counts = np.unique(data, return_counts=True)
class_labels = {0: "Outside/Water", 1: "Optimal", 2: "Risk", 3: "Dead", 4: "Topo Obstacle"}
for cls, cnt in zip(classes, counts):
    label = class_labels.get(cls, f"Class {cls}")
    print(f"  {label}: {cnt:,} px ({cnt/data.size*100:.1f}%)", flush=True)

# ── 2. Mask Class 1 (Optimal) → binary ────────────────────────────────
binary = np.zeros(data.shape, dtype="uint8")
binary[data == 1] = 1
opt_px = int(binary.sum())
print(f"  Optimal pixels: {opt_px:,}", flush=True)

# ── 3. Sieve filter ───────────────────────────────────────────────────
print(f"\nSieve filter (MMU={MMU_PX} px, ~1 ha)...", flush=True)
binary_sieved = features.sieve(binary, size=MMU_PX, connectivity=4)
opt_sieved_px = int(binary_sieved.sum())
removed = opt_px - opt_sieved_px
print(f"  After sieve: {opt_sieved_px:,} px ({removed:,} removed, {removed/opt_px*100:.1f}%)", flush=True)

if opt_sieved_px == 0:
    print("ERROR: No optimal pixels remain after sieve!", flush=True)
    sys.exit(1)

# ── 4. Vectorize ──────────────────────────────────────────────────────
print(f"\nVectorizing...", flush=True)
results = (
    {"properties": {"v": v}, "geometry": s}
    for s, v in features.shapes(binary_sieved, mask=binary_sieved, transform=transform)
)
geoms = []
raw_count = 0
for r in results:
    raw_count += 1
    poly = shape(r["geometry"])
    if poly.geom_type == "Polygon":
        geoms.append(poly.simplify(SIMPLIFY_TOL, preserve_topology=True))
    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            geoms.append(p.simplify(SIMPLIFY_TOL, preserve_topology=True))

t2 = time.time()
print(f"  Raw polygons: {raw_count:,}", flush=True)
print(f"  After simplify: {len(geoms):,} in {t2-t1:.1f}s", flush=True)

# ── 5. Filter by area ─────────────────────────────────────────────────
min_m2 = 10000  # 1 ha in m²
final = [g for g in geoms if g.area * (111320**2) > min_m2]

# Project to UTM 41N for accurate area (Aral Sea is in UTM 41N)
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32641", always_xy=True).transform
final_with_area = []
for g in final:
    try:
        g_utm = shapely_transform(project, g)
        area_ha = g_utm.area / 10000
        if area_ha >= 1.0:
            final_with_area.append((g, area_ha))
    except Exception:
        pass

final_with_area.sort(key=lambda x: x[1], reverse=True)
print(f"  After 1ha filter: {len(final_with_area):,} clusters", flush=True)
total_ha = sum(a for _, a in final_with_area)
print(f"  Total optimal area: {total_ha/100:,.0f} km² ({total_ha:,.0f} ha)", flush=True)

# ── 6. Clip to BBOX ───────────────────────────────────────────────────
print(f"\nClipping to BBOX...", flush=True)
clipped = []
for g, area_ha in final_with_area:
    if BBOX_POLYGON.contains(g):
        clipped.append((g, area_ha))
    elif BBOX_POLYGON.intersects(g):
        g_clipped = g.intersection(BBOX_POLYGON)
        if g_clipped.geom_type == "Polygon" and g_clipped.area > 0.0001:
            project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32641", always_xy=True).transform
            try:
                g_utm = shapely_transform(project, g_clipped)
                area_ha_clipped = g_utm.area / 10000
                if area_ha_clipped >= 1.0:
                    clipped.append((g_clipped, area_ha_clipped))
            except Exception:
                pass
        elif g_clipped.geom_type == "MultiPolygon":
            for p in g_clipped.geoms:
                if p.area > 0.0001:
                    try:
                        p_utm = shapely_transform(project, p)
                        area_ha_p = p_utm.area / 10000
                        if area_ha_p >= 1.0:
                            clipped.append((p, area_ha_p))
                    except Exception:
                        pass

clipped.sort(key=lambda x: x[1], reverse=True)
total_ha_clipped = sum(a for _, a in clipped)
print(f"  After BBOX clip: {len(clipped):,} clusters, {total_ha_clipped/100:,.0f} km² ({total_ha_clipped:,.0f} ha)", flush=True)

# ── 7. Export GeoJSON ─────────────────────────────────────────────────
print(f"\nExporting: {GEOJSON_PATH.name}", flush=True)
features_out = []
for g, area_ha in clipped:
    features_out.append({
        "type": "Feature",
        "properties": {
            "area_ha": round(area_ha, 1),
            "area_km2": round(area_ha / 100, 2),
        },
        "geometry": mapping(g),
    })

geojson = {"type": "FeatureCollection", "features": features_out}
GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False)

t3 = time.time()
print(f"  Saved: {len(features_out):,} features ({os.path.getsize(GEOJSON_PATH)/1024:.0f} KB) in {t3-t2:.1f}s", flush=True)

# ── 7. Folium map ─────────────────────────────────────────────────────
print(f"\nGenerating Folium map: {MAP_PATH.name}", flush=True)

m = folium.Map(location=[45.0, 60.0], zoom_start=8,
               tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
               attr="Google Satellite")

folium.TileLayer(
    "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    name="Google Hybrid", attr="Google"
).add_to(m)

# Top N clusters for the map (by area)
top_features = features_out[:TOP_N_CLUSTERS]
top_geojson = {"type": "FeatureCollection", "features": top_features}

folium.GeoJson(
    top_geojson,
    name="Optimal Zones (V4)",
    style_function=lambda f: {
        "fillColor": "#2ecc40",
        "color": "#2ecc40",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["area_km2", "area_ha"],
        aliases=["Area (km²):", "Area (ha):"],
        localize=True,
    ),
    highlight_function=lambda f: {"weight": 2, "color": "#007bff"},
).add_to(m)

# ── AOI boundary: strict BBOX rectangle ─────────────────────────────
print(f"  Adding BBOX rectangle: [{MIN_LON}, {MIN_LAT}] — [{MAX_LON}, {MAX_LAT}]", flush=True)
folium.Rectangle(
    bounds=[[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]],
    color="#1f77b4",
    weight=2,
    dashArray="5, 5",
    fill=False,
    name="AOI BBOX (Область вычислений)",
).add_to(m)

# Stats
summary_html = f"""
<div style="position: fixed; top: 20px; right: 20px; z-index: 9999;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2); font-family: Arial;
            min-width: 220px;">
    <h4 style="margin: 0 0 10px;">Aral Saxaul AI — V4</h4>
    <b>Optimal:</b> {total_ha_clipped/100:,.0f} km²<br>
    <b>Clusters:</b> {len(clipped):,}<br>
    <b>Topo filter:</b> Slope &le; 5°<br>
    <b>AOI:</b> BBOX (Rectangle)<br>
    <b>MMU:</b> 1 ha<br>
    <b>Resolution:</b> 30 m<br>
    <b>Showing top:</b> {min(TOP_N_CLUSTERS, len(features_out)):,} clusters
</div>
"""
m.get_root().html.add_child(folium.Element(summary_html))

folium.LayerControl().add_to(m)
plugins.Fullscreen().add_to(m)
plugins.MeasureControl(position="bottomleft").add_to(m)
plugins.MousePosition().add_to(m)

MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
m.save(str(MAP_PATH))

t4 = time.time()
print(f"  Saved: {MAP_PATH.name} ({os.path.getsize(MAP_PATH)/1024:.0f} KB) in {t4-t3:.1f}s", flush=True)

# ── 8. Summary ────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Total pipeline:         {t4-t0:.1f}s")
print(f"  Optimal area (AOI):     {total_ha_clipped/100:,.0f} km² ({total_ha_clipped:,.0f} ha)")
print(f"  Clusters (>=1ha):       {len(clipped):,}")
print(f"  Slope filter:           ON (&le;5°)")
print(f"  AOI boundary:           BBOX Rectangle (clean)")
print(f"  GeoJSON:                {GEOJSON_PATH}")
print(f"  Map:                    {MAP_PATH}")
print(f"\n{'=' * 60}")
print("DONE. V4 ready for field brigade deployment.", flush=True)
