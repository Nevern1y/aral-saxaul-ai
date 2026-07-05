"""
diag_aoi_vectorization_loss.py — one-off diagnostic (not part of the pipeline).
Quantifies exactly where area is lost between aoi_mask_v5.tif (raster, gates
predictions) and aral_sea_1960.geojson (vector, used for point-in-polygon
splits), by redoing the vectorization step from build_aoi_mask.py with full
accounting at each stage: raw shapes -> area filter -> simplify -> holes.
"""
import warnings; warnings.filterwarnings("ignore")
import json, sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
MASK_PATH = BASE / "outputs/data/aoi_mask_v5.tif"
VECTOR_PATH = BASE / "outputs/aoi/aral_sea_1960.geojson"

AEA = "+proj=aea +lat_1=43 +lat_2=47 +lat_0=45 +lon_0=60 +datum=WGS84"
to_aea = Transformer.from_crs("EPSG:4326", AEA, always_xy=True).transform

def geo_area_km2(poly):
    return shp_transform(to_aea, poly).area / 1e6

print("=" * 70)
print("AOI VECTORIZATION LOSS DIAGNOSTIC")
print("=" * 70)

with rasterio.open(MASK_PATH) as src:
    data = src.read(1)
    transform = src.transform
    px_area_km2 = abs(transform.a * transform.e) * 111_320 * 111_320 / 1e6  # rough deg->km2, refined below
    n_mask_px = int((data == 1).sum())

# Pixel size in degrees -> use actual per-pixel geodesic area via a tiny reference square at mask centroid,
# but for a pure pixel-count cross-check simple 30m x 30m nominal is fine (already established: 900 m2/px).
raster_area_km2 = n_mask_px * 900 / 1e6
print(f"\n[Raster] AOI=1 pixels: {n_mask_px:,}  ->  area = {raster_area_km2:,.1f} km2 (900 m2/px nominal)")

print("\nRe-running features.shapes() on aoi_mask_v5.tif ...")
raw_shapes = list(features.shapes(data, mask=data, transform=transform))
print(f"  Total raw shapes yielded: {len(raw_shapes):,}")

all_polys = []
for s, v in raw_shapes:
    geom = shape(s)
    if geom.geom_type == "Polygon":
        all_polys.append(geom)
    elif geom.geom_type == "MultiPolygon":
        all_polys.extend(list(geom.geoms))

total_raw_area_deg2 = sum(p.area for p in all_polys)
print(f"  Total polygons (post multipoly-split): {len(all_polys):,}")
print(f"  Total raw area (deg2, before ANY filter/simplify): {total_raw_area_deg2:.4f}")

# Stage A: apply the >0.01 deg2 area filter exactly as build_aoi_mask.py does, BEFORE simplify
kept_raw = [p for p in all_polys if p.area > 0.01]
dropped_raw = [p for p in all_polys if p.area <= 0.01]
kept_raw_area = sum(p.area for p in kept_raw)
dropped_raw_area = sum(p.area for p in dropped_raw)
print(f"\n[Stage A: area>0.01 deg2 filter, BEFORE simplify]")
print(f"  Kept polygons:    {len(kept_raw):,}  area={kept_raw_area:.4f} deg2")
print(f"  Dropped polygons: {len(dropped_raw):,}  area={dropped_raw_area:.4f} deg2")
print(f"  Dropped fraction of total: {100*dropped_raw_area/total_raw_area_deg2:.2f}%")

# Stage B: now simplify the KEPT polygons exactly as build_aoi_mask.py does, and re-measure area
simplified = [p.simplify(0.001, preserve_topology=True) for p in kept_raw]
simp_area = sum(p.area for p in simplified)
print(f"\n[Stage B: simplify(0.001, preserve_topology=True) on kept polygons]")
print(f"  Area after simplify: {simp_area:.4f} deg2  (vs {kept_raw_area:.4f} before simplify)")
print(f"  Change from simplify: {simp_area - kept_raw_area:+.4f} deg2 ({100*(simp_area-kept_raw_area)/kept_raw_area:+.2f}%)")

# Convert everything to true geodesic km2 for the headline numbers
kept_raw_km2 = sum(geo_area_km2(p) for p in kept_raw)
dropped_raw_km2 = sum(geo_area_km2(p) for p in dropped_raw)
simp_km2 = sum(geo_area_km2(p) for p in simplified)

print(f"\n[Geodesic areas, Albers Equal-Area projection]")
print(f"  Raster (all AOI=1 px):                 {raster_area_km2:,.1f} km2")
print(f"  Vectorized, BEFORE any filter/simplify: {sum(geo_area_km2(p) for p in all_polys):,.1f} km2")
print(f"  Kept after >0.01 deg2 filter (pre-simp):{kept_raw_km2:,.1f} km2")
print(f"  Dropped by >0.01 deg2 filter:           {dropped_raw_km2:,.1f} km2  <-- SMALL FRAGMENTS LOST")
print(f"  After simplify(0.001):                 {simp_km2:,.1f} km2")
print(f"  Committed aral_sea_1960.geojson (net of holes, from earlier measurement): 55,061.8 km2")

# Now separately: what's the hole area (already measured before, re-derive briefly here to confirm)
with open(VECTOR_PATH, encoding="utf-8") as f:
    committed = json.load(f)
outer_only_km2 = 0.0
net_km2 = 0.0
for feat in committed["features"]:
    poly = shape(feat["geometry"])
    net_km2 += geo_area_km2(poly)
    from shapely.geometry import Polygon
    outer_only = Polygon(poly.exterior)
    outer_only_km2 += geo_area_km2(outer_only)
print(f"\n[Committed file hole check]")
print(f"  Outer rings only (ignoring holes): {outer_only_km2:,.1f} km2")
print(f"  Net (holes subtracted, shapely .area default): {net_km2:,.1f} km2")
print(f"  Hole area = outer - net: {outer_only_km2 - net_km2:,.1f} km2")

print(f"\n[RECONCILIATION]")
print(f"  Raster area:                  {raster_area_km2:,.1f} km2")
print(f"  - small-fragment filter loss: -{dropped_raw_km2:,.1f} km2")
print(f"  - simplify distortion:        {simp_km2-kept_raw_km2:+,.1f} km2")
print(f"  - interior holes (net):       -{outer_only_km2 - net_km2:,.1f} km2")
predicted_final = raster_area_km2 - dropped_raw_km2 + (simp_km2-kept_raw_km2) - (outer_only_km2-net_km2)
print(f"  => predicted final area:      {predicted_final:,.1f} km2")
print(f"  => actual committed area:     {net_km2:,.1f} km2")
print(f"  => residual (rounding/simplify-of-holes interaction): {predicted_final-net_km2:,.1f} km2")

print("\nDone.")
