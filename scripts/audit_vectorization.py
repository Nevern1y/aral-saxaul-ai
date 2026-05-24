"""
audit_vectorization.py — QA audit of V3.2 vectorization
Samples 1000 random points inside optimal zones, checks NDMI compliance.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import random
import numpy as np
import rasterio
from pathlib import Path
from shapely.geometry import shape, Point
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
GEOJSON_PATH = BASE / "outputs/data/optimal_zones_v3.geojson"
RASTER_PATH = BASE / "outputs/data/feature_stack_30m.vrt"
TILE1_PATH = BASE / "outputs/data/feature_stack_30m_tile1.tif"
TILE0_PATH = BASE / "outputs/data/feature_stack_30m_tile0_redo.tif"

NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
N_SAMPLES = 1000
RANDOM_SEED = 42

print("=" * 60)
print("V3.2 QA AUDIT — Vectorization Sanity Check")
print("=" * 60, flush=True)

# ── 1. Load GeoJSON ───────────────────────────────────────────────────
print(f"\n1. Loading: {GEOJSON_PATH.name}", flush=True)
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson = json.load(f)

polygons = []
for feat in geojson["features"]:
    poly = shape(feat["geometry"])
    if poly.geom_type == "Polygon":
        polygons.append(poly)
    elif poly.geom_type == "MultiPolygon":
        polygons.extend(list(poly.geoms))

total_polys = len(polygons)
print(f"   Loaded {len(geojson['features'])} features, {total_polys} polygons", flush=True)

# ── 2. Merge all polygons → single geometry for point sampling ────────
print(f"\n2. Merging polygons for point sampling...", flush=True)
merged = unary_union(polygons)
bounds = merged.bounds  # (minx, miny, maxx, maxy)
print(f"   Bounds: {bounds}", flush=True)

# ── 3. Generate 1000 random points within polygons ────────────────────
print(f"\n3. Generating {N_SAMPLES} random points inside polygons...", flush=True)
random.seed(RANDOM_SEED)

points = []
while len(points) < N_SAMPLES:
    x = random.uniform(bounds[0], bounds[2])
    y = random.uniform(bounds[1], bounds[3])
    pt = Point(x, y)
    if merged.contains(pt):
        points.append(pt)

print(f"   Generated {len(points)} valid points", flush=True)

# ── 4. Sample NDMI from raster ────────────────────────────────────────
print(f"\n4. Sampling NDMI from raster...", flush=True)

coords = [(pt.x, pt.y) for pt in points]

ndmi_values = None
for raster_path in [RASTER_PATH, TILE1_PATH, TILE0_PATH]:
    if raster_path.exists():
        try:
            with rasterio.open(raster_path) as src:
                samples = list(src.sample(coords, indexes=1))
                ndmi_values = np.array([s[0] for s in samples], dtype=np.float32)
            print(f"   Sampled from: {raster_path.name}", flush=True)
            break
        except Exception as e:
            print(f"   Failed: {raster_path.name} — {e}", flush=True)
            continue

if ndmi_values is None:
    print("ERROR: No raster available for sampling!", flush=True)
    sys.exit(1)

# ── 5. Analysis ───────────────────────────────────────────────────────
print(f"\n5. Analysis", flush=True)

valid = ~(np.isnan(ndmi_values) | np.isinf(ndmi_values))
ndmi_valid = ndmi_values[valid]
invalid_count = int((~valid).sum())

optimal = (ndmi_valid < NDMI_OPTIMAL).sum()
dead = (ndmi_valid > NDMI_DEAD).sum()
risk = ((ndmi_valid >= NDMI_OPTIMAL) & (ndmi_valid <= NDMI_DEAD)).sum()
total_valid = len(ndmi_valid)

print(f"\n   {'Metric':<25} {'Count':>8} {'%':>8}")
print(f"   {'-'*41}")
print(f"   {'Total samples':<25} {N_SAMPLES:>8} {'100.0%':>8}")
print(f"   {'Invalid (NaN/Inf)':<25} {invalid_count:>8} {invalid_count/N_SAMPLES*100:>7.1f}%")
print(f"   {'Valid NDMI reads':<25} {total_valid:>8} {total_valid/N_SAMPLES*100:>7.1f}%")
print(f"   {'---':<25}")
print(f"   {'OPTIMAL (NDMI < -0.055)':<25} {optimal:>8} {optimal/total_valid*100:>7.1f}%")
print(f"   {'RISK (-0.055 ≤ NDMI ≤ -0.025)':<25} {risk:>8} {risk/total_valid*100:>7.1f}%")
print(f"   {'DEAD (NDMI > -0.025)':<25} {dead:>8} {dead/total_valid*100:>7.1f}%")

accuracy = optimal / total_valid * 100
fp_rate = dead / total_valid * 100

print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
print(f"   Accuracy (NDMI < -0.055):       {accuracy:.1f}%")
print(f"   False Positives (NDMI > -0.025): {fp_rate:.1f}%")
print(f"   {'PASS' if accuracy >= 95 and fp_rate < 1 else 'REVIEW NEEDED'}")

if ndmi_values is not None and total_valid > 0:
    print(f"\n   NDMI stats among sampled points:")
    print(f"     Min: {ndmi_valid.min():.4f}")
    print(f"     Max: {ndmi_valid.max():.4f}")
    print(f"     Mean: {ndmi_valid.mean():.4f}")
    print(f"     P50: {np.median(ndmi_valid):.4f}")
    print(f"     P05: {np.percentile(ndmi_valid, 5):.4f}")
    print(f"     P95: {np.percentile(ndmi_valid, 95):.4f}")

print(f"\nDone. Audit complete.", flush=True)
