"""
v5_extract_stats.py — Pre-compute zone statistics and operational GeoJSON.

Decimates the filtered suitability map 10x (100 m resolution, 1 px = 1 ha),
labels connected optimal zones, filters by >= 10 ha, and exports:
  - outputs/data/v5_stats.json
  - outputs/data/operational_zones_v5.geojson
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import Resampling
from scipy.ndimage import label
from shapely.geometry import mapping as shp_mapping, shape as shp_shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"
DATA.mkdir(parents=True, exist_ok=True)

TIF_PATH = DATA / "suitability_map_v5_filtered.tif"
STATS_PATH = DATA / "v5_stats.json"
GEOJSON_PATH = DATA / "operational_zones_v5.geojson"

t0 = time.time()
print("=" * 56)
print("  V5.0 EXTRACT STATS — Pre-compute zone statistics")
print("=" * 56, flush=True)

# ── 1. Read with 10x decimation (100 m / pixel) ──────────────────────
print("\n[1/4] Reading TIF with 10x decimation ...", end=" ", flush=True)
with rasterio.open(TIF_PATH) as src:
    h, w = src.height, src.width
    dh, dw = h // 10, w // 10
    arr = src.read(1, out_shape=(1, dh, dw), resampling=Resampling.mode)
    transform = src.transform * src.transform.scale(
        (w / dw), (h / dh)
    )
    src_crs = src.crs
print(f"done ({arr.shape[1]}x{arr.shape[0]}, ~{arr.size / 1e6:.1f}M px)")

# ── 2. Label connected optimal zones ──────────────────────────────────
print("[2/4] Labeling connected zones ...", end=" ", flush=True)
opt_mask = (arr == 1).astype(np.int8)
labeled, n_labels = label(opt_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
zone_sizes = np.bincount(labeled.ravel())[1:]  # px count per zone
print(f"done ({n_labels:,} raw zones)")

# ── 3. Filter >= 10 ha (>= 10 px at 100 m) ────────────────────────────
print("[3/4] Filtering >= 10 ha, generating GeoJSON ...", flush=True)
large_mask = zone_sizes >= 10
large_indices = np.where(large_mask)[0] + 1
large_sizes_px = zone_sizes[large_mask]

n_clusters = int(len(large_indices))
top10_px = sorted(large_sizes_px, reverse=True)[:10]
top10_ha = [int(x) for x in top10_px]
top10_km2 = sum(top10_px) / 100
area_ha = int(zone_sizes.sum())
area_km2 = area_ha / 100

# Build histogram bins
bins_def = [(10, 100), (100, 1000), (1000, 5000), (5000, int(large_sizes_px.max()) + 1)]
hist_labels = ["10-100", "100-1000", "1000-5000", ">5000"]
hist_counts = []
for lo, hi in bins_def:
    hist_counts.append(int(((large_sizes_px >= lo) & (large_sizes_px < hi)).sum()))

stats = {
    "area_ha": area_ha,
    "area_km2": round(area_km2, 1),
    "clusters": n_clusters,
    "top10_ha": top10_ha,
    "top10_km2": round(top10_km2, 1),
    "histogram": {
        "labels": hist_labels,
        "counts": hist_counts,
    },
}

# Generate operational GeoJSON with simplified geometries
large_indices_set = set(int(x) for x in large_indices)
keep_mask = np.isin(labeled, large_indices).astype(np.uint8)
transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
op_features = []
for s, label_val in shapes(labeled, mask=keep_mask, transform=transform):
    label_val = int(label_val)
    if label_val not in large_indices_set:
        continue
    geom = shp_shape(s)
    geom_wgs84 = shp_transform(transformer.transform, geom)
    geom_simple = geom_wgs84.simplify(tolerance=0.0005, preserve_topology=True)
    if geom_simple.is_empty or geom_simple.geom_type not in ("Polygon", "MultiPolygon"):
        continue
    px_count = zone_sizes[label_val - 1]
    op_features.append({
        "type": "Feature",
        "properties": {
            "class": 1,
            "zone": "Optimal",
            "area_ha": round(float(px_count), 1),
        },
        "geometry": shp_mapping(geom_simple),
    })

with open(GEOJSON_PATH, "w") as f:
    json.dump({"type": "FeatureCollection", "features": op_features}, f, indent=2)

print(f"    Zones >= 10 ha: {n_clusters:,}")
print(f"    GeoJSON: {GEOJSON_PATH.name} ({len(op_features):,} polygons, "
      f"{GEOJSON_PATH.stat().st_size / 1e6:.1f} MB)")

# ── 4. Write stats JSON ───────────────────────────────────────────────
print(f"[4/4] Writing {STATS_PATH.name} ...", end=" ", flush=True)
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

elapsed = time.time() - t0
print("done")
print(f"\n{'=' * 56}")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Outputs:")
print(f"    {STATS_PATH.name}")
print(f"    {GEOJSON_PATH.name}")
print(f"{'=' * 56}")
