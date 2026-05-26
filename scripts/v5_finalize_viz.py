"""
v5_finalize_viz.py — Morphological clean-up, reproject to EPSG:4326, premium palette.

Renders a PNG at ~1200 px wide in EPSG:4326 pixel space via rasterio.warp.reproject,
then overlays it on a Folium map via ImageOverlay. Zero projection shift.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import folium
import numpy as np
from PIL import Image
from rasterio.features import sieve, shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import rasterio
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"
REPORTS = BASE / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RAW_MAP = DATA / "suitability_map_v5.tif"
FILTERED_MAP = DATA / "suitability_map_v5_filtered.tif"
GEOJSON_PATH = DATA / "optimal_zones_v5.geojson"
HTML_PATH = REPORTS / "suitability_map_v5.html"

PALETTE = {
    0:  "#E2E8F0",   # WATER_NODATA
    1:  "#065F46",   # OPTIMAL
    3:  "#FDE68A",   # RISK_DRY_SALT
    4:  "#991B1B",   # DEAD_WET_TOXIC
    5:  "#1F2937",   # OBSTACLE_TOPO
    10: "#A7F3D0",   # VEGETATION
}
CLASS_NAMES = {0: "Water / NoData", 1: "Optimal", 3: "Dry Salt Crust",
               4: "Dead / Capillary Brine", 5: "Terrain Obstacle", 10: "Vegetation"}

t0 = time.time()
print("=" * 56)
print("  V5.0 FINALIZE — Morphological Filter + Premium Viz")
print("=" * 56, flush=True)

# ── 1. Median filter ───────────────────────────────────────────────────
print("\n[1/4] Median filter (3x3) ...", end=" ", flush=True)
with rasterio.open(RAW_MAP) as src:
    profile = src.profile
    src_crs = src.crs
    raw = src.read(1)

filtered = median_filter(raw, size=3).astype(np.uint8)
with rasterio.open(FILTERED_MAP, "w", **profile) as dst:
    dst.write(filtered, 1)
print(f"done -> {FILTERED_MAP.name}")

# ── 2. Class distribution ──────────────────────────────────────────────
print("\n[2/4] Class distribution (after filter):")
for cls, name in sorted(CLASS_NAMES.items()):
    cnt = int((filtered == cls).sum())
    pct = cnt / filtered.size * 100
    print(f"    {cls} ({name:<24}) {cnt:>12,}  ({pct:6.2f}%)", flush=True)

# ── 3. Vectorize Optimal zones only (for app.py compatibility) ─────────
print("\n[3/4] Vectorizing Optimal zones -> GeoJSON ...", end=" ", flush=True)
transformer = Transformer.from_crs("EPSG:32641", "EPSG:4326", always_xy=True)
mask = (filtered == 1).astype(np.uint8)
sieved = sieve(mask, size=5, connectivity=4)
opt_features = []
for s, v in shapes(sieved, mask=sieved, transform=profile["transform"]):
    geom = shape(s)
    geom_wgs84 = shapely_transform(transformer.transform, geom)
    opt_features.append({
        "type": "Feature",
        "properties": {"class": 1, "zone": "Optimal", "area_px": int(v)},
        "geometry": mapping(geom_wgs84),
    })
with open(GEOJSON_PATH, "w") as f:
    json.dump({"type": "FeatureCollection", "features": opt_features}, f, indent=2)
print(f"{len(opt_features):,} polygons -> {GEOJSON_PATH.name}")

# ── 4. Reproject filtered raster to EPSG:4326 + generate PNG + Folium ─
print("\n[4/4] Reprojecting to EPSG:4326 and generating Folium map ...", flush=True)

with rasterio.open(FILTERED_MAP) as src:
    # Reproject a downsampled version to EPSG:4326
    tgt_res = 0.00015  # ~15 m at Aral latitude — keeps detail but manageable
    tgt_crs = "EPSG:4326"
    transform, tgt_w, tgt_h = calculate_default_transform(
        src_crs, tgt_crs, src.width, src.height,
        *src.bounds, resolution=tgt_res,
    )

    # Cap at ~1400 px wide
    max_dim = 1400
    if tgt_w > max_dim:
        scale = max_dim / tgt_w
        tgt_w = max_dim
        tgt_h = int(tgt_h * scale)
        transform, tgt_w, tgt_h = calculate_default_transform(
            src_crs, tgt_crs, src.width, src.height,
            *src.bounds, dst_width=tgt_w, dst_height=tgt_h,
        )

    # Reproject using mode resampling (preserves class values)
    warped = np.empty((tgt_h, tgt_w), dtype=np.uint8)
    reproject(
        source=src.read(1),
        destination=warped,
        src_transform=src.profile["transform"],
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=tgt_crs,
        resampling=Resampling.mode,  # mode for categorical data
    )

    # EPSG:4326 bounds for ImageOverlay
    west = transform.c
    north = transform.f
    east = west + transform.a * tgt_w
    south = north + transform.e * tgt_h
    img_bounds = [[south, west], [north, east]]

    # Build RGBA image
    img = np.zeros((tgt_h, tgt_w, 4), dtype=np.uint8)
    for cls in sorted(PALETTE):
        hex_c = PALETTE[cls].lstrip("#")
        rgba = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        if cls == 0:
            rgba = (0, 0, 0, 0)  # fully transparent
        else:
            rgba = rgba + (220,)
        img[warped == cls] = rgba

    pil_img = Image.fromarray(img, "RGBA")
    png_path = HTML_PATH.with_suffix(".png")
    pil_img.save(png_path)

# Build Folium map
m = folium.Map(location=[45.0, 60.0], zoom_start=7,
               tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
               attr="Google Satellite", control_scale=True)

folium.raster_layers.ImageOverlay(
    image=str(png_path),
    bounds=img_bounds,
    opacity=0.80,
    name="Suitability V5.0",
).add_to(m)

# AOI bounding box (dashed white, faint fill)
folium.Rectangle(
    bounds=[[south, west], [north, east]],
    color="white",
    weight=2,
    dash_array="5, 5",
    fill=True,
    fill_opacity=0.03,
    fill_color="white",
    tooltip=f"AOI: {west:.2f}E {east:.2f}E | {south:.2f}N {north:.2f}N",
).add_to(m)

# Legend
legend_html = """
<div style="position:fixed; bottom:20px; left:20px; z-index:9999;
            background:white; padding:10px 14px; border-radius:8px;
            box-shadow:0 0 10px rgba(0,0,0,0.15); font-size:13px;
            font-family:'Segoe UI',Arial,sans-serif;">
    <b style="font-size:14px;">Aral Saxaul V5.0</b><br>
"""
for cls in sorted(PALETTE):
    name = CLASS_NAMES.get(cls, f"Class {cls}")
    hex_c = PALETTE[cls]
    legend_html += (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{hex_c};border-radius:2px;margin-right:6px;"></span>'
        f'{name}<br>'
    )
legend_html += """
    <hr style="margin:6px 0;">
    <span style="color:#666;font-size:11px;">3x3 median filter | NDWI + SCL water mask</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
m.save(str(HTML_PATH))

elapsed = time.time() - t0
print(f"  PNG: {png_path.name} ({pil_img.size[0]}x{pil_img.size[1]})")
print(f"  HTML: {HTML_PATH.name} ({HTML_PATH.stat().st_size / 1e3:.0f} KB)")
print(f"  Bounds: south={south:.4f} west={west:.4f} north={north:.4f} east={east:.4f}")
print(f"\n{'=' * 56}")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Outputs:")
print(f"    {FILTERED_MAP.name}")
print(f"    {GEOJSON_PATH.name}  ({len(opt_features):,} polygons)")
print(f"    {HTML_PATH.name}  (reprojected EPSG:4326)")
print(f"{'=' * 56}")
