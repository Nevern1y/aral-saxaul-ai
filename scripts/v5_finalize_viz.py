"""
v5_finalize_viz.py — categorical clean-up, reproject to EPSG:4326, premium palette.

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
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
from scipy.ndimage import convolve

from v5_rules import CLASS_NAMES

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"
REPORTS = BASE / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RAW_MAP = DATA / "suitability_map_v5.tif"
FILTERED_MAP = DATA / "suitability_map_v5_filtered.tif"
GEOJSON_PATH = DATA / "optimal_zones_v5.geojson"
HTML_PATH = REPORTS / "suitability_map_v5.html"

CLASSES = sorted(CLASS_NAMES)
MAJORITY_KERNEL = np.ones((3, 3), dtype=np.uint8)

PALETTE = {
    0:  "#E2E8F0",   # WATER_NODATA
    1:  "#065F46",   # CANDIDATE SUITABLE
    3:  "#FDE68A",   # RISK_DRY_SALT
    4:  "#991B1B",   # DEAD_WET_TOXIC
    5:  "#1F2937",   # OBSTACLE_TOPO
    10: "#A7F3D0",   # VEGETATION
}
DISPLAY_NAMES_RU = {
    0: "Вода / тень / нет данных",
    1: "Кандидатные зоны",
    3: "Риск сухой соли",
    4: "Риск влажной рапы",
    5: "Сложный рельеф",
    10: "Есть растительность",
}
t0 = time.time()
print("=" * 56)
print("  V5.1 FINALIZE — Categorical Filter + Screening Viz")
print("=" * 56, flush=True)


def majority_filter_block(padded: np.ndarray) -> np.ndarray:
    """3x3 categorical majority filter; ties keep the original center class."""
    center = padded[1:-1, 1:-1]
    center_count = np.zeros(center.shape, dtype=np.uint8)
    counts_by_class: list[tuple[int, np.ndarray]] = []

    for cls in CLASSES:
        counts = convolve(
            (padded == cls).astype(np.uint8),
            MAJORITY_KERNEL,
            mode="constant",
            cval=0,
        )[1:-1, 1:-1]
        counts_by_class.append((cls, counts))
        center_count[center == cls] = counts[center == cls]

    out = center.copy()
    best_count = center_count.copy()
    for cls, counts in counts_by_class:
        update = counts > best_count
        out[update] = cls
        best_count[update] = counts[update]
    return out.astype(np.uint8)


def write_majority_filtered_raster() -> tuple[dict[int, int], int]:
    class_counts = {cls: 0 for cls in CLASSES}
    total_px = 0
    block_rows = 512

    with rasterio.open(RAW_MAP) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=None, compress="lzw")

        with rasterio.open(FILTERED_MAP, "w", **profile) as dst:
            n_blocks = (src.height + block_rows - 1) // block_rows
            for block_idx, row0 in enumerate(range(0, src.height, block_rows), start=1):
                nrows = min(block_rows, src.height - row0)
                read_row0 = max(0, row0 - 1)
                read_row1 = min(src.height, row0 + nrows + 1)
                data = src.read(
                    1,
                    window=Window(0, read_row0, src.width, read_row1 - read_row0),
                )

                top_pad = 1 if row0 == 0 else 0
                bottom_pad = 1 if row0 + nrows >= src.height else 0
                padded = np.pad(data, ((top_pad, bottom_pad), (1, 1)), mode="edge")
                filtered_block = majority_filter_block(padded)

                dst.write(filtered_block, 1, window=Window(0, row0, src.width, nrows))
                total_px += int(filtered_block.size)
                for cls in CLASSES:
                    class_counts[cls] += int((filtered_block == cls).sum())

                if block_idx == 1 or block_idx == n_blocks or block_idx % 10 == 0:
                    print(f"{block_idx}/{n_blocks}", end=" ", flush=True)

    return class_counts, total_px

# ── 1. Categorical majority filter ─────────────────────────────────────
print("\n[1/4] Majority filter (3x3 categorical) ...", end=" ", flush=True)
class_counts, total_px = write_majority_filtered_raster()
print(f"done -> {FILTERED_MAP.name}")

# ── 2. Class distribution ──────────────────────────────────────────────
print("\n[2/4] Class distribution (after filter):")
for cls, name in sorted(CLASS_NAMES.items()):
    cnt = class_counts.get(cls, 0)
    pct = cnt / total_px * 100 if total_px else 0.0
    print(f"    {cls} ({name:<24}) {cnt:>12,}  ({pct:6.2f}%)", flush=True)

# ── 3. Write lightweight vector note ───────────────────────────────────
print("\n[3/4] Writing vector export note -> GeoJSON ...", end=" ", flush=True)
vector_note = {
    "type": "FeatureCollection",
    "features": [],
    "metadata": {
        "status": "moved_to_operational_export",
        "reason": "Full-resolution 10 m candidate vectorization creates a very large file and is not used by the dashboard.",
        "use_instead": "outputs/data/operational_zones_v5.geojson generated by scripts/v5_extract_stats.py",
        "source_raster": str(FILTERED_MAP),
    },
}
with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
    json.dump(vector_note, f, ensure_ascii=False, indent=2)
print(f"done -> {GEOJSON_PATH.name}")

# ── 4. Reproject filtered raster to EPSG:4326 + generate PNG + Folium ─
print("\n[4/4] Reprojecting to EPSG:4326 and generating Folium map ...", flush=True)

with rasterio.open(FILTERED_MAP) as src:
    src_crs = src.crs
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
    name="Карта предварительного отбора V5.1",
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
    <b style="font-size:14px;">Aral Saxaul V5.1</b><br>
"""
for cls in sorted(PALETTE):
    name = DISPLAY_NAMES_RU.get(cls, CLASS_NAMES.get(cls, f"Класс {cls}"))
    hex_c = PALETTE[cls]
    legend_html += (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{hex_c};border-radius:2px;margin-right:6px;"></span>'
        f'{name}<br>'
    )
legend_html += """
    <hr style="margin:6px 0;">
    <span style="color:#666;font-size:11px;">Предварительная карта | категориальный фильтр 3x3 | маска воды и теней</span>
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
print(f"    {GEOJSON_PATH.name}  (vector export note)")
print(f"    {HTML_PATH.name}  (reprojected EPSG:4326)")
print(f"{'=' * 56}")
