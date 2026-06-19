"""Phase 8b — render the V6 suitability layer as a self-contained Folium map.

Produces outputs/reports/suitability_map_v6.html with two togglable overlays
(base64-embedded PNGs, so the HTML works on Streamlit Cloud where the source
.tif files are gitignored):

  1. "Зоны пригодности V6" — suitability_zones_v6.tif in the SAME palette/legend
     codes as V5.1 (candidate / moderate-salt / strong-salt / vegetation), so the
     map reads like the existing product but is built on the lab-calibrated layer.
  2. "Балл пригодности V6 (0..1)" — suitability_index_v6.tif as a green→red
     gradient (high score = less saline = greener), off by default.

Both rasters are already EPSG:4326 (no reprojection needed). They are downsampled
to ~1400 px wide for a light HTML; mode-resampling preserves zone classes, average
for the continuous score.

This does NOT touch V5: the V5 map, thresholds, and logistics stay frozen. The
dashboard shows V6 as the primary map and keeps V5.1 as a 10 m detail layer.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import folium
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"render_v6_map needs rasterio + folium + Pillow: {e}")

BASE = Path(__file__).resolve().parent.parent.parent
ODATA = BASE / "outputs" / "data"
REPORTS = BASE / "outputs" / "reports"

ZONES = ODATA / "suitability_zones_v6.tif"
INDEX = ODATA / "suitability_index_v6.tif"
HTML_PATH = REPORTS / "suitability_map_v6.html"
PNG_ZONES = REPORTS / "suitability_map_v6_zones.png"
PNG_INDEX = REPORTS / "suitability_map_v6_score.png"

NODATA_F = -9999.0
MAX_DIM = 1400

# Same palette as V5.1 (UX parity). V6 has no topo(5) class; 3/4 are salinity severity.
PALETTE = {
    0:  (0, 0, 0, 0),          # water / nodata -> transparent
    1:  (6, 95, 70, 220),      # candidate (low salinity)  #065F46
    3:  (253, 230, 138, 220),  # moderate salinity         #FDE68A
    4:  (153, 27, 27, 220),    # strong salinity           #991B1B
    10: (167, 243, 208, 220),  # vegetation                #A7F3D0
}
DISPLAY_NAMES_RU = {
    1: "Кандидат (низкое засоление)",
    3: "Умеренное засоление",
    4: "Сильное засоление",
    10: "Есть растительность",
    0: "Вода / нет данных",
}


def downsampled(src, band, resampling):
    """Read a band downsampled so width <= MAX_DIM, returning (array, bounds)."""
    scale = min(1.0, MAX_DIM / src.width)
    out_w = max(1, int(src.width * scale))
    out_h = max(1, int(src.height * scale))
    arr = src.read(band, out_shape=(out_h, out_w), resampling=resampling)
    b = src.bounds
    bounds = [[b.bottom, b.left], [b.top, b.right]]
    return arr, bounds, out_w, out_h


def zones_rgba(arr):
    img = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    for cls, rgba in PALETTE.items():
        img[arr == cls] = rgba
    return img


def score_rgba(arr):
    """Continuous score -> green(high)→yellow→red(low); NoData transparent."""
    img = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    valid = arr != NODATA_F
    v = np.clip(arr, 0.0, 1.0)
    # simple 3-stop ramp: 0=red(153,27,27) -> 0.5=amber(253,230,138) -> 1=green(6,95,70)
    r = np.where(v < 0.5, 153 + (253 - 153) * (v / 0.5), 253 + (6 - 253) * ((v - 0.5) / 0.5))
    g = np.where(v < 0.5, 27 + (230 - 27) * (v / 0.5), 230 + (95 - 230) * ((v - 0.5) / 0.5))
    bl = np.where(v < 0.5, 27 + (138 - 27) * (v / 0.5), 138 + (70 - 138) * ((v - 0.5) / 0.5))
    img[..., 0] = np.where(valid, r, 0).astype(np.uint8)
    img[..., 1] = np.where(valid, g, 0).astype(np.uint8)
    img[..., 2] = np.where(valid, bl, 0).astype(np.uint8)
    img[..., 3] = np.where(valid, 210, 0).astype(np.uint8)
    return img


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not ZONES.exists():
        raise SystemExit(f"missing {ZONES} — run build_suitability_index.py first")

    print("rendering V6 zones overlay ...", flush=True)
    with rasterio.open(ZONES) as src:
        zarr, bounds, w, h = downsampled(src, 1, Resampling.mode)
    Image.fromarray(zones_rgba(zarr), "RGBA").save(PNG_ZONES)

    score_ok = INDEX.exists()
    if score_ok:
        print("rendering V6 score overlay ...", flush=True)
        with rasterio.open(INDEX) as src:
            iarr, _, _, _ = downsampled(src, 1, Resampling.average)
        Image.fromarray(score_rgba(iarr), "RGBA").save(PNG_INDEX)

    south, west = bounds[0]
    north, east = bounds[1]

    m = folium.Map(location=[45.0, 60.5], zoom_start=7,
                   tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                   attr="Google Satellite", control_scale=True)

    folium.raster_layers.ImageOverlay(
        image=str(PNG_ZONES), bounds=bounds, opacity=0.80,
        name="Зоны пригодности V6 (засоление)", show=True,
    ).add_to(m)

    if score_ok:
        folium.raster_layers.ImageOverlay(
            image=str(PNG_INDEX), bounds=bounds, opacity=0.80,
            name="Балл пригодности V6 (0..1)", show=False,
        ).add_to(m)

    folium.Rectangle(
        bounds=bounds, color="white", weight=2, dash_array="5, 5",
        fill=True, fill_opacity=0.02, fill_color="white",
        tooltip=f"AOI: {west:.2f}E {east:.2f}E | {south:.2f}N {north:.2f}N",
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend = """
<div style="position:fixed; bottom:20px; left:20px; z-index:9999;
            background:white; padding:10px 14px; border-radius:8px;
            box-shadow:0 0 10px rgba(0,0,0,0.15); font-size:13px;
            font-family:'Segoe UI',Arial,sans-serif;">
    <b style="font-size:14px;">Aral Saxaul V6 — слой солёности</b><br>
"""
    for cls in (1, 3, 4, 10, 0):
        r, g, b, a = PALETTE[cls]
        sw = "background:#ffffff;border:1px solid #bbb;" if cls == 0 else f"background:rgb({r},{g},{b});"
        legend += (f'<span style="display:inline-block;width:12px;height:12px;'
                   f'{sw}border-radius:2px;margin-right:6px;"></span>'
                   f'{DISPLAY_NAMES_RU[cls]}<br>')
    legend += """
    <hr style="margin:6px 0;">
    <span style="color:#666;font-size:11px;">30 м · модель солёности по 70 лаб. профилям (LOO AUC 0.68)<br>
    зоны 3/4 — градация засоления по NDMI, не физика V5</span>
</div>
"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(HTML_PATH))

    print(f"  PNG zones: {PNG_ZONES.name} ({w}x{h})")
    if score_ok:
        print(f"  PNG score: {PNG_INDEX.name}")
    print(f"  HTML: {HTML_PATH.name} ({HTML_PATH.stat().st_size / 1e3:.0f} KB)")
    print(f"  bounds south={south:.4f} west={west:.4f} north={north:.4f} east={east:.4f}")
    print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
