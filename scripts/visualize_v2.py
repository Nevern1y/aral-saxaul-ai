"""
visualize_v2.py — Create interactive folium map of salinity_map_v2_alpha.tif
"""

import warnings
warnings.filterwarnings("ignore")

import rasterio
import numpy as np
from pathlib import Path
from math import floor, ceil

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
RASTER = BASE / "outputs/data/salinity_map_v2_alpha.tif"
OUT_HTML = BASE / "outputs/maps/salinity_v2_alpha.html"

Path(BASE / "outputs/maps").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("V2.0 MAP VISUALIZATION — Folium + ImageOverlay")
print("=" * 60)

# ── 1. Read raster, downsample for web ────────────────────────────────
with rasterio.open(RASTER) as src:
    bounds = src.bounds
    crs = src.crs
    data = src.read(1)
    profile = src.profile

print(f"Raster: {data.shape}, CRS: {crs}")
print(f"Bounds: {bounds}")

# Downsample: target ~2000px width for web
width = data.shape[1]
target_w = 2000
factor = max(1, width // target_w)

# Block mean downsampling
h, w = data.shape
h2, w2 = h // factor, w // factor
data_ds = data[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))

print(f"Downsampled: {data_ds.shape} (factor={factor})")

# ── 2. Create RGBA overlay (transparent NoData) ───────────────────────
# Color scale: blue (low salt) → yellow → red (high salt)
valid = data_ds > -9990
vmin, vmax = 1.25, 2.26  # from earlier stats, use dynamic if different
vmin = float(np.percentile(data_ds[valid], 2))
vmax = float(np.percentile(data_ds[valid], 98))
print(f"Color range: {vmin:.3f} – {vmax:.3f}%")

# Normalize
norm = np.clip((data_ds - vmin) / (vmax - vmin), 0, 1)

# Create RGBA (R,G,B,A) — light blue→yellow→red colormap
rgba = np.zeros((*data_ds.shape, 4), dtype=np.uint8)

# Blue → Cyan → Yellow → Red
for i in range(h2):
    for j in range(w2):
        if not valid[i, j]:
            rgba[i, j] = [0, 0, 0, 0]
            continue
        v = norm[i, j]
        if v < 0.5:
            # Blue → Cyan
            t = v / 0.5
            rgba[i, j] = [int(255 * t), int(255 * t), 255, 200]
        else:
            # Cyan → Yellow → Red
            t = (v - 0.5) / 0.5
            rgba[i, j] = [255, int(255 * (1 - t)), int(255 * (1 - t)), 200]

# ── 3. Save as PNG overlay ────────────────────────────────────────────
from PIL import Image
png_path = BASE / "outputs/maps/salinity_overlay.png"
img = Image.fromarray(rgba, "RGBA")
img.save(png_path)
print(f"Overlay saved: {png_path} ({png_path.stat().st_size/1024:.0f} KB)")

# ── 4. Create Folium map ──────────────────────────────────────────────
import folium
import base64
from io import BytesIO

# Embed PNG as base64 data URI (avoids browser file:// restrictions)
buf = BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
data_uri = f"data:image/png;base64,{b64}"

# Center of raster bounds
center_lat = (bounds.top + bounds.bottom) / 2
center_lon = (bounds.left + bounds.right) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
               tiles="OpenStreetMap", control_scale=True)

# Add satellite basemap
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satellite", show=True
).add_to(m)

folium.TileLayer("OpenStreetMap", name="OSM", show=False).add_to(m)

# Overlay the salinity PNG
# The overlay bounds must match the raster bounds geo-referencing
overlay_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

folium.raster_layers.ImageOverlay(
    image=data_uri,
    bounds=overlay_bounds,
    opacity=0.65,
    name="Salinity V2.0 Alpha",
    show=True,
).add_to(m)

# Add colorbar legend via HTML
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; 
            background: white; padding: 12px; border-radius: 6px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.3); font-family: Arial; font-size: 12px;">
    <b>Salinity % (top layer)</b><br>
    <div style="width: 200px; height: 20px; 
                background: linear-gradient(to right, #0000FF, #00FFFF, #FFFF00, #FF0000);
                margin: 4px 0; border-radius: 2px;"></div>
    <div style="display: flex; justify-content: space-between;">
        <span>{:.2f}%</span>
        <span>{:.2f}%</span>
    </div>
    <i style="font-size: 10px; color: #888;">n=7 points, Ridge alpha=1000</i>
</div>
""".format(vmin, vmax)
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)

m.save(str(OUT_HTML))
print(f"\nMap saved: {OUT_HTML}")
print(f"  Open in browser to view")
print(f"  Center: {center_lat:.4f}, {center_lon:.4f}")
print(f"  Overlay bounds: {overlay_bounds}")
print(f"\nDone.")
