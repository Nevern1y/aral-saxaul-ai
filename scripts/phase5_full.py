import rasterio
import numpy as np
import folium
import folium.plugins as plugins
import json, os
from rasterio import features
from shapely.geometry import shape
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config

data_dir = str(config.output_dir / "data")
reports_dir = str(config.output_dir / "reports")

with rasterio.open(os.path.join(data_dir, "suitability_full.tif")) as src:
    proba = src.read(1)
    transform = src.transform
    bounds = src.bounds

# --- Statistics ---
valid = proba != -1.0
high = proba > 0.5
n_valid = int(valid.sum())
n_high = int(high.sum())
n_total = proba.size

pixel_ha = (30 * 30) / 10000
print(f"Total pixels: {n_total:,}")
print(f"Valid: {n_valid:,} ({n_valid/n_total*100:.1f}%)")
print(f"High suitability: {n_high:,} ({n_high/n_total*100:.1f}% of total)")
print(f"Suitable area: {n_high * pixel_ha:,.0f} ha ({n_high * 30 * 30 / 1e6:,.0f} km²)")
print(f"Mean suitability (valid): {float(proba[valid].mean()):.4f}")

# --- Downsampled polygons for map ---
# Aggregate to 300m (factor 10)
factor = 10
h, w = proba.shape
h_ds = h // factor
w_ds = w // factor
proba_ds = proba[:h_ds*factor, :w_ds*factor].reshape(h_ds, factor, w_ds, factor).mean(axis=(1, 3))
binary_ds = proba_ds > 0.5
ds_transform = (transform[0]*factor, 0, transform[2], 0, transform[4]*factor, transform[5])

print(f"\nDownsampled: {binary_ds.shape} at ~300m")
print(f"High-suitability pixels: {int(binary_ds.sum())}")

# Extract polygons from downsampled binary
shapes = list(features.shapes(np.uint8(binary_ds), mask=binary_ds, transform=ds_transform))
min_px = 2
geojson = {"type": "FeatureCollection", "features": []}
for geom, val in shapes:
    if val == 1:
        poly = shape(geom)
        if poly.area > 0:
            n_pix = int(poly.area / abs(ds_transform[0] * ds_transform[4]))
            if n_pix >= min_px:
                area_ha = round(n_pix * pixel_ha * factor * factor, 1)
                area_km2 = round(area_ha / 100, 1)
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"area_ha": area_ha, "area_km2": area_km2},
                })

ng = len(geojson["features"])
total_ha = sum(f["properties"]["area_ha"] for f in geojson["features"])
print(f"\nPolygons: {ng}, total area: {total_ha:,.0f} ha ({total_ha/100:,.0f} km²)")

# --- Folium map ---
center_lat = (bounds.top + bounds.bottom) / 2
center_lon = (bounds.left + bounds.right) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Satellite",
)

if geojson["features"]:
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            "fillColor": "#00ff00",
            "color": "#006600",
            "weight": 1,
            "fillOpacity": 0.4,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["area_ha", "area_km2"],
            aliases=["Area (ha)", "Area (km²)"],
        ),
    ).add_to(m)

# Centroid
folium.Marker(
    [center_lat, center_lon],
    popup=f"Full AOI<br>Suitable: {total_ha:,.0f} ha ({total_ha/100:,.0f} km²)",
    icon=folium.Icon(color="green", icon="info-sign"),
).add_to(m)

folium.Rectangle(
    [[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    color="blue", weight=2, fill=False, popup="AOI Bounds",
).add_to(m)

folium.LayerControl().add_to(m)

formatter = "function(num) {return L.Util.formatNum(num, 4) + ' &deg;';};"
plugins.MousePosition(
    position="bottomright",
    separator=" | ",
    empty_string="NaN",
    lng_first=False,
    num_digits=20,
    prefix="Координаты:",
    lat_formatter=formatter,
    lng_formatter=formatter,
).add_to(m)

os.makedirs(reports_dir, exist_ok=True)
html_path = os.path.join(reports_dir, "suitability_map_full.html")
m.save(html_path)
print(f"\nMap saved: {html_path}")
