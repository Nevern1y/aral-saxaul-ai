import rasterio
import numpy as np
import folium
import folium.plugins as plugins
import os
from rasterio import features
from shapely.geometry import shape, box
from src.config import config

out_dir = str(config.output_dir / "data")
reports_dir = str(config.output_dir / "reports")

with rasterio.open(os.path.join(out_dir, "suitability_test.tif")) as src:
    proba = src.read(1)
    transform = src.transform
    bounds = src.bounds

binary = proba > 0.5

# --- Polygon extraction ---
# The raster is in EPSG:4326. features.shapes gives polygon outlines in degrees.
# Compute approximate pixel area in m^2 at the mean latitude for filtering.
pixel_w = abs(transform[0] * 111320 * np.cos(np.deg2rad((bounds.top + bounds.bottom) / 2)))
pixel_h = abs(transform[4] * 111320)
pixel_ha = pixel_w * pixel_h / 10000  # hectares per pixel

print(f"Pixel dimensions: {pixel_w:.1f} x {pixel_h:.1f}m = {pixel_ha:.2f} ha")

shapes = list(features.shapes(np.uint8(binary), mask=binary, transform=transform))

min_px = 5
geojson = {"type": "FeatureCollection", "features": []}
for geom, val in shapes:
    if val == 1:
        poly = shape(geom)
        if poly.area > 0:
            area_deg2 = poly.area
            n_pixels = int(area_deg2 / (abs(transform[0] * transform[4])))
            if n_pixels >= min_px:
                area_ha = round(n_pixels * pixel_ha, 1)
                geojson["features"].append(
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"area_ha": area_ha},
                    }
                )

print(f"Found {len(geojson['features'])} suitable polygons")
total_ha = sum(f["properties"]["area_ha"] for f in geojson["features"])
print(f"Total suitable area: {total_ha:.0f} ha")
print(f"Binary pixel count: {binary.sum()}, pixel_ha: {pixel_ha:.2f}, product: {binary.sum() * pixel_ha:.0f} ha")

# --- Folium map ---
center_lat = (bounds.top + bounds.bottom) / 2
center_lon = (bounds.left + bounds.right) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
)

if geojson["features"]:
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            "fillColor": "#00ff00",
            "color": "#006600",
            "weight": 1,
            "fillOpacity": 0.5,
        },
        tooltip=folium.GeoJsonTooltip(fields=["area_ha"], aliases=["Area (ha)"]),
    ).add_to(m)

# Add centroid marker
folium.Marker(
    [center_lat, center_lon],
    popup=f"AOI Center<br>Suitable: {total_ha:.0f} ha ({binary.sum() * pixel_ha:.0f} ha total)",
    icon=folium.Icon(color="green", icon="info-sign"),
).add_to(m)

# Add bounds
folium.Rectangle(
    [[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    color="blue", weight=2, fill=False, popup="AOI Bounds",
).add_to(m)

folium.LayerControl().add_to(m)

os.makedirs(reports_dir, exist_ok=True)
html_path = os.path.join(reports_dir, "suitability_map_test.html")
m.save(html_path)
print(f"Map saved: {html_path}")
