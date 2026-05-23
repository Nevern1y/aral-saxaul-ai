import warnings; warnings.filterwarnings("ignore")
import ee, os, time, json
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import shape
import folium
from src.utils import initialize_gee, logger
from src.phase1_ingestion import AOIBuilder
from src.config import config
from src.phase2_local import run_phase2_local
from src.phase3_training import run_phase3
from src.phase4_local import run_phase4

initialize_gee(project="tribal-dispatch-494405-u4")

# Phase 1: AOI
test_bbox = (60.0, 44.5, 60.2, 44.7)
builder = AOIBuilder(bbox=test_bbox, vectorize=False)
aoi = builder.run()
logger.info("PHASE 1 DONE")

# Phase 2
csv_path = run_phase2_local(aoi, scale_m=100, n_samples=5000)
logger.info("PHASE 2 DONE: csv=%s", csv_path)

# Phase 3
df = pd.read_csv(csv_path)
feature_cols = list(config.FEATURE_COLUMNS)
df = df[feature_cols + ["label"]]
model, scaler, metrics = run_phase3(df, n_optuna_trials=10)
logger.info("PHASE 3 DONE: P=%.4f", metrics.get("precision", 0))

# Phase 4
fs_path = str(config.output_dir / "data" / "feature_stack_100m.tif")
output_path = str(config.output_dir / "data" / "suitability_full.tif")
run_phase4(input_path=fs_path, output_path=output_path)
logger.info("PHASE 4 DONE")

# Phase 5
with rasterio.open(output_path) as src:
    proba = src.read(1)
    transform = src.transform
    bounds = src.bounds

binary = proba > 0.5
shape_list = list(features.shapes(np.uint8(binary), mask=binary, transform=transform))
geojson = {"type": "FeatureCollection", "features": []}
pixel_w = abs(transform[0] * 111320 * np.cos(np.deg2rad((bounds.top + bounds.bottom) / 2)))
pixel_h = abs(transform[4] * 111320)
pixel_ha = pixel_w * pixel_h / 10000
for geom, val in shape_list:
    if val == 1:
        n_px = int(shape(geom).area / (abs(transform[0] * transform[4])))
        if n_px >= 5:
            geojson["features"].append(
                {"type": "Feature", "geometry": geom, "properties": {"area_ha": round(n_px * pixel_ha, 1)}}
            )

center = ((bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2)
m = folium.Map(
    location=center, zoom_start=11,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
)
if geojson["features"]:
    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillColor": "#00ff00", "color": "#006600", "weight": 1, "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["area_ha"], aliases=["Area (ha)"]),
    ).add_to(m)
folium.LayerControl().add_to(m)
html_path = str(config.output_dir / "reports" / "pipeline_test.html")
m.save(html_path)
total_ha = sum(f["properties"]["area_ha"] for f in geojson["features"])

print("=" * 60)
print("  FULL PIPELINE TEST PASSED")
print("=" * 60)
print(f"  CSV:      {csv_path}")
print(f"  Model:    {str(config.output_dir / 'models' / 'xgb_classifier.pkl')}")
print(f"  Raster:   {output_path}")
print(f"  Map:      {html_path}")
print(f"  Suitable: {total_ha:.0f} ha  ({binary.sum() * pixel_ha:.0f} ha total)")
print("=" * 60)
