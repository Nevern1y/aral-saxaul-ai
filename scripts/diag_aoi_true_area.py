"""
diag_aoi_true_area.py — one-off diagnostic (not part of the pipeline).
Computes the TRUE geodesic area of aoi_mask_v5.tif, correcting the
"900 m2/px nominal" assumption (wrong for an EPSG:4326 degree-grid away
from the equator) by summing per-row pixel counts x correct per-row
geodesic pixel area (accounts for cos(latitude) longitude compression).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, math
from pathlib import Path
import numpy as np
import rasterio

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
MASK_PATH = BASE / "outputs/data/aoi_mask_v5.tif"

M_PER_DEG_LAT = 111_320.0  # standard approximation, adequate at this precision

with rasterio.open(MASK_PATH) as src:
    data = src.read(1)
    t = src.transform
    dx_deg = t.a
    dy_deg = -t.e
    height, width = src.height, src.width
    top = src.bounds.top

print(f"Grid: {width} x {height},  pixel size {dx_deg:.8f} x {dy_deg:.8f} deg")

# Per-row pixel count of AOI==1
row_counts = (data == 1).sum(axis=1).astype(np.float64)  # shape (height,)
row_indices = np.arange(height)
row_center_lat = top - (row_indices + 0.5) * dy_deg  # deg

# True geodesic pixel area per row (equirectangular approx, adequate for this precision)
m_per_deg_lon_row = M_PER_DEG_LAT * np.cos(np.radians(row_center_lat))
pixel_area_m2_row = (dx_deg * m_per_deg_lon_row) * (dy_deg * M_PER_DEG_LAT)

total_area_m2 = float((row_counts * pixel_area_m2_row).sum())
total_area_km2 = total_area_m2 / 1e6

# Old (wrong) nominal estimate for comparison
n_mask_px = int((data == 1).sum())
nominal_km2 = n_mask_px * 900 / 1e6

print(f"\nTotal AOI=1 pixels: {n_mask_px:,}")
print(f"OLD nominal estimate (900 m2/px flat assumption): {nominal_km2:,.1f} km2  <-- WRONG, ignores cos(lat)")
print(f"CORRECTED geodesic estimate (per-row cos(lat) pixel area): {total_area_km2:,.1f} km2")
print(f"Correction factor: {nominal_km2/total_area_km2:.4f}x (expected ~= 1/cos(45deg) = {1/math.cos(math.radians(45)):.4f})")

print(f"\nFor reference:")
print(f"  Vector polygon (Albers equal-area, net of holes, measured earlier): 55,061.8 km2")
print(f"  Documented true 1960 Aral Sea extent (per download_aoi.py comment): ~68,000 km2")
print(f"  Corrected raster mask area: {total_area_km2:,.1f} km2")
diff_pct = 100*(total_area_km2 - 55061.8)/55061.8
print(f"  Relative diff raster vs vector (corrected): {diff_pct:+.1f}%")
