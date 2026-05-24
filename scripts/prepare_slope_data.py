"""
prepare_slope_data.py — Download SRTM tiles, compute slope & elevation
at 30m, resample to match feature stack grid.
Output: outputs/data/dem_slope_30m.tif, outputs/data/dem_elevation_30m.tif
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time, struct, math, os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import sobel

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
OUT_PATH = BASE / "outputs/data/dem_slope_30m.tif"

# AOI bounds (matching V4 feature stack)
LON_MIN, LON_MAX = 57.5, 62.0
LAT_MIN, LAT_MAX = 43.3, 46.7

# SRTM3: 3 arc-second = 3/3600 degree, grid 1201x1201 per 1° tile
SRTM_RES = 3.0 / 3600
SRTM_SIZE = 1201

print("=" * 60)
print("DEM / SLOPE DATA PREPARATION")
print("=" * 60, flush=True)
t0 = time.time()

# ── 1. Determine tiles needed ─────────────────────────────────────────
lat_start = int(math.floor(LAT_MIN))
lat_end = int(math.floor(LAT_MAX))
lon_start = int(math.floor(LON_MIN))
lon_end = int(math.floor(LON_MAX))

tiles = []
for lat in range(lat_start, lat_end + 1):
    for lon in range(lon_start, lon_end + 1):
        tiles.append((lat, lon))

print(f"\nSRTM tiles needed: {len(tiles)}")
for lat, lon in sorted(tiles):
    print(f"  N{lat:02d}E{lon:03d}.hgt", flush=True)

# ── 2. Download missing tiles ─────────────────────────────────────────
import srtm
geo = srtm.get_data(srtm1=False, srtm3=True)

cache_dir = Path(os.path.expanduser("~")) / ".cache" / "srtm"
cache_dir.mkdir(parents=True, exist_ok=True)

for lat, lon in tiles:
    fname = f"N{lat:02d}E{lon:03d}.hgt"
    fpath = cache_dir / fname
    if not fpath.exists():
        print(f"\n  Downloading: {fname}...", flush=True)
        try:
            elev = geo.get_elevation(lat + 0.5, lon + 0.5)  # warm up cache
            # Now it should be cached
            if fpath.exists():
                print(f"  Downloaded OK ({fpath.stat().st_size / 1024:.0f} KB)", flush=True)
            else:
                print(f"  WARNING: {fname} not in cache after request", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    else:
        print(f"  Cached: {fname} ({fpath.stat().st_size / 1024:.0f} KB)", flush=True)

# ── 3. Read HGT tiles and mosaic ──────────────────────────────────────
print(f"\nReading tiles and building mosaic...", flush=True)

def read_hgt(lat, lon):
    """Read SRTM .hgt file as 2D numpy array (3601x3601)."""
    fname = f"N{lat:02d}E{lon:03d}.hgt"
    fpath = cache_dir / fname
    if not fpath.exists():
        return None

    with open(fpath, "rb") as f:
        raw = f.read()

    arr = np.frombuffer(raw, dtype=">i2").astype(np.float32)
    arr = arr.reshape((SRTM_SIZE, SRTM_SIZE))
    arr[arr < -1000] = np.nan  # invalid data
    return arr

# Calculate output grid (SRTM3: 1200 cells per degree)
cells_per_deg = 1200
nrows = int(round((LAT_MAX - LAT_MIN) * cells_per_deg)) + 1
ncols = int(round((LON_MAX - LON_MIN) * cells_per_deg)) + 1

print(f"  Output size: {ncols}x{nrows} ({ncols * nrows:,} px)", flush=True)

mosaic = np.full((nrows, ncols), np.nan, dtype=np.float32)

for lat, lon in tiles:
    print(f"  Reading N{lat:02d}E{lon:03d}.hgt...", flush=True)
    tile = read_hgt(lat, lon)
    if tile is None:
        print(f"    SKIP: tile not found", flush=True)
        continue

    # Tile covers [lat, lat+1] x [lon, lon+1] in degrees
    # Mosaic covers [LAT_MIN, LAT_MAX] x [LON_MIN, LON_MAX]
    # Calculate overlap in pixel space
    tile_top = lat + 1.0
    tile_bottom = lat
    tile_left = lon
    tile_right = lon + 1.0

    # Overlap with AOI
    ov_top = min(tile_top, LAT_MAX)
    ov_bottom = max(tile_bottom, LAT_MIN)
    ov_left = max(tile_left, LON_MIN)
    ov_right = min(tile_right, LON_MAX)

    if ov_top <= ov_bottom or ov_left >= ov_right:
        continue  # no overlap

    # Pixel coordinates within tile (top-left origin, rows increase southward)
    tile_row_start = int(round((tile_top - ov_top) * cells_per_deg))
    tile_col_start = int(round((ov_left - tile_left) * cells_per_deg))
    tile_row_end = int(round((tile_top - ov_bottom) * cells_per_deg))
    tile_col_end = int(round((ov_right - tile_left) * cells_per_deg))

    # Pixel coordinates within mosaic
    mos_row_start = int(round((ov_bottom - LAT_MIN) * cells_per_deg))
    mos_col_start = int(round((ov_left - LON_MIN) * cells_per_deg))
    mos_row_end = int(round((ov_top - LAT_MIN) * cells_per_deg))
    mos_col_end = int(round((ov_right - LON_MIN) * cells_per_deg))

    h = tile_row_end - tile_row_start
    w = tile_col_end - tile_col_start
    if h <= 0 or w <= 0:
        continue

    mosaic[mos_row_start:mos_row_end, mos_col_start:mos_col_end] = \
        tile[tile_row_start:tile_row_end, tile_col_start:tile_col_end]

t1 = time.time()
print(f"  Mosaic built: {t1-t0:.1f}s", flush=True)

# ── 4. Compute slope ───────────────────────────────────────────────────
print(f"\nComputing slope (Sobel 3x3)...", flush=True)

# Mask of valid data
valid_mask = ~np.isnan(mosaic)
if valid_mask.sum() == 0:
    print("ERROR: No valid elevation data in AOI!", flush=True)
    sys.exit(1)

# Fill NaN for gradient computation (use nearest interpolation)
from scipy.ndimage import distance_transform_edt

# Fill NaN with nearest valid value
if np.any(np.isnan(mosaic)):
    mask = np.isnan(mosaic)
    # Simple fill: use minimum valid value
    mosaic_filled = mosaic.copy()
    mosaic_filled[mask] = np.nanmin(mosaic)
    # Apply slight Gaussian blur to reduce noise
    from scipy.ndimage import gaussian_filter
    mosaic_filled = gaussian_filter(mosaic_filled, sigma=0.5)
else:
    mosaic_filled = mosaic

# Cell size in meters at mean latitude
lat_center = (LAT_MIN + LAT_MAX) / 2
lat_rad = math.radians(lat_center)
deg_to_m = 111320.0
cellsize_x = deg_to_m * math.cos(lat_rad) * SRTM_RES
cellsize_y = deg_to_m * SRTM_RES

print(f"  Cell size: {cellsize_x:.1f}m x {cellsize_y:.1f}m", flush=True)

dzdx = sobel(mosaic_filled, axis=1) / (8 * cellsize_x)
dzdy = sobel(mosaic_filled, axis=0) / (8 * cellsize_y)

slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
slope[~valid_mask] = np.nan

t2 = time.time()
print(f"  Slope computed: {t2-t1:.1f}s", flush=True)
print(f"  Slope range: {np.nanmin(slope):.2f}° – {np.nanmax(slope):.2f}°", flush=True)
print(f"  Slope > 5°: {np.sum(slope > 5):,} px ({np.sum(slope > 5) / valid_mask.sum() * 100:.1f}%)", flush=True)

# ── 5. Resample to match VRT grid ─────────────────────────────────────
print(f"\nResampling to feature stack grid (30m)...", flush=True)

# Read one window from VRT to get exact grid
vrt_path = BASE / "outputs/data/feature_stack_30m.vrt"
with rasterio.open(vrt_path) as ref_src:
    ref_transform = ref_src.transform
    ref_width = ref_src.width
    ref_height = ref_src.height
    ref_crs = ref_src.crs

print(f"  Reference grid: {ref_width}x{ref_height}, CRS={ref_crs}", flush=True)

# Resample slope to reference grid using scipy
from scipy.ndimage import zoom

scale_y = ref_height / nrows
scale_x = ref_width / ncols
print(f"  Scale factors: x={scale_x:.4f}, y={scale_y:.4f}", flush=True)

# Fill NaN for resampling
slope_filled = slope.copy()
nan_mask = np.isnan(slope_filled)
if nan_mask.any():
    slope_filled[nan_mask] = 0.0  # flat terrain where no data

slope_resampled = zoom(slope_filled, (scale_y, scale_x), order=1)
# Re-apply valid mask resampled
valid_resampled = zoom(valid_mask.astype(np.float32), (scale_y, scale_x), order=0) > 0.5
slope_resampled[~valid_resampled] = np.nan

# Ensure correct shape (handle rounding)
slope_resampled = slope_resampled[:ref_height, :ref_width]

t3 = time.time()
print(f"  Resampled: {slope_resampled.shape}, {t3-t2:.1f}s", flush=True)

# ── 6. Save as GeoTIFF ────────────────────────────────────────────────
print(f"\nSaving: {OUT_PATH.name}", flush=True)

profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "width": ref_width,
    "height": ref_height,
    "crs": ref_crs,
    "transform": ref_transform,
    "compress": "lzw",
    "nodata": np.nan,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(slope_resampled.astype(np.float32), 1)

t4 = time.time()
size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"  Saved: {size_mb:.1f} MB in {t4-t3:.1f}s", flush=True)

# ── 6a. Save raw elevation (before slope) resampled to 30m grid ──────
ELEV_OUT_PATH = BASE / "outputs/data/dem_elevation_30m.tif"
print(f"\nSaving raw elevation: {ELEV_OUT_PATH.name}", flush=True)

elev_for_export = mosaic.copy()
elev_nan_mask = np.isnan(elev_for_export)
if elev_nan_mask.any():
    elev_for_export[elev_nan_mask] = np.nanmin(elev_for_export)

elev_resampled = zoom(elev_for_export, (scale_y, scale_x), order=1)
elev_resampled[~valid_resampled] = np.nan
elev_resampled = elev_resampled[:ref_height, :ref_width]

with rasterio.open(ELEV_OUT_PATH, "w", **profile) as dst:
    dst.write(elev_resampled.astype(np.float32), 1)

t4b = time.time()
size_mb_elev = os.path.getsize(ELEV_OUT_PATH) / 1024 / 1024
print(f"  Saved elevation: {size_mb_elev:.1f} MB in {t4b-t4:.1f}s", flush=True)

# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Total time:        {t4b-t0:.1f}s")
print(f"  Slope output:      {OUT_PATH}")
print(f"  Slope size:        {size_mb:.1f} MB")
print(f"  Elevation output:  {ELEV_OUT_PATH}")
print(f"  Elevation size:    {size_mb_elev:.1f} MB")
print(f"  Slope range:       {np.nanmin(slope_resampled):.2f}° – {np.nanmax(slope_resampled):.2f}°")
pct_steep = np.sum(slope_resampled > 5) / np.sum(~np.isnan(slope_resampled)) * 100
print(f"  Slope > 5°:        {np.sum(slope_resampled > 5):,} px ({pct_steep:.1f}%)")
print(f"\nDone. Run V4 inference again to enable topographic filter.", flush=True)
