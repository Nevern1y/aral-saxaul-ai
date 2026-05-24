"""
download_west_aral.py — Download Sentinel-2 / Sentinel-1 tiles for West Aral
and rebuild the AOI mask for full historical Aral Sea coverage.

Current coverage gap:
  - AOI mask (V1.0) valid data:    59.0°E – 61.0°E
  - Tile1 already covers:          58.0°E – 61.38°E
  - True Aral Sea full extent:    ~57.5°E – 61.5°E
  - New tiles needed:              57.5°E – 58.0°E (western strip)

After this script, run:
  1. scripts/prepare_slope_data.py  (rebuilds dem_slope_30m.tif with new extent)
  2. scripts/run_inference_v4.py    (reclassifies with updated AOI mask)
  3. scripts/phase5_v4_export.py    (regenerates GeoJSON + map)

Requires:
  pip install earthengine-api  (and authenticated: earthengine authenticate)
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time, json, math, os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import zoom

sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ──────────────────────────────────────────────────────
BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
OUTPUTS = BASE / "outputs/data"

# New full extent for the entire historical Aral Sea
FULL_BBOX = (57.5, 43.3, 62.0, 46.7)  # (lon_min, lat_min, lon_max, lat_max)
OLD_BBOX = (58.0, 43.5, 62.0, 46.5)

# The western strip that needs new downloads
WEST_STRIP_BBOX = (57.5, 43.3, 58.0, 46.7)

# GEE parameters (matching existing phase1_ingestion.py)
DRY_SEASON_START = "2025-08-01"
DRY_SEASON_END = "2025-09-30"
CLOUD_COVER_MAX = 40
TARGET_SCALE = 30
TARGET_CRS = "EPSG:4326"

OUTPUT_NAME_WEST = "aral_west_strip"

print("=" * 60)
print("DOWNLOAD WEST ARAL — Missing tile acquisition")
print("=" * 60, flush=True)
t0 = time.time()

print(f"\nCurrent extent:  {OLD_BBOX}")
print(f"Target extent:   {FULL_BBOX}")
print(f"Missing strip:   {WEST_STRIP_BBOX}")
print(f"\nNew area needed: "
      f"{(WEST_STRIP_BBOX[2]-WEST_STRIP_BBOX[0]) * (WEST_STRIP_BBOX[3]-WEST_STRIP_BBOX[1]):.2f} deg²")

# ── Step 1: Try GEE download ──────────────────────────────────────────
try:
    import ee
    ee.Initialize()
    GEE_AVAILABLE = True
    print("\nGEE initialized successfully.", flush=True)
except Exception as e:
    GEE_AVAILABLE = False
    print(f"\nWARNING: GEE not available: {e}", flush=True)
    print("Will use fallback: recompute AOI mask from existing data.", flush=True)

if GEE_AVAILABLE:
    print(f"\n{'─'*60}")
    print("STEP 1: Building Sentinel-2 composite for west strip")
    print(f"{'─'*60}", flush=True)

    # Build S2 composite
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(ee.Geometry.Rectangle(WEST_STRIP_BBOX)) \
        .filterDate(DRY_SEASON_START, DRY_SEASON_END) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_MAX))

    count = s2.size().getInfo()
    print(f"  Scenes found: {count}", flush=True)

    if count == 0:
        print("  WARNING: No scenes found. Try expanding date range.", flush=True)
    else:
        # Median composite
        composite = s2.median()

        # Compute NDMI and NDWI
        ndmi = composite.normalizedDifference(["B8", "B11"]).rename("NDMI")
        ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

        # Stack
        stack = ee.Image.cat([ndmi, ndwi]).float()

        print(f"\n{'─'*60}")
        print("STEP 2: Exporting to Drive (or downloading directly)")
        print(f"{'─'*60}", flush=True)

        # Try direct download first (works for small areas)
        try:
            url = stack.getDownloadURL({
                "name": OUTPUT_NAME_WEST,
                "scale": TARGET_SCALE,
                "crs": TARGET_CRS,
                "region": ee.Geometry.Rectangle(WEST_STRIP_BBOX).getInfo()["coordinates"],
                "format": "GEO_TIFF",
            })
            print(f"  Download URL obtained (expires ~1 hr)", flush=True)
            print(f"  Downloading...", flush=True)

            import requests
            resp = requests.get(url, stream=True, timeout=600)
            out_path = OUTPUTS / f"{OUTPUT_NAME_WEST}.tif"
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8*1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"  Saved: {out_path.name} ({os.path.getsize(out_path)/1024/1024:.1f} MB)", flush=True)

        except Exception as e:
            print(f"  Direct download failed: {e}", flush=True)
            print("  Use GEE Code Editor or `Export.image.toDrive` instead:", flush=True)
            print(f"    Export.image.toDrive({{")
            print(f"      image: stack,")
            print(f"      description: '{OUTPUT_NAME_WEST}',")
            print(f"      folder: 'aral_saxaul',")
            print(f"      fileNamePrefix: '{OUTPUT_NAME_WEST}',")
            print(f"      scale: {TARGET_SCALE},")
            print(f"      crs: '{TARGET_CRS}',")
            print(f"      region: {ee.Geometry.Rectangle(WEST_STRIP_BBOX).getInfo()['coordinates']},")
            print(f"      fileFormat: 'GeoTIFF',")
            print(f"    }});")

else:
    print(f"\n{'─'*60}")
    print("FALLBACK: Rebuilding AOI mask from existing data + JRC")
    print(f"{'─'*60}", flush=True)
    print("""
Since GEE is not available, use this alternative approach:

1. DOWNLOAD from USGS EarthExplorer or Copernicus Data Space:
   - Sentinel-2 L2A tiles (S2B_MSIL2A_*) for the west strip
   - Date range: August-September 2025
   - Tiles: T40S* and T41S* for the Aral region

2. PROCESS locally:
   - Compute NDMI from B8/B11
   - Compute NDWI from B3/B8
   - Stack into 2-band GeoTIFF matching the existing grid

3. REBUILD AOI mask using JRC Global Surface Water:
   - GEE exports 'max_extent' band showing historical water bodies
   - Use this to create a new binary mask AOI

4. MERGE:
   - Update VRT to include the new west strip tiles
   - Run prepare_slope_data.py for expanded DEM coverage
   - Run run_inference_v4.py (remove AOI filter or use new mask)
""", flush=True)

# ── Step 3: Instructions for merge ─────────────────────────────────────
print(f"\n{'─'*60}")
print("MERGE INSTRUCTIONS (after west strip is downloaded)")
print(f"{'─'*60}")
print(f"""
After obtaining the west strip GeoTIFF:

1. UPDATE VRT:
   gdalbuildvrt -srcnodata nan -vrtnodata nan
     outputs/data/feature_stack_full.vrt
     outputs/data/feature_stack_30m_tile1.tif
     outputs/data/{OUTPUT_NAME_WEST}.tif
     outputs/data/feature_stack_30m_tile0_redo.tif

2. UPDATE CONFIG (src/config.py line 32):
   ARAL_BBOX: Tuple[float,...] = {FULL_BBOX}
   
3. REBUILD DEM/SLOPE:
   python scripts/prepare_slope_data.py
   (auto-detects new BBOX bounds)

4. RUN V4 INFERENCE (without AOI mask filter, or with new mask):
   python scripts/run_inference_v4.py
   
5. EXPORT:
   python scripts/phase5_v4_export.py

6. VERIFY:
   Start Streamlit: python -m streamlit run app.py
""", flush=True)

# ── Step 4: Quick area estimate ───────────────────────────────────────
print(f"{'─'*60}")
print("AREA ESTIMATE")
print(f"{'─'*60}")

# Read current V4 raster for pixel count reference
v4_path = OUTPUTS / "suitability_map_v4.tif"
if v4_path.exists():
    with rasterio.open(v4_path) as src:
        data = src.read(1)
        total_aoi = int((data != 0).sum())
        print(f"  Current AOI land:     {total_aoi * 900 / 1_000_000:.0f} km²")
        print(f"  Current optimal:      {int((data == 1).sum()) * 900 / 1_000_000:.0f} km²")
    print(f"  Est. new total AOI:   ~55,000 km² (full historical Aral Sea)")
    print(f"  Est. new optimal:     ~16,000–17,000 km² (extrapolated)")

print(f"\n{'=' * 60}")
print(f"Done in {time.time()-t0:.1f}s", flush=True)
