"""
build_aoi_mask.py — Build AOI mask from SRTM elevation (Elevation Hack)
Historical Aral Sea level in 1960: ~53.4m above sea level.
Mask = 1 where elevation <= 54.0m AND elevation > -50.0m AND NDMI valid.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time, json, os, numpy as np
from pathlib import Path
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
OUTPUTS = BASE / "outputs/data"
AOI_DIR = BASE / "outputs/aoi"
AOI_MASK_PATH = OUTPUTS / "aoi_mask_v5.tif"
ELEV_PATH = OUTPUTS / "dem_elevation_30m.tif"
AOI_VECTOR_PATH = AOI_DIR / "aral_sea_1960.geojson"

INPUT_CANDIDATES = [
    OUTPUTS / "feature_stack_30m.vrt",
    OUTPUTS / "feature_stack_30m_tile1.tif",
    OUTPUTS / "feature_stack_30m_tile0_redo.tif",
]

BAND_NDMI = 1

print("=" * 60)
print("BUILD AOI MASK — Elevation Hack (SRTM3)")
print("Historical Aral Sea level 1960: 53.4m -> threshold <= 54.0m")
print("=" * 60, flush=True)
t0 = time.time()

# -- 1. Load elevation ------------------------------------------------
if not ELEV_PATH.exists():
    print(f"ERROR: Elevation file not found: {ELEV_PATH}", flush=True)
    print("  Run: python scripts/prepare_slope_data.py", flush=True)
    sys.exit(1)

elev_src = rasterio.open(ELEV_PATH)
print(f"\nElevation: {ELEV_PATH.name}", flush=True)
print(f"  Size: {elev_src.width}x{elev_src.height}", flush=True)

# -- 2. Open feature stack for valid data mask -----------------------
src = None
for p in INPUT_CANDIDATES:
    if p.exists():
        try:
            src = rasterio.open(p)
            print(f"Feature stack: {p.name}", flush=True)
            break
        except Exception:
            continue

if src is None:
    print("ERROR: No feature stack found for valid data mask!", flush=True)
    sys.exit(1)

print(f"  Size: {src.width}x{src.height}", flush=True)

if src.width != elev_src.width or src.height != elev_src.height:
    print(f"  WARNING: Dimension mismatch! Elev={elev_src.width}x{elev_src.height}, Stack={src.width}x{src.height}", flush=True)

# -- 3. Profile ------------------------------------------------------
profile = src.profile.copy()
profile.update(driver="GTiff", dtype="uint8", count=1, compress="lzw", nodata=0)

# -- 4. Build elevation mask block-wise ------------------------------
print(f"\nBuilding elevation mask (<= 54.0 m ASL, > -50.0 m)...", flush=True)

windows = list(src.block_windows(1))

with rasterio.open(AOI_MASK_PATH, "w", **profile) as dst:
    for idx, (_, window) in enumerate(windows):
        if (idx + 1) % 500 == 0 or idx == 0:
            pct = (idx + 1) / len(windows) * 100
            print(f"  [{idx+1:5d}/{len(windows)}] {pct:.0f}%  ({time.time()-t0:.0f}s)", flush=True)

        try:
            elevation = elev_src.read(1, window=window)
            ndmi = src.read(BAND_NDMI, window=window)
        except Exception:
            dst.write(np.zeros((window.height, window.width), dtype=np.uint8), 1, window=window)
            continue

        elevation_valid = ~np.isnan(elevation)
        elevation_mask = elevation_valid & (elevation <= 54.0) & (elevation > -50.0)

        data_valid = ~(np.isnan(ndmi) | np.isinf(ndmi))

        mask = (elevation_mask & data_valid).astype(np.uint8)

        dst.write(mask, 1, window=window)

elev_src.close()
src.close()

elapsed = time.time() - t0
print(f"  Done: {elapsed:.1f}s", flush=True)

# -- 5. Stats --------------------------------------------------------
with rasterio.open(AOI_MASK_PATH) as res:
    data = res.read(1)
    land = int(data.sum())
    water = data.size - land
    print(f"\n{'='*60}")
    print("AOI MASK STATS — Elevation Hack")
    print(f"{'='*60}")
    print(f"  Land pixels (<=54m ASL):  {land:>12,}")
    print(f"  Outside/water:            {water:>12,}")
    print(f"  Total pixels:             {data.size:>12,}")
    print(f"  Land area:                {land * 900 / 1_000_000:>8.0f} km2  ({land * 900 / 10_000:>8.0f} ha)")
    print(f"  Saved:                    {AOI_MASK_PATH}")
    print(f"  Size:                     {AOI_MASK_PATH.stat().st_size / 1024 / 1024:.0f} MB")

# -- 6. Vectorize for coastline --------------------------------------
print(f"\n{'='*60}")
print("COASTLINE EXTRACTION")
print(f"{'='*60}")

try:
    with rasterio.open(AOI_MASK_PATH) as res:
        data = res.read(1)
        transform = res.transform

    coast_results = (
        {"geometry": s}
        for s, v in features.shapes(data, mask=data, transform=transform)
    )
    coast_polys = []
    for r in coast_results:
        poly = shape(r["geometry"])
        if poly.geom_type == "Polygon" and poly.area > 0.01:
            coast_polys.append(poly.simplify(0.001, preserve_topology=True))
        elif poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                if p.area > 0.01:
                    coast_polys.append(p.simplify(0.001, preserve_topology=True))

    coast_polys.sort(key=lambda p: p.area, reverse=True)
    print(f"  Coastline parts: {len(coast_polys):,}", flush=True)
    if coast_polys:
        print(f"  Main body: {coast_polys[0].area:.1f} deg2", flush=True)

    AOI_DIR.mkdir(parents=True, exist_ok=True)
    coast_features = [{
        "type": "Feature",
        "properties": {"area_deg2": round(p.area, 2)},
        "geometry": mapping(p),
    } for p in coast_polys]
    coast_geojson = {"type": "FeatureCollection", "features": coast_features}
    with open(AOI_VECTOR_PATH, "w", encoding="utf-8") as f:
        json.dump(coast_geojson, f, ensure_ascii=False)
    print(f"  Saved: {AOI_VECTOR_PATH} ({os.path.getsize(AOI_VECTOR_PATH)/1024:.0f} KB)", flush=True)

except Exception as e:
    print(f"  WARNING: Coastline extraction failed: {e}", flush=True)

print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
print("Ready for V4 inference.", flush=True)
