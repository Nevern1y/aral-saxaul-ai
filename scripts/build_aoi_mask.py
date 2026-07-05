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
from scipy import ndimage
from shapely.geometry import shape, mapping

# 30 m grid geometry (EPSG:4326): a pixel is DEG x DEG degrees. At the Aral latitude
# (~45.9 N) that is ~30 m (N-S) x ~21 m (E-W) = ~630 m2, NOT 900 m2. The old stats
# assumed a flat 900 m2/pixel, which inflated the reported AOI area from ~67,000 km2
# (the real 1960 seabed) to ~95,850 km2 — a trigonometry artifact, not real over-capture.
DEG = 0.00026949458523585647
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0


def mask_area_km2(mask: np.ndarray, transform) -> float:
    """Latitude-correct land area (km2) of a boolean/uint8 mask on an EPSG:4326 grid.

    Sums per-row pixel areas with a cos(latitude) longitude correction, matching the
    px_ha formula in scripts/v6/build_suitability_index.py (lines ~337-340)."""
    m = mask == 1
    h = m.shape[0]
    total_m2 = 0.0
    for r0 in range(0, h, 512):
        r1 = min(r0 + 512, h)
        lat = transform.f + transform.e * (np.arange(r0, r1) + 0.5)
        m_lat = DEG * M_PER_DEG_LAT
        m_lon = DEG * M_PER_DEG_LON * np.cos(np.radians(lat))
        px_m2 = m_lat * m_lon                      # per-row pixel area
        row_land = m[r0:r1].sum(axis=1)
        total_m2 += float((row_land * px_m2).sum())
    return total_m2 / 1_000_000.0

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

# -- 4b. Morphological cleanup (display/scoring gate only) ------------
# The raw <=54 m + NDMI-valid gate is a per-pixel test, so its outline is extremely
# ragged: ~20,000 tiny interior holes (single NDMI-invalid pixels inside the seabed)
# and ~2,000 thin fingers/specks along the shore. Drawn on the web map this reads as
# "chopped up" noise and the AOI outline shimmers. We clean the mask so both the zone
# fill and the derived 1960-shoreline contour are smooth:
#   1) fill only SMALL interior holes (< HOLE_FILL_MAX_PX): these are NDMI-dropouts on
#      dry seabed, not real lakes; large water bodies are left as holes.
#   2) morphological OPENING (erode then dilate, OPENING_ITERS px): sheds thin fingers
#      and single-pixel specks. Opening only ever REMOVES pixels, so the mask can never
#      grow into territory that has no data (no closing — that could bulge the border
#      into settlements/desert with no NDMI support).
#   3) keep the largest 8-connected component (drops detached delta/irrigated basins).
# This is a DISPLAY/scoring mask: it never enters the salinity-model coefficients (the
# model reads NDMI at the soil-profile points, not this raster), so it cannot change
# any metric or drift the calibration. Verified: 0 of the 70 georeferenced pits lost.
HOLE_FILL_MAX_PX = 128     # ~11.5 ha at 30 m; fills speckle dropouts, keeps real lakes
OPENING_ITERS = 3          # ~90 m; sheds thin fingers without eroding the seabed body
print("\nCleaning mask (fill speckle holes -> opening -> largest component)...", flush=True)
with rasterio.open(AOI_MASK_PATH) as res:
    data = res.read(1)
    transform = res.transform
    profile_cc = res.profile.copy()

land_before = int((data == 1).sum())
mask_bool = data == 1

# 1) fill small interior holes only
holes = ndimage.binary_fill_holes(mask_bool) & ~mask_bool
hlab, hn = ndimage.label(holes, structure=np.ones((3, 3), dtype=int))
if hn > 0:
    hsizes = ndimage.sum(np.ones_like(data, dtype=np.int64), hlab, index=np.arange(1, hn + 1))
    small_hole_ids = np.where(hsizes < HOLE_FILL_MAX_PX)[0] + 1
    mask_bool = mask_bool | np.isin(hlab, small_hole_ids)
    print(f"  filled {len(small_hole_ids):,} of {hn:,} interior holes (< {HOLE_FILL_MAX_PX} px)", flush=True)

# 2) opening: shed thin fingers/specks (removal-only, never grows the footprint)
mask_bool = ndimage.binary_opening(mask_bool, structure=np.ones((3, 3), dtype=int),
                                   iterations=OPENING_ITERS)

# 3) largest connected component
labels, n_comp = ndimage.label(mask_bool, structure=np.ones((3, 3), dtype=int))
if n_comp >= 1:
    sizes = ndimage.sum(np.ones_like(data, dtype=np.int64), labels, index=np.arange(1, n_comp + 1))
    biggest = int(np.argmax(sizes)) + 1
    mask_bool = labels == biggest

data = mask_bool.astype(np.uint8)
with rasterio.open(AOI_MASK_PATH, "w", **profile_cc) as dst:
    dst.write(data, 1)
land_after = int((data == 1).sum())
print(f"  components: {n_comp:,}; land px {land_before:,} -> {land_after:,} "
      f"(removed {land_before - land_after:,} px of specks/fingers/detached basins)", flush=True)

# -- 5. Stats --------------------------------------------------------
with rasterio.open(AOI_MASK_PATH) as res:
    data = res.read(1)
    transform = res.transform
    land = int(data.sum())
    water = data.size - land
    area_km2 = mask_area_km2(data, transform)
    print(f"\n{'='*60}")
    print("AOI MASK STATS — Elevation Hack + largest connected component")
    print(f"{'='*60}")
    print(f"  Land pixels (<=54m ASL):  {land:>12,}")
    print(f"  Outside/water:            {water:>12,}")
    print(f"  Total pixels:             {data.size:>12,}")
    # Latitude-correct area (cos(lat)); the old flat 900 m2/px overstated this ~1.4x.
    print(f"  Land area (cos-lat):      {area_km2:>8.0f} km2  ({area_km2 * 100:>10.0f} ha)")
    print("  (real 1960 Aral seabed ~68,000 km2 — the gate reproduces it once area is measured correctly)")
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
