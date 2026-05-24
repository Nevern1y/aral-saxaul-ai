"""
fetch_historic_aral_gee.py — Generate historical Aral Sea coastline from JRC GSW max_extent.

Uses Google Earth Engine to extract maximum water extent polygon (1984–present),
which approximates the pre-crisis Aral Sea coastline (~55-60k km²).

1. Load JRC/GSW1_4/GlobalSurfaceWater max_extent
2. Clip to BBOX (57.5, 43.3, 62.0, 46.7)
3. Download binary mask as GeoTIFF via getDownloadURL
4. Polygonize locally with rasterio
5. Save to outputs/aoi/aral_sea_1960.geojson
6. Auto-run build_aoi_mask -> inference -> export
"""
import sys, json, os, time, subprocess, urllib.request, ssl, shutil, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
AOI_DIR = BASE / 'outputs' / 'aoi'
OUTPUTS = BASE / 'outputs' / 'data'
TMP = BASE / 'outputs' / 'tmp' / 'gee'
OUT = AOI_DIR / 'aral_sea_1960.geojson'

BBOX = (57.5, 43.3, 62.0, 46.7)  # minLon, minLat, maxLon, maxLat

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def log(msg): print(f"  {msg}", flush=True)

t0 = time.time()

# ── 1. Initialize GEE ──────────────────────────────────────────────────
log("Step 1: Initializing Google Earth Engine...")
import ee
try:
    ee.Initialize()
    log("  GEE initialized OK")
except Exception as e:
    log(f"  ERROR: {e}")
    log("  Trying ee.Authenticate()...")
    try: ee.Authenticate()
    except: pass
    try: ee.Initialize()
    except Exception as e2:
        log(f"  GEE auth failed: {e2}")
        log("  Falling back to existing coastline.")
        sys.exit(1)

# ── 2. Load JRC GSW max_extent ─────────────────────────────────────────
log(f"\nStep 2: Loading JRC/GSW1_4/GlobalSurfaceWater...")
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
max_ext = gsw.select("max_extent")
roi = ee.Geometry.Rectangle(list(BBOX), "EPSG:4326", False)

log("  Clipping to BBOX...")
image = max_ext.clip(roi).updateMask(max_ext.eq(1)).unmask(0).rename("aral_mask")

# Get metadata
stats = image.reduceRegion(
    reducer=ee.Reducer.sum().unweighted(),
    geometry=roi,
    scale=30,
    maxPixels=1e10,
    bestEffort=True
)
try:
    px_count = stats.getInfo().get("aral_mask", 0)
    area_km2 = px_count * 900 / 1_000_000
    log(f"  GEE detected ~{px_count:,.0f} water pixels ≈ {area_km2:,.0f} km²")
except: pass

# ── 3. Download as GeoTIFF via getDownloadURL ───────────────────────────
log(f"\nStep 3: Downloading mask from GEE...")
TMP.mkdir(parents=True, exist_ok=True)
tif_path = TMP / "jrc_max_extent_clipped.tif"

try:
    url = image.getDownloadURL({
        "region": roi,
        "scale": 30,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
        "filePerBand": False,
    })
    log(f"  Download URL obtained ({len(url)} chars)")

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    resp = urllib.request.urlopen(req, timeout=300, context=ssl_ctx)
    with open(tif_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    size_mb = tif_path.stat().st_size / 1024 / 1024
    log(f"  Saved: {tif_path.name} ({size_mb:.1f} MB)")
except Exception as e:
    log(f"  Download failed: {e}")
    log("  Trying geemap alternative...")
    try:
        import geemap
        geemap.ee_export_image(
            image, filename=str(tif_path), scale=30, region=roi,
            file_per_band=False, crs="EPSG:4326"
        )
        log(f"  geemap download: {tif_path.name} ({tif_path.stat().st_size/1024/1024:.1f} MB)")
    except Exception as e2:
        log(f"  geemap also failed: {e2}")
        log("  FALLING BACK to existing coastline.")
        sys.exit(1)

# ── 4. Polygonize ───────────────────────────────────────────────────────
log(f"\nStep 4: Polygonizing to GeoJSON...")
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, polygonize, clip_by_rect

with rasterio.open(tif_path) as src:
    data = src.read(1)
    transform = src.transform

log(f"  Raster: {data.shape}, band values: {int(data.min())}..{int(data.max())}")

# Mask: only max_extent=1
valid_mask = data == 1
water_px = int(valid_mask.sum())
log(f"  Water pixels: {water_px:,} ({water_px * 900 / 1_000_000:.0f} km²)")

# Polygonize
results = (
    {"geometry": s, "properties": {"value": v}}
    for s, v in features.shapes(data, mask=valid_mask, transform=transform)
)
polys = []
for r in results:
    g = shape(r["geometry"])
    if g.area > 0.001:  # filter tiny slivers
        polys.append(g)

log(f"  Raw polygons: {len(polys):,}")

# Merge, simplify, filter
if polys:
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        merged = merged.simplify(0.0005, preserve_topology=True)
    log(f"  Merged type: {merged.geom_type}")
    
    # Clip to BBOX
    merged = clip_by_rect(merged, BBOX[0], BBOX[1], BBOX[2], BBOX[3])

    # Sort by area
    if merged.geom_type == "MultiPolygon":
        parts = sorted(list(merged.geoms), key=lambda p: p.area, reverse=True)
    else:
        parts = [merged]
    
    log(f"  Parts: {len(parts)}, largest: {parts[0].area:.2f} deg² = {parts[0].area * 111 * 111:.0f} km²")

    # Build features
    features_out = []
    for p in parts:
        if p.area > 0.01:  # >~100 km²
            features_out.append({
                "type": "Feature",
                "properties": {
                    "area_deg2": round(p.area, 2),
                    "source": "JRC GSW max_extent (1984-present)"
                },
                "geometry": mapping(p)
            })
    
    fc = {"type": "FeatureCollection", "features": features_out}
    AOI_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    
    total_deg2 = sum(f["properties"]["area_deg2"] for f in features_out)
    log(f"\n  Saved: {OUT}")
    log(f"  Features: {len(features_out)}, total area: {total_deg2:.1f} deg² ≈ {total_deg2 * 111 * 111:.0f} km²")
else:
    log("  ERROR: No polygons found!")
    sys.exit(1)

# Cleanup
if tif_path.exists():
    tif_path.unlink()
    log("  Temp file cleaned")

elapsed = time.time() - t0
log(f"\n  GEE fetch complete: {elapsed:.1f}s")

# ── 5. Run pipeline ────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("Step 5: Running full V4 pipeline...")
log("=" * 60)

scripts = [
    ("build_aoi_mask.py", "AOI mask (with vector clip)"),
    ("run_inference_v4.py", "V4 inference"),
    ("phase5_v4_export.py", "Vector export + map"),
]

for script, desc in scripts:
    log(f"\n--- {desc} ---")
    ret = os.system(f'python "{BASE / "scripts" / script}"')
    if ret != 0:
        log(f"  ERROR: exit code {ret}")
        sys.exit(1)
    log(f"  OK")

log(f"\n{'='*60}")
log(f"  ALL DONE. Total elapsed: {time.time()-t0:.1f}s")
log(f"{'='*60}")
