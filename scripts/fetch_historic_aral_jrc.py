"""
fetch_historic_aral_jrc.py — Download JRC GSW max_extent tiles from EU FTP.

Tile grid: 14 columns × 36 rows, each 40000×40000 px.
Col: floor((lon+180) / (360/14)), Row: floor((lat+90) / (180/36))

Output: outputs/aoi/aral_sea_1960.geojson with JRC max_extent coastline.
Auto-runs full V4 pipeline after download.
"""
import sys, json, os, time, urllib.request, ssl, shutil, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
AOI_DIR = BASE / 'outputs' / 'aoi'
OUTPUTS = BASE / 'outputs' / 'data'
TMP = BASE / 'outputs' / 'tmp' / 'jrc'
OUT = AOI_DIR / 'aral_sea_1960.geojson'
BBOX = (57.5, 43.3, 62.0, 46.7)

N_COLS = 14  # 360°/14 = 25.71° per column
N_ROWS = 36  # 180°/36 = 5° per row
TILE_PX = 40000
TILE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/Aggregated/LATEST/extent/tiles/extent-{x:010d}-{y:010d}.tif"

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def log(msg): print(f"  {msg}", flush=True)

t0 = time.time()
log("Downloading JRC GSW max_extent from EU FTP...")

# Determine tiles covering BBOX
min_lon, min_lat, max_lon, max_lat = BBOX
deg_per_col = 360 / N_COLS
deg_per_row = 180 / N_ROWS

col_min = int((min_lon + 180) // deg_per_col)
col_max = int((max_lon + 180) // deg_per_col)
row_min = int((min_lat + 90) // deg_per_row)
row_max = int((max_lat + 90) // deg_per_row)

log(f"  Tile grid: {N_COLS} cols × {N_ROWS} rows ({deg_per_col:.1f}°×{deg_per_row:.1f}° per tile)")
log(f"  Bounding tiles: col {col_min}-{col_max}, row {row_min}-{row_max}")

TMP.mkdir(parents=True, exist_ok=True)
downloaded = []

for col in range(col_min, col_max + 1):
    for row in range(row_min, row_max + 1):
        if col >= N_COLS or row >= N_ROWS:
            continue
        x_px = col * TILE_PX
        y_px = row * TILE_PX
        tif = TMP / f"extent-{x_px:010d}-{y_px:010d}.tif"
        url = TILE_URL.format(x=x_px, y=y_px)
        
        lon0 = col * deg_per_col - 180
        lat0 = row * deg_per_row - 90
        log(f"  Download col={col} row={row} ({lon0:.0f}E to {lon0+deg_per_col:.0f}E, {lat0:.0f}N to {lat0+deg_per_row:.0f}N)...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=120, context=ssl_ctx)
            with open(tif, 'wb') as f:
                shutil.copyfileobj(resp, f)
            size_mb = tif.stat().st_size / 1024 / 1024
            log(f"    OK: {size_mb:.1f} MB")
            downloaded.append(tif)
        except Exception as e:
            log(f"    FAILED: {e}")

if not downloaded:
    log("ERROR: No tiles downloaded!")
    sys.exit(1)

# Mosaic: read each tile, extract water polygons
log(f"\nProcessing {len(downloaded)} downloaded tiles...")
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, clip_by_rect

all_polys = []
for tif in downloaded:
    try:
        with rasterio.open(tif) as src:
            data = src.read(1)
            # JRC max_extent: 1 = water ever detected (1984-2021)
            water = data == 1
            water_px = int(water.sum())
            if water_px < 1000:  # skip near-empty tiles
                log(f"  {tif.name}: {water_px:,} water px — skipping (nearly empty)")
                continue
            log(f"  {tif.name}: {water_px:,} water px")
            
            for s, v in rio_shapes(data, mask=water, transform=src.transform):
                g = shape(s)
                if g is not None and not g.is_empty and g.area > 0.0001:
                    all_polys.append(g)
    except Exception as e:
        log(f"  Error: {tif.name}: {e}")

log(f"  Raw polygons: {len(all_polys):,}")

if not all_polys:
    log("ERROR: No water polygons found!")
    sys.exit(1)

log(f"  Merging...")
merged = unary_union(all_polys)
if merged.is_empty:
    log("ERROR: Empty merge!")
    sys.exit(1)

if merged.geom_type == "MultiPolygon":
    merged = merged.simplify(0.0005, preserve_topology=True)
log(f"  Type: {merged.geom_type}")

# Clip to BBOX
merged = clip_by_rect(merged, BBOX[0], BBOX[1], BBOX[2], BBOX[3])

parts = sorted(list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged],
               key=lambda p: p.area, reverse=True)
total_deg2 = sum(p.area for p in parts)
total_km2 = total_deg2 * 111 * 111
log(f"  Parts: {len(parts)}, total area: {total_deg2:.1f} deg² ≈ {total_km2:.0f} km²")
if parts:
    log(f"  Largest: {parts[0].area:.2f} deg² ≈ {parts[0].area*111*111:.0f} km²")

# Save GeoJSON
features_out = [{
    "type": "Feature",
    "properties": {
        "area_deg2": round(p.area, 4),
        "source": "JRC GSW1_4 max_extent (1984-2021)"
    },
    "geometry": mapping(p)
} for p in parts if p.area > 0.01]

fc = {"type": "FeatureCollection", "features": features_out}
AOI_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(fc, f, ensure_ascii=False, indent=2)

log(f"\n  Saved: {OUT}")
log(f"  Features: {len(features_out)}, total: {sum(f['properties']['area_deg2'] for f in features_out)*111*111:.0f} km²")

# Cleanup
for tif in downloaded:
    tif.unlink()
log(f"  Temp files cleaned ({time.time()-t0:.1f}s)")

# Run pipeline
log("\n" + "="*60)
log("Step 2: Running V4 pipeline...")
log("="*60)
for script, desc in [
    ("build_aoi_mask.py", "AOI mask"),
    ("run_inference_v4.py", "Inference"),
    ("phase5_v4_export.py", "Export"),
]:
    log(f"\n--- {desc} ---")
    ret = os.system(f'python "{BASE / "scripts" / script}"')
    if ret != 0:
        log(f"  ERROR: exit code {ret}")
        sys.exit(1)

log(f"\n{'='*60}")
log(f"  ALL DONE. Total: {time.time()-t0:.1f}s")
log(f"{'='*60}")
