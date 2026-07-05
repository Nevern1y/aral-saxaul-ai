"""
build_v6_vectors.py — Vectorize the V6 salinity-risk raster into field-usable
Shapefile + KML zone files (rasterio is unavailable on this workstation due to
an Application Control policy blocking its bundled GDAL DLL, so this uses
tifffile + scipy + shapely instead of rasterio.features.shapes).

Mirrors the approach already used for V5.1 in scripts/v5_extract_stats.py:
  1. Read outputs/data/suitability_zones_v6.tif (uint8 zone codes, EPSG:4326).
  2. Downsample by mode-pooling to a coarser grid (default factor 10, i.e.
     30 m -> ~300 m) so zone polygons are a manageable size/count.
  3. Label 4-connected zones per class with scipy.ndimage.label.
  4. Filter out zones below a minimum area (default 10 ha).
  5. Trace each zone's boundary with skimage.measure.find_contours (marching
     squares) so we get real polygon outlines (with holes) instead of blocky
     unit-cell squares.
  6. Export to Shapefile and KML with class/zone/area_ha attributes, same
     schema as outputs/data/operational_zones_v5.geojson.

Outputs:
  outputs/data/suitability_zones_v6.shp  (+ .dbf/.shx/.prj/.cpg)
  outputs/data/suitability_zones_v6.kml
  outputs/data/suitability_zones_v6_vector_stats.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import tifffile
from scipy.ndimage import label as cc_label
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "outputs" / "data"

SRC_TIF = DATA / "suitability_zones_v6.tif"
OUT_SHP = DATA / "suitability_zones_v6.shp"
OUT_KML = DATA / "suitability_zones_v6.kml"
OUT_STATS = DATA / "suitability_zones_v6_vector_stats.json"

DOWNSAMPLE_FACTOR = 10          # 30 m -> ~300 m pixels
MIN_ZONE_AREA_HA = 10.0         # drop specks below this size (matches V5.1)
SIMPLIFY_TOLERANCE_DEG = 0.0015  # ~1.5 pixel at 300 m grid
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_AT_EQUATOR = 111_320.0

CLASS_NAMES_RU = {
    1: ("Кандидатная зона (низкий риск соли)", "Candidate (low salinity)"),
    3: ("Умеренный риск засоления", "Moderate salinity risk"),
    4: ("Сильный риск засоления", "Strong salinity risk"),
    10: ("Существующая растительность", "Existing vegetation"),
}
# class 0 (water/nodata) is intentionally never vectorized.


def mode_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mode-pool a categorical uint8 array by `factor` in both dimensions."""
    h, w = arr.shape
    hc, wc = (h // factor) * factor, (w // factor) * factor
    cropped = arr[:hc, :wc]
    blocks = cropped.reshape(hc // factor, factor, wc // factor, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(hc // factor, wc // factor, factor * factor)
    # mode along last axis; classes are few (0,1,3,4,10) so bincount per-pixel is fine
    out = np.zeros((hc // factor, wc // factor), dtype=np.uint8)
    flat = blocks.reshape(-1, factor * factor)
    # vectorized mode via sorting (small factor*factor, e.g. 100 elements)
    sorted_flat = np.sort(flat, axis=1)
    # mode = value with the longest run in the sorted row; approximate via bincount per row
    # (factor*factor small enough that a python loop over unique classes is fine)
    classes = np.unique(arr)
    counts = np.zeros((flat.shape[0], len(classes)), dtype=np.int32)
    for i, c in enumerate(classes):
        counts[:, i] = (flat == c).sum(axis=1)
    best = classes[np.argmax(counts, axis=1)]
    out = best.reshape(hc // factor, wc // factor).astype(np.uint8)
    return out


def contours_to_polygon(mask: np.ndarray) -> Polygon | MultiPolygon | None:
    """Trace a binary mask's boundary (with holes) into a shapely polygon."""
    from skimage import measure

    contours = measure.find_contours(mask.astype(np.float64), level=0.5)
    if not contours:
        return None

    rings = []
    for c in contours:
        # skimage gives (row, col); need at least a triangle
        if len(c) < 4:
            continue
        # (row, col) -> (col, row) so downstream x=col, y=row
        pts = [(pt[1], pt[0]) for pt in c]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        rings.append(poly)

    if not rings:
        return None
    if len(rings) == 1:
        return rings[0]

    # Largest ring is exterior; subtract any rings nested inside it as holes,
    # union any that are disjoint (separate blobs the tracer split up).
    rings.sort(key=lambda p: p.area, reverse=True)
    exterior = rings[0]
    result = exterior
    for r in rings[1:]:
        if exterior.contains(r.representative_point()):
            result = result.difference(r)
        else:
            result = result.union(r)
    return result


def px_area_ha(lat_deg: float, deg_per_px: float) -> float:
    m_lat = deg_per_px * M_PER_DEG_LAT
    m_lon = deg_per_px * M_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat_deg))
    return (m_lat * m_lon) / 10_000.0


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("  V6 VECTORIZE — Shapefile + KML from suitability_zones_v6.tif")
    print("=" * 60)

    if not SRC_TIF.exists():
        raise FileNotFoundError(f"Missing {SRC_TIF}; run scripts/v6/build_suitability_index.py first")

    print(f"\n[1/5] Reading {SRC_TIF.name} ...", end=" ", flush=True)
    arr = tifffile.imread(SRC_TIF)
    with tifffile.TiffFile(SRC_TIF) as tf:
        tags = tf.pages[0].tags
        scale = tags["ModelPixelScaleTag"].value
        tie = tags["ModelTiepointTag"].value
    deg_per_px = float(scale[0])
    origin_lon = float(tie[3])
    origin_lat = float(tie[4])
    print(f"done ({arr.shape[1]}x{arr.shape[0]}, {arr.nbytes / 1e6:.0f} MB)")

    print(f"[2/5] Mode-downsampling x{DOWNSAMPLE_FACTOR} "
          f"(~{deg_per_px * DOWNSAMPLE_FACTOR * M_PER_DEG_LON_AT_EQUATOR:.0f} m/px) ...",
          end=" ", flush=True)
    ds = mode_downsample(arr, DOWNSAMPLE_FACTOR)
    ds_deg_per_px = deg_per_px * DOWNSAMPLE_FACTOR
    print(f"done ({ds.shape[1]}x{ds.shape[0]})")

    print("[3/5] Labeling connected zones per class ...", flush=True)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    features = []
    dropped_small = 0
    kept = 0

    for cls, (name_ru, name_en) in CLASS_NAMES_RU.items():
        mask = (ds == cls)
        if not mask.any():
            continue
        labeled, n = cc_label(mask, structure=structure)
        sizes = np.bincount(labeled.ravel())
        print(f"    class {cls} ({name_en}): {n:,} raw components")

        for zid in range(1, n + 1):
            px_count = int(sizes[zid])
            if px_count == 0:
                continue
            zone_mask = labeled == zid
            rows, cols = np.where(zone_mask)
            r0, r1 = rows.min(), rows.max() + 1
            c0, c1 = cols.min(), cols.max() + 1
            lat_center = origin_lat - ds_deg_per_px * ((r0 + r1) / 2.0)
            area_ha = px_count * px_area_ha(lat_center, ds_deg_per_px)
            if area_ha < MIN_ZONE_AREA_HA:
                dropped_small += 1
                continue

            sub = zone_mask[r0:r1, c0:c1]
            # pad 1px so boundary tracing closes cleanly at tile edges
            sub_p = np.pad(sub, 1, mode="constant", constant_values=False)
            poly = contours_to_polygon(sub_p)
            if poly is None or poly.is_empty:
                continue

            def to_geo(x, y, r0=r0, c0=c0):
                # x=col (with -1 for pad), y=row (with -1 for pad)
                lon = origin_lon + (c0 + x - 1) * ds_deg_per_px
                lat = origin_lat - (r0 + y - 1) * ds_deg_per_px
                return (lon, lat)

            def transform_geom(geom):
                from shapely.ops import transform as shp_transform
                return shp_transform(lambda x, y, z=None: to_geo(x, y), geom)

            geom_wgs84 = transform_geom(poly)
            geom_simple = geom_wgs84.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
            if geom_simple.is_empty or geom_simple.geom_type not in ("Polygon", "MultiPolygon"):
                continue

            features.append({
                "class": int(cls),
                "zone": name_ru,
                "zone_en": name_en,
                "area_ha": round(area_ha, 1),
                "geometry": geom_simple,
            })
            kept += 1

    print(f"[4/5] Kept {kept:,} zones >= {MIN_ZONE_AREA_HA:.0f} ha "
          f"(dropped {dropped_small:,} smaller specks)")

    if not features:
        raise RuntimeError("No V6 zones survived filtering; check MIN_ZONE_AREA_HA / input raster.")

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf = gdf[["class", "zone", "zone_en", "area_ha", "geometry"]]

    print(f"[5/5] Writing outputs ...", flush=True)
    if OUT_SHP.exists():
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = OUT_SHP.with_suffix(ext)
            if p.exists():
                p.unlink()
    gdf.to_file(OUT_SHP, driver="ESRI Shapefile")
    print(f"    {OUT_SHP.name} ({OUT_SHP.stat().st_size / 1e6:.1f} MB)")

    if OUT_KML.exists():
        OUT_KML.unlink()
    gdf_kml = gdf.copy()
    gdf_kml["Name"] = gdf_kml["zone"] + " (" + gdf_kml["area_ha"].astype(str) + " ha)"
    gdf_kml.to_file(OUT_KML, driver="KML")
    print(f"    {OUT_KML.name} ({OUT_KML.stat().st_size / 1e6:.1f} MB)")

    area_by_class = gdf.groupby("class")["area_ha"].agg(["sum", "count"]).to_dict("index")
    stats = {
        "source": SRC_TIF.name,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "approx_resolution_m": round(deg_per_px * DOWNSAMPLE_FACTOR * M_PER_DEG_LON_AT_EQUATOR, 0),
        "min_zone_area_ha": MIN_ZONE_AREA_HA,
        "zones_kept": kept,
        "zones_dropped_small": dropped_small,
        "area_ha_by_class": {
            str(k): {"area_ha": round(v["sum"], 1), "count": int(v["count"])}
            for k, v in area_by_class.items()
        },
        "total_area_ha": round(float(gdf["area_ha"].sum()), 1),
        "class_names": {str(k): v[1] for k, v in CLASS_NAMES_RU.items()},
        "note": (
            "Vectorized with tifffile+scipy+shapely (no rasterio) because rasterio's "
            "bundled GDAL DLL is blocked by this workstation's Application Control policy. "
            "Zones are traced at ~"
            f"{deg_per_px * DOWNSAMPLE_FACTOR * M_PER_DEG_LON_AT_EQUATOR:.0f} m resolution "
            "(mode-pooled from the native 30 m V6 grid), not pixel-exact."
        ),
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    {OUT_STATS.name}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
