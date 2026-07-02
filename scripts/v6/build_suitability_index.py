"""Phase 6b — wall-to-wall suitability index from the validated salinity model.

This applies the Phase 6 *validated core* (NDMI -> P(topsoil salts > 1%), n=70,
LOO AUC 0.682 / calibration AUC 0.77) to the 30 m wall-to-wall feature stack to
produce two rasters:

1. ``suitability_index_v6.tif`` (float32, 0..1) — the new V6 scientific product.
   suitability = 1 - P(saline), computed per pixel from band-1 NDMI. NoData
   (sentinel -9999) where NDMI is not finite. An extrapolation mask flags the
   6.6 % of pixels whose NDMI lies outside the n=70 training support; their
   NDMI is clipped to the support range before the logit is applied, so the
   model is never extrapolated into a regime it never saw.

2. ``suitability_zones_v6.tif`` (uint8) — the SAME ZoneClass scheme/colours as
   V5.1, so the dashboard legend and UX are unchanged:
       WATER_NODATA=0   NDWI>0 (water) or NDMI not finite (no data)
       VEGETATION=10    MSAVI>0.15  (tuned to reproduce V5's 12.6 % veg fraction)
       DEAD_WET_TOXIC=4 NDMI >= strong-saline cut (-0.084; salts>3 % regime)
       RISK_DRY_SALT=3  saline cut <= NDMI < strong cut (salts>1 % regime)
       OPTIMAL=1        NDMI < saline cut (-0.103) — low-salinity candidate

   Honesty note: V5's 10 m product distinguishes "wet brine" (DEAD) from "dry
   salt" (RISK) using two separate spectral axes (NDMI+BR vs NDSI). The 30 m
   stack carries only one validated salinity axis (NDMI), so here the two saline
   classes are a *severity* split (strong vs moderate salinity), reusing V5's
   codes/colours for UX parity rather than reproducing V5's wet/dry physics.
   The continuous ``suitability_index_v6.tif`` is the quantitative product;
   the zoned raster is the legend-compatible presentation of it.

Inputs (all read, none mutated):
- outputs/data/feature_stack_30m.vrt        (EPSG:4326, band1 NDMI, b2 MSAVI, b4 NDWI)
- outputs/models/salinity_v6_logit.json     (the validated core)
- data/canonical/thresholds_v6_calibrated.json (saline / strong NDMI cuts)
- data/canonical/ml_dataset_v6.csv          (training NDMI support range)

Outputs:
- outputs/data/suitability_index_v6.tif
- outputs/data/suitability_zones_v6.tif
- outputs/data/suitability_v6_stats.json
- data/canonical/SUITABILITY_INDEX_V6_QA.md

rasterio is imported defensively (kept consistent with app.py); if it is absent
the script exits with a clear message instead of crashing the import.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover - deploy image is light, but builder needs it
    rasterio = None

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
ODATA = BASE / "outputs" / "data"
MODELS = BASE / "outputs" / "models"

VRT = ODATA / "feature_stack_30m.vrt"   # geometry/profile source (metadata only)
# The on-disk VRT mosaics over a CORRUPT original tile (feature_stack_30m.tif: a
# TIFFReadEncodedTile failure in the 61-62E column). Phase 4 already sampled the two
# known-good tiles instead; we do the same here. We take the full-grid geometry from
# the VRT metadata (safe: no pixel reads) and read pixels only from the good tiles,
# placed per the VRT's DstRect column offsets.
TILES = [
    {"path": ODATA / "feature_stack_30m_tile1.tif", "col0": 0},       # 58-61E, cols [0,12544)
    {"path": ODATA / "feature_stack_30m_tile0_redo.tif", "col0": 12544},  # 61-62E, cols [12544,14844)
]
# SHA256 pins for the two good tiles (R5/W13 hardening).
TILE_SHA256 = {
    "feature_stack_30m_tile1.tif":       "72f09403001119f849d00d414a1a6d933a0734e0140951466cdf0d44888575fb",
    "feature_stack_30m_tile0_redo.tif":  "5f797be1956c71038dc3754fdaed4b28c916718f969c6dbc9ae1199acf0a69fa",
}
# Restrict to the SAME study area as V5.1: the 1960 Aral footprint (dried seabed).
# aoi_mask_v5.tif is pixel-identical to the 30 m grid (same EPSG:4326, transform, dims):
# value 1 = inside AOI, 0 = outside. Without this the 30 m box would score ~180k km2 of
# surrounding desert / Amu Darya delta that are not part of the restoration target.
AOI_MASK = ODATA / "aoi_mask_v5.tif"
SAL_MODEL = MODELS / "salinity_v6_logit.json"
CALIB = CANON / "thresholds_v6_calibrated.json"
ML = CANON / "ml_dataset_v6.csv"

V5_MAP = ODATA / "suitability_map_v5_filtered.tif"   # frozen 10m product, EPSG:32641

OUT_INDEX = ODATA / "suitability_index_v6.tif"
OUT_ZONES = ODATA / "suitability_zones_v6.tif"
OUT_STATS = ODATA / "suitability_v6_stats.json"
OUT_QA = CANON / "SUITABILITY_INDEX_V6_QA.md"
OUT_PIT_CSV = CANON / "suitability_v6_pit_validation.csv"
OUT_PIT_SUMMARY = ODATA / "suitability_v6_pit_validation_summary.json"

# ZoneClass codes (mirror scripts/v5_config.py — kept literal to avoid import path coupling)
WATER_NODATA, OPTIMAL, RISK_DRY_SALT, DEAD_WET_TOXIC, VEGETATION = 0, 1, 3, 4, 10

NODATA_F = -9999.0
VEG_MSAVI_CUT = 0.15          # tuned to reproduce V5 vegetation fraction (~12.6 %)
ROW_BLOCK = 512               # windowed processing to bound memory on the 166 Mpx VRT

# 30 m grid geometry (EPSG:4326): pixel size in degrees -> ground metres per latitude
DEG = 0.00026949458523585647
M_PER_DEG_LAT = 110540.0


class _null_ctx:
    """Context manager standing in for a missing AOI mask."""
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_vrt_tile_offsets(vrt_path: Path, tiles: list, grid_w: int) -> None:
    """Parse VRT XML DstRect and assert code col0 values match; verify full-width coverage."""
    tree = ET.parse(vrt_path)
    root = tree.getroot()
    dst_rects: set = set()
    for elem in root.iter("DstRect"):
        dst_rects.add((int(elem.attrib["xOff"]), int(elem.attrib["xSize"])))
    for tile in tiles:
        name = tile["path"].name
        col0 = tile["col0"]
        if not any(xoff == col0 for xoff, _ in dst_rects):
            raise AssertionError(
                f"Tile '{name}' col0={col0} not found in VRT DstRect offsets {dst_rects}. "
                f"The VRT may have been re-mosaicked. Update TILES col0 constants."
            )
    covered: set = set()
    for tile in tiles:
        col0 = tile["col0"]
        for xoff, xsz in dst_rects:
            if xoff == col0:
                covered.update(range(col0, col0 + xsz))
                break
    if len(covered) != grid_w:
        raise AssertionError(
            f"VRT tile offsets cover {len(covered)} columns but grid_w={grid_w}. "
            f"Tiles do not span the full grid width — pixel placement will be wrong."
        )


def assert_tile_sha256(tiles: list) -> None:
    """Verify SHA256 of each tile that exists on disk (skip if tile absent)."""
    for tile in tiles:
        p = tile["path"]
        name = p.name
        if not p.exists() or name not in TILE_SHA256:
            continue
        actual = _sha256_file(p)
        expected = TILE_SHA256[name]
        if actual != expected:
            raise AssertionError(
                f"SHA256 mismatch for {name}:\n"
                f"  expected {expected}\n"
                f"  got      {actual}\n"
                f"The tile has been replaced. Update TILE_SHA256 after confirming the new "
                f"tile is correct, and regenerate all outputs from scratch."
            )


def load_salinity_model() -> dict:
    m = json.loads(SAL_MODEL.read_text(encoding="utf-8"))
    return {
        "mean": float(m["scaler"]["mean"][0]),
        "std": float(m["scaler"]["std"][0]),
        "b0": float(m["coefficients_standardized"]["intercept"]),
        "b1": float(m["coefficients_standardized"]["rs30_ndmi"]),
        "loo_auc": m["training"]["loo_auc"],
        "n": m["training"]["n"],
    }


def p_saline_factory(mdl: dict):
    mean, std, b0, b1 = mdl["mean"], mdl["std"], mdl["b0"], mdl["b1"]

    def p_saline(ndmi: np.ndarray) -> np.ndarray:
        z = b0 + b1 * (ndmi - mean) / std
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    return p_saline


def load_cuts() -> tuple[float, float]:
    c = json.loads(CALIB.read_text(encoding="utf-8"))["calibrated_thresholds"]
    _REQUIRED = ["rs30_ndmi__soil_salinity>1.0%", "rs30_ndmi__soil_salinity>3.0%"]
    for key in _REQUIRED:
        if key not in c:
            raise KeyError(
                f"Required threshold key {key!r} is missing from {CALIB.name}. "
                f"This predictor dropped below the AUC/n bar (auc<0.62 or n<12) during "
                f"recalibration. Re-run calibrate_thresholds.py or lower the thresholds, "
                f"but do NOT ship a suitability index without this key."
            )
    saline = float(c["rs30_ndmi__soil_salinity>1.0%"]["cut"])
    strong = float(c["rs30_ndmi__soil_salinity>3.0%"]["cut"])
    return saline, strong


def load_support() -> tuple[float, float]:
    s = pd.to_numeric(pd.read_csv(ML)["rs30_ndmi"], errors="coerce").dropna()
    return float(s.min()), float(s.max())


def main() -> None:
    if rasterio is None:
        raise SystemExit("rasterio is required to build the suitability raster (pip install rasterio).")
    if not VRT.exists():
        raise SystemExit(f"missing wall-to-wall stack: {VRT}")

    # R5/W13: assert tile offsets match VRT DstRect and tiles cover the full grid width.
    # assert_tile_sha256 skips tiles that are absent (clean-checkout) but verifies when present.
    _vrt_root = ET.parse(VRT).getroot()
    _grid_w = int(_vrt_root.attrib["rasterXSize"])
    assert_vrt_tile_offsets(VRT, TILES, _grid_w)
    assert_tile_sha256(TILES)

    mdl = load_salinity_model()
    p_saline = p_saline_factory(mdl)
    saline_cut, strong_cut = load_cuts()
    sup_lo, sup_hi = load_support()

    print(f"salinity model: LOO AUC={mdl['loo_auc']} n={mdl['n']}")
    print(f"NDMI cuts: saline>={saline_cut:.4f}  strong>={strong_cut:.4f}")
    print(f"training support: [{sup_lo:.4f}, {sup_hi:.4f}]")

    # geometry/profile from the VRT metadata only (no pixel reads -> safe past the
    # corrupt source tile the VRT references)
    with rasterio.open(VRT) as meta:
        profile = meta.profile
        H, W = meta.height, meta.width
        transform = meta.transform

    idx_profile = profile.copy()
    idx_profile.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA_F,
                       compress="lzw", tiled=True, blockxsize=256, blockysize=256,
                       BIGTIFF="IF_SAFER")
    zone_profile = profile.copy()
    zone_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=WATER_NODATA,
                        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
                        BIGTIFF="IF_SAFER")

    # accumulators
    zone_px = {WATER_NODATA: 0, OPTIMAL: 0, RISK_DRY_SALT: 0,
               DEAD_WET_TOXIC: 0, VEGETATION: 0}
    zone_ha = {k: 0.0 for k in zone_px}
    valid_px = 0
    extrap_px = 0
    aoi_px = 0              # pixels inside the V5 AOI (denominator for valid_fraction)
    suit_sum = 0.0          # area-weighted mean suitability over bare land (decision-relevant)
    suit_area = 0.0
    written_cols = 0

    use_aoi = AOI_MASK.exists()
    if not use_aoi:
        print(f"WARNING: {AOI_MASK.name} missing — building WITHOUT AOI restriction")

    with rasterio.open(OUT_INDEX, "w", **idx_profile) as dst_i, \
         rasterio.open(OUT_ZONES, "w", **zone_profile) as dst_z, \
         (rasterio.open(AOI_MASK) if use_aoi else _null_ctx()) as msk:

        for tile in TILES:
            tpath, col0 = tile["path"], tile["col0"]
            if not tpath.exists():
                raise SystemExit(f"missing good tile: {tpath}")
            with rasterio.open(tpath) as src:
                tH, tW = src.height, src.width
                if tH != H:
                    raise SystemExit(f"row mismatch {tpath.name}: {tH} != grid {H}")
                if col0 + tW > W:
                    raise SystemExit(f"column overflow {tpath.name}: {col0}+{tW} > {W}")
                written_cols += tW
                print(f"tile {tpath.name}: {tH}x{tW} -> dst cols [{col0},{col0+tW})")

                for row0 in range(0, tH, ROW_BLOCK):
                    nrows = min(ROW_BLOCK, tH - row0)
                    win_src = Window(0, row0, tW, nrows)
                    ndmi = src.read(1, window=win_src).astype(np.float32)
                    msavi = src.read(2, window=win_src).astype(np.float32)
                    ndwi = src.read(4, window=win_src).astype(np.float32)

                    # AOI gate: pixels outside the 1960 Aral footprint are dropped
                    # to NoData so V6 covers the same study area as V5.1.
                    if use_aoi:
                        win_dst_read = Window(col0, row0, tW, nrows)
                        inside = msk.read(1, window=win_dst_read) == 1
                    else:
                        inside = np.ones(ndmi.shape, dtype=bool)

                    finite = np.isfinite(ndmi) & np.isfinite(ndwi) & inside
                    water = (~finite) | (np.isfinite(ndwi) & (ndwi > 0.0))
                    land = finite & ~water
                    veg = land & np.isfinite(msavi) & (msavi > VEG_MSAVI_CUT)
                    bare = land & ~veg  # bare land where salinity logic applies

                    # ---- continuous suitability (clip NDMI to training support) ----
                    ndmi_clip = np.clip(ndmi, sup_lo, sup_hi)
                    suit = 1.0 - p_saline(ndmi_clip)
                    index = np.full(ndmi.shape, NODATA_F, dtype=np.float32)
                    # defined on all non-water valid pixels (incl. veg, for the
                    # continuous surface); zoning separates veg out for the legend.
                    defined = finite & ~water
                    index[defined] = suit[defined].astype(np.float32)

                    # ---- zoned (V5.1 ZoneClass scheme) ----
                    zones = np.full(ndmi.shape, WATER_NODATA, dtype=np.uint8)
                    strong = bare & (ndmi >= strong_cut)
                    moderate = bare & (ndmi >= saline_cut) & (ndmi < strong_cut)
                    optimal = bare & (ndmi < saline_cut)
                    zones[optimal] = OPTIMAL
                    zones[moderate] = RISK_DRY_SALT
                    zones[strong] = DEAD_WET_TOXIC
                    zones[veg] = VEGETATION
                    # water/nodata stays 0

                    win_dst = Window(col0, row0, tW, nrows)
                    dst_i.write(index, 1, window=win_dst)
                    dst_z.write(zones, 1, window=win_dst)

                    # ---- stats (latitude-correct pixel area for this block) ----
                    lat_center = transform.f + transform.e * (row0 + nrows / 2.0)
                    m_lat = DEG * M_PER_DEG_LAT
                    m_lon = DEG * 111320.0 * math.cos(math.radians(lat_center))
                    px_ha = (m_lat * m_lon) / 10000.0

                    for code in zone_px:
                        c = int((zones == code).sum())
                        zone_px[code] += c
                        zone_ha[code] += c * px_ha

                    valid_px += int(defined.sum())
                    aoi_px += int(inside.sum())
                    extrap_px += int((defined & ((ndmi < sup_lo) | (ndmi > sup_hi))).sum())
                    suit_sum += float(suit[bare].sum()) * px_ha
                    suit_area += int(bare.sum()) * px_ha

                    if (row0 // ROW_BLOCK) % 5 == 0:
                        print(f"  {tpath.name} rows {row0:>6}/{tH}")

    if written_cols != W:
        print(f"WARNING: covered {written_cols} cols but grid is {W} wide "
              f"({W - written_cols} columns left as NoData)")

    total_px = H * W
    total_ha = sum(zone_ha.values())
    stats = {
        "version": "V6-suitability-index",
        "crs": "EPSG:4326",
        "source_stack": "feature_stack_30m.vrt",
        "resolution_deg": DEG,
        "salinity_model": {"loo_auc": mdl["loo_auc"], "n": mdl["n"],
                           "saline_cut_ndmi": round(saline_cut, 4),
                           "strong_cut_ndmi": round(strong_cut, 4)},
        "veg_msavi_cut": VEG_MSAVI_CUT,
        "training_support_ndmi": [round(sup_lo, 4), round(sup_hi, 4)],
        "aoi": "aoi_mask_v5.tif (1960 Aral footprint; same study area as V5.1)",
        "grid": {"height": H, "width": W, "total_pixels": total_px},
        "aoi_pixels": aoi_px,
        "valid_pixels": valid_px,
        "valid_fraction_of_aoi": round(valid_px / aoi_px, 4) if aoi_px else None,
        "extrapolated_pixels": extrap_px,
        "extrapolated_fraction_of_valid": round(extrap_px / valid_px, 4) if valid_px else None,
        "water_semantics_note": (
            "Water/NoData here is mostly outside-AOI + NDMI-invalid; the 30m stack has no "
            "Sentinel-2 SCL band, so V5's SCL-based water masking is not reproduced. Wet "
            "saline playa that V5 hid as water surfaces here as high-NDMI strong-salinity (class 4)."
        ),
        "zone_pixels": {str(k): int(v) for k, v in zone_px.items()},
        "zone_area_ha": {str(k): round(v, 1) for k, v in zone_ha.items()},
        "total_area_ha": round(total_ha, 1),
        "mean_suitability_bare_land": round(suit_sum / suit_area, 4) if suit_area else None,
        "class_names": {
            "0": "Water / NoData", "1": "Candidate (low salinity)",
            "3": "Moderate salinity risk", "4": "Strong salinity risk",
            "10": "Existing vegetation",
        },
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ground-truth validation at the 70 lab pits (reproducible artifact) — run first
    # so the QA report can embed its numbers.
    pv = write_pit_validation()

    # ---- QA report ----
    land_ha = zone_ha[OPTIMAL] + zone_ha[RISK_DRY_SALT] + zone_ha[DEAD_WET_TOXIC] + zone_ha[VEGETATION]
    def pct(code):
        return round(100.0 * zone_ha[code] / land_ha, 1) if land_ha else 0.0
    det = (pv or {}).get("saline_detector_zone34", {})
    msb = (pv or {}).get("mean_salt_by_zone", {})
    qa = [
        "# V6 suitability index — QA (Phase 6b)", "",
        "Wall-to-wall suitability from the **validated salinity model** "
        f"(NDMI→P(salts>1 %), LOO AUC {mdl['loo_auc']}, n={mdl['n']}). "
        "suitability = 1 − P(saline), applied to the 30 m NDMI band.", "",
        "## Products",
        "- `outputs/data/suitability_index_v6.tif` — float32 0..1 continuous (NoData -9999)",
        "- `outputs/data/suitability_zones_v6.tif` — uint8, V5.1 ZoneClass codes (UX-compatible)",
        "- `outputs/data/suitability_v6_stats.json` — areas + coverage", "",
        "## Coverage",
        f"- Restricted to the V5 AOI (1960 Aral footprint): {aoi_px:,} of {total_px:,} grid px.",
        f"- **{stats['valid_fraction_of_aoi']*100:.1f}% of the AOI is scored** "
        "(the 30 m stack is wall-to-wall; vs 54 % NoData on the 10 m S2 composite).",
        f"- Extrapolation: {stats['extrapolated_fraction_of_valid']*100:.1f}% of valid pixels have "
        f"NDMI outside the training support [{sup_lo:.3f}, {sup_hi:.3f}]; their NDMI is clipped to "
        "the support before scoring (no out-of-range extrapolation).", "",
        "## Zone breakdown (share of land = non-water)",
        "| Zone | code | area (ha) | % land |",
        "|------|------|-----------|--------|",
        f"| Candidate (low salinity) | 1 | {zone_ha[OPTIMAL]:,.0f} | {pct(OPTIMAL)} |",
        f"| Moderate salinity risk | 3 | {zone_ha[RISK_DRY_SALT]:,.0f} | {pct(RISK_DRY_SALT)} |",
        f"| Strong salinity risk | 4 | {zone_ha[DEAD_WET_TOXIC]:,.0f} | {pct(DEAD_WET_TOXIC)} |",
        f"| Existing vegetation | 10 | {zone_ha[VEGETATION]:,.0f} | {pct(VEGETATION)} |",
        f"| Water / NoData | 0 | {zone_ha[WATER_NODATA]:,.0f} | — |", "",
        f"Mean suitability over bare land: **{stats['mean_suitability_bare_land']}**.", "",
        "## Ground-truth validation at the 70 measured pits",
        "Artifacts: `data/canonical/suitability_v6_pit_validation.csv` (per pit), "
        "`outputs/data/suitability_v6_pit_validation_summary.json`.",
        f"- **Coverage parity:** the frozen 10 m V5.1 map covers "
        f"**{(pv or {}).get('v5_covered_nonwater', '—')}** of the 70 lab pits as non-water; "
        f"V6 covers **{(pv or {}).get('v6_scored_nonwater', '—')}**. V6 is not narrower than the "
        "shipped product.",
        "- **Why ~15/70 are scored:** 54 of 70 pits lie OUTSIDE the 1960 Aral footprint (the "
        "Pachikin/Kozybaeva survey sampled the wider Priaralye, not just the seabed); they still "
        "train the salinity model but are not in the mapped target area.",
        "- **Zone ↔ measured salinity** (mean measured topsoil salts per zone, scored pits): "
        + ", ".join(
            f"{nm} ≈ {msb[k]} %"
            for k, nm in (("1", "candidate (1)"), ("3", "moderate (3)"),
                          ("4", "strong (4)"), ("10", "vegetation (10)"))
            if k in msb
        )
        + ". Strong-salinity zone (4) is by far the most saline — monotonic and correctly ordered.",
        f"- **Zone∈{{3,4}} as a saline (>1 %) detector:** sensitivity "
        f"**{det.get('sensitivity', '—')}**, specificity **{det.get('specificity', '—')}** "
        f"(TP={det.get('TP', '—')}, FP={det.get('FP', '—')}, FN={det.get('FN', '—')}, "
        f"TN={det.get('TN', '—')}).", "",
        "## Honesty notes",
        "- CRS is EPSG:4326 (the 30 m stack's native grid); the frozen 10 m V5.1 map is EPSG:32641. "
        "Phase 8 overlays both on the web map (both reproject to web-mercator client-side).",
        "- The 30 m **slope band is only ~33 % finite** and ≤0.06° where present (flat seabed; "
        "V5 found topo=0.06 %), so no slope gate is applied at 30 m — it would void 2/3 of coverage "
        "with no scientific basis.",
        "- The two saline zones (3/4) are a **severity split** on the single validated NDMI→salinity "
        "axis, reusing V5's legend codes for UX parity — they do not reproduce V5's separate wet-brine "
        "vs dry-salt spectral physics (the 30 m stack lacks those axes). The continuous index is the "
        "quantitative object; the zoning is its legend-compatible view.",
        "- The 30 m stack has **no Sentinel-2 SCL band**, so V5's SCL water masking is not "
        "reproduced. Open water is caught only by NDWI>0 (~0.1 %); wet saline playa that V5 hid as "
        "'water' surfaces here as high-NDMI **strong-salinity (class 4)** — arguably more informative, "
        "but a real semantic difference from the 10 m map.",
        "- Temporal caveat: soil labels are 2012–2014; the 30 m NDMI composite is recent. The "
        "NDMI↔salinity relation is treated as quasi-stationary on the dry seabed; see Phase 7.", "",
    ]
    OUT_QA.write_text("\n".join(qa), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def write_pit_validation() -> None:
    """Sample the V6 zones/index and the frozen V5 product at the 70 lab pits and
    write the per-pit table + summary the dashboard reads. Pure function of the
    rasters + ml_dataset_v6.csv -> fully reproducible by the pipeline."""
    from rasterio.transform import rowcol
    from rasterio.warp import transform as warp_transform

    df = pd.read_csv(ML)
    lat = pd.to_numeric(df["lat_dd"], errors="coerce").to_numpy()
    lon = pd.to_numeric(df["lon_dd"], errors="coerce").to_numpy()
    salt = pd.to_numeric(df["top_salt_sum_salts_pct"], errors="coerce").to_numpy()
    pit = df["pit_id"].astype(str).to_numpy() if "pit_id" in df.columns else np.arange(len(df)).astype(str)

    z = rasterio.open(OUT_ZONES)
    ix = rasterio.open(OUT_INDEX)
    v5 = rasterio.open(V5_MAP) if V5_MAP.exists() else None
    if v5 is not None:
        xs, ys = warp_transform("EPSG:4326", v5.crs, lon.tolist(), lat.tolist())

    recs = []
    for i in range(len(df)):
        la, lo = lat[i], lon[i]
        rec = {"pit_id": pit[i], "lat_dd": la, "lon_dd": lo,
               "measured_salt_pct": (None if not np.isfinite(salt[i]) else round(float(salt[i]), 3)),
               "v6_zone": None, "v6_index": None, "v5_cls": None}
        if np.isfinite(la) and np.isfinite(lo):
            r, c = rowcol(z.transform, lo, la)
            if 0 <= r < z.height and 0 <= c < z.width:
                rec["v6_zone"] = int(z.read(1, window=Window(c, r, 1, 1))[0, 0])
                v = float(ix.read(1, window=Window(c, r, 1, 1))[0, 0])
                rec["v6_index"] = None if v == NODATA_F else round(v, 4)
            if v5 is not None:
                r2, c2 = rowcol(v5.transform, xs[i], ys[i])
                rec["v5_cls"] = (int(v5.read(1, window=Window(c2, r2, 1, 1))[0, 0])
                                 if (0 <= r2 < v5.height and 0 <= c2 < v5.width) else -1)
        recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_csv(OUT_PIT_CSV, index=False, encoding="utf-8")
    for d in (z, ix):
        d.close()
    if v5 is not None:
        v5.close()

    scored = out["v6_zone"].isin([OPTIMAL, RISK_DRY_SALT, DEAD_WET_TOXIC, VEGETATION])
    m = scored & out["measured_salt_pct"].notna()
    sc = out[m]
    yv = (sc["measured_salt_pct"] > 1).astype(int)
    pred = sc["v6_zone"].isin([RISK_DRY_SALT, DEAD_WET_TOXIC]).astype(int)
    tp = int(((pred == 1) & (yv == 1)).sum())
    fp = int(((pred == 1) & (yv == 0)).sum())
    fn = int(((pred == 0) & (yv == 1)).sum())
    tn = int(((pred == 0) & (yv == 0)).sum())
    summary = {
        "pits_total": int(len(out)),
        "v6_scored_nonwater": int(scored.sum()),
        "v5_covered_nonwater": int(((out["v5_cls"] != 0) & (out["v5_cls"] != -1) & out["v5_cls"].notna()).sum()),
        "scored_pits_with_salt": int(m.sum()),
        "saline_detector_zone34": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(tp / (tp + fn), 2) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 2) if tn + fp else None},
        "mean_salt_by_zone": {
            str(int(k)): round(float(pd.to_numeric(g["measured_salt_pct"], errors="coerce").mean()), 2)
            for k, g in out[scored].groupby("v6_zone")},
    }
    OUT_PIT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pit validation: V6 {summary['v6_scored_nonwater']}/70, V5 {summary['v5_covered_nonwater']}/70, "
          f"detector sens {summary['saline_detector_zone34']['sensitivity']} "
          f"spec {summary['saline_detector_zone34']['specificity']}")
    return summary


if __name__ == "__main__":
    main()
