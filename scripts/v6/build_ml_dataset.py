# -*- coding: utf-8 -*-
"""Phase 4 — assemble the ML-ready habitat table and recompute correlations.

For every georeferenced profile we combine three feature groups:

1. Soil (from soil_layers_v6.csv) — aggregated to a TOPSOIL window (0-30 cm,
   the rooting zone that matters for saxaul establishment) and a whole-profile
   mean, depth-weighted by layer thickness. Both salinity (mass %) and the
   agro-chemical battery (humus, N, CaCO3, gypsum, exchangeable bases, pH).
2. Remote sensing — Sentinel-2 indices computed at each point with the SAME
   formulas as production (`v5_rules.compute_indices`): NDMI, NDVI, NDWI(NIR),
   NDSI(green,SWIR2), BR(NIR/SWIR2), BI. Plus Copernicus DEM slope and the V5.1
   zone class the production map assigns there. DN are scaled /10000 to
   reflectance exactly as the pipeline does.
3. Label (from saxaul_labels_v6.csv) — saxaul_status, label_role/strength/weight.

The output `ml_dataset_v6.csv` is the single training/analysis table for
Phases 5-7. We also recompute Spearman correlations of every soil + RS feature
against a binary suitability target on the trainable rows, replacing the n=11
correlations in the old ground_truth_v2 analysis.

Honesty guards:
- Rows are never silently dropped; non-trainable rows (uncertain/exclude) are
  kept with label_weight 0 and a flag.
- Temporal mismatch (soil 2012-2014 vs S2 ~2024) is recorded in the manifest.
- recovered_offset coordinates are flagged so Phase 7 can re-run without them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
DATA = BASE / "outputs" / "data"
OUT = CANON / "ml_dataset_v6.csv"

sys.path.insert(0, str(BASE / "scripts"))
from v5_rules import compute_indices, classify_pixel, CLASS_NAMES  # noqa: E402

THRESH_PATH = DATA / "thresholds_v5.json"
TOPSOIL_CM = 30.0

# soil columns to aggregate (layer-level -> profile features)
SALT_COLS = ["salt_sum_salts_pct", "salt_cl_pct", "salt_so4_pct", "salt_na_pct",
             "salt_alk_hco3_pct", "salt_ca_pct", "salt_mg_pct"]
CHEM_COLS = ["humus_pct", "n_total_pct", "c_n_ratio", "caco3_pct", "caso4_pct",
             "exch_ca", "exch_mg", "exch_na", "exch_k", "exch_sum",
             "p2o5_mobile", "k2o_mobile", "ph_water"]
TEX_COLS = ["sand_pct", "silt_pct", "clay_pct", "physical_clay_lt0p01",
            "hygroscopic_water_pct"]
AGG_COLS = SALT_COLS + CHEM_COLS + TEX_COLS

MORPH_FEATURES_PATH = CANON / "morph_features_v6.csv"

# Morphological feature columns derived by morph_features.py (per-pit, depth-aware).
# These join directly on pit_id (no depth aggregation — already pit-level summaries).
# Coverage notes are in data/canonical/morph_features_manifest.json and morph_vocab.json.
MORPH_COLS = [
    "depth_to_moist_cm",    # capillary fringe proxy; NaN = all-dry profile (~93% coverage)
    "depth_to_salt_cm",     # depth to salic horizon; NaN = no salt layer (~63% coverage)
    "rust_mottling_flag",   # waterlogging indicator; 0/1, all 76 pits
    "gley_flag",            # reductive anoxia; 0/1, all 76 pits
    "solum_depth_cm",       # max recorded depth; 100% coverage
    "hcl_effervescence_class",  # 2=surface, 1=weak, 0=none, NaN=missing (~80% coverage)
    "surface_crust_flag",   # physical barrier for seedling emergence; 0/1, all 76 pits
    "marine_shell_flag",    # relict seabed marker; 0/1, all 76 pits (~14% positive rate)
    "horizon_salic_flag",   # salic horizon suffix (зс/сн) present; 0/1, all 76 pits
    "horizon_ploughed_flag",  # anthropogenic disturbance (пах); 0/1, all 76 pits
]

BAND_PATHS = {
    "B3": DATA / "B3_10m.tif", "B4": DATA / "B4_10m.tif", "B8": DATA / "B8_10m.tif",
    "B11": DATA / "B11_20m.tif", "B12": DATA / "B12_20m.tif", "SCL": DATA / "SCL_10m.tif",
}
SLOPE_PATH = DATA / "dem_slope_30m.tif"   # EPSG:4326, degrees
MAP_PATH = DATA / "suitability_map_v5.tif"  # EPSG:32641, uint8 zone class

# Legacy 30 m feature stack (EPSG:4326) — bands: NDMI, MSAVI, SI, NDWI, Slope,
# TWI, VH. Unlike the 10 m production composite it has full 70/70 coverage at
# the ground-truth points, so it is used for analysis power while the 10 m
# production indices (rs_*) give continuity with the live V5.1 cascade.
LEGACY_TILES = [DATA / "feature_stack_30m_tile0_redo.tif",
                DATA / "feature_stack_30m_tile1.tif"]
LEGACY_BANDS = ["ndmi", "msavi", "si", "ndwi", "slope", "twi", "vh"]


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# ---------------------------------------------------------------------------
# Soil aggregation: depth-weighted topsoil + whole-profile means
# ---------------------------------------------------------------------------

def aggregate_soil() -> pd.DataFrame:
    L = pd.read_csv(CANON / "soil_layers_v6.csv", dtype=str, keep_default_na=False)
    L["top"] = to_num(L["depth_top_cm"])
    L["bot"] = to_num(L["depth_bottom_cm"])
    for c in AGG_COLS:
        if c in L.columns:
            L[c] = to_num(L[c])

    def wmean(group: pd.DataFrame, cols, top_only: bool) -> dict:
        g = group.copy()
        if top_only:
            g = g[g["top"] < TOPSOIL_CM]
        out = {}
        if g.empty:
            return {c: np.nan for c in cols}
        thick = (g["bot"].clip(upper=TOPSOIL_CM if top_only else g["bot"]) - g["top"]).clip(lower=0)
        thick = thick.fillna(0)
        for c in cols:
            if c not in g.columns:
                out[c] = np.nan
                continue
            v = g[c]
            m = v.notna() & (thick > 0)
            if m.sum() == 0:
                out[c] = float(v.mean()) if v.notna().any() else np.nan
            else:
                out[c] = float(np.average(v[m], weights=thick[m]))
        return out

    rows = []
    for pid, grp in L.groupby("pit_id"):
        rec = {"pit_id": pid}
        top = wmean(grp, AGG_COLS, top_only=True)
        prof = wmean(grp, AGG_COLS, top_only=False)
        for c in AGG_COLS:
            rec[f"top_{c}"] = top[c]
            rec[f"prof_{c}"] = prof[c]
        rec["n_layers"] = int(len(grp))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Remote-sensing sampling at each point (production formulas)
# ---------------------------------------------------------------------------

def sample_rs(points: pd.DataFrame) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform as rio_transform

    thresholds = json.loads(THRESH_PATH.read_text())
    lon = points["lon_dd"].astype(float).to_numpy()
    lat = points["lat_dd"].astype(float).to_numpy()

    def sample_one(path, xs, ys, src_is_4326):
        """Sample raster at lon/lat arrays; reproject lon/lat into raster CRS."""
        with rasterio.open(path) as src:
            if src.crs and src.crs.to_epsg() != 4326:
                xx, yy = rio_transform("EPSG:4326", src.crs, list(xs), list(ys))
            else:
                xx, yy = list(xs), list(ys)
            out = []
            band = src.read(1)
            for x, y in zip(xx, yy):
                try:
                    r, c = src.index(x, y)
                    if 0 <= r < band.shape[0] and 0 <= c < band.shape[1]:
                        out.append(float(band[r, c]))
                    else:
                        out.append(np.nan)
                except Exception:
                    out.append(np.nan)
        return np.array(out, dtype=float)

    bands = {k: sample_one(p, lon, lat, False) for k, p in BAND_PATHS.items()}
    slope = sample_one(SLOPE_PATH, lon, lat, True)
    zone_map = sample_one(MAP_PATH, lon, lat, False)

    refl = {k: bands[k] / 10000.0 for k in ("B3", "B4", "B8", "B11", "B12")}
    idx = compute_indices(refl["B3"], refl["B4"], refl["B8"], refl["B11"], refl["B12"])

    df = pd.DataFrame({"pit_id": points["pit_id"].to_numpy()})
    df["rs_ndmi"] = idx["ndmi"]
    df["rs_ndvi"] = idx["ndvi"]
    df["rs_ndwi_nir"] = idx["ndwi_nir"]
    df["rs_ndsi_green_swir2"] = idx["ndsi_green_swir2"]
    df["rs_br_nir_swir2"] = idx["br_nir_swir2"]
    df["rs_bi"] = idx["bi"]
    df["rs_slope_deg"] = slope
    df["rs_scl"] = bands["SCL"]

    # V5.1 zone class as produced AND as recomputed (consistency check)
    df["v5_zone_map"] = zone_map
    recomputed = []
    for i in range(len(df)):
        scalar = {k: float(np.asarray(idx[k]).reshape(-1)[i]) for k in idx}
        if not np.isfinite(slope[i]) or any(not np.isfinite(v) for v in scalar.values()):
            recomputed.append(np.nan)
        else:
            recomputed.append(classify_pixel(bands["SCL"][i], slope[i], scalar, thresholds))
    df["v5_zone_recomputed"] = recomputed
    df["v5_zone_name"] = df["v5_zone_map"].map(
        lambda z: CLASS_NAMES.get(int(z), "?") if np.isfinite(z) else "nodata")

    # production-10m RS coverage flag (SCL NoData => point outside S2 footprint)
    df["rs_prod_valid"] = df["rs_ndmi"].notna() & (df["rs_scl"] > 0)

    # ---- legacy 30 m stack: full-coverage indices for analysis -----------
    n = len(df)
    legacy = {f"rs30_{b}": np.full(n, np.nan) for b in LEGACY_BANDS}
    for tp in LEGACY_TILES:
        if not tp.exists():
            continue
        with rasterio.open(tp) as src:
            # Validate band order matches LEGACY_BANDS contract (item 7: positional-contract guard).
            # Tile descriptions are uppercase; LEGACY_BANDS are lowercase — compare case-insensitively.
            actual_descs = [d.lower() if d else "" for d in (src.descriptions or [])]
            if actual_descs:
                for bi, expected_band in enumerate(LEGACY_BANDS):
                    if bi >= len(actual_descs):
                        break
                    if actual_descs[bi] != expected_band:
                        raise ValueError(
                            f"Band order mismatch in {tp.name}: "
                            f"band {bi + 1} description is '{actual_descs[bi]}' "
                            f"but LEGACY_BANDS[{bi}] expects '{expected_band}'. "
                            f"Update LEGACY_BANDS or regenerate the feature stack."
                        )
            arrs = [src.read(i + 1) for i in range(min(src.count, len(LEGACY_BANDS)))]
            if src.crs and src.crs.to_epsg() != 4326:
                xx, yy = rio_transform("EPSG:4326", src.crs, list(lon), list(lat))
            else:
                xx, yy = list(lon), list(lat)
            for i, (x, y) in enumerate(zip(xx, yy)):
                try:
                    r, c = src.index(x, y)
                except Exception:
                    continue
                for bi, b in enumerate(LEGACY_BANDS):
                    if bi >= len(arrs):
                        continue
                    if 0 <= r < arrs[bi].shape[0] and 0 <= c < arrs[bi].shape[1]:
                        v = arrs[bi][r, c]
                        # first valid tile wins
                        if np.isfinite(v) and np.isnan(legacy[f"rs30_{b}"][i]):
                            legacy[f"rs30_{b}"][i] = float(v)
    for k, v in legacy.items():
        df[k] = v
    df["rs30_valid"] = df["rs30_ndmi"].notna()
    return df


# ---------------------------------------------------------------------------
# Main: join soil + RS + labels; recompute correlations
# ---------------------------------------------------------------------------

def load_morph_features() -> pd.DataFrame:
    """Load per-pit morphological features derived by morph_features.py.

    Returns a DataFrame with pit_id + MORPH_COLS. If morph_features_v6.csv is absent,
    returns an empty DataFrame so that the rest of the pipeline degrades gracefully
    (morph columns will be NaN for all pits, and the raster-dependent path can still run).
    Coverage notes per feature are in data/canonical/morph_features_manifest.json.
    """
    if not MORPH_FEATURES_PATH.exists():
        import warnings
        warnings.warn(
            f"morph_features_v6.csv not found at {MORPH_FEATURES_PATH}. "
            "Run scripts/v6/morph_features.py first. Morph columns will be NaN.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["pit_id"] + MORPH_COLS)
    mf = pd.read_csv(MORPH_FEATURES_PATH, dtype=str, keep_default_na=False)
    for col in MORPH_COLS:
        if col in mf.columns:
            mf[col] = to_num(mf[col])
    keep_cols = ["pit_id"] + [c for c in MORPH_COLS if c in mf.columns]
    return mf[keep_cols].copy()


def main() -> None:
    labels = pd.read_csv(CANON / "saxaul_labels_v6.csv", dtype=str, keep_default_na=False)
    labels["label_weight"] = to_num(labels["label_weight"]).fillna(0.0)
    geo = labels[labels["has_coordinates"].isin(["True", "true"])].copy()

    soil = aggregate_soil()
    morph = load_morph_features()
    rs = sample_rs(geo[["pit_id", "lat_dd", "lon_dd"]])

    df = (geo
          .merge(soil, on="pit_id", how="left")
          .merge(morph, on="pit_id", how="left")
          .merge(rs, on="pit_id", how="left"))

    # binary suitability target for correlation / training (trainable rows only)
    df["y_suitable"] = np.where(df["label_role"] == "positive", 1,
                        np.where(df["label_role"] == "negative", 0, np.nan))
    df.to_csv(OUT, index=False, encoding="utf-8")

    # ---- correlations on trainable rows -----------------------------------
    # ml_correlations_v6.csv: Spearman rho vs y_suitable (saxaul suitability target).
    # Note: binary morph flags are included (nunique()>2 gate removed for flag columns).
    # REPRODUCIBILITY NOTE: this table uses the saxaul y_suitable target. A separate
    # morph_salinity_correlations_v6.csv (below) uses the salinity target and covers all
    # morph features (including binary flags) to make ablation feature-selection numbers
    # reproducible and NOT buried as inline code comments.
    from scipy.stats import spearmanr
    train = df[(df["label_weight"] > 0) & df["y_suitable"].notna()].copy()
    feat_cols = [c for c in df.columns
                 if (c.startswith(("top_", "prof_", "rs_", "rs30_")) or c in MORPH_COLS)
                 and c not in ("rs_scl", "rs_prod_valid", "rs30_valid")]
    corrs = []
    for c in feat_cols:
        v = to_num(train[c])
        m = v.notna() & train["y_suitable"].notna()
        # Allow binary flags (nunique <= 2) through; only require n >= 8 and non-constant
        if m.sum() >= 8 and v[m].nunique() >= 2:
            rho, p = spearmanr(v[m], train.loc[m, "y_suitable"])
            corrs.append({"feature": c, "spearman_rho": round(float(rho), 3),
                          "p_value": round(float(p), 4), "n": int(m.sum())})
    corr_df = pd.DataFrame(corrs).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(CANON / "ml_correlations_v6.csv", index=False, encoding="utf-8")

    # ---- morph features vs SALINITY target (producer for ablation feature-selection) ---
    # CORE REPRODUCIBILITY: the rho values used in spatial_validation.py to select
    # ablation predictors MUST be produced here, not asserted as magic comments.
    # Target: binary saline indicator (top_salt_sum_salts_pct > 1%) on ALL 70 rows
    # (not just trainable rows), matching the ablation target in spatial_validation.py.
    # Binary flags are included (no nunique filter beyond requiring variance).
    df["y_saline"] = (to_num(df["top_salt_sum_salts_pct"]) > 1.0).astype(float)
    morph_salt_corrs = []
    for c in MORPH_COLS:
        if c not in df.columns:
            continue
        v = to_num(df[c])
        m = v.notna() & df["y_saline"].notna()
        if m.sum() >= 8 and v[m].nunique() >= 2:
            rho, p = spearmanr(v[m], df.loc[m, "y_saline"])
            morph_salt_corrs.append({
                "feature": c,
                "spearman_rho_vs_salinity": round(float(rho), 3),
                "p_value": round(float(p), 4),
                "n": int(m.sum()),
                "coverage_pct": round(100.0 * m.sum() / len(df), 1),
            })
    morph_salt_df = pd.DataFrame(morph_salt_corrs).sort_values(
        "spearman_rho_vs_salinity", key=lambda s: s.abs(), ascending=False)
    morph_salt_df.to_csv(CANON / "morph_salinity_correlations_v6.csv",
                         index=False, encoding="utf-8")

    # RS consistency: does sampled point class match recomputed?
    both = df[df["v5_zone_map"].notna() & df["v5_zone_recomputed"].notna()]
    consistent = int((both["v5_zone_map"] == both["v5_zone_recomputed"]).sum())

    # Coverage stats for morph features (from morph_features_manifest.json if available)
    morph_coverage_note = "see data/canonical/morph_features_manifest.json for per-feature coverage"
    morph_low_cov = {c: f"{int(df[c].notna().sum())}/{len(df)}"
                     for c in MORPH_COLS if c in df.columns
                     and df[c].notna().sum() / len(df) < 0.60}

    manifest = {
        "version": "V6-ml-dataset",
        "n_rows": int(len(df)),
        "n_trainable": int(len(train)),
        "n_positive": int((df["y_suitable"] == 1).sum()),
        "n_negative": int((df["y_suitable"] == 0).sum()),
        "n_features_soil": len([c for c in feat_cols if c.startswith(("top_", "prof_"))]),
        "n_features_rs": len([c for c in feat_cols if c.startswith(("rs_", "rs30_"))]),
        "n_features_morph": len([c for c in feat_cols if c in MORPH_COLS]),
        "morph_features": MORPH_COLS,
        "morph_coverage_note": morph_coverage_note,
        "morph_low_coverage_features": morph_low_cov,
        "morph_missingness_policy": (
            "Morph features with <60% non-null coverage are NOT imputed — NaN is preserved. "
            "The ablation in spatial_validation.py uses only rows where all features are finite."
        ),
        "rs_coverage": {
            "production_10m_valid": f"{int(df['rs_prod_valid'].sum())}/{len(df)} points",
            "legacy_30m_valid": f"{int(df['rs30_valid'].sum())}/{len(df)} points",
            "note": ("The production 2025 spring S2 composite is ~54% NoData over the "
                     "Syrdarya delta, so only ~14/70 ground-truth points fall on valid "
                     "10m pixels. The legacy 30m stack covers all points and is used for "
                     "analysis. RECOMMENDED FUTURE STEP: re-fetch a gap-free annual "
                     "cloud-masked S2 median to lift 10m coverage."),
        },
        "rs_zone_consistency": f"{consistent}/{len(both)} sampled points match production map",
        "top_correlations": corr_df.head(15).to_dict(orient="records"),
        "temporal_mismatch_note": ("soil/vegetation observed 2012-2014 vs Sentinel-2 "
                                   "composite 2025; correlations are indicative, not "
                                   "contemporaneous"),
        "outputs": {
            "ml_dataset": str(OUT),
            "correlations_vs_saxaul": str(CANON / "ml_correlations_v6.csv"),
            "correlations_vs_salinity_morph": str(CANON / "morph_salinity_correlations_v6.csv"),
        },
    }
    (CANON / "ml_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
