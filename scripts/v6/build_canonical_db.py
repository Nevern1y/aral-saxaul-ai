# -*- coding: utf-8 -*-
"""Phase 2 — build the canonical V6 soil-profile database.

Inputs
------
- data/raw_external/morph_*.csv          : morphology with DMS coordinates
- data/interim/docx_*_long.csv           : appendix-B lab tables (Phase 1)

Outputs (data/canonical/)
-------------------------
- profiles_v6.csv        : one row per profile  (id, year, lat/lon DD, elevation,
                           landform/vegetation text, in_aoi, coordinate provenance)
- soil_layers_v6.csv     : one row per (profile, depth) sample with chemistry,
                           water-extract salinity (mass %), granulometry
- canonical_manifest.json: row counts, join coverage, QA flags

Design choices
--------------
- Coordinates: DMS -> decimal degrees, taken from morphology tables. A profile
  without a morphology row is kept as lab-only (no coordinates).
- Water extract: the report gives two rows per sample (mass % then meq/100g).
  We take the mass-% row as primary to stay consistent with the existing
  ground_truth_v2.csv (which is in %), and keep meq columns as *_meq for audit.
- Values are parsed with a single tolerant float parser (comma decimal, "нет"
  / "сл" / "н/д" -> NaN). Nothing is imputed here.
- AOI / 1960-coastline membership is flagged, never used to silently drop rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "data" / "raw_external"
INTERIM = BASE / "data" / "interim"
CANON = BASE / "data" / "canonical"

ARAL_BBOX = (57.5, 43.3, 62.0, 46.7)  # lon_min, lat_min, lon_max, lat_max (v5_config)

MORPH_FILES = ["morph_2012", "morph_2013", "morph_2012_degradation", "morph_2014_beloe"]

NULL_TOKENS = {"", "-", "–", "нет", "сл", "сл.", "следы", "н/д", "n/a", "na", "nan", "не опр", "не опр."}


def parse_float(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace("\xa0", " ")
    if text.lower() in NULL_TOKENS:
        return np.nan
    # water-extract cells can be "0,01 / 0,44" (mass% / meq) — take first number here
    text = text.split("/")[0].strip()
    text = text.replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else np.nan


def parse_second(value) -> float:
    """Second number after a slash (meq side of '%/meq' cells)."""
    if value is None:
        return np.nan
    parts = str(value).split("/")
    if len(parts) < 2:
        return np.nan
    return parse_float(parts[1])


def dms_to_dd(value) -> float:
    if not isinstance(value, str):
        return np.nan
    s = value.replace("о", " ").replace("°", " ").replace('"', " ").replace("'", " ").replace(",", ".")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        return np.nan
    d, m = float(nums[0]), float(nums[1])
    sec = float(nums[2]) if len(nums) > 2 else 0.0
    return d + m / 60 + sec / 3600


def norm_id(value) -> str:
    text = str(value).strip().replace(" ", "")
    text = text.replace("A", "А")  # latin A -> cyrillic А
    return text


def is_profile_id(value) -> bool:
    return bool(re.search(r"\d[АA]?/\d", str(value)))


def parse_depth(value) -> tuple[float, float]:
    text = "" if value is None else str(value).replace(",", ".")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan, np.nan
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return float(nums[0]), float(nums[1])


# ---------------------------------------------------------------------------
# Profiles (coordinates + context) from morphology tables
# ---------------------------------------------------------------------------

def build_profiles() -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for fname in MORPH_FILES:
        df = pd.read_csv(RAW / f"{fname}.csv", dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        if len(df.columns) < 4:
            raise ValueError(
                f"{fname}.csv has only {len(df.columns)} columns; expected at least 4 "
                f"(id, lat, lon, elevation at positions 0–3). "
                f"Do not reorder the morph CSV columns — it silently corrupts coordinates."
            )
        # Item 7: positional-contract guard. Columns 1 and 2 must carry Lat/Lon keywords.
        _latc_hint, _lonc_hint = df.columns[1].lower(), df.columns[2].lower()
        if "lat" not in _latc_hint:
            raise ValueError(
                f"{fname}.csv: columns[1]='{df.columns[1]}' does not contain 'lat'. "
                f"Expected Latitude at position 1. Reorder will silently corrupt coordinates."
            )
        if "lon" not in _lonc_hint:
            raise ValueError(
                f"{fname}.csv: columns[2]='{df.columns[2]}' does not contain 'lon'. "
                f"Expected Longitude at position 2. Reorder will silently corrupt coordinates."
            )
        idc, latc, lonc, hc = df.columns[0], df.columns[1], df.columns[2], df.columns[3]
        vegc = next((c for c in df.columns if "Растительност" in c or "рельеф" in c.lower()), df.columns[-2])
        for _, r in df.iterrows():
            pid = norm_id(r[idc])
            if not is_profile_id(pid):
                continue
            if pid in rows:
                continue
            lat, lon = dms_to_dd(r[latc]), dms_to_dd(r[lonc])
            rows[pid] = {
                "pit_id": pid,
                "lat_dd": lat,
                "lon_dd": lon,
                "elevation_m": parse_float(r[hc]),
                "landform_vegetation": str(r[vegc]).strip(),
                "coord_source_file": fname,
            }
    prof = pd.DataFrame(list(rows.values()))
    prof["coord_provenance"] = np.where(
        prof["lat_dd"].notna() & prof["lon_dd"].notna(), "surveyed", "missing"
    )
    prof = recover_offset_coords(prof)
    prof["src_year"] = prof["pit_id"].str.extract(r"/(\d{2})$")[0].map(
        lambda x: 2000 + int(x) if pd.notna(x) else np.nan
    )
    lon, lat = prof["lon_dd"], prof["lat_dd"]
    prof["in_aoi"] = (
        (lon >= ARAL_BBOX[0]) & (lon <= ARAL_BBOX[2]) & (lat >= ARAL_BBOX[1]) & (lat <= ARAL_BBOX[3])
    )
    prof["has_coordinates"] = prof["lat_dd"].notna() & prof["lon_dd"].notna()
    return prof.sort_values("pit_id").reset_index(drop=True)


# Coordinates recovered from a documented relative offset in the field report.
# Only points with (a) a KNOWN direction and (b) a coordinate-bearing
# predecessor are recovered. The offset (<=50 m) is far below the 10 m pixel +
# 2012-2014 vs ~2024 temporal uncertainty, so this is defensible. Recovered
# points are flagged coord_provenance="recovered_offset" so Phase 4/7 can run a
# sensitivity check with them excluded. 24/14 is the gold negative (documented
# saxaul-plantation failure) whose own coordinates are "н/д" in the source.
OFFSET_RECOVERY = {
    "24/14": {"ref": "23/14", "east_m": 50.0, "north_m": 0.0,
              "evidence": "Разрез 24/14: «В 50 м восточнее предыдущего разреза» (предыдущий = 23/14)"},
}


def recover_offset_coords(prof: pd.DataFrame) -> pd.DataFrame:
    """Fill coordinates for profiles documented as a metric offset from a peer."""
    idx = {r["pit_id"]: i for i, r in prof.iterrows()}
    for pid, spec in OFFSET_RECOVERY.items():
        if pid not in idx or spec["ref"] not in idx:
            continue
        tgt, ref = prof.loc[idx[pid]], prof.loc[idx[spec["ref"]]]
        if pd.notna(tgt["lat_dd"]) or pd.isna(ref["lat_dd"]):
            continue  # already has coords, or reference has none
        ref_lat = float(ref["lat_dd"])
        dlat = spec["north_m"] / 110_540.0
        dlon = spec["east_m"] / (111_320.0 * np.cos(np.radians(ref_lat)))
        prof.loc[idx[pid], "lat_dd"] = ref_lat + dlat
        prof.loc[idx[pid], "lon_dd"] = float(ref["lon_dd"]) + dlon
        prof.loc[idx[pid], "coord_provenance"] = "recovered_offset"
    return prof


# ---------------------------------------------------------------------------
# Soil layers: chemistry + water extract + granulometry, joined on (id, depth)
# ---------------------------------------------------------------------------

def load_chem() -> pd.DataFrame:
    df = pd.read_csv(INTERIM / "docx_chem_physchem_long.csv", dtype=str, keep_default_na=False)
    df = df[df["pit_id"].map(is_profile_id)].copy()
    df["pit_id"] = df["pit_id"].map(norm_id)
    out = pd.DataFrame({"pit_id": df["pit_id"]})
    out[["depth_top_cm", "depth_bottom_cm"]] = df["depth_cm"].apply(lambda v: pd.Series(parse_depth(v)))
    for col in ["humus_pct", "n_total_pct", "c_n_ratio", "co2_pct", "caco3_pct", "caso4_pct",
                "exch_ca", "exch_mg", "exch_na", "exch_k", "exch_sum",
                "p2o5_mobile", "k2o_mobile", "n_hydrolyzable", "ph_water"]:
        out[col] = df[col].map(parse_float) if col in df.columns else np.nan
    out["src_year"] = pd.to_numeric(df["src_year"], errors="coerce")
    return out


def load_water() -> pd.DataFrame:
    df = pd.read_csv(INTERIM / "docx_water_extract_long.csv", dtype=str, keep_default_na=False)
    df = df[df["pit_id"].map(is_profile_id)].copy()
    df["pit_id"] = df["pit_id"].map(norm_id)
    # two rows per (pit,depth): first = mass %, second = meq. Take first as primary.
    df["_rank"] = df.groupby(["pit_id", "depth_cm"]).cumcount()
    primary = df[df["_rank"] == 0].copy()
    out = pd.DataFrame({"pit_id": primary["pit_id"]})
    out[["depth_top_cm", "depth_bottom_cm"]] = primary["depth_cm"].apply(lambda v: pd.Series(parse_depth(v)))
    ion_cols = ["sum_salts", "alk_hco3", "alk_co3", "cl", "so4", "ca", "mg", "na", "k"]
    for col in ion_cols:
        out[f"salt_{col}_pct"] = primary[col].map(parse_float)
    out["src_year"] = pd.to_numeric(primary["src_year"], errors="coerce")
    return out


def load_granulometry() -> pd.DataFrame:
    df = pd.read_csv(INTERIM / "docx_granulometry_long.csv", dtype=str, keep_default_na=False)
    df = df[df["pit_id"].map(is_profile_id)].copy()
    df["pit_id"] = df["pit_id"].map(norm_id)
    out = pd.DataFrame({"pit_id": df["pit_id"]})
    out[["depth_top_cm", "depth_bottom_cm"]] = df["depth_cm"].apply(lambda v: pd.Series(parse_depth(v)))
    frac = ["sand_1_0p25", "sand_0p25_0p05", "silt_0p05_0p01", "silt_0p01_0p005",
            "silt_0p005_0p001", "clay_lt0p001", "physical_clay_lt0p01", "hygroscopic_water_pct"]
    for col in frac:
        out[col] = df[col].map(parse_float) if col in df.columns else np.nan
    # aggregate to the 3-way texture split used by ground_truth_v2 (sand/silt/clay)
    out["sand_pct"] = out[["sand_1_0p25", "sand_0p25_0p05"]].sum(axis=1, min_count=1)
    out["silt_pct"] = out[["silt_0p05_0p01", "silt_0p01_0p005", "silt_0p005_0p001"]].sum(axis=1, min_count=1)
    out["clay_pct"] = out["clay_lt0p001"]
    out["src_year"] = pd.to_numeric(df["src_year"], errors="coerce")
    return out


def merge_layers(chem, water, gran) -> pd.DataFrame:
    keys = ["pit_id", "depth_top_cm", "depth_bottom_cm"]
    layers = chem.merge(water.drop(columns=["src_year"]), on=keys, how="outer")
    layers = layers.merge(gran.drop(columns=["src_year"]), on=keys, how="outer")
    return layers.sort_values(["pit_id", "depth_top_cm"]).reset_index(drop=True)


def main() -> None:
    CANON.mkdir(parents=True, exist_ok=True)

    profiles = build_profiles()
    chem, water, gran = load_chem(), load_water(), load_granulometry()
    layers = merge_layers(chem, water, gran)

    # Attach coordinates / context to each layer for convenience.
    layers = layers.merge(
        profiles[["pit_id", "lat_dd", "lon_dd", "elevation_m", "in_aoi",
                  "has_coordinates", "coord_provenance"]],
        on="pit_id", how="left",
    )
    layers["georeferenced"] = layers["has_coordinates"].fillna(False)

    profiles.to_csv(CANON / "profiles_v6.csv", index=False, encoding="utf-8")
    layers.to_csv(CANON / "soil_layers_v6.csv", index=False, encoding="utf-8")

    lab_ids = set(chem["pit_id"]) | set(water["pit_id"]) | set(gran["pit_id"])
    coord_ids = set(profiles.loc[profiles["has_coordinates"], "pit_id"])
    manifest = {
        "version": "V6-canonical",
        "n_profiles_total": int(len(profiles)),
        "n_profiles_with_coords": int(profiles["has_coordinates"].sum()),
        "coord_provenance_counts": profiles["coord_provenance"].value_counts().to_dict(),
        "recovered_offset_ids": sorted(
            profiles.loc[profiles["coord_provenance"] == "recovered_offset", "pit_id"]),
        "n_profiles_in_aoi": int(profiles["in_aoi"].sum()),
        "n_soil_layers": int(len(layers)),
        "n_lab_profiles": len(lab_ids),
        "n_lab_with_coords": len(lab_ids & coord_ids),
        "n_lab_without_coords": len(lab_ids - coord_ids),
        "lab_without_coords": sorted(lab_ids - coord_ids),
        "outputs": {
            "profiles": str(CANON / "profiles_v6.csv"),
            "soil_layers": str(CANON / "soil_layers_v6.csv"),
        },
    }
    (CANON / "canonical_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
