"""Build normalized V5.1 science/provenance tables.

The script does not alter V5 map products. It prepares field, soil, mapping,
and provenance tables under ``outputs/science`` so validation can be reproduced
and coordinate uncertainty is explicit.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
SCIENCE = BASE / "outputs" / "science"

FIELD_PATH = BASE / "AralField(Sheet1).csv"
CHEM_PATH = BASE / "Результаты EC,TDS, pH, в.в., гумус(в.в.csv"
HUMUS_PATH = BASE / "результаты Арал(Лист2).csv"
TEXTURE_PATH = BASE / "мехсостав 2020, 2021 по типам(Лист4).csv"
GT_PATH = BASE / "outputs" / "data" / "ground_truth_v2.csv"


CONFIRMED_MAPPING = {"S124", "S125", "S126", "S127", "S128", "S131", "S132", "S133", "S134"}
TENTATIVE_MAPPING = {"S129", "S130"}


def safe_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", ".")
    if text in {"", "-", "NA", "N/A", "nan", "None", "np.float64(nan)"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return value
    return np.nan


def normalize_pit_code(value: Any, pit_number: int | None = None) -> str:
    text = "" if value is None else str(value).strip()
    text = text.replace(" ", "").replace("A", "А")
    if text:
        match = re.search(r"(\d+)", text)
        if match:
            return f"{int(match.group(1)):02d}/20А"
    if pit_number is not None and not pd.isna(pit_number):
        return f"{int(pit_number):02d}/20А"
    return ""


def extract_pit_number(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def parse_depth(depth_raw: Any) -> tuple[float, float]:
    text = "" if depth_raw is None else str(depth_raw).replace(",", ".")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan, np.nan
    if len(nums) == 1:
        val = float(nums[0])
        return val, val
    return float(nums[0]), float(nums[1])


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if any(pd.isna(v) for v in [lon1, lat1, lon2, lat2]):
        return np.nan
    radius_m = 6_371_000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return radius_m * 2 * math.asin(math.sqrt(a))


def build_field_sites() -> pd.DataFrame:
    raw = pd.read_csv(FIELD_PATH, sep=";", header=None, skiprows=3, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        s_point = str(row.iloc[0]).strip()
        if not s_point or s_point.lower() == "nan":
            continue
        rows.append(
            {
                "S_Point": s_point,
                "location_text": first_non_empty(row.iloc[1]),
                "lon_aralfield_dd": safe_float(row.iloc[11]),
                "lat_aralfield_dd": safe_float(row.iloc[12]),
                "elevation_m": safe_float(row.iloc[13]),
                "field_date": first_non_empty(row.iloc[14]),
                "field_time": first_non_empty(row.iloc[15]),
                "field_salinity_0_20": safe_float(row.iloc[24]),
                "field_salinity_20_50": safe_float(row.iloc[25]),
                "field_salinity_50_100": safe_float(row.iloc[26]),
                "field_ec_0_20": safe_float(row.iloc[27]),
                "field_ec_20_50": safe_float(row.iloc[28]),
                "field_ec_50_100": safe_float(row.iloc[29]),
                "field_soil_temp_0_20_c": safe_float(row.iloc[30]),
                "field_soil_temp_20_50_c": safe_float(row.iloc[31]),
                "field_soil_temp_50_100_c": safe_float(row.iloc[32]),
                "field_moisture_0_20": safe_float(row.iloc[33]),
                "field_moisture_20_50": safe_float(row.iloc[34]),
                "field_moisture_50_100": safe_float(row.iloc[35]),
                "mean_canopy_temp_c": safe_float(row.iloc[36]),
                "mean_soil_surface_temp_c": safe_float(row.iloc[37]),
                "landcover_type": first_non_empty(row.iloc[38]),
                "landcover_type2": first_non_empty(row.iloc[39]),
                "vegetation_type": first_non_empty(row.iloc[40]),
                "vegetation_name": first_non_empty(row.iloc[41]),
                "vegetation_density_pct_text": first_non_empty(row.iloc[42]),
                "terrain_type": first_non_empty(row.iloc[43]),
                "field_soil_texture": first_non_empty(row.iloc[44]),
                "lithology": first_non_empty(row.iloc[45]),
                "soil_color": first_non_empty(row.iloc[46]),
                "field_note": first_non_empty(row.iloc[47]),
                "source_file": FIELD_PATH.name,
                "coordinate_source": "AralField decimal degrees",
            }
        )
    return pd.DataFrame(rows)


def build_chem_layers() -> pd.DataFrame:
    raw = pd.read_csv(CHEM_PATH, sep=";", header=None, skiprows=2, encoding="utf-8-sig", low_memory=False)
    rows: list[dict[str, Any]] = []
    current_pit: int | None = None
    for _, row in raw.iterrows():
        profile_name = first_non_empty(row.iloc[1])
        if isinstance(profile_name, str) and "Разрез" in profile_name:
            current_pit = extract_pit_number(profile_name)
        if current_pit is None:
            continue
        depth_raw = first_non_empty(row.iloc[2])
        if pd.isna(depth_raw):
            continue
        top_cm, bottom_cm = parse_depth(depth_raw)
        rows.append(
            {
                "pit_number": current_pit,
                "pit_code": normalize_pit_code(None, current_pit),
                "source_table": "chemistry_salinity_ions",
                "depth_raw": depth_raw,
                "depth_top_cm": top_cm,
                "depth_bottom_cm": bottom_cm,
                "CO3_mmol_100g": safe_float(row.iloc[3]),
                "CO3_pct": safe_float(row.iloc[4]),
                "HCO3_mmol_100g": safe_float(row.iloc[5]),
                "HCO3_pct": safe_float(row.iloc[6]),
                "Cl_mmol_100g": safe_float(row.iloc[7]),
                "Cl_pct": safe_float(row.iloc[8]),
                "SO4_mmol_100g": safe_float(row.iloc[9]),
                "SO4_pct": safe_float(row.iloc[10]),
                "Ca_mmol_100g": safe_float(row.iloc[11]),
                "Ca_pct": safe_float(row.iloc[12]),
                "Mg_mmol_100g": safe_float(row.iloc[13]),
                "Mg_pct": safe_float(row.iloc[14]),
                "Na_mmol_100g": safe_float(row.iloc[15]),
                "Na_pct": safe_float(row.iloc[16]),
                "K_mmol_100g": safe_float(row.iloc[17]),
                "K_pct": safe_float(row.iloc[18]),
                "salinity_pct": safe_float(row.iloc[19]),
                "sum_anions": safe_float(row.iloc[20]),
                "sum_cations": safe_float(row.iloc[21]),
                "source_file": CHEM_PATH.name,
            }
        )
    return pd.DataFrame(rows)


def build_humus_layers() -> pd.DataFrame:
    raw = pd.read_csv(HUMUS_PATH, sep=";", encoding="utf-8-sig", low_memory=False)
    pit_ffill = raw["№ разреза"].ffill()
    rows: list[dict[str, Any]] = []
    for idx, row in raw.iterrows():
        pit_number = extract_pit_number(pit_ffill.iloc[idx])
        if pit_number is None:
            continue
        depth_raw = first_non_empty(row.get("Глубина"))
        if pd.isna(depth_raw):
            continue
        top_cm, bottom_cm = parse_depth(depth_raw)
        rows.append(
            {
                "pit_number": pit_number,
                "pit_code": normalize_pit_code(pit_ffill.iloc[idx], pit_number),
                "source_table": "humus_co2_ph",
                "depth_raw": depth_raw,
                "depth_top_cm": top_cm,
                "depth_bottom_cm": bottom_cm,
                "humus_pct": safe_float(row.get("гумус , % ")),
                "CO2": safe_float(row.get("СО2")),
                "pH": safe_float(row.get("рН")),
                "source_file": HUMUS_PATH.name,
            }
        )
    return pd.DataFrame(rows)


def build_texture_profiles() -> pd.DataFrame:
    raw = pd.read_csv(TEXTURE_PATH, sep=";", header=None, encoding="utf-8-sig", low_memory=False)
    rows: list[dict[str, Any]] = []
    for idx in range(len(raw)):
        pit_2020 = extract_pit_number(raw.iloc[idx, 0])
        if pit_2020 is not None and 1 <= pit_2020 <= 21:
            rows.append(
                {
                    "pit_number": pit_2020,
                    "pit_code": normalize_pit_code(None, pit_2020),
                    "texture_year": 2020,
                    "sand_pct": safe_float(raw.iloc[idx, 1]),
                    "silt_pct": safe_float(raw.iloc[idx, 3]),
                    "clay_pct": safe_float(raw.iloc[idx, 6]),
                    "texture_label": first_non_empty(raw.iloc[idx, 7]),
                    "source_file": TEXTURE_PATH.name,
                }
            )
        pit_2021 = extract_pit_number(raw.iloc[idx, 13]) if raw.shape[1] > 13 else None
        if pit_2021 is not None and 1 <= pit_2021 <= 21:
            rows.append(
                {
                    "pit_number": pit_2021,
                    "pit_code": normalize_pit_code(None, pit_2021),
                    "texture_year": 2021,
                    "sand_pct": safe_float(raw.iloc[idx, 14]),
                    "silt_pct": safe_float(raw.iloc[idx, 16]),
                    "clay_pct": safe_float(raw.iloc[idx, 19]),
                    "texture_label": first_non_empty(raw.iloc[idx, 20]),
                    "source_file": TEXTURE_PATH.name,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["pit_number", "texture_year"]).sort_values(["texture_year", "pit_number"])


def build_mapping(field_sites: pd.DataFrame) -> pd.DataFrame:
    if GT_PATH.exists():
        gt = pd.read_csv(GT_PATH)
        gt_cols = gt[["S_Point", "pit_code", "pit_number", "Lon_DD", "Lat_DD"]].copy()
        gt_cols.rename(
            columns={"Lon_DD": "lon_shifted_odt_dd", "Lat_DD": "lat_shifted_odt_dd"},
            inplace=True,
        )
    else:
        gt_cols = pd.DataFrame(columns=["S_Point", "pit_code", "pit_number", "lon_shifted_odt_dd", "lat_shifted_odt_dd"])

    mapping = field_sites[["S_Point", "lon_aralfield_dd", "lat_aralfield_dd"]].merge(gt_cols, on="S_Point", how="left")
    mapping["pit_number"] = mapping["pit_number"].fillna(mapping["S_Point"].str.extract(r"S(\d+)")[0].astype(float) - 123)
    mapping["pit_code"] = [normalize_pit_code(code, num) for code, num in zip(mapping["pit_code"], mapping["pit_number"])]
    mapping["mapping_confidence"] = mapping["S_Point"].map(
        lambda point: "confirmed_from_project_notes" if point in CONFIRMED_MAPPING else (
            "tentative_from_project_notes" if point in TENTATIVE_MAPPING else "unreviewed"
        )
    )
    mapping["coordinate_conflict_m"] = [
        round(haversine_m(raw_lon, raw_lat, shifted_lon, shifted_lat), 1)
        for raw_lon, raw_lat, shifted_lon, shifted_lat in zip(
            mapping["lon_aralfield_dd"],
            mapping["lat_aralfield_dd"],
            mapping["lon_shifted_odt_dd"],
            mapping["lat_shifted_odt_dd"],
        )
    ]
    mapping["coordinate_policy"] = "dual_source_unresolved"
    mapping["notes"] = (
        "Raw AralField DD and shifted/ODT-derived coordinates are both retained. "
        "Do not collapse to one coordinate source until the ODT/GPS authority is resolved."
    )
    return mapping.sort_values("pit_number")


def build_profile_summary(
    chem_layers: pd.DataFrame,
    humus_layers: pd.DataFrame,
    texture_profiles: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    pits = pd.DataFrame({"pit_number": list(range(1, 22))})
    pits["pit_code"] = [normalize_pit_code(None, n) for n in pits["pit_number"]]

    chem_top = chem_layers.sort_values(["pit_number", "depth_top_cm", "depth_bottom_cm"]).groupby("pit_number").first().reset_index()
    chem_cols = [
        "pit_number", "depth_raw", "depth_top_cm", "depth_bottom_cm", "salinity_pct",
        "CO3_mmol_100g", "HCO3_mmol_100g", "Cl_mmol_100g", "SO4_mmol_100g",
        "Ca_mmol_100g", "Mg_mmol_100g", "Na_mmol_100g", "K_mmol_100g",
    ]
    chem_top = chem_top[[c for c in chem_cols if c in chem_top.columns]].rename(
        columns={
            "depth_raw": "top_chem_depth_raw",
            "depth_top_cm": "top_chem_depth_top_cm",
            "depth_bottom_cm": "top_chem_depth_bottom_cm",
            "salinity_pct": "top_salinity_pct",
        }
    )

    humus_top = humus_layers.sort_values(["pit_number", "depth_top_cm", "depth_bottom_cm"]).groupby("pit_number").first().reset_index()
    humus_top = humus_top[["pit_number", "depth_raw", "depth_top_cm", "depth_bottom_cm", "humus_pct", "CO2", "pH"]].rename(
        columns={
            "depth_raw": "top_humus_depth_raw",
            "depth_top_cm": "top_humus_depth_top_cm",
            "depth_bottom_cm": "top_humus_depth_bottom_cm",
            "humus_pct": "top_humus_pct",
            "CO2": "top_CO2",
            "pH": "top_pH",
        }
    )

    texture_2020 = texture_profiles[texture_profiles["texture_year"] == 2020].copy()
    texture_2020 = texture_2020[["pit_number", "sand_pct", "silt_pct", "clay_pct", "texture_label"]].rename(
        columns={
            "sand_pct": "sand_pct_2020",
            "silt_pct": "silt_pct_2020",
            "clay_pct": "clay_pct_2020",
            "texture_label": "texture_label_2020",
        }
    )

    mapped = mapping[[
        "pit_number", "S_Point", "lon_aralfield_dd", "lat_aralfield_dd",
        "lon_shifted_odt_dd", "lat_shifted_odt_dd", "mapping_confidence", "coordinate_conflict_m",
    ]].copy()
    mapped["pit_number"] = mapped["pit_number"].astype(int)

    out = pits.merge(chem_top, on="pit_number", how="left")
    out = out.merge(humus_top, on="pit_number", how="left")
    out = out.merge(texture_2020, on="pit_number", how="left")
    out = out.merge(mapped, on="pit_number", how="left")
    out["georeference_status"] = np.where(out["S_Point"].notna(), "mapped_to_field_site_dual_source", "lab_only_no_coordinates")
    out["usable_for_spatial_validation"] = False
    out["spatial_validation_note"] = np.where(
        out["S_Point"].notna(),
        "Coordinate source unresolved; validate raw and shifted coordinates separately.",
        "No coordinates available for this profile.",
    )
    return out


def main() -> None:
    SCIENCE.mkdir(parents=True, exist_ok=True)

    field_sites = build_field_sites()
    chem_layers = build_chem_layers()
    humus_layers = build_humus_layers()
    texture_profiles = build_texture_profiles()
    mapping = build_mapping(field_sites)
    profile_summary = build_profile_summary(chem_layers, humus_layers, texture_profiles, mapping)

    all_layers = pd.concat([chem_layers, humus_layers], ignore_index=True, sort=False)

    outputs = {
        "field_sites": SCIENCE / "field_sites_v5.csv",
        "soil_layers": SCIENCE / "soil_layers_v5.csv",
        "texture_profiles": SCIENCE / "texture_profiles_v5.csv",
        "site_profile_mapping": SCIENCE / "site_profile_mapping_v5.csv",
        "soil_profile_summary": SCIENCE / "soil_profile_summary_v5.csv",
    }
    field_sites.to_csv(outputs["field_sites"], index=False, encoding="utf-8")
    all_layers.to_csv(outputs["soil_layers"], index=False, encoding="utf-8")
    texture_profiles.to_csv(outputs["texture_profiles"], index=False, encoding="utf-8")
    mapping.to_csv(outputs["site_profile_mapping"], index=False, encoding="utf-8")
    profile_summary.to_csv(outputs["soil_profile_summary"], index=False, encoding="utf-8")

    manifest = {
        "version": "V5.1-science",
        "purpose": "Normalized field/soil/provenance tables for V5 scientific validation.",
        "coordinate_policy": "dual_source_unresolved",
        "important_limitation": (
            "Profiles 12-21 are lab-only until coordinates are provided. The 11 mapped points "
            "retain both raw AralField coordinates and shifted/ODT-derived coordinates."
        ),
        "source_files": {
            "field_sites": str(FIELD_PATH),
            "chemistry_salinity_ions": str(CHEM_PATH),
            "humus_co2_ph": str(HUMUS_PATH),
            "texture": str(TEXTURE_PATH),
            "legacy_ground_truth_mapping": str(GT_PATH),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "row_counts": {
            "field_sites": int(len(field_sites)),
            "chemistry_layers": int(len(chem_layers)),
            "humus_layers": int(len(humus_layers)),
            "soil_layers_total": int(len(all_layers)),
            "texture_profiles": int(len(texture_profiles)),
            "site_profile_mapping": int(len(mapping)),
            "soil_profile_summary": int(len(profile_summary)),
        },
    }
    manifest_path = SCIENCE / "v5_science_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("V5.1 science dataset built")
    for key, path in outputs.items():
        print(f"  {key}: {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
