"""Sample V5 map/bands at field coordinates and write validation reports.

The report evaluates both coordinate sources currently present in the project:
raw AralField decimal degrees and shifted/ODT-derived coordinates. This avoids
silently treating one unresolved coordinate system as ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import spearmanr

from build_v5_science_dataset import main as build_science_dataset
from v5_rules import CLASS_NAMES, classify_pixel, compute_indices


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"
SCIENCE = BASE / "outputs" / "science"

MAPPING_PATH = SCIENCE / "site_profile_mapping_v5.csv"
PROFILE_PATH = SCIENCE / "soil_profile_summary_v5.csv"
FIELD_PATH = SCIENCE / "field_sites_v5.csv"
AUTHORITY_TEMPLATE_PATH = SCIENCE / "v5_coordinate_authority_template.csv"

BAND_PATHS = {
    "B3": DATA / "B3_10m.tif",
    "B4": DATA / "B4_10m.tif",
    "B8": DATA / "B8_10m.tif",
    "B11": DATA / "B11_20m.tif",
    "B12": DATA / "B12_20m.tif",
    "SCL": DATA / "SCL_10m.tif",
    "Slope": DATA / "dem_slope_30m.tif",
    "class_raw": DATA / "suitability_map_v5.tif",
    "class_filtered": DATA / "suitability_map_v5_filtered.tif",
}
THRESHOLDS_PATH = DATA / "thresholds_v5.json"

RAW_SOURCE = "raw_aralfield_dd"
SHIFTED_SOURCE = "shifted_odt_derived_dd"
THIRD_PARTY_SOURCE = "third_party_gps_dd"
UNRESOLVED_SOURCE = "unresolved"
AUTHORITATIVE_SOURCE = "authoritative_dd"
VALID_SELECTED_SOURCES = {RAW_SOURCE, SHIFTED_SOURCE, THIRD_PARTY_SOURCE, UNRESOLVED_SOURCE, ""}


def finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def selected_source_coordinates(row: pd.Series, source: str) -> tuple[float, float]:
    if source == RAW_SOURCE:
        return finite_or_nan(row.get("lon_aralfield_dd")), finite_or_nan(row.get("lat_aralfield_dd"))
    if source == SHIFTED_SOURCE:
        return finite_or_nan(row.get("lon_shifted_odt_dd")), finite_or_nan(row.get("lat_shifted_odt_dd"))
    if source == THIRD_PARTY_SOURCE:
        return finite_or_nan(row.get("authoritative_lon_dd")), finite_or_nan(row.get("authoritative_lat_dd"))
    return np.nan, np.nan


def validate_authority_row(row: pd.Series) -> tuple[str, str, float, float]:
    source = clean_text(row.get("selected_coordinate_source")) or UNRESOLVED_SOURCE
    if source not in VALID_SELECTED_SOURCES:
        return "invalid", f"Invalid selected_coordinate_source: {source}", np.nan, np.nan
    if source == UNRESOLVED_SOURCE:
        return "unresolved", "Pending external GPS/ODT authority evidence.", np.nan, np.nan
    if not clean_text(row.get("evidence_reference")):
        return "incomplete", "A selected coordinate source requires evidence_reference.", np.nan, np.nan

    lon, lat = selected_source_coordinates(row, source)
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return "incomplete", "Selected source does not resolve to valid lon/lat.", lon, lat

    if source in {RAW_SOURCE, SHIFTED_SOURCE}:
        authored_lon = finite_or_nan(row.get("authoritative_lon_dd"))
        authored_lat = finite_or_nan(row.get("authoritative_lat_dd"))
        if math.isfinite(authored_lon) and math.isfinite(authored_lat):
            if abs(authored_lon - lon) > 1e-6 or abs(authored_lat - lat) > 1e-6:
                return (
                    "invalid",
                    "Authoritative lon/lat differs from selected raw/shifted source; use third_party_gps_dd.",
                    lon,
                    lat,
                )
    return "resolved", "Coordinate source has reviewer evidence and valid coordinates.", lon, lat


def authority_template_review(mapping: pd.DataFrame) -> pd.DataFrame:
    review_cols = [
        "S_Point", "selected_coordinate_source", "authoritative_lon_dd", "authoritative_lat_dd",
        "evidence_reference", "reviewer", "review_date", "decision_note",
    ]
    if not AUTHORITY_TEMPLATE_PATH.exists():
        out = mapping[["S_Point", "pit_number", "pit_code"]].copy()
        out["authority_row_status"] = "unresolved"
        out["authority_validation_note"] = "Authority template not created yet."
        return out

    template = pd.read_csv(AUTHORITY_TEMPLATE_PATH, dtype=str, keep_default_na=False)
    if "S_Point" not in template.columns:
        out = mapping[["S_Point", "pit_number", "pit_code"]].copy()
        out["authority_row_status"] = "invalid"
        out["authority_validation_note"] = "Authority template is missing S_Point column."
        return out

    for col in review_cols:
        if col not in template.columns:
            template[col] = ""
    review = template[review_cols].drop_duplicates("S_Point", keep="last")
    merged = mapping.merge(review, on="S_Point", how="left")

    statuses = [validate_authority_row(row) for _, row in merged.iterrows()]
    merged["authority_row_status"] = [status for status, _, _, _ in statuses]
    merged["authority_validation_note"] = [note for _, note, _, _ in statuses]
    merged["authority_lon_dd"] = [lon for _, _, lon, _ in statuses]
    merged["authority_lat_dd"] = [lat for _, _, _, lat in statuses]
    return merged


def sample_raster(src: rasterio.io.DatasetReader, lon: float, lat: float) -> float:
    if pd.isna(lon) or pd.isna(lat):
        return np.nan
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    row, col = src.index(x, y)
    if row < 0 or col < 0 or row >= src.height or col >= src.width:
        return np.nan
    value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    return finite_or_nan(value)


def authority_status_counts(authority_review: pd.DataFrame) -> dict[str, int]:
    if authority_review.empty or "authority_row_status" not in authority_review.columns:
        return {}
    return {str(key): int(value) for key, value in authority_review["authority_row_status"].value_counts().items()}


def coordinate_policy_from_authority(authority_review: pd.DataFrame) -> str:
    counts = authority_status_counts(authority_review)
    resolved = counts.get("resolved", 0)
    invalid = counts.get("invalid", 0)
    incomplete = counts.get("incomplete", 0)
    total = int(len(authority_review))
    if invalid:
        return "authority_template_invalid_dual_source_retained"
    if incomplete:
        return "authority_template_incomplete_dual_source_retained"
    if resolved and resolved == total:
        return "authoritative_coordinates_available_dual_source_retained_for_audit"
    if resolved:
        return "partial_authoritative_coordinates_available_dual_source_retained"
    return "dual_source_unresolved"


def make_coordinate_rows(mapping: pd.DataFrame, authority_review: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in mapping.iterrows():
        base = {
            "S_Point": row["S_Point"],
            "pit_number": int(row["pit_number"]),
            "pit_code": row["pit_code"],
            "mapping_confidence": row["mapping_confidence"],
            "coordinate_conflict_m": row["coordinate_conflict_m"],
            "authority_selected_source": "",
            "authority_evidence_reference": "",
            "authority_row_status": "audit_source_not_authoritative",
            "authority_validation_note": "Raw/shifted candidate retained for coordinate audit.",
        }
        if pd.notna(row.get("lon_aralfield_dd")) and pd.notna(row.get("lat_aralfield_dd")):
            rows.append(
                {
                    **base,
                    "coordinate_source": RAW_SOURCE,
                    "lon": row["lon_aralfield_dd"],
                    "lat": row["lat_aralfield_dd"],
                }
            )
        if pd.notna(row.get("lon_shifted_odt_dd")) and pd.notna(row.get("lat_shifted_odt_dd")):
            rows.append(
                {
                    **base,
                    "coordinate_source": SHIFTED_SOURCE,
                    "lon": row["lon_shifted_odt_dd"],
                    "lat": row["lat_shifted_odt_dd"],
                }
            )
    for _, row in authority_review[authority_review["authority_row_status"] == "resolved"].iterrows():
        rows.append(
            {
                "S_Point": row["S_Point"],
                "pit_number": int(row["pit_number"]),
                "pit_code": row["pit_code"],
                "mapping_confidence": row["mapping_confidence"],
                "coordinate_conflict_m": row["coordinate_conflict_m"],
                "coordinate_source": AUTHORITATIVE_SOURCE,
                "lon": row["authority_lon_dd"],
                "lat": row["authority_lat_dd"],
                "authority_selected_source": clean_text(row.get("selected_coordinate_source")),
                "authority_evidence_reference": clean_text(row.get("evidence_reference")),
                "authority_row_status": row["authority_row_status"],
                "authority_validation_note": row["authority_validation_note"],
            }
        )
    return pd.DataFrame(rows)


def sample_points(points: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    missing = [str(path) for path in BAND_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V5 raster inputs: " + "; ".join(missing))

    datasets = {key: rasterio.open(path) for key, path in BAND_PATHS.items()}
    try:
        rows: list[dict[str, Any]] = []
        for _, point in points.iterrows():
            out = point.to_dict()
            for key, src in datasets.items():
                out[key] = sample_raster(src, point["lon"], point["lat"])

            reflectance = {
                band: out[band] / 10000.0 if pd.notna(out[band]) else np.nan
                for band in ["B3", "B4", "B8", "B11", "B12"]
            }
            indices = compute_indices(
                reflectance["B3"],
                reflectance["B4"],
                reflectance["B8"],
                reflectance["B11"],
                reflectance["B12"],
            )
            scalar_indices = {key: finite_or_nan(np.asarray(value).reshape(-1)[0]) for key, value in indices.items()}
            out.update(scalar_indices)
            out["class_recomputed"] = classify_pixel(out["SCL"], out["Slope"], scalar_indices, thresholds)

            for class_col in ["class_raw", "class_filtered", "class_recomputed"]:
                class_val = out.get(class_col)
                out[f"{class_col}_name"] = CLASS_NAMES.get(int(class_val), "Unknown") if pd.notna(class_val) else "No sample"

            rows.append(out)
    finally:
        for src in datasets.values():
            src.close()
    return pd.DataFrame(rows)


def spearman_with_bootstrap(df: pd.DataFrame, target: str, feature: str) -> dict[str, Any]:
    valid = df[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(valid))
    result: dict[str, Any] = {
        "target": target,
        "feature": feature,
        "n": n,
        "spearman_r": None,
        "p_value": None,
        "bootstrap_ci95": None,
        "status": "ok",
    }
    if n < 3:
        result["status"] = "insufficient_n"
        return result
    if valid[target].nunique() < 2 or valid[feature].nunique() < 2:
        result["status"] = "constant_input"
        return result

    r, p = spearmanr(valid[target], valid[feature])
    result["spearman_r"] = round(float(r), 6)
    result["p_value"] = round(float(p), 6)

    if n >= 5:
        rng = np.random.default_rng(51)
        boot = []
        values = valid.to_numpy(dtype=float)
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            sample = values[idx]
            if len(np.unique(sample[:, 0])) < 2 or len(np.unique(sample[:, 1])) < 2:
                continue
            br, _ = spearmanr(sample[:, 0], sample[:, 1])
            if math.isfinite(br):
                boot.append(br)
        if boot:
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            result["bootstrap_ci95"] = [round(float(lo), 6), round(float(hi), 6)]
    return result


def build_correlations(samples: pd.DataFrame) -> list[dict[str, Any]]:
    targets = ["top_salinity_pct", "field_salinity_0_20", "field_ec_0_20"]
    features = ["ndmi", "ndvi", "ndwi_nir", "ndsi_green_swir2", "br_nir_swir2", "bi"]
    rows: list[dict[str, Any]] = []
    for source, group in samples.groupby("coordinate_source"):
        for target in targets:
            if target not in group.columns:
                continue
            for feature in features:
                stat = spearman_with_bootstrap(group, target, feature)
                stat["coordinate_source"] = source
                rows.append(stat)
    return rows


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No rows."
    tmp = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in tmp.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (list, tuple)):
                val = json.dumps(val)
            elif isinstance(val, float):
                val = "" if pd.isna(val) else f"{val:.6g}"
            elif pd.isna(val):
                val = ""
            else:
                val = str(val)
            vals.append(val.replace("|", "\\|"))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, separator, *rows])


def write_report(samples: pd.DataFrame, summary: dict[str, Any], report_path: Path) -> None:
    class_rows = []
    for source, group in samples.groupby("coordinate_source"):
        counts = group["class_filtered_name"].value_counts(dropna=False)
        for class_name, count in counts.items():
            class_rows.append({"coordinate_source": source, "class_filtered": class_name, "count": int(count)})
    class_df = pd.DataFrame(class_rows)

    corr_df = pd.DataFrame(summary["correlations"])
    corr_view = corr_df[
        (corr_df["target"] == "top_salinity_pct")
        & (corr_df["feature"].isin(["ndmi", "ndsi_green_swir2", "br_nir_swir2"]))
    ].copy()

    point_cols = [
        "coordinate_source", "S_Point", "pit_code", "top_salinity_pct", "field_salinity_0_20",
        "field_ec_0_20", "class_filtered_name", "ndmi", "ndsi_green_swir2", "br_nir_swir2",
        "authority_selected_source", "authority_row_status",
    ]
    for col in ["top_salinity_pct", "field_salinity_0_20", "field_ec_0_20", "ndmi", "ndsi_green_swir2", "br_nir_swir2"]:
        if col in samples.columns:
            samples[col] = samples[col].round(4)

    text = f"""# V5.1 Scientific Validation Report

This report samples the V5 map and Sentinel-derived indices at mapped field/soil profiles. It is a pilot validation report, not an accuracy certificate.

## Coordinate Policy

Coordinate status: `{summary['coordinate_policy']}`.

Both coordinate candidates are evaluated separately:

- `raw_aralfield_dd`: decimal-degree coordinates from `AralField(Sheet1).csv`.
- `shifted_odt_derived_dd`: legacy shifted/ODT-derived coordinates from `ground_truth_v2.csv`.
- `authoritative_dd`: only appears after `v5_coordinate_authority_template.csv` contains resolved reviewer decisions with evidence.

The raw and shifted coordinate systems differ by roughly `{summary['coordinate_conflict_median']:.0f}` m median distance. Do not merge raw and shifted validation samples. If `authoritative_dd` rows are present, use them as the evidence-backed validation subset and keep raw/shifted rows as coordinate audit diagnostics.

Authority template status counts: `{json.dumps(summary['authority_template_status_counts'], ensure_ascii=False)}`.

## Sample Counts

- Point samples: {summary['n_point_samples']}
- Unique field sites: {summary['n_field_sites']}
- Profiles with coordinates: {summary['n_profiles_with_coordinates']}
- Lab-only profiles without coordinates: {summary['n_lab_only_profiles']}
- Authoritative point samples: {summary['n_authoritative_point_samples']}
- Authority template resolved rows: {summary['authority_template_resolved_rows']} / {summary['authority_template_rows']}

## Filtered V5 Class Distribution At Points

{markdown_table(class_df, ['coordinate_source', 'class_filtered', 'count'])}

## Selected Spearman Correlations

Target is top-layer lab salinity (`top_salinity_pct`). Bootstrap intervals are unstable with `n=11`; use them only as sensitivity diagnostics.

{markdown_table(corr_view, ['coordinate_source', 'target', 'feature', 'n', 'spearman_r', 'p_value', 'bootstrap_ci95', 'status'])}

## Point-Level Audit

{markdown_table(samples, point_cols)}

## Scientific Limitations

- V5 is a rule-based suitability screening product, not a trained habitat model.
- `Candidate suitable` is a residual class after excluding water/shadow, steep slope, vegetation, dry-salt proxy, and wet-brine proxy.
- Only 11 profiles currently have any field-site mapping; profiles 12-21 are lab-only until coordinates are provided.
- Coordinate authority is unresolved, so all reported correlations are exploratory.
- Remote-sensing indices are surface proxies and cannot replace root-zone salinity, EC, groundwater depth, or planting survival observations.
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    SCIENCE.mkdir(parents=True, exist_ok=True)
    if not (MAPPING_PATH.exists() and PROFILE_PATH.exists() and FIELD_PATH.exists()):
        build_science_dataset()

    mapping = pd.read_csv(MAPPING_PATH)
    profiles = pd.read_csv(PROFILE_PATH)
    fields = pd.read_csv(FIELD_PATH)

    authority_review = authority_template_review(mapping)
    points = make_coordinate_rows(mapping, authority_review)
    points = points.merge(
        profiles[["pit_number", "top_salinity_pct", "top_pH", "top_humus_pct", "sand_pct_2020", "silt_pct_2020", "clay_pct_2020"]],
        on="pit_number",
        how="left",
    )
    points = points.merge(
        fields[["S_Point", "field_salinity_0_20", "field_ec_0_20", "landcover_type", "vegetation_name", "vegetation_density_pct_text"]],
        on="S_Point",
        how="left",
    )

    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    samples = sample_points(points, thresholds)

    sample_path = SCIENCE / "v5_point_samples.csv"
    samples.to_csv(sample_path, index=False, encoding="utf-8")

    correlations = build_correlations(samples)
    conflict = mapping["coordinate_conflict_m"].replace([np.inf, -np.inf], np.nan).dropna()
    authority_counts = authority_status_counts(authority_review)
    summary = {
        "version": "V5.1-science-validation",
        "coordinate_policy": coordinate_policy_from_authority(authority_review),
        "n_point_samples": int(len(samples)),
        "n_authoritative_point_samples": int((samples["coordinate_source"] == AUTHORITATIVE_SOURCE).sum()),
        "n_field_sites": int(mapping["S_Point"].nunique()),
        "n_profiles_with_coordinates": int(profiles["S_Point"].notna().sum()),
        "n_lab_only_profiles": int(profiles["S_Point"].isna().sum()),
        "authority_template_present": bool(AUTHORITY_TEMPLATE_PATH.exists()),
        "authority_template_rows": int(len(authority_review)),
        "authority_template_status_counts": authority_counts,
        "authority_template_resolved_rows": int(authority_counts.get("resolved", 0)),
        "authority_template_unresolved_rows": int(authority_counts.get("unresolved", 0)),
        "authority_template_incomplete_rows": int(authority_counts.get("incomplete", 0)),
        "authority_template_invalid_rows": int(authority_counts.get("invalid", 0)),
        "coordinate_conflict_median": float(conflict.median()) if not conflict.empty else None,
        "coordinate_conflict_min": float(conflict.min()) if not conflict.empty else None,
        "coordinate_conflict_max": float(conflict.max()) if not conflict.empty else None,
        "correlations": correlations,
        "outputs": {
            "point_samples": str(sample_path),
            "summary_json": str(SCIENCE / "v5_validation_summary.json"),
            "report_md": str(SCIENCE / "v5_validation_report.md"),
        },
    }

    summary_path = SCIENCE / "v5_validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = SCIENCE / "v5_validation_report.md"
    write_report(samples.copy(), summary, report_path)

    print("V5.1 validation report built")
    print(f"  samples: {sample_path}")
    print(f"  summary: {summary_path}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
