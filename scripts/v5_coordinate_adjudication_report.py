"""Build a coordinate adjudication appendix for V5.1 validation.

This workflow does not choose an authoritative coordinate source. It packages
the two competing coordinate candidates, their V5 point diagnostics, and a
review template so the final GPS/ODT decision can be documented explicitly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v5_science_dataset import main as build_science_dataset
from v5_validation_report import main as build_validation_report


BASE = Path(__file__).resolve().parent.parent
SCIENCE = BASE / "outputs" / "science"

FIELD_PATH = SCIENCE / "field_sites_v5.csv"
MAPPING_PATH = SCIENCE / "site_profile_mapping_v5.csv"
PROFILE_PATH = SCIENCE / "soil_profile_summary_v5.csv"
POINT_SAMPLES_PATH = SCIENCE / "v5_point_samples.csv"

CANDIDATES_PATH = SCIENCE / "v5_coordinate_candidates.csv"
TEMPLATE_PATH = SCIENCE / "v5_coordinate_authority_template.csv"
SUMMARY_PATH = SCIENCE / "v5_coordinate_adjudication_summary.json"
REPORT_PATH = SCIENCE / "v5_coordinate_adjudication_report.md"
MANIFEST_PATH = SCIENCE / "v5_science_manifest.json"

WATER_CLASS = "Water / NoData / Shadow"
RAW_SOURCE = "raw_aralfield_dd"
SHIFTED_SOURCE = "shifted_odt_derived_dd"
THIRD_PARTY_SOURCE = "third_party_gps_dd"
UNRESOLVED_SOURCE = "unresolved"
AUTHORITATIVE_SOURCE = "authoritative_dd"
VALID_SELECTED_SOURCES = {RAW_SOURCE, SHIFTED_SOURCE, THIRD_PARTY_SOURCE, UNRESOLVED_SOURCE, ""}
REVIEW_COLUMNS = [
    "selected_coordinate_source",
    "authoritative_lon_dd",
    "authoritative_lat_dd",
    "evidence_reference",
    "reviewer",
    "review_date",
    "decision_note",
]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def is_blank(value: Any) -> bool:
    return clean_text(value) == ""


def fmt_float(value: Any, digits: int = 6) -> str:
    out = finite_float(value)
    if math.isnan(out):
        return ""
    return f"{out:.{digits}g}"


def ensure_inputs() -> None:
    if not (FIELD_PATH.exists() and MAPPING_PATH.exists() and PROFILE_PATH.exists()):
        build_science_dataset()
    if not POINT_SAMPLES_PATH.exists():
        build_validation_report()


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No rows."
    view = df[[col for col in columns if col in df.columns]].copy()
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows: list[str] = []
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = "" if pd.isna(value) else f"{float(value):.6g}"
            elif pd.isna(value):
                value = ""
            else:
                value = str(value)
            values.append(value.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def source_review_note(source: str) -> str:
    if source == RAW_SOURCE:
        return "AralField decimal-degree candidate; keep pending GPS/ODT/photo authority review."
    if source == SHIFTED_SOURCE:
        return "Legacy shifted/ODT-derived candidate; keep pending original ODT/GPS authority review."
    if source == AUTHORITATIVE_SOURCE:
        return "Evidence-backed authoritative coordinate selected from the review template."
    return "Coordinate candidate requires source review."


def map_support_note(class_name: Any) -> str:
    if class_name == WATER_CLASS:
        return "V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof."
    if pd.isna(class_name):
        return "V5 could not sample this coordinate."
    return "V5 samples a non-water class; this is diagnostic only, not coordinate proof."


def build_candidates(samples: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    field_cols = [
        "S_Point",
        "location_text",
        "field_note",
        "field_soil_texture",
        "lithology",
        "soil_color",
    ]
    candidates = samples.merge(fields[[col for col in field_cols if col in fields.columns]], on="S_Point", how="left")
    candidates["authority_decision"] = np.where(candidates["coordinate_source"] == AUTHORITATIVE_SOURCE, "selected_authoritative", "unresolved")
    candidates["candidate_review_status"] = np.where(
        candidates["coordinate_source"] == AUTHORITATIVE_SOURCE,
        "resolved_by_authority_template",
        "candidate_not_authoritative",
    )
    candidates["source_review_note"] = candidates["coordinate_source"].map(source_review_note)
    candidates["map_support_note"] = candidates["class_filtered_name"].map(map_support_note)

    ordered_cols = [
        "S_Point",
        "pit_number",
        "pit_code",
        "mapping_confidence",
        "coordinate_source",
        "lon",
        "lat",
        "coordinate_conflict_m",
        "authority_decision",
        "candidate_review_status",
        "class_filtered_name",
        "class_recomputed_name",
        "top_salinity_pct",
        "top_pH",
        "top_humus_pct",
        "field_salinity_0_20",
        "field_ec_0_20",
        "ndmi",
        "ndsi_green_swir2",
        "br_nir_swir2",
        "landcover_type",
        "vegetation_name",
        "vegetation_density_pct_text",
        "field_soil_texture",
        "lithology",
        "soil_color",
        "location_text",
        "field_note",
        "authority_selected_source",
        "authority_evidence_reference",
        "source_review_note",
        "map_support_note",
    ]
    existing_cols = [col for col in ordered_cols if col in candidates.columns]
    candidates = candidates[existing_cols].sort_values(["pit_number", "coordinate_source"])
    return candidates


def selected_source_coordinates(row: pd.Series) -> tuple[float, float]:
    source = clean_text(row.get("selected_coordinate_source"))
    if source == RAW_SOURCE:
        return finite_float(row.get("raw_lon_aralfield_dd")), finite_float(row.get("raw_lat_aralfield_dd"))
    if source == SHIFTED_SOURCE:
        return finite_float(row.get("shifted_lon_odt_dd")), finite_float(row.get("shifted_lat_odt_dd"))
    if source == THIRD_PARTY_SOURCE:
        return finite_float(row.get("authoritative_lon_dd")), finite_float(row.get("authoritative_lat_dd"))
    return float("nan"), float("nan")


def validate_authority_row(row: pd.Series) -> tuple[str, str]:
    source = clean_text(row.get("selected_coordinate_source"))
    if source == "":
        source = UNRESOLVED_SOURCE

    if source not in VALID_SELECTED_SOURCES:
        return "invalid", f"Invalid selected_coordinate_source: {source}"
    if source == UNRESOLVED_SOURCE:
        return "unresolved", "Pending external GPS/ODT authority evidence."

    if is_blank(row.get("evidence_reference")):
        return "incomplete", "A selected coordinate source requires evidence_reference."

    lon, lat = selected_source_coordinates(row)
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return "incomplete", "Selected source does not resolve to valid lon/lat."

    if source in {RAW_SOURCE, SHIFTED_SOURCE} and (
        not is_blank(row.get("authoritative_lon_dd")) or not is_blank(row.get("authoritative_lat_dd"))
    ):
        authored_lon = finite_float(row.get("authoritative_lon_dd"))
        authored_lat = finite_float(row.get("authoritative_lat_dd"))
        if math.isfinite(authored_lon) and math.isfinite(authored_lat):
            if abs(authored_lon - lon) > 1e-6 or abs(authored_lat - lat) > 1e-6:
                return "invalid", "Authoritative lon/lat differs from selected raw/shifted source; use third_party_gps_dd."

    return "resolved", "Coordinate source has reviewer evidence and valid coordinates."


def apply_template_validation(template: pd.DataFrame) -> pd.DataFrame:
    statuses = [validate_authority_row(row) for _, row in template.iterrows()]
    template = template.copy()
    template["authority_row_status"] = [status for status, _ in statuses]
    template["authority_validation_note"] = [note for _, note in statuses]
    return template


def preserve_existing_review_fields(template: pd.DataFrame) -> pd.DataFrame:
    if not TEMPLATE_PATH.exists():
        return template

    previous = pd.read_csv(TEMPLATE_PATH, dtype=str, keep_default_na=False)
    if "S_Point" not in previous.columns:
        return template

    review_cols = [col for col in REVIEW_COLUMNS if col in previous.columns]
    if not review_cols:
        return template

    previous_review = previous[["S_Point", *review_cols]].drop_duplicates("S_Point", keep="last")
    merged = template.merge(previous_review, on="S_Point", how="left", suffixes=("", "_existing"))
    for col in REVIEW_COLUMNS:
        existing_col = f"{col}_existing"
        if existing_col not in merged.columns:
            continue
        merged[col] = [
            clean_text(existing) if not is_blank(existing) else clean_text(current)
            for current, existing in zip(merged[col], merged[existing_col])
        ]
        merged.drop(columns=[existing_col], inplace=True)
    return merged


def build_authority_template(mapping: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in mapping.sort_values("pit_number").iterrows():
        rows.append(
            {
                "S_Point": row.get("S_Point"),
                "pit_number": row.get("pit_number"),
                "pit_code": row.get("pit_code"),
                "raw_lon_aralfield_dd": row.get("lon_aralfield_dd"),
                "raw_lat_aralfield_dd": row.get("lat_aralfield_dd"),
                "shifted_lon_odt_dd": row.get("lon_shifted_odt_dd"),
                "shifted_lat_odt_dd": row.get("lat_shifted_odt_dd"),
                "coordinate_conflict_m": row.get("coordinate_conflict_m"),
                "selected_coordinate_source": "",
                "authoritative_lon_dd": "",
                "authoritative_lat_dd": "",
                "evidence_reference": "",
                "reviewer": "",
                "review_date": "",
                "decision_note": "",
            }
        )
    template = pd.DataFrame(rows)
    template = preserve_existing_review_fields(template)
    return apply_template_validation(template)


def class_distribution(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, group in candidates.groupby("coordinate_source"):
        counts = group["class_filtered_name"].fillna("No sample").value_counts()
        for class_name, count in counts.items():
            rows.append(
                {
                    "coordinate_source": source,
                    "class_filtered_name": class_name,
                    "count": int(count),
                }
            )
    return rows


def source_diagnostics(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, group in candidates.groupby("coordinate_source"):
        classes = group["class_filtered_name"].fillna("No sample")
        total = int(len(group))
        water = int((classes == WATER_CLASS).sum())
        no_sample = int((classes == "No sample").sum())
        non_water = total - water - no_sample
        candidate = int((classes == "Candidate suitable").sum())
        vegetation = int((classes == "Existing vegetation").sum())
        rows.append(
            {
                "coordinate_source": source,
                "total_points": total,
                "water_nodata_shadow_points": water,
                "non_water_sampled_points": non_water,
                "candidate_suitable_points": candidate,
                "existing_vegetation_points": vegetation,
                "water_nodata_shadow_share_pct": round(water / total * 100, 1) if total else 0.0,
                "diagnostic_interpretation": (
                    "All sampled points fall in water/nodata/shadow; this source is suspect but not rejected without ODT/GPS authority."
                    if total and water == total
                    else "Mixed sampled classes; usable for pilot diagnostics but still not authoritative."
                ),
            }
        )
    return rows


def template_status_counts(template: pd.DataFrame) -> dict[str, int]:
    if "authority_row_status" not in template.columns:
        return {}
    return {str(key): int(value) for key, value in template["authority_row_status"].value_counts().items()}


def decision_status_from_template(template: pd.DataFrame) -> str:
    counts = template_status_counts(template)
    if counts.get("invalid", 0) > 0:
        return "invalid_template"
    if counts.get("incomplete", 0) > 0:
        return "incomplete_template"
    if counts.get("resolved", 0) == len(template) and len(template) > 0:
        return "resolved_with_evidence"
    if counts.get("resolved", 0) > 0:
        return "partially_resolved"
    return "unresolved"


def build_summary(mapping: pd.DataFrame, profiles: pd.DataFrame, candidates: pd.DataFrame, template: pd.DataFrame) -> dict[str, Any]:
    conflicts = pd.to_numeric(mapping["coordinate_conflict_m"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    status_counts = template_status_counts(template)
    return {
        "version": "V5.1-coordinate-adjudication",
        "coordinate_policy": "adjudication_required_dual_source_unresolved",
        "decision_status": decision_status_from_template(template),
        "n_field_sites": int(mapping["S_Point"].nunique()),
        "n_coordinate_candidates": int(len(candidates)),
        "n_profiles_with_coordinate_candidates": int(mapping["pit_number"].nunique()),
        "n_lab_only_profiles_without_coordinates": int(profiles["S_Point"].isna().sum()),
        "authority_template_rows": int(len(template)),
        "authority_template_status_counts": status_counts,
        "authority_template_resolved_rows": int(status_counts.get("resolved", 0)),
        "authority_template_invalid_rows": int(status_counts.get("invalid", 0)),
        "authority_template_incomplete_rows": int(status_counts.get("incomplete", 0)),
        "authority_template_unresolved_rows": int(status_counts.get("unresolved", 0)),
        "coordinate_conflict_median": float(conflicts.median()) if not conflicts.empty else None,
        "coordinate_conflict_min": float(conflicts.min()) if not conflicts.empty else None,
        "coordinate_conflict_max": float(conflicts.max()) if not conflicts.empty else None,
        "source_diagnostics": source_diagnostics(candidates),
        "class_distribution_by_coordinate_source": class_distribution(candidates),
        "decision_rule": {
            "current_position": "No coordinate source is authoritative until external GPS/ODT/photo evidence is recorded.",
            "do_not_do": "Do not merge raw and shifted validation samples, and do not use spectral class alone as coordinate proof.",
            "minimum_evidence_needed": [
                "Original GPS/ODT export or field notebook for S124-S134.",
                "Reviewer decision for each S_Point in v5_coordinate_authority_template.csv.",
                "Coordinates for lab-only profiles 12/20A-21/20A before spatial validation uses those profiles.",
            ],
        },
        "outputs": {
            "coordinate_candidates": str(CANDIDATES_PATH),
            "authority_template": str(TEMPLATE_PATH),
            "summary_json": str(SUMMARY_PATH),
            "report_md": str(REPORT_PATH),
        },
    }


def write_report(candidates: pd.DataFrame, template: pd.DataFrame, summary: dict[str, Any]) -> None:
    diagnostics = pd.DataFrame(summary["source_diagnostics"])
    class_dist = pd.DataFrame(summary["class_distribution_by_coordinate_source"])

    candidate_cols = [
        "S_Point",
        "pit_code",
        "coordinate_source",
        "lon",
        "lat",
        "coordinate_conflict_m",
        "class_filtered_name",
        "top_salinity_pct",
        "field_salinity_0_20",
        "field_ec_0_20",
        "authority_selected_source",
        "map_support_note",
    ]
    template_cols = [
        "S_Point",
        "pit_code",
        "raw_lon_aralfield_dd",
        "raw_lat_aralfield_dd",
        "shifted_lon_odt_dd",
        "shifted_lat_odt_dd",
        "coordinate_conflict_m",
        "selected_coordinate_source",
        "authoritative_lon_dd",
        "authoritative_lat_dd",
        "evidence_reference",
        "authority_row_status",
        "authority_validation_note",
    ]

    median_conflict = summary.get("coordinate_conflict_median")
    median_text = f"{median_conflict / 1000:.1f} km" if median_conflict is not None else "n/a"

    text = f"""# V5.1 Coordinate Adjudication Appendix

This appendix prepares the coordinate-authority decision for the 11 mapped field points. It does not change V5 map products or V5 rule logic.

## Decision State

- Status: `{summary['decision_status']}`.
- Policy: `{summary['coordinate_policy']}`.
- Field sites with coordinate candidates: {summary['n_field_sites']}.
- Coordinate candidates evaluated: {summary['n_coordinate_candidates']}.
- Authority template rows resolved: {summary['authority_template_resolved_rows']} / {summary['authority_template_rows']}.
- Authority template unresolved rows: {summary['authority_template_unresolved_rows']}.
- Authority template incomplete/invalid rows: {summary['authority_template_incomplete_rows']} / {summary['authority_template_invalid_rows']}.
- Median raw-vs-shifted conflict: {median_text}.
- Lab-only profiles without coordinates: {summary['n_lab_only_profiles_without_coordinates']}.

## Use Policy

- Do not merge `raw_aralfield_dd` and `shifted_odt_derived_dd` validation samples.
- Do not treat V5 spectral class as coordinate proof.
- Use this appendix to record an external GPS/ODT/photo-backed decision, then rerun `python scripts/run_v5_science_suite.py --qa`.

## Source-Level Diagnostics

{markdown_table(diagnostics, ['coordinate_source', 'total_points', 'water_nodata_shadow_points', 'non_water_sampled_points', 'candidate_suitable_points', 'existing_vegetation_points', 'water_nodata_shadow_share_pct', 'diagnostic_interpretation'])}

## Filtered V5 Class Distribution

{markdown_table(class_dist, ['coordinate_source', 'class_filtered_name', 'count'])}

## Coordinate Candidate Audit

{markdown_table(candidates, candidate_cols)}

## Authority Review Template

Fill `outputs/science/v5_coordinate_authority_template.csv` with the selected source or a third-party GPS coordinate for each point. Leave a reference to the original evidence in `evidence_reference`.

Manual review fields are preserved on regeneration; running the science suite will not erase reviewer decisions already present in the template.

Allowed `selected_coordinate_source` values: `raw_aralfield_dd`, `shifted_odt_derived_dd`, `third_party_gps_dd`, or `unresolved`.

{markdown_table(template, template_cols)}

## Remaining Blockers

- Authoritative coordinates for `S124-S134` / profiles `01/20A-11/20A`.
- Coordinates for lab-only profiles `12/20A-21/20A`.
- Independent survival or planting outcome observations if the product is later upgraded beyond screening.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def update_science_manifest(candidates: pd.DataFrame, template: pd.DataFrame, summary: dict[str, Any]) -> None:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {}

    manifest.setdefault("outputs", {}).update(
        {
            "coordinate_candidates": str(CANDIDATES_PATH),
            "coordinate_authority_template": str(TEMPLATE_PATH),
            "coordinate_adjudication_summary": str(SUMMARY_PATH),
            "coordinate_adjudication_report": str(REPORT_PATH),
        }
    )
    manifest.setdefault("row_counts", {}).update(
        {
            "coordinate_candidates": int(len(candidates)),
            "coordinate_authority_template": int(len(template)),
        }
    )
    manifest["coordinate_adjudication_status"] = summary["decision_status"]
    manifest["coordinate_authority_template_status_counts"] = summary["authority_template_status_counts"]
    manifest["coordinate_adjudication_note"] = (
        "Fill v5_coordinate_authority_template.csv with GPS/ODT evidence before treating any coordinate source as authoritative."
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    SCIENCE.mkdir(parents=True, exist_ok=True)
    ensure_inputs()

    mapping = pd.read_csv(MAPPING_PATH)
    profiles = pd.read_csv(PROFILE_PATH)
    fields = pd.read_csv(FIELD_PATH)
    samples = pd.read_csv(POINT_SAMPLES_PATH)

    candidates = build_candidates(samples, fields)
    template = build_authority_template(mapping)
    summary = build_summary(mapping, profiles, candidates, template)

    candidates.to_csv(CANDIDATES_PATH, index=False, encoding="utf-8")
    template.to_csv(TEMPLATE_PATH, index=False, encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(candidates.copy(), template.copy(), summary)
    update_science_manifest(candidates, template, summary)

    print("V5.1 coordinate adjudication appendix built")
    print(f"  candidates: {CANDIDATES_PATH}")
    print(f"  template: {TEMPLATE_PATH}")
    print(f"  summary: {SUMMARY_PATH}")
    print(f"  report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
