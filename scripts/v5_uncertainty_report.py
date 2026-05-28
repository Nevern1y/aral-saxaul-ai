"""V5.1 threshold sensitivity and uncertainty diagnostics.

This script estimates how much the candidate screening area changes when the
adaptive percentile thresholds are made more conservative or more sensitive.
It works on a 200 m decimated grid to keep the diagnostic lightweight and
explicitly reports that values are approximate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from v5_rules import CLASS_NAMES, WATER_SCL_VALUES, classify_arrays, classify_pixel, compute_indices


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "outputs" / "data"
SCIENCE = BASE / "outputs" / "science"

BAND_PATHS = {
    "B3": DATA / "B3_10m.tif",
    "B4": DATA / "B4_10m.tif",
    "B8": DATA / "B8_10m.tif",
    "B11": DATA / "B11_20m.tif",
    "B12": DATA / "B12_20m.tif",
    "SCL": DATA / "SCL_10m.tif",
    "Slope": DATA / "dem_slope_30m.tif",
}
THRESHOLDS_PATH = DATA / "thresholds_v5.json"
POINT_SAMPLES_PATH = SCIENCE / "v5_point_samples.csv"
VALIDATION_SUMMARY_PATH = SCIENCE / "v5_validation_summary.json"

DECIMATION_FACTOR = 20  # 10 m -> ~200 m grid, one sample represents ~4 ha


def read_decimated_stack() -> tuple[dict[str, np.ndarray], float]:
    missing = [str(path) for path in BAND_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V5 raster inputs: " + "; ".join(missing))

    with rasterio.open(BAND_PATHS["B8"]) as ref:
        out_h = ref.height // DECIMATION_FACTOR
        out_w = ref.width // DECIMATION_FACTOR
        area_per_sample_ha = abs(ref.res[0] * DECIMATION_FACTOR * ref.res[1] * DECIMATION_FACTOR) / 10000.0
        vrt_kw: dict[str, Any] = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
        }
        stack = {
            "B8": ref.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32),
        }

        for key in ["B3", "B4", "SCL"]:
            with rasterio.open(BAND_PATHS[key]) as src:
                resampling = Resampling.nearest
                stack[key] = src.read(1, out_shape=(out_h, out_w), resampling=resampling).astype(np.float32)

        for key in ["B11", "B12", "Slope"]:
            with rasterio.open(BAND_PATHS[key]) as src:
                resampling = Resampling.bilinear if key != "Slope" else Resampling.cubic
                with WarpedVRT(src, resampling=resampling, **vrt_kw) as vrt:
                    stack[key] = vrt.read(1, out_shape=(out_h, out_w), resampling=resampling).astype(np.float32)

    return stack, area_per_sample_ha


def percentile_thresholds(indices: dict[str, np.ndarray], scl: np.ndarray) -> dict[str, float]:
    valid = (
        ~np.isin(scl, WATER_SCL_VALUES)
        & (indices["ndwi_nir"] <= 0)
        & np.isfinite(indices["ndwi_nir"])
        & np.isfinite(indices["ndmi"])
        & np.isfinite(indices["ndsi_green_swir2"])
        & np.isfinite(indices["br_nir_swir2"])
    )
    values = {
        "NDMI": indices["ndmi"][valid],
        "NDSI_Green_SWIR2": indices["ndsi_green_swir2"][valid],
        "BR_NIR_SWIR2": indices["br_nir_swir2"][valid],
    }
    out: dict[str, float] = {}
    for name, arr in values.items():
        for pct in [10, 15, 20, 80, 85, 90]:
            out[f"{name}_P{pct}"] = float(np.nanpercentile(arr, pct))
    out["valid_sample_count"] = int(valid.sum())
    out["total_sample_count"] = int(valid.size)
    return out


def scenario_thresholds(percentiles: dict[str, float], baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "risk_conservative_P10_P90": {
            "NDMI_P15": percentiles["NDMI_P10"],
            "NDMI_P85": percentiles["NDMI_P90"],
            "NDSI_Green_SWIR2_P15": percentiles["NDSI_Green_SWIR2_P10"],
            "NDSI_Green_SWIR2_P85": percentiles["NDSI_Green_SWIR2_P90"],
            "BR_NIR_SWIR2_P15": percentiles["BR_NIR_SWIR2_P10"],
            "BR_NIR_SWIR2_P85": percentiles["BR_NIR_SWIR2_P90"],
        },
        "baseline_P15_P85": baseline,
        "risk_sensitive_P20_P80": {
            "NDMI_P15": percentiles["NDMI_P20"],
            "NDMI_P85": percentiles["NDMI_P80"],
            "NDSI_Green_SWIR2_P15": percentiles["NDSI_Green_SWIR2_P20"],
            "NDSI_Green_SWIR2_P85": percentiles["NDSI_Green_SWIR2_P80"],
            "BR_NIR_SWIR2_P15": percentiles["BR_NIR_SWIR2_P20"],
            "BR_NIR_SWIR2_P85": percentiles["BR_NIR_SWIR2_P80"],
        },
    }


def class_area_summary(classes: np.ndarray, area_per_sample_ha: float) -> list[dict[str, Any]]:
    rows = []
    total_area = classes.size * area_per_sample_ha
    for class_val in sorted(CLASS_NAMES):
        count = int((classes == class_val).sum())
        area_ha = count * area_per_sample_ha
        rows.append(
            {
                "class": int(class_val),
                "class_name": CLASS_NAMES[class_val],
                "sample_count": count,
                "area_ha_approx": round(area_ha, 1),
                "area_km2_approx": round(area_ha / 100.0, 1),
                "pct_of_sample_grid": round(area_ha / total_area * 100.0, 3) if total_area else 0.0,
            }
        )
    return rows


def point_stability(scenarios: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    if not POINT_SAMPLES_PATH.exists():
        return []
    points = pd.read_csv(POINT_SAMPLES_PATH)
    required = ["S_Point", "coordinate_source", "SCL", "Slope", "ndwi_nir", "ndvi", "ndmi", "ndsi_green_swir2", "br_nir_swir2", "bi"]
    if any(col not in points.columns for col in required):
        return []
    rows = []
    for _, point in points.iterrows():
        indices = {key: point[key] for key in ["ndwi_nir", "ndvi", "ndmi", "ndsi_green_swir2", "br_nir_swir2", "bi"]}
        scenario_classes = {}
        for name, thresholds in scenarios.items():
            if any(pd.isna(v) for v in [point["SCL"], point["Slope"], *indices.values()]):
                scenario_classes[name] = None
            else:
                scenario_classes[name] = classify_pixel(point["SCL"], point["Slope"], indices, thresholds)
        non_null = [v for v in scenario_classes.values() if v is not None]
        stable = len(set(non_null)) <= 1 if non_null else False
        rows.append(
            {
                "S_Point": point["S_Point"],
                "coordinate_source": point["coordinate_source"],
                "stable_across_scenarios": stable,
                "classes": {
                    scenario: (CLASS_NAMES.get(value) if value is not None else "No sample")
                    for scenario, value in scenario_classes.items()
                },
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.6g}"
            else:
                val = str(val)
            vals.append(val.replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, Any]) -> None:
    area_rows = []
    for scenario, rows in summary["class_area_by_scenario"].items():
        candidate = next(row for row in rows if row["class"] == 1)
        area_rows.append(
            {
                "scenario": scenario,
                "candidate_area_ha_approx": candidate["area_ha_approx"],
                "candidate_area_km2_approx": candidate["area_km2_approx"],
                "candidate_pct_grid": candidate["pct_of_sample_grid"],
            }
        )

    stability_rows = []
    for source in sorted(set(row["coordinate_source"] for row in summary["point_stability"])):
        source_rows = [row for row in summary["point_stability"] if row["coordinate_source"] == source]
        stable = sum(1 for row in source_rows if row["stable_across_scenarios"])
        stability_rows.append(
            {
                "coordinate_source": source,
                "stable_points": stable,
                "total_points": len(source_rows),
                "stable_pct": round(stable / len(source_rows) * 100, 1) if source_rows else 0,
            }
        )

    text = f"""# V5.1 Uncertainty And Sensitivity Report

This report estimates threshold sensitivity on a decimated ~200 m grid. It is a diagnostic uncertainty layer, not a replacement for field validation.

## Method

- Read V5 Sentinel-2/SRTM stack on a `{DECIMATION_FACTOR}x` decimated grid.
- Recompute V5 indices from reflectance-scaled bands.
- Compare three threshold scenarios: conservative P10/P90 risk flags, baseline P15/P85, and sensitive P20/P80 risk flags.
- Candidate area estimates are approximate because the grid is sampled, not full-resolution vectorized.

## Candidate Area Sensitivity

{markdown_table(area_rows, ['scenario', 'candidate_area_ha_approx', 'candidate_area_km2_approx', 'candidate_pct_grid'])}

## Point Class Stability

{markdown_table(stability_rows, ['coordinate_source', 'stable_points', 'total_points', 'stable_pct'])}

## Interpretation

- A stable candidate pixel is more robust to percentile-threshold choices.
- An unstable pixel should be treated as low-confidence and prioritized for field checking before operational planting.
- Coordinate authority status from validation summary: `{summary.get('coordinate_policy', 'unknown')}`.
"""
    (SCIENCE / "v5_uncertainty_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    SCIENCE.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))

    stack, area_per_sample_ha = read_decimated_stack()
    for key in ["B3", "B4", "B8", "B11", "B12"]:
        stack[key] = stack[key] / 10000.0
    indices = compute_indices(stack["B3"], stack["B4"], stack["B8"], stack["B11"], stack["B12"])
    percentiles = percentile_thresholds(indices, stack["SCL"])
    scenarios = scenario_thresholds(percentiles, baseline)

    class_area_by_scenario = {}
    for scenario, thresholds in scenarios.items():
        classes = classify_arrays(stack["SCL"], stack["Slope"], indices, thresholds)
        class_area_by_scenario[scenario] = class_area_summary(classes, area_per_sample_ha)

    validation_summary = {}
    if VALIDATION_SUMMARY_PATH.exists():
        validation_summary = json.loads(VALIDATION_SUMMARY_PATH.read_text(encoding="utf-8"))

    summary = {
        "version": "V5.1-uncertainty",
        "method": "Decimated 200m threshold sensitivity diagnostic.",
        "coordinate_policy": validation_summary.get("coordinate_policy", "unknown"),
        "n_authoritative_point_samples": validation_summary.get("n_authoritative_point_samples", 0),
        "decimation_factor": DECIMATION_FACTOR,
        "area_per_sample_ha": area_per_sample_ha,
        "percentiles": percentiles,
        "scenarios": scenarios,
        "class_area_by_scenario": class_area_by_scenario,
        "point_stability": point_stability(scenarios),
        "limitations": [
            "Area estimates are approximate because this is a decimated diagnostic grid.",
            "Candidate suitability remains a screening class, not field-proven planting success.",
            "Point stability follows the coordinate sources available in v5_point_samples.csv.",
        ],
    }
    summary_path = SCIENCE / "v5_uncertainty_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)

    print("V5.1 uncertainty report built")
    print(f"  summary: {summary_path}")
    print(f"  report: {SCIENCE / 'v5_uncertainty_report.md'}")


if __name__ == "__main__":
    main()
