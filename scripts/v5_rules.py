"""Shared V5 spectral formulas and classification rules.

This module is the single source of truth for V5.1 rule-based screening.
It intentionally keeps the current V5 logic unchanged while making the
formulas reusable by inference, QA, validation, and documentation scripts.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

try:  # Running as ``python scripts/name.py``
    from v5_config import ZoneClass
except ImportError:  # Running as a package/module
    from scripts.v5_config import ZoneClass


WATER_SCL_VALUES: tuple[int, ...] = (3, 6, 7, 8, 9, 10)

CLASS_NAMES: dict[int, str] = {
    int(ZoneClass.WATER_NODATA): "Water / NoData / Shadow",
    int(ZoneClass.OPTIMAL): "Candidate suitable",
    int(ZoneClass.RISK_DRY_SALT): "Dry salt risk",
    int(ZoneClass.DEAD_WET_TOXIC): "Wet brine risk",
    int(ZoneClass.OBSTACLE_TOPO): "Topographic obstacle",
    int(ZoneClass.VEGETATION): "Existing vegetation",
}

CLASS_LIMITATIONS: dict[int, str] = {
    int(ZoneClass.OPTIMAL): (
        "Residual candidate class: pixels not flagged as water/shadow, steep slope, "
        "vegetation, dry salt, or wet brine by the V5 rules. This is a screening "
        "class, not proof of planting success."
    ),
    int(ZoneClass.RISK_DRY_SALT): "Spectral dry-salt proxy; requires field confirmation.",
    int(ZoneClass.DEAD_WET_TOXIC): "Wet/brine proxy; requires soil EC or salinity confirmation.",
}


def safe_divide(numerator, denominator):
    """Elementwise division with non-finite results converted to NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(numerator, denominator)
    if np.isscalar(out):
        return out if np.isfinite(out) else np.nan
    out = np.asarray(out, dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def compute_indices(b3, b4, b8, b11, b12) -> Dict[str, np.ndarray]:
    """Compute V5 indices from reflectance-scaled bands.

    Inputs must already be reflectance values, not Sentinel-2 DN integers.
    Band naming follows Sentinel-2: B3=green, B4=red, B8=NIR,
    B11=SWIR1, B12=SWIR2.
    """
    b3 = np.asarray(b3, dtype=np.float32)
    b4 = np.asarray(b4, dtype=np.float32)
    b8 = np.asarray(b8, dtype=np.float32)
    b11 = np.asarray(b11, dtype=np.float32)
    b12 = np.asarray(b12, dtype=np.float32)

    return {
        "ndwi_nir": safe_divide(b3 - b8, b3 + b8),
        "ndvi": safe_divide(b8 - b4, b8 + b4),
        "ndmi": safe_divide(b8 - b11, b8 + b11),
        "ndsi_green_swir2": safe_divide(b3 - b12, b3 + b12),
        "br_nir_swir2": safe_divide(b8, b12),
        "bi": np.sqrt(b3 ** 2 + b4 ** 2),
    }


def classify_arrays(
    scl,
    slope,
    indices: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """Classify arrays using the V5 cascade.

    The cascade is conservative about missing data:
    water/shadow/no-data -> topographic obstacle -> vegetation -> dark shadow ->
    wet brine risk -> dry salt risk -> residual candidate suitable.
    """
    scl = np.asarray(scl)
    slope = np.asarray(slope, dtype=np.float32)

    ndwi_nir = np.asarray(indices["ndwi_nir"], dtype=np.float32)
    ndvi = np.asarray(indices["ndvi"], dtype=np.float32)
    ndmi = np.asarray(indices["ndmi"], dtype=np.float32)
    ndsi_green_swir2 = np.asarray(indices["ndsi_green_swir2"], dtype=np.float32)
    br_nir_swir2 = np.asarray(indices["br_nir_swir2"], dtype=np.float32)
    bi = np.asarray(indices["bi"], dtype=np.float32)

    cond_invalid = (
        ~np.isfinite(ndwi_nir)
        | ~np.isfinite(ndvi)
        | ~np.isfinite(ndmi)
        | ~np.isfinite(ndsi_green_swir2)
        | ~np.isfinite(br_nir_swir2)
        | ~np.isfinite(bi)
        | ~np.isfinite(slope)
    )

    cond_water = (
        np.isin(scl, WATER_SCL_VALUES)
        | (ndwi_nir > 0)
        | cond_invalid
    )

    cond_topo = (
        ~cond_water
        & np.isfinite(slope)
        & (slope > 5.0)
    )

    cond_veg = (
        ~cond_water
        & ~cond_topo
        & np.isfinite(ndvi)
        & (ndvi > 0.08)
    )

    cond_shadow = (
        ~cond_water
        & ~cond_topo
        & ~cond_veg
        & np.isfinite(bi)
        & (bi < 0.15)
    )

    cond_dead = (
        ~cond_water
        & ~cond_topo
        & ~cond_veg
        & ~cond_shadow
        & np.isfinite(ndmi)
        & np.isfinite(br_nir_swir2)
        & (ndmi > thresholds["NDMI_P85"])
        & (br_nir_swir2 > thresholds["BR_NIR_SWIR2_P85"])
    )

    cond_risk = (
        ~cond_water
        & ~cond_topo
        & ~cond_veg
        & ~cond_shadow
        & ~cond_dead
        & np.isfinite(ndsi_green_swir2)
        & np.isfinite(ndmi)
        & (ndsi_green_swir2 > thresholds["NDSI_Green_SWIR2_P85"])
        & (ndmi < thresholds["NDMI_P15"])
    )

    return np.select(
        [cond_water, cond_topo, cond_veg, cond_shadow, cond_dead, cond_risk],
        [
            ZoneClass.WATER_NODATA,
            ZoneClass.OBSTACLE_TOPO,
            ZoneClass.VEGETATION,
            ZoneClass.WATER_NODATA,
            ZoneClass.DEAD_WET_TOXIC,
            ZoneClass.RISK_DRY_SALT,
        ],
        default=ZoneClass.OPTIMAL,
    ).astype(np.uint8)


def classify_pixel(
    scl: float,
    slope: float,
    indices: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> int:
    """Scalar wrapper for validation and forensic checks."""
    arr_indices = {k: np.asarray([v], dtype=np.float32) for k, v in indices.items()}
    out = classify_arrays(
        np.asarray([scl]),
        np.asarray([slope], dtype=np.float32),
        arr_indices,
        thresholds,
    )
    return int(out[0])
