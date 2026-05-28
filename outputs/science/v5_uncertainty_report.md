# V5.1 Uncertainty And Sensitivity Report

This report estimates threshold sensitivity on a decimated ~200 m grid. It is a diagnostic uncertainty layer, not a replacement for field validation.

## Method

- Read V5 Sentinel-2/SRTM stack on a `20x` decimated grid.
- Recompute V5 indices from reflectance-scaled bands.
- Compare three threshold scenarios: conservative P10/P90 risk flags, baseline P15/P85, and sensitive P20/P80 risk flags.
- Candidate area estimates are approximate because the grid is sampled, not full-resolution vectorized.

## Candidate Area Sensitivity

| scenario | candidate_area_ha_approx | candidate_area_km2_approx | candidate_pct_grid |
| --- | --- | --- | --- |
| risk_conservative_P10_P90 | 1.66735e+06 | 16673.5 | 18.301 |
| baseline_P15_P85 | 1.48988e+06 | 14898.8 | 16.353 |
| risk_sensitive_P20_P80 | 1.50778e+06 | 15077.8 | 16.55 |

## Point Class Stability

| coordinate_source | stable_points | total_points | stable_pct |
| --- | --- | --- | --- |
| raw_aralfield_dd | 10 | 11 | 90.9 |
| shifted_odt_derived_dd | 11 | 11 | 100 |

## Interpretation

- A stable candidate pixel is more robust to percentile-threshold choices.
- An unstable pixel should be treated as low-confidence and prioritized for field checking before operational planting.
- Coordinate authority status from validation summary: `dual_source_unresolved`.
