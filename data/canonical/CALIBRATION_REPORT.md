# RS / soil threshold calibration — Phase 5

Replaces blind P15/P85 percentiles with measured, AUC-justified cuts.

## Policy

- **10 m V5.1 cascade: kept frozen.** KEPT AS-IS (frozen). n=14 valid 10m points, non-significant, low-quality NoData-edge pixels; recalibration not defensible.
- **30 m suitability layer: calibrated.** CALIBRATED below. 30m NDMI vs measured topsoil salinity rho=+0.66 p<1e-9 n=70; stack is 99% valid across AOI.

## Evaluations (rank-AUC, Youden-optimal cut)

| predictor | target | dir | cut | AUC | sens | spec | n | n_pos |
|-----------|--------|-----|-----|-----|------|------|---|-------|
| top_caco3_pct | saxaul_suitable | >= | 5.3 | 0.855 | 1.0 | 0.7 | 26 | 3 |
| rs30_ndwi | soil_salinity>1.0% | >= | -0.142 | 0.797 | 0.7 | 0.93 | 70 | 27 |
| rs30_ndwi | soil_salinity>3.0% | >= | -0.142 | 0.787 | 0.86 | 0.82 | 70 | 14 |
| rs30_msavi | soil_salinity>1.0% | <= | 0.133 | 0.784 | 0.7 | 0.88 | 70 | 27 |
| rs30_msavi | soil_salinity>3.0% | <= | 0.122 | 0.781 | 0.79 | 0.86 | 70 | 14 |
| rs30_ndmi | soil_salinity>3.0% | >= | -0.084 | 0.776 | 0.93 | 0.61 | 70 | 14 |
| rs30_ndmi | soil_salinity>1.0% | >= | -0.103 | 0.767 | 0.93 | 0.63 | 70 | 27 |
| rs30_si | soil_salinity>1.0% | >= | 2741.58 | 0.744 | 0.63 | 0.95 | 70 | 27 |
| rs30_ndwi | saxaul_suitable | <= | -0.162 | 0.737 | 1.0 | 0.46 | 56 | 6 |
| top_salt_cl_pct | saxaul_suitable | <= | 0.059 | 0.733 | 1.0 | 0.42 | 49 | 6 |
| rs30_si | soil_salinity>3.0% | >= | 2852.926 | 0.724 | 0.64 | 0.88 | 70 | 14 |
| top_sand_pct | saxaul_suitable | >= | 62.56 | 0.695 | 1.0 | 0.5 | 47 | 5 |
| top_exch_na | saxaul_suitable | <= | 0.262 | 0.692 | 0.5 | 0.98 | 52 | 6 |
| rs30_si | saxaul_suitable | <= | 2634.741 | 0.69 | 1.0 | 0.44 | 56 | 6 |
| rs30_msavi | saxaul_suitable | >= | 0.129 | 0.683 | 1.0 | 0.38 | 56 | 6 |
| rs30_ndmi | saxaul_suitable | <= | -0.103 | 0.647 | 0.83 | 0.56 | 56 | 6 |
| top_salt_sum_salts_pct | saxaul_suitable | <= | 0.311 | 0.642 | 1.0 | 0.52 | 56 | 6 |
| top_ph_water | saxaul_suitable | <= | 8.692 | 0.533 | 0.75 | 0.63 | 42 | 4 |

Published thresholds (AUC≥0.62, n≥12): 17 of 18 evaluations.

## Interpretation

All published cuts have the agronomically correct sign:

- **Saxaul suitability** ↑ with low chloride (Cl ≤ 0.06 %, AUC 0.73), low total
  salts (≤ 0.31 %, AUC 0.64), low sodium (exch. Na ≤ 0.26, AUC 0.69), low NDWI
  /NDMI (drier surface), and ↑ with carbonate (CaCO3 ≥ 5.3 %, AUC 0.86) and
  sandy texture (sand ≥ 63 %, AUC 0.70). This matches saxaul autecology:
  it establishes on light, non-saline, carbonate substrates and is excluded by
  chloride-sodium solonchaks.
- **NDMI as a salinity proxy** is now *measured*, not assumed: 30 m NDMI
  separates saline soils (>1 %) at AUC 0.77 and strongly saline (>3 %) at 0.78,
  optimal cut ≈ -0.10. This replaces the blind P15/P85 percentile with a
  ground-truth-anchored breakpoint and confirms the CLAUDE.md polarity
  (high NDMI ⇒ saline brine, NOT favorable).

## What changes / what does not

- The frozen V5.1 10 m map and its thresholds_v5.json are **unchanged** (golden
  snapshot stays valid).
- These calibrated cuts parameterize the Phase 6 soil-anchored suitability
  model on the 30 m wall-to-wall stack, where they are statistically supported.
- Specificities are modest for the saxaul-label cuts (only 6 positives); these
  are screening aids feeding a multivariate model, not standalone classifiers.
  Phase 7 spatial CV quantifies combined skill.
