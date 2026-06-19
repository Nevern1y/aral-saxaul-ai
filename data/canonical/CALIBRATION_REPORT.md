# RS / soil threshold calibration — Phase 5

Replaces blind P15/P85 percentiles with measured, AUC-justified cuts.

## Policy

- **10 m V5.1 cascade: kept frozen.** KEPT AS-IS (frozen). n=14 valid 10m points, non-significant, low-quality NoData-edge pixels; recalibration not defensible.
- **30 m suitability layer: calibrated.** CALIBRATED below. 30m NDMI vs measured topsoil salinity rho=+0.66 p<1e-9 n=70; stack is 99% valid across AOI.

## Stability gates

- `MIN_N = 12`: minimum total n for any published threshold.
- `MIN_N_POS = 8`: minimum positives for a *robust* threshold.
  Entries with n_pos < 8 are published but flagged `indicative_only=true` in the JSON.
- `MIN_AUC = 0.62`: minimum oriented AUC (below this the entry is dropped).

## Evaluations (rank-AUC, Youden-optimal cut)

| predictor | target | dir | cut | AUC (oriented) | AUC (raw) | inverted | sens | spec | n | n_pos | stability |
|-----------|--------|-----|-----|----------------|-----------|----------|------|------|---|-------|-----------|
| top_caco3_pct | saxaul_suitable | >= | 5.3 | 0.855 | 0.855 | no | 1.0 | 0.7 | 26 | 3 | low(n_pos=3) |
| rs30_ndwi | soil_salinity>1.0% | >= | -0.142 | 0.797 | 0.797 | no | 0.7 | 0.93 | 70 | 27 | robust |
| rs30_ndwi | soil_salinity>3.0% | >= | -0.142 | 0.787 | 0.787 | no | 0.86 | 0.82 | 70 | 14 | robust |
| rs30_msavi | soil_salinity>1.0% | <= | 0.133 | 0.784 | 0.216 | yes | 0.7 | 0.88 | 70 | 27 | robust |
| rs30_msavi | soil_salinity>3.0% | <= | 0.122 | 0.781 | 0.219 | yes | 0.79 | 0.86 | 70 | 14 | robust |
| rs30_ndmi | soil_salinity>3.0% | >= | -0.084 | 0.776 | 0.776 | no | 0.93 | 0.61 | 70 | 14 | robust |
| rs30_ndmi | soil_salinity>1.0% | >= | -0.103 | 0.767 | 0.767 | no | 0.93 | 0.63 | 70 | 27 | robust |
| rs30_si | soil_salinity>1.0% | >= | 2741.58 | 0.744 | 0.744 | no | 0.63 | 0.95 | 70 | 27 | robust |
| rs30_ndwi | saxaul_suitable | <= | -0.162 | 0.737 | 0.263 | yes | 1.0 | 0.46 | 56 | 6 | low(n_pos=6) |
| top_salt_cl_pct | saxaul_suitable | <= | 0.059 | 0.733 | 0.267 | yes | 1.0 | 0.42 | 49 | 6 | low(n_pos=6) |
| rs30_si | soil_salinity>3.0% | >= | 2852.926 | 0.724 | 0.724 | no | 0.64 | 0.88 | 70 | 14 | robust |
| top_sand_pct | saxaul_suitable | >= | 62.56 | 0.695 | 0.695 | no | 1.0 | 0.5 | 47 | 5 | low(n_pos=5) |
| top_exch_na | saxaul_suitable | <= | 0.262 | 0.692 | 0.308 | yes | 0.5 | 0.98 | 52 | 6 | low(n_pos=6) |
| rs30_si | saxaul_suitable | <= | 2634.741 | 0.69 | 0.31 | yes | 1.0 | 0.44 | 56 | 6 | low(n_pos=6) |
| rs30_msavi | saxaul_suitable | >= | 0.129 | 0.683 | 0.683 | no | 1.0 | 0.38 | 56 | 6 | low(n_pos=6) |
| rs30_ndmi | saxaul_suitable | <= | -0.103 | 0.647 | 0.353 | yes | 0.83 | 0.56 | 56 | 6 | low(n_pos=6) |
| top_salt_sum_salts_pct | saxaul_suitable | <= | 0.311 | 0.642 | 0.358 | yes | 1.0 | 0.52 | 56 | 6 | low(n_pos=6) |
| top_ph_water | saxaul_suitable | <= | 8.692 | 0.533 | 0.467 | yes | 0.75 | 0.63 | 42 | 4 | low(n_pos=4) |

Published thresholds (AUC≥0.62, n≥12): 17 of 18 evaluations (8 robust, 9 indicative-only [n_pos<8]).

## Interpretation

All published cuts have the agronomically correct sign.
Numbers below are generated directly from the computed results and update automatically on recalibration.

### NDMI as a measured salinity proxy

30 m NDMI separates saline soils (>1.0 %) at oriented AUC 0.767 (raw 0.767, direction >=, optimal cut -0.103, n=70, n_pos=27) and strongly saline (>3.0 %) at AUC 0.776 (raw 0.776, cut -0.084, n_pos=14). This replaces the blind P15/P85 percentile with a ground-truth-anchored breakpoint and confirms the CLAUDE.md polarity (high NDMI ⇒ saline brine, NOT favorable).

### Saxaul suitability predictors

Saxaul suitability (screening-aid direction, n_pos=6, all cuts in-sample) increases with: low chloride (Cl <= 0.059 %, oriented AUC 0.733, raw 0.267, n_pos=6, stability=low); low total salts (<= 0.311 %, AUC 0.642, n_pos=6); low exchangeable Na (<= 0.262, AUC 0.692, n_pos=6); carbonate (CaCO3 >= 5.3 %, AUC 0.855, n_pos=3 [indicative only]); sandy texture (sand >= 62.6 %, AUC 0.695, n_pos=5 [indicative only]); low NDMI/NDWI (drier surface, AUC 0.647, n_pos=6). This matches saxaul autecology: it establishes on light, non-saline, carbonate substrates and is excluded by chloride-sodium solonchaks.

### Small-n instability warning (W8)

The following 9 published entry/entries have n_pos < 8 and are marked `indicative_only=true` in the JSON. They have correct signs but are statistically underpowered — the TPR axis has very few attainable values and a single relabel can substantially move the AUC. Do NOT treat them as validated thresholds:

- **rs30_ndmi__saxaul_suitable**: cut=-0.1026, AUC=0.647 (raw=0.353), n_pos=6, stability=low (n_pos<=6)
- **rs30_ndwi__saxaul_suitable**: cut=-0.1616, AUC=0.737 (raw=0.263), n_pos=6, stability=low (n_pos<=6)
- **rs30_si__saxaul_suitable**: cut=2634.7415, AUC=0.69 (raw=0.31), n_pos=6, stability=low (n_pos<=6)
- **rs30_msavi__saxaul_suitable**: cut=0.1288, AUC=0.683 (raw=0.683), n_pos=6, stability=low (n_pos<=6)
- **top_salt_sum_salts_pct__saxaul_suitable**: cut=0.311, AUC=0.642 (raw=0.358), n_pos=6, stability=low (n_pos<=6)
- **top_salt_cl_pct__saxaul_suitable**: cut=0.059, AUC=0.733 (raw=0.267), n_pos=6, stability=low (n_pos<=6)
- **top_exch_na__saxaul_suitable**: cut=0.2618, AUC=0.692 (raw=0.308), n_pos=6, stability=low (n_pos<=6)
- **top_caco3_pct__saxaul_suitable**: cut=5.3, AUC=0.855 (raw=0.855), n_pos=3, stability=low (n_pos<=3)
- **top_sand_pct__saxaul_suitable**: cut=62.56, AUC=0.695 (raw=0.695), n_pos=5, stability=low (n_pos<=5)

### Inverted predictors (W7)

The following entries have raw AUC < 0.5 and were evaluated in the flipped direction (oriented AUC = 1 − raw). The `direction` and `cut` already encode the correct sign; `auc_raw` is the original rank-AUC before flipping so downstream tools can distinguish genuinely strong predictors from inverted proxies of equal oriented AUC:

- **rs30_msavi__soil_salinity>1.0%**: direction=<=, AUC(oriented)=0.784, AUC(raw)=0.216 → flipped by 0.568
- **rs30_msavi__soil_salinity>3.0%**: direction=<=, AUC(oriented)=0.781, AUC(raw)=0.219 → flipped by 0.562
- **rs30_ndmi__saxaul_suitable**: direction=<=, AUC(oriented)=0.647, AUC(raw)=0.353 → flipped by 0.294
- **rs30_ndwi__saxaul_suitable**: direction=<=, AUC(oriented)=0.737, AUC(raw)=0.263 → flipped by 0.474
- **rs30_si__saxaul_suitable**: direction=<=, AUC(oriented)=0.69, AUC(raw)=0.31 → flipped by 0.38
- **top_salt_sum_salts_pct__saxaul_suitable**: direction=<=, AUC(oriented)=0.642, AUC(raw)=0.358 → flipped by 0.284
- **top_salt_cl_pct__saxaul_suitable**: direction=<=, AUC(oriented)=0.733, AUC(raw)=0.267 → flipped by 0.466
- **top_exch_na__saxaul_suitable**: direction=<=, AUC(oriented)=0.692, AUC(raw)=0.308 → flipped by 0.384

## What changes / what does not

- The frozen V5.1 10 m map and its thresholds_v5.json are **unchanged** (golden
  snapshot stays valid).
- These calibrated cuts parameterize the Phase 6 soil-anchored suitability
  model on the 30 m wall-to-wall stack, where they are statistically supported.
- Saxaul-label cuts (n_pos=6, all in-sample Youden) are screening aids only;
  those with n_pos < MIN_N_POS are additionally flagged indicative_only.
  Phase 7 spatial CV quantifies combined model skill.
