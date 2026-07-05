# V6 models — fit report (Phase 6)

## Why not a free-fit saxaul classifier
Direct NDMI+MSAVI→saxaul logit: **LOO AUC = 0.48** (in-sample 0.667), n=56, 6 positive. Near chance out-of-sample → not trustworthy. Demoted to exploratory.

## Validated core: NDMI → soil salinity, suitability = 1 − P(saline)
- NDMI→P(salts>1%): **LOO AUC = 0.682**, n=70, 27 saline (independent of saxaul labels).
- Derived suitability vs saxaul labels: AUC = 0.647 (n=56, 6 positive). **Consistency check, NOT independent validation:** the saxaul positives are all low-salinity pits and the suitability score is a monotone function of the same salinity target, so this AUC reflects the shared salinity axis, not a separately demonstrated habitat signal.
- Derived suitability vs INDEPENDENT AralField 2018 Haloxylon: **AUC = 0.5** (n=11, 3 present) — the only genuinely external saxaul check; at ≈0.5 it shows no demonstrated saxaul skill and must not be sold as one.

## Salinity model coefficients (standardized)

| term | coef |
|------|------|
| intercept | -0.470 |
| rs30_ndmi | +0.325 |

Positive NDMI coef ⇒ higher NDMI raises P(saline) ⇒ lowers suitability — matches the documented polarity and the calibration.

## Honest conclusion
The trustworthy quantitative object is the **salinity model** (n=70, validated). Saxaul suitability is built on it (Phase 6b) rather than fit directly, because 6 positives cannot support a credible classifier. All skill numbers are reported with their small n; see Phase 7 for spatial CV and bootstrap CIs.
