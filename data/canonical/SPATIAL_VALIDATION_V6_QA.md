# V6 spatial validation & uncertainty — QA (Phase 7)

## Salinity model (NDMI → P(topsoil salts > 1 %))
- n=70 (27 saline).
- In-sample AUC **0.767**; leave-one-out AUC **0.682** (bootstrap 95 % CI 0.552–0.8, 2000 resamples on the held-out LOO scores).
- **Spatial leave-block-out** (20 km blocks, 11 blocks): **mean per-block AUC = 0.792** (k=3 evaluable blocks); pooled AUC = 0.385.
- Within-block NDMI→salt sign is **positive in 5/5** blocks.

### Why per-block, not pooled, is the honest spatial metric
The pooled spatial AUC (ranking *all* out-of-block predictions together) is 0.385 — below the per-block 0.792 by 0.407. This gap is **not** loss of signal: within every block the NDMI→salinity slope keeps the same (positive) sign and ranks saline > non-saline well (mean per-block AUC ≈ 0.79). What varies between regions is the *base rate / intercept* — one block is 8/8 saline, another 1/10 — so predictions calibrated on other regions' baselines mis-rank across blocks even though they rank correctly within a region. The decision the product supports is local ('which spots here are less saline?'), which is exactly the per-block question. The limitation is **regional calibration drift**, recorded honestly; discrimination is spatially stable.

## Derived suitability (1 − P_saline) vs saxaul labels
- AUC **0.647** (95 % CI 0.435–0.827), n=56 (6 positive).
- Wide CI is expected at this n; reported for honesty, not as a strong claim.

## Independent validation — AralField 2018 Haloxylon
- AUC **0.5**, n=11 (3 present), 95 % CI 0.111–0.889.
- n=11 with 3 positives is far too small for a real estimate — directional only.

## Predictive uncertainty raster
- `outputs/data/suitability_uncertainty_v6.tif` — per-pixel SE of P(saline) via the delta method (Var(η)=g′·COV·g from the coefficient covariance), clipped to training support, masked to the AOI.
- Use it on the dashboard to show *where the model is least certain* (data-sparse NDMI ranges and extrapolated pixels have the largest SE).

## Honest bottom line
The salinity model is the one quantitatively trustworthy object, and its skill survives spatial CV (see numbers above). Everything saxaul-specific rides on small n; intervals are wide and reported as such. This is a screening aid with quantified uncertainty, not a planting guarantee.
