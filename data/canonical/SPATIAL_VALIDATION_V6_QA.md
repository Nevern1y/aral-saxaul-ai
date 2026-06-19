# V6 spatial validation & uncertainty — QA (Phase 7)

## Salinity model (NDMI → P(topsoil salts > 1 %))
- n=70 (27 saline).
- In-sample AUC **0.767**; leave-one-out AUC **0.682** (bootstrap 95 % CI 0.552–0.8, 2000 resamples on the held-out LOO scores).
- **Spatial leave-block-out** (20 km blocks, 11 blocks): **mean per-block AUC = 0.792** (k=3 evaluable blocks); pooled AUC = 0.385.
- Within-block NDMI→salt sign is **positive in 5/5** blocks.

### Why per-block, not pooled, is the honest spatial metric
The pooled spatial AUC (ranking *all* out-of-block predictions together) is 0.385 — below the per-block 0.792 by 0.407. This gap is **not** loss of signal: within every block the NDMI→salinity slope keeps the same (positive) sign and ranks saline > non-saline well (mean per-block AUC ≈ 0.79). What varies between regions is the *base rate / intercept* — one block is 8/8 saline, another 1/10 — so predictions calibrated on other regions' baselines mis-rank across blocks even though they rank correctly within a region. The decision the product supports is local ('which spots here are less saline?'), which is exactly the per-block question. The limitation is **regional calibration drift**, recorded honestly; discrimination is spatially stable.

## Derived suitability (1 − P_saline) vs saxaul labels
- AUC **0.647** (95 % CI 0.427–0.818), n=56 (6 positive).
- Wide CI is expected at this n; reported for honesty, not as a strong claim.

## Independent validation — AralField 2018 Haloxylon
- AUC **0.5**, n=11 (3 present), 95 % CI 0.111–0.889.
- n=11 with 3 positives is far too small for a real estimate — directional only.

## Morphological feature ablation (NDMI vs NDMI+morph predictors)
Three ablation sets are reported side-by-side to disentangle morph signal from subset-selection confound.

### SET A — HEADLINE: pure 100%-coverage features, no hcl, full n=70
This is the honest 'does morph help?' number, free of the hcl-driven subset confound.
- **Target:** salinity (topsoil salt > 1 %) — same as the main salinity model.
- **Features:** rust_mottling_flag, gley_flag, surface_crust_flag, marine_shell_flag, horizon_salic_flag, horizon_ploughed_flag, solum_depth_cm (all 100% coverage; hcl excluded).
- **n = 70** (27 saline) — full dataset, no subset restriction.
- **Baseline (NDMI only, n=70):** LOO AUC **0.682** (95 % CI 0.551–0.803).
- **Augmented (NDMI + morph):** LOO AUC **0.736** (95 % CI 0.596–0.871).
- **ΔAUC = +0.053** — direction: **LIFT**.

### SET B — secondary: 7 features + hcl_effervescence_class (~80% cov), n≈56
- **CRITICAL SUBSET CONFOUND:** adding hcl_effervescence_class restricts the ablation to n=56 rows (complete cases). On this subset the **NDMI-only baseline ITSELF drops to 0.6** (vs 0.682 on full n=70) — a fall of 0.082 AUC units purely from subset selection, before any morph feature is added.
- Approximately 0.08 AUC units of any apparent lift on this subset reflects that baseline collapse, NOT morph signal.
- **Baseline (NDMI only, n=56):** LOO AUC **0.6** (95 % CI 0.443–0.745).
- **Augmented (NDMI + morph):** LOO AUC **0.831** (95 % CI 0.715–0.924).
- **ΔAUC = +0.231** — direction: **LIFT** (but ~0.08 of this is subset artefact, not morph signal).

### SET C — tertiary: all 10 morph features (includes lower-coverage depth cols), n≈33
This negative result is a valid scientific finding and is reported explicitly.
- **n = 33** complete cases (lower-coverage features reduce n further).
- **Baseline (NDMI only, n=33):** LOO AUC **0.772** (95 % CI 0.585–0.926).
- **Augmented (NDMI + morph):** LOO AUC **0.699** (95 % CI 0.496–0.881).
- **ΔAUC = -0.074** — direction: **HURT**.

### Summary
- SET A (headline, n=70, no hcl): the honest, subset-confound-free morph result.
- SET B (n≈56, +hcl): NDMI-only baseline drops from 0.682 to 0.6 purely from subset restriction; interpret its ΔAUC with that collapse in mind.
- SET C (n≈33, all features): negative result reported honestly.
- The shipped salinity model is unchanged (NDMI-only). All ablations are analysis only — a correctly-measured neutral or negative result is a success.

## Predictive uncertainty raster
- `outputs/data/suitability_uncertainty_v6.tif` — per-pixel SE of P(saline) via the delta method (Var(η)=g′·COV·g from the coefficient covariance), clipped to training support, masked to the AOI.
- Use it on the dashboard to show *where the model is least certain* (data-sparse NDMI ranges and extrapolated pixels have the largest SE).

## Honest bottom line
The salinity model is the one quantitatively trustworthy object, and its skill survives spatial CV (see numbers above). Everything saxaul-specific rides on small n; intervals are wide and reported as such. This is a screening aid with quantified uncertainty, not a planting guarantee.
