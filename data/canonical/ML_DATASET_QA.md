# ML dataset + correlations — QA notes (Phase 4)

Script: `scripts/v6/build_ml_dataset.py` → `ml_dataset_v6.csv` (70 georef rows,
64 features), `ml_correlations_v6.csv`, `ml_dataset_manifest.json`.

## What the table contains

- **Soil** (50 features): topsoil 0-30 cm + whole-profile depth-weighted means
  of salinity (mass %), agro-chemistry (humus, N, CaCO3, gypsum, exchangeable
  Ca/Mg/Na/K, P2O5, K2O, pH) and texture (sand/silt/clay, hygroscopic water).
- **RS production 10 m** (`rs_*`): NDMI/NDVI/NDWI/NDSI/BR/BI via the SAME
  `v5_rules.compute_indices` as production, + slope + V5.1 zone class.
- **RS legacy 30 m** (`rs30_*`): NDMI/MSAVI/SI/NDWI/Slope/TWI/VH from the
  feature stack (full coverage).
- **Label**: saxaul_status, label_role/strength/weight, y_suitable (1/0).

## DECISIVE coverage finding (changed the modeling strategy)

The production Sentinel-2 composite (2025-04..06, <10 % cloud) is **~54 % NoData
over the Syrdarya delta**, so only **14 / 70** ground-truth points fall on valid
10 m pixels; the V5.1 zone map covers 13/70. The DEM slope and the legacy 30 m
stack cover **70/70**.

Implication: you cannot calibrate an RS threshold where the RS input is NoData.
Recalibrating production thresholds on 14 points would be statistically hollow.

**Strategic pivot (within the approved freedom to improve the plan):**
1. **Soil chemistry is the scientific anchor** of the trained model — coverage
   is near-complete (humus 233 layers), it is the causal driver of saxaul
   establishment, and it is contemporaneous (2012-2014) with the labels.
2. **RS stays the wall-to-wall screening layer** (the existing V5.1 cascade),
   with provenance/coverage flagged. Both 10 m (`rs_*`, continuity) and 30 m
   (`rs30_*`, analysis power) are kept in SEPARATE columns — never silently
   mixed.
3. The Phase 6 suitability raster is still produced wall-to-wall from the RS
   indices that exist everywhere, but its thresholds/weights are calibrated
   against the soil-anchored labels instead of blind percentiles.
4. **Recommended future upgrade** (logged, not silently skipped): re-fetch a
   gap-free annual cloud-masked S2 median to lift 10 m coverage to ~70/70.

## Pipeline validation

`v5_zone_recomputed` vs `v5_zone_map`: **14/14 match** where S2 data exists →
the V6 sampler reproduces the production cascade exactly (DN/10000 →
compute_indices → classify_pixel). No silent formula drift.

## Correlations vs y_suitable (Spearman, trainable rows)

Ecologically coherent signs (weak magnitudes, expected at n≈50 with 6 positives):

| feature | rho | p | n | reading |
|---------|-----|---|---|---------|
| prof_caso4_pct | +0.47 | 0.18 | 10 | gypsum favorable |
| top_caco3_pct | +0.39 | **0.05** | 26 | carbonate favorable (only sig.) |
| salt_cl_pct | -0.27 | 0.07 | 49 | chloride salinity unfavorable |
| rs30_ndwi | -0.25 | 0.06 | 56 | surface wetness unfavorable (matches NDMI story) |
| exch_na / exch_sum | -0.21 | 0.13 | 52 | sodicity unfavorable |
| sand_pct | +0.21 | 0.16 | 47 | sandy texture favorable |

Direction agrees with agronomy: saxaul favors gypsiferous/carbonate, sandy,
low-chloride, low-sodium substrates. Magnitudes are honest — small positive n
limits significance; Phase 6/7 will quantify model skill with spatial CV.

## Limitations carried to Phase 7

- 6 positives is small; correlations are screening hints, not confirmed effects.
- Temporal mismatch: soil 2012-2014 vs S2 2025.
- 1 positive (13/14) and several rows lack 10 m RS; soil model is the fallback
  where RS is absent.
