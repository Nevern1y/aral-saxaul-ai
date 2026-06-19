# AUDIT-B — Scientific Red-Team of the V6 Saxaul Suitability Layer

**Scope:** read-only audit of logic, leakage, and statistical assumptions. No code or model was modified.
**Repo root:** `F:\OPENCODE PROJECTS\aral-saxaul-ai`
**Date:** 2026-06-19
**Method:** read README/CLAUDE, all `data/canonical/*.md` QA reports, the calibrated-threshold and model JSON artifacts, and every `scripts/v6/*.py` plus `scripts/run_v6_pipeline.py`. Every weakness below cites a concrete file:line or report figure.

---

## Framing — what is genuinely strong (so we critique the right things)

The V6 layer is unusually honest, and three design choices pre-empt the most common fake-accuracy traps. The audit credits them so the critique stays focused on what actually limits accuracy:

- **No label circularity.** Saxaul labels are derived from field *text* (morphology + DOCX narrative), and hard negatives are explicitly restricted to documented plantation failures and barren ground, **not** saline solonchaks — so soil chemistry is *learned*, not baked into the target (`scripts/v6/extract_saxaul_labels.py:49-55`). The salinity model is then fit on measured topsoil salts, independent of saxaul labels (`scripts/v6/train_suitability_model.py:152`).
- **No free-fit on 6 positives.** The direct NDMI+MSAVI→saxaul logit is demoted to "exploratory, NOT for decisions" because its LOO AUC ≈ chance; suitability is instead derived as `1 − P(saline)` to borrow the n=70 salinity signal (`scripts/v6/train_suitability_model.py:4-19, 189-201`).
- **Honest metrics.** Bootstrap CIs are computed on held-out LOO scores (not refit in-sample), with the in-sample-bootstrap bug explicitly avoided (`scripts/v6/spatial_validation.py:221-238`), and a red-team QA script enforces CI/polarity/coverage invariants (`scripts/v6/qa_science_audit_v6.py`).

The weaknesses below are therefore mostly about **small n, domain transfer, and silent misuse downstream** — not about leakage or dishonest reporting in the current artifacts.

---

## Ranked weaknesses

### W1 — Cross-region calibration drift: pooled spatial AUC 0.385 vs per-block 0.792 — SEVERITY: HIGH

- **Evidence:** `outputs/data/spatial_validation_v6.json` / `data/canonical/SPATIAL_VALIDATION_V6_QA.md:6,10`: `spatial_lbo_pooled_auc = 0.385`, `spatial_lbo_perblock_auc = 0.792`, `pooled_minus_perblock = -0.407`. Computed at `scripts/v6/spatial_validation.py:206-207`.
- **Why it hurts accuracy:** the salinity logit carries a single global intercept β₀ (`train_suitability_model.py:145-146`). Block base rates range from 8/8 saline to 1/10 (`SPATIAL_VALIDATION_V6_QA.md:10`), so predictions calibrated on one region's baseline mis-rank pixels in another. Discrimination is locally stable but the model **cannot rank suitability across regions** — pooled AUC is below chance.
- **Risk if ignored:** any consumer who aggregates the continuous `suitability_index_v6.tif` across regions (e.g., "is Ustyurt better than the seabed floor?") gets a near-random ranking while believing the per-block 0.79 applies. This is the single largest accuracy hazard, and it is a *usage-scope* hazard, not a model bug.
- **Honest fix:** ship the per-block-vs-pooled gap as a first-class scope statement in the README and as GeoTIFF metadata (`regional_calibration=true`, `pooled_auc=0.385`); for genuine cross-region ranking, add region/base-rate covariates or a random intercept (mixed-effects / hierarchical logit) and re-measure pooled AUC.

### W2 — Saxaul-suitability skill is weak and unstable: AUC 0.647, 95% CI [0.435, 0.827] — SEVERITY: HIGH

- **Evidence:** `data/canonical/SPATIAL_VALIDATION_V6_QA.md:13`; `spatial_validation.py:247-257`. n=56, **6 positives**. CI lower bound 0.435 is below chance.
- **Why it hurts accuracy:** with 6 positives, each positive moves AUC by ≈1/(6·50)·… on the order of 0.03 per rank swap; the CI width (0.39) is an order of magnitude larger, so the *sign* of the suitability gradient is not secured out-of-sample.
- **Risk if ignored:** a downstream threshold ("suitable if P>0.3") can invert under minor label noise; presenting 0.647 without the CI overstates confidence.
- **Honest fix:** keep reporting the CI everywhere the point estimate appears; treat saxaul suitability as a screening aid only; prioritize collecting more presence labels (target ≥30 positives) before any standalone saxaul classifier is shipped. The AralField independent check (AUC 0.5, n=11, CI 0.111–0.889; `SPATIAL_VALIDATION_V6_QA.md:17`) confirms the external signal is currently too thin to validate against.

### W3 — Univariate NDMI salinity model ignores known soil-texture modifiers — SEVERITY: HIGH

- **Evidence:** model has one predictor (`train_suitability_model.py:42-43,128-146`, `salinity_v6_logit.json` predictor `rs30_ndmi`). Yet calibration shows texture/chemistry carry independent signal: CaCO₃ AUC 0.855, chloride 0.733, sand 0.695, exch. Na 0.692 (`data/canonical/thresholds_v6_calibrated.json:164-183,144-163`).
- **Why it hurts accuracy:** the NDMI→salinity coefficient is a population average. Sandy, well-leached pixels with high NDMI are over-predicted as saline; clay pixels that retain salts are under-predicted. A single axis cannot separate these regimes, producing texture-correlated residual bias.
- **Risk if ignored:** systematic, spatially-clustered error in fine-textured or sandy zones — invisible in a pooled AUC but damaging to local planting decisions.
- **Honest fix:** add sand%/clay% (and optionally Cl⁻) as covariates or interactions and re-validate with the same LOO+spatial-block protocol; report AUC stratified by texture class so the bias is quantified, not assumed away. Note the owner permits adding out-of-AOI points (see W4) which would also raise n for a multivariate fit.

### W4 — Geographic domain shift: 54/70 training pits lie OUTSIDE the mapped AOI — SEVERITY: HIGH

- **Evidence:** `scripts/v6/build_suitability_index.py:359-360` — "54 of 70 pits lie OUTSIDE the 1960 Aral footprint … sampled the wider Priaralye, not just the seabed". The model trains on all 70 but is applied to the seabed AOI.
- **Why it hurts accuracy:** β₀/β₁ are dominated (77%) by Priaralye soils whose NDMI↔salinity baseline may differ from the exposed seabed; this is textbook covariate/label shift from train to deploy domain.
- **Risk if ignored:** the entire seabed map can be uniformly biased (over- or under-estimating salinity) with no in-AOI check to catch it. The owner explicitly permits adding out-of-AOI points, which makes **weighting/stratification mandatory** — adding them raw deepens the shift.
- **Honest fix:** (a) report AUC for the 16 in-AOI pits separately from the 54 out-of-AOI; (b) when ingesting more out-of-AOI points, stratify by region and either weight toward in-AOI density or fit a region-aware intercept (ties to W1); (c) state explicitly that current validation is partly in-sample on out-of-AOI training pits.

### W5 — Ground-truth "coverage parity" conflates training pits with test pits — SEVERITY: MEDIUM

- **Evidence:** QA check "V6 covers >= as many GT pits as V5" (`scripts/v6/qa_science_audit_v6.py:136-138`) operates over all 70 pits; coverage parity reported as V5=13 / V6=15 (`build_suitability_index.py:355-358`, `qa_science_audit_v6.py:162`). 54 of those pits are training data (W4).
- **Why it hurts accuracy (reporting):** validating on pits that fed β₀/β₁ is circular; the headline "parity" figure mixes train and test and can mask overfitting.
- **Risk if ignored:** readers infer independent validation where little exists.
- **Honest fix:** split the pit-validation summary into in-AOI (true test) vs out-of-AOI (training) AUC/sensitivity, and add an audit check that in-AOI skill ≥ a stated floor.

### W6 — Youden thresholds tuned in-sample on 6 positives; instability not carried into the artifact — SEVERITY: MEDIUM

- **Evidence:** thresholds are Youden-optimal cuts chosen on the same data, no held-out (`scripts/v6/calibrate_thresholds.py:63-105`). Saxaul-label cuts all have n_pos=6 with modest specificity, e.g. NDMI cut spec 0.56, MSAVI spec 0.38 (`thresholds_v6_calibrated.json:94-133`). The "screening aid, not classifier" caveat exists in the report (`calibrate_thresholds.py:205-207`) but **not inside `thresholds_v6_calibrated.json`**.
- **Why it hurts accuracy:** moving a single positive shifts Youden J by ≈1/6−1/50 ≈ 0.16; cuts are over-fit to these exact pits.
- **Risk if ignored:** a tool consuming the JSON gets point cuts with no stability/n_pos warning and treats them as production thresholds.
- **Honest fix:** add `n_pos` and a `stability="low (n_pos≤6)"` flag to each entry in `calibrated_thresholds`; prefer cuts validated by nested CV when n permits.

### W7 — "Oriented AUC" can present a weak inverse predictor as strong — SEVERITY: MEDIUM

- **Evidence:** for `rs30_msavi__soil_salinity>1.0%` the raw rank-AUC is 0.216 but the published `auc` is 0.784 (`thresholds_v6_calibrated.json:74-83,265-275`; inversion logic `calibrate_thresholds.py:69-90`). The `calibrated_thresholds` summary block omits `auc_raw`.
- **Why it hurts accuracy (interpretation):** orientation flipping is statistically correct, but a dashboard ranking predictors by the summary `auc` field will rank an inverted proxy alongside genuinely strong direct predictors.
- **Risk if ignored:** mis-prioritization of predictors in any future feature-selection step.
- **Honest fix:** carry `auc_raw` and `direction` into the `calibrated_thresholds` entries (they already exist in `all_evaluations`), and label inverted predictors explicitly.

### W8 — Soil-feature thresholds rest on n_pos = 3–5 — SEVERITY: MEDIUM

- **Evidence:** CaCO₃ cut uses n_pos=3 (AUC 0.855, `thresholds_v6_calibrated.json:164-173`); sand n_pos=5 (`:174-183`). `MIN_N=12` gates total n but there is no minimum on positives (`calibrate_thresholds.py:45,99-100,143`).
- **Why it hurts accuracy:** with 3 positives the TPR axis has ≈4 attainable values; the "high" AUC is barely powered and fragile to one relabel.
- **Risk if ignored:** impressive-looking soil thresholds (CaCO₃ 0.86) are quoted as if robust.
- **Honest fix:** add an explicit `MIN_N_POS` (e.g. ≥8) gate before publishing a cut, or mark sub-threshold entries as "indicative only".

### W9 — Spatial-CV refits use a hardcoded λ=0.1, not the tuned λ from training — SEVERITY: MEDIUM

- **Evidence:** `scripts/v6/spatial_validation.py:202` refits each block with `L2` fixed; the constant is set to match the committed model rather than re-selected. By contrast the training script tunes λ by LOO over a grid `[0.1,0.3,1,3,10]` (`train_suitability_model.py:43,101-114`).
- **Why it hurts accuracy (measurement):** the spatial AUC is reported for one regularization strength that was not validated per-fold; for small/odd-base-rate blocks λ=0.1 may over-shrink the intercept, understating achievable local skill (and interacting with the W1 drift story).
- **Risk if ignored:** the spatial-CV numbers are taken as the model's ceiling when they are λ-specific.
- **Honest fix:** either re-select λ inside each spatial fold (nested) or report a λ-sensitivity sweep alongside the headline spatial AUC.

### W10 — Temporal mismatch: 2012–2014 soil chemistry vs a recent NDMI composite — SEVERITY: MEDIUM

- **Evidence:** labels/soil derive from the 2012–2014 Pachikin/Kozybaeva survey (`extract_saxaul_labels.py:73`); the QA notes the NDMI composite is recent and the relation is treated as quasi-stationary (`data/canonical/SUITABILITY_INDEX_V6_QA.md`, temporal caveat enforced by `qa_science_audit_v6.py:161`).
- **Why it hurts accuracy:** over 6–15 years on an active seabed, vegetation and salt distributions shift; ρ(NDMI,salt)=0.66 measured against decade-old soils may overstate the present-day link.
- **Risk if ignored:** the model's effective AUC today may be below the reported figures, with no monitoring.
- **Honest fix:** re-sample a subset of pits for current salinity to estimate drift; surface the calibration vintage on the dashboard; consider a date-matched NDMI composite where possible.

### W11 — 30 m stack limitations: no SCL water band, weak slope/TWI coverage — SEVERITY: MEDIUM

- **Evidence:** stats note the 30 m stack has no Sentinel-2 SCL band, so V5's SCL water masking is not reproduced and wet saline playa surfaces as high-NDMI strong-salinity (`build_suitability_index.py:300-304`); slope-band gap documented and enforced (`qa_science_audit_v6.py:158`). Water is approximated by `ndwi > 0` (`build_suitability_index.py:228`).
- **Why it hurts accuracy:** an NDWI-only water gate will misclassify thin-water/wet-playa edges, and absent slope/TWI removes terrain controls on salt accumulation from the bare-land logic.
- **Risk if ignored:** systematic mislabeling of wet saline margins and loss of a physically-relevant predictor.
- **Honest fix:** source an SCL or alternative water mask for the 30 m grid; restore a slope/TWI band where coverage allows, and validate the bare/water gate against known playa polygons.

### W12 — 10 m cascade frozen on a non-significant n=14 basis (latent obsolescence) — SEVERITY: LOW

- **Evidence:** the V5.1 10 m thresholds are kept frozen because only 14/70 pits have valid 10 m NDMI, the 10 m NDMI↔salinity link is non-significant (rho≈0.42, p≈0.13), and 10 m vs 30 m NDMI even anti-correlate (r≈−0.35) at NoData-edge pixels (`calibrate_thresholds.py:10-16`).
- **Why it hurts accuracy:** the decision is defensible *now*, but the 14 points sit on NoData edges (anomalous NDMI), so non-significance is partly an artifact, not proof of independence; the frozen thresholds may be wrong yet are shipped unchanged.
- **Risk if ignored:** if better 10 m coverage later shows a real link, the frozen cascade is silently stale.
- **Honest fix:** record this as a known-limitation with a re-evaluation trigger ("re-test 10 m calibration when valid-pixel n>30"); keep the golden snapshot but flag it as provisional.

### W13 — Corrupt-tile workaround uses hardcoded tile paths and column offsets — SEVERITY: LOW

- **Evidence:** the on-disk VRT mosaics over a corrupt original tile (TIFFReadEncodedTile failure), so the code reads two "good" tiles with hardcoded `col0` offsets (`scripts/v6/build_suitability_index.py:63-75`, and the parallel list in `spatial_validation.py`). A column-coverage warning exists (`build_suitability_index.py:277-279`) but no checksum.
- **Why it hurts accuracy/reproducibility:** if tiles are moved/renamed the offsets silently address the wrong columns; the workaround is not surfaced in `run_v6_pipeline.py` or README.
- **Risk if ignored:** a future rebuild produces a subtly mis-registered map with no hard error.
- **Honest fix:** regenerate a clean VRT, or pin tile identity with SHA256 checksums and assert total covered columns == grid width.

---

## Explicit "do NOT do" list (anti-patterns that would manufacture fake accuracy)

1. **Do NOT fit a standalone saxaul classifier on the 6 positives and report its in-sample AUC.** The exploratory logit already shows LOO ≈ chance (`train_suitability_model.py:4-6`); presenting in-sample skill would be fabricated accuracy.
2. **Do NOT bake soil chemistry / salinity into the saxaul labels.** Hard negatives must stay restricted to documented failures and barren ground, never solonchaks (`extract_saxaul_labels.py:51-55`); doing so makes suitability=1−P(saline) circular against its own labels.
3. **Do NOT reuse the saxaul labels to also tune the salinity model.** Independence (`train_suitability_model.py:152`) is what lets the label AUC count as validation; tuning on it collapses train/test.
4. **Do NOT drop or hide the confidence intervals.** Especially the 0.435–0.827 saxaul CI and the 0.111–0.889 AralField CI — quoting bare point estimates overstates skill at these n.
5. **Do NOT quote the pooled-across-regions story using the per-block AUC, or vice-versa.** Report both (0.385 pooled, 0.792 per-block) and state which decision each supports (`SPATIAL_VALIDATION_V6_QA.md:9-10`).
6. **Do NOT add out-of-AOI points without stratification/weighting.** Raw addition deepens the existing 54/70 domain shift (W4); region must be modeled or weighted.
7. **Do NOT pick Youden cuts on all the data and present them as validated thresholds.** In-sample Youden on n_pos≤6 is optimistic (W6); use held-out evaluation or flag as indicative.
8. **Do NOT report "oriented" AUC without the raw AUC and direction.** A 0.216→0.784 flip must be transparent (W7).
9. **Do NOT count out-of-AOI training pits as independent validation coverage.** Separate in-AOI test skill from training coverage (W5).
10. **Do NOT bootstrap a refit model's in-sample AUC.** Resample (score,label) pairs on held-out LOO predictions only — the existing correct procedure (`spatial_validation.py:221-238`).

---

## Severity tally

| Severity | Count | IDs |
|----------|-------|-----|
| HIGH | 4 | W1, W2, W3, W4 |
| MEDIUM | 7 | W5, W6, W7, W8, W9, W10, W11 |
| LOW | 2 | W12, W13 |
| **Total** | **13** | |

**Bottom line:** the V6 layer is honest and largely leakage-free; its accuracy is capped not by dishonesty but by small n (6 saxaul positives, 16 in-AOI pits), a single-axis salinity model, temporal drift, and — most importantly — **regional calibration drift** that makes cross-region suitability ranking unreliable (pooled AUC 0.385). The highest-value honest fixes are: model region/base-rate explicitly (W1), add soil-texture covariates (W3), report in-AOI vs out-of-AOI skill separately (W4/W5), and gather more saxaul ground truth (W2).
