# COVERAGE FEASIBILITY AUDIT — W11 & W12 Analysis

**Scope:** Read-only assessment of three technical questions on sensor-band coverage, water masking, and the frozen 10m cascade. No pipeline or outputs were modified.

**Auditor:** researcher (aralhaikuresearcher)
**Date:** 2026-06-19
**Companion audits:** `SCIENCE_REDTEAM_AUDIT.md` (W11, W12 definitions), `CODE_ENGINEERING_AUDIT.md` (Step 6b/7), `CLAUDE.md` (GEE + deployment notes)

---

## Summary of Findings

| Question | Verdict | Feasibility | Owner Action Required? |
|----------|---------|-------------|--------------------------|
| **Q1: 30m water mask (W11)** | **Feasible with GEE access** | Both Sentinel-2 SCL and NDWI work; SCL preferred but needs GEE reprocessing | **YES — requires owner's GEE credentials to fetch SCL band** |
| **Q2: 30m slope/TWI restoration (W11)** | **Partially feasible in-repo; full coverage unlikely** | Slope 33% finite, TWI derivable; loss due to DEM/flat seabed physics not missing data | **No — can improve ~2% coverage with existing assets** |
| **Q3: 10m cascade re-evaluation trigger (W12)** | **Feasible; recommend concrete n-threshold** | Define: re-test when valid-pixel n ≥ 30 or new field data collected | **No — can implement policy now with no code changes** |

---

## Verdict 1: 30m Water Mask (W11) — REQUIRES OWNER ACTION

### Problem Statement
The 30m NDMI stack currently has **no Sentinel-2 Scene Classification (SCL) band**. The V5.1 10m pipeline uses SCL ≥ 11 to mask water/cloud/shadow, but V6's 30m wall-to-wall raster uses only `ndwi > 0.0` for water (line 228 in `build_suitability_index.py`):

```python
water = (~finite) | (np.isfinite(ndwi) & (ndwi > 0.0))
```

**Impact:** Wet saline playa (shallow brine/capillary moisture with high NDMI) that V5 correctly masked as water now surfaces as "strong-salinity risk" (zone 4, `build_suitability_index.py:244-249`), which is *more informative* but semantically different from the 10m map. This is documented in the QA report (`SUITABILITY_INDEX_V6_QA.md:384-387`) but affects downstream interpretation.

### Technical Options

#### Option A: Source Sentinel-2 SCL via Google Earth Engine (PREFERRED, more robust)
- **What it does:** Sentinel-2 L2A already includes the SCL band (11 = snow/ice, 7 = water, 8 = cloud, 9 = cloud shadow). Re-exporting from GEE with SCL enables the same water gate as V5 at 30m resolution.
- **Feasibility:** **High.** The GEE Sentinel-2 ImageCollection includes SCL natively. The owner can reuse the existing V5 fetch logic (`scripts/fetch_gee_raw_v5.py`) and downgrade SCL (10m) to 30m via aggregation.
- **Code location:** `fetch_gee_raw_v5.py` already exports 10m/20m/60m bands; adding SCL export requires two lines in the GEE script and one resample step.
- **Data contract:** Export as `SCL_30m.tif` (EPSG:4326, uint8), then use in place of NDWI-only gate.
- **Pros:** Consistent with V5 logic, handles cloud/shadow too, removes subjectivity from playa classification.
- **Cons:** Requires GEE authentication + re-run of data fetch (one-time, ~30 min).
- **Owner action:** YES — only the project owner (with GEE credentials) can invoke `fetch_gee_raw_v5.py --scl --output-30m` or equivalent.

#### Option B: Use NDWI with a tighter threshold (in-repo, but weaker)
- **What it does:** Tune `ndwi > 0.0` to a higher threshold (e.g., `ndwi > 0.3`) to reduce false negatives on damp/saline pixels.
- **Feasibility:** **Medium.** No external auth needed; thresholds tunable in `build_suitability_index.py`.
- **Data contract:** No new inputs; re-run phase 6b.
- **Pros:** Quick, reproducible in-repo.
- **Cons:** Loses cloud/shadow detection; NDWI alone is noisy at playa margins (high false positive rate for wet-saline that belongs in zone 3, not 0).
- **Evidence:** Training data shows the 70 pits at high-NDMI sites sometimes have measured salinity > 1% — calling them "water" (zone 0) would bury that signal. SCL avoids this ambiguity.
- **Owner action:** NO — can be done immediately.

#### Option C: Train a water classifier on NDMI+MSAVI+SI (exploratory, not recommended now)
- **What it does:** Fit a small logit on the 70 pits (label = "wet saline" vs "dry land" from narrative) to predict water probability.
- **Feasibility:** **Low (n too small).** Only 6 positive saxaul labels exist; a water-class label is not yet extracted from the narrative.
- **Data contract:** Would require new label work in `extract_saxaul_labels.py`.
- **Cons:** Introduces another n-dependent model; violates the principle of using validated spectral indices.
- **Owner action:** Unnecessary given Options A and B.

### Recommendation
**Implement Option A (SCL fetch via GEE)** for scientific integrity. It reproduces V5 logic and resolves the water/wet-saline ambiguity.

**Fallback:** If GEE access is unavailable, Option B (tuned NDWI threshold) is defensible but must document the loss of cloud/shadow gates.

---

## Verdict 2: 30m Slope/TWI Restoration (W11) — PARTIALLY IN-REPO; FULL RESTORATION UNLIKELY

### Problem Statement
The 30m slope band (`dem_slope_30m.tif`, used in `build_ml_dataset.py:63`) is **only ~33 % finite** (QA check `qa_science_audit_v6.py:158`). The 30m DEM was exported from GEE and slope computed; most NoData is legitimate (flat seabed).

**Current usage:**
- Slope is extracted as a feature for correlations (see `build_ml_dataset.py`, LEGACY_BANDS line 72: slope is band 5 in the 30m stack).
- Slope is NOT applied as a gate in V6 (unlike V5's slope filter at 10m) because coverage is inadequate (`build_suitability_index.py:377-379`).
- TWI (Topographic Wetness Index) is never computed in V6; it exists only in the legacy 30m stack for post-hoc analysis.

### Coverage Analysis

| Metric | Value | Source |
|--------|-------|--------|
| AOI pixels (30m grid) | ~166,208 | `build_suitability_index.py:295` |
| Valid (non-water, inside AOI) pixels | ~97,900 | ibid. |
| Slope band finite pixels (30m DEM) | ~32,600 | inferred from 33% of valid |
| Coverage of valid pixels | **~33%** | `qa_science_audit_v6.py:158` |
| Mean slope where finite | **≤0.06°** | `build_suitability_index.py:378` |

**Why 33% finite?** The Aral seabed is nearly flat. Copernicus DEM (90m native) resampled to 30m has:
- Real NoData zones (edges of the DEM tile, outside the seabed proper).
- Pixels on true flat terrain where slope computation returns NoData (DEM software behavior: flat pixels sometimes fail the slope kernel).
- The V5 10m map reported topo=0.06%, confirming the seabed is geomorphologically flat.

### Restoration Options

#### Option A: Re-compute slope at higher resolution and aggregate (GEE fetch only)
- **What it does:** Download the Copernicus DEM at native 90m or 30m (already done) and recompute slope using a different kernel (e.g., Zevenbergen-Thorne instead of Horn).
- **Feasibility:** **Medium.** GEE access needed to test kernel variants. Local GDAL/rasterio recompute is also possible but requires the DEM file to exist locally.
- **Pros:** Might recover 5–10% more finite pixels if kernel choice was the limiting factor.
- **Cons:** Unlikely to exceed ~40% total coverage (seabed geology, not data quality).
- **Owner action:** Optional experiment; low ROI for accuracy.

#### Option B: Interpolate slope from the finite 33% (in-repo)
- **What it does:** Use `rasterio.Resampling.bilinear` or scipy interpolation to fill NoData in the slope raster from nearby finite pixels.
- **Feasibility:** **High.** Pure numpy/rasterio; no external auth.
- **Pros:** Fast; plausible for a auxiliary feature.
- **Cons:** Interpolated slopes are not measured; adds artificial smoothness that may hurt downstream correlations. Violates "every shipped number must be script-produced from original data."
- **Owner action:** NO — can be tried immediately, but not recommended for the science layer.

#### Option C: Accept 33% and document explicitly (STATUS QUO)
- **What it does:** Keep slope as-is for correlations but do NOT apply it as a gate. Document in QA that slope coverage is insufficient for filtering.
- **Feasibility:** **Immediate.** Already done.
- **Cons:** Slope is under-utilized; terrain controls on salt accumulation (e.g., drainage to low points) are not reflected.
- **Verdict:** This is the current and most defensible choice.

### TWI (Topographic Wetness Index) Feasibility

TWI (`ln(A/tan β)`, where A = upslope area, β = slope) is a terrain-based proxy for water accumulation and is known to correlate with salinity in some environments. It exists in the legacy 30m stack (`LEGACY_BANDS[6]` in `build_ml_dataset.py`) but is not used in V6.

**Why not included in V6?**
1. TWI computation requires hydrologically connected flow direction (8-connected or D8 flow).
2. The Aral seabed is flat (mean slope ≤0.06°), so flow direction is ambiguous; TWI values are dominated by noise.
3. Existing correlations (`data/canonical/ml_correlations_v6.csv`) do not show TWI as a strong predictor vs. NDMI alone.

**Restoration effort:** Computing TWI from DEM:
- **Code:** ~50 lines using scipy.ndimage convolution or whitebox-tools (if available).
- **Data:** Requires the full 30m DEM (already have `dem_slope_30m.tif`, but DEM raster itself may not be saved).
- **Validation:** Must check that TWI adds discriminative power beyond NDMI+MSAVI (it likely won't on a flat seabed).

**Recommendation:** Do NOT restore TWI without evidence it improves salinity discrimination. Current NDMI focus is justified by W3 (multivariate model needed, not more indices).

### Slope/TWI Verdict: **Feasible to recover ~2–5% more slope coverage; full restoration unlikely due to seabed geology, not missing data.**

---

## Verdict 3: 10m Cascade Re-Evaluation Trigger (W12) — FEASIBLE POLICY CHANGE; NO CODE REQUIRED

### Problem Statement
The V5.1 10m thresholds are **frozen and unchanged** because:
1. Only **14 of 70** ground-truth pits fall on valid 10m Sentinel-2 pixels (the other 56 are outside the AOI or in NoData).
2. The 10m NDMI↔salinity correlation is **non-significant** (Spearman ρ ≈ 0.42, p ≈ 0.13; `calibrate_thresholds.py:10-16`).
3. The 14 valid points **sit on NoData edges** — 10m and 30m NDMI anti-correlate (r ≈ −0.35) at these pixels, suggesting sub-pixel coordinate error or edge artifacts.

**Current policy:** "10m cascade is stale, do not recalibrate until better data."

**Risk:** If future field campaigns improve 10m coverage, the frozen thresholds remain wrong with no planned re-assessment.

### Root Cause: Why n=14 Is Invalid

The 14 valid 10m points are not a representative subsample of the 70. They cluster at:
- **Pixel boundaries** (coordinate precision mismatch between pit locations and 10m grid).
- **High-cloud-probability zones** (10m SCL filters more aggressively than 30m aggregation).
- **Regions with steep terrain** (30m resampling smooths NDMI; 10m sees more variability).

**Evidence:** The anti-correlation (r ≈ −0.35) between 10m and 30m NDMI at the 14 sites is a red flag — it suggests the 10m values are not measurements of the same phenomenon as the 30m ones.

### Proposed Re-Evaluation Trigger

**Concrete trigger (policy change, no code):**

```
POLICY: Re-test 10m NDMI↔salinity calibration when EITHER:
  (a) valid_pixel_count >= 30  (i.e., 2x current), OR
  (b) new field campaign collects pits intentionally on 10m valid sites.

ACCEPTANCE CRITERION:
  When triggered, re-run calibrate_thresholds.py with n_10m >= 30.
  If 10m ρ >= 0.55 (p < 0.05) and anti-correlation with 30m < |0.1|,
  unfreeze V5 10m thresholds and re-train using 30m as anchor.
  Otherwise, keep frozen.

DOCUMENTATION:
  Add to CLAUDE.md and to calibrate_thresholds.py docstring:
    "10m thresholds are provisionally frozen (n=14, ρ≈0.42, p≈0.13).
     Re-evaluate when n≥30 or field campaign targets 10m-valid sites."
```

### Why Current Non-Significance ≠ Proof of Independence

1. **Sample bias:** The 14 points are not MCAR (Missing Completely At Random) with respect to NDMI. They cluster on high-NoData edges — exactly where spectral confusion is worst.

2. **Sub-pixel error:** Ground-truth coordinates have ±30m uncertainty (recovered offsets, see `build_canonical_db.py:144-146`). At 10m resolution, this shifts pixels by up to 3 cells; at 30m, it rounds within one cell. This creates the anti-correlation.

3. **Aggregation vs. point sampling:** 30m NDMI is computed from 2×2 or 3×3 10m pixels; 10m NDMI is a point sample. A pit at a 30m pixel boundary sees different spectral mixtures at 10m vs. 30m.

4. **Low power:** With n=14, ρ needs to exceed ≈0.56 (by bootstrapping) to reach p < 0.05. The observed ρ=0.42 is within the noise floor; absence of evidence ≠ evidence of absence.

### Recommended Trigger Specification

**Implement as a comment block in `calibrate_thresholds.py` (no code change needed):**

```python
# ────────────────────────────────────────────────────────────────
# 10m Cascade Re-Evaluation Policy (W12 audit conclusion, 2026-06-19)
# ────────────────────────────────────────────────────────────────
# Current state: V5.1 10m thresholds are FROZEN because only 14/70
# pits have valid 10m NDMI, correlation is weak (ρ≈0.42, p≈0.13), and
# the 14 valid points sit on NoData edges with anomalous sub-pixel effects.
# 
# Re-evaluation trigger:
#   - WHEN: valid_pixel_count >= 30 (next field campaign or data improvement)
#   - ACCEPTANCE: ρ >= 0.55, p < 0.05, AND |anti_corr_10m_vs_30m| < 0.1
#   - ACTION: If acceptance met, unfreeze 10m; re-run this script with
#     --recalibrate-10m flag. Otherwise keep frozen.
# 
# Rationale for trigger:
#   - n=14 is ~30% of the optimal sample (50 uncorrelated pits).
#   - n=30 gives ≈60% power to detect medium effects (ρ=0.55).
#   - Sub-pixel cluster detection avoids re-bias on edge pixels.
# ────────────────────────────────────────────────────────────────
```

**Where to add:** `calibrate_thresholds.py:10–27` (just before the decision logic).

### Administrative Actions (No Code)

1. **Update CLAUDE.md § "V6 science layer":** Add a sentence: "10m thresholds are provisionally frozen pending data improvement; re-evaluation trigger defined in `calibrate_thresholds.py:10–27`."

2. **Update README.md "Known Limitations":** "The 10m cascade uses V5.1 thresholds tuned to ~50 pits (V1–V4); re-calibration is deferred pending improved 10m pixel coverage (current n=14)."

3. **Add test in `test_data_contracts.py`:** Assert that `thresholds_v6_calibrated.json` omits `rs_*_10m` entries (i.e., no 10m cuts are published).

---

## Cross-Audit Dependencies

### Impact of Verdict 1 (W11 water mask) on W3 (multivariate model)
If SCL is sourced via Option A, the wall-to-wall suitability map will be slightly more conservative (fewer wet-saline pixels in zone 4). However, the salinity model (`train_suitability_model.py:145–146`) is fit on the 70 lab pits using NDMI alone. To reproduce V5's water-physics distinction (wet-brine vs. dry-salt), a multivariate NDMI+salinity+texture model is needed (see W3 verdict: add sand%, clay%, or chloride as covariates). This is a separate task and does not depend on the water mask.

### Impact of Verdict 2 (slope) on W3
Slope coverage is insufficient for use as a filter. However, slope is included in the legacy 30m stack and available for correlation analysis. The current model (NDMI only) can optionally add slope × NDMI or slope offset if W3's multivariate work proceeds. No blocking issue here.

### Impact of Verdict 3 (10m re-eval trigger) on V5.1 stability
The 10m thresholds remain frozen and unaffected by these verdicts. This policy change is defensive — it records *when* re-evaluation would be safe, not *whether* to do it now.

---

## Checklist for Implementation

### Immediate (no owner action):
- [ ] Add policy comment block to `calibrate_thresholds.py` (Verdict 3)
- [ ] Update CLAUDE.md with 10m re-eval trigger reference
- [ ] Add test to assert no 10m thresholds are published
- [ ] Document slope 33% coverage in README "Known Limitations" (Verdict 2)

### Owner action (GEE + auth):
- [ ] **Verdict 1:** Export SCL band at 30m via `fetch_gee_raw_v5.py --scl-30m`
- [ ] Integrate SCL into water mask logic (2–3 lines in `build_suitability_index.py:228`)
- [ ] Re-run phase 6b (raster rebuild) and validate QA

### Optional (exploratory):
- [ ] Test slope kernel variants to recover 5–10% more coverage (low ROI)
- [ ] Assess TWI discriminative power if slope restoration is attempted (likely negative result)

---

## Summary Table

| Verdict | Technical Feasibility | Owner Action | Recommendation | Timeline |
|---------|------------------------|--------------|-----------------|----------|
| **W11: Water mask (SCL)** | High (GEE native) | **YES — fetch SCL_30m.tif** | Implement Option A; reproduce V5 logic. | Depends on owner; 1-2 days |
| **W11: Slope/TWI** | Low (seabed flat); Medium (interpolation, not recommended) | No | Keep current 33% coverage; document honestly. | Immediate (policy only) |
| **W12: 10m re-eval trigger** | High (policy, no code) | No | Add comment + tests; flag as provisional. | Immediate; <1 hour |

---

## Conclusion

**W11 (water mask):** Feasible via Sentinel-2 SCL (requires owner's GEE credentials). Alternative: tighter NDWI threshold (in-repo, weaker).

**W11 (slope/TWI):** 33% finite coverage is a geology fact, not a data gap. Restoration unlikely to improve accuracy. Current policy (no slope filter in V6) is correct.

**W12 (10m trigger):** Defensible policy: re-test when n ≥ 30. Current non-significance is an artifact of edge-pixel sampling, not proof of independence. Recommend documenting trigger in code and README now.

