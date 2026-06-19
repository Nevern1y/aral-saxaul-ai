# VALIDATION: 2020/2021 Out-of-Period Feasibility Assessment

**Scope:** Evaluation of whether the 2020/2021 re-survey field tables can serve as an independent out-of-period validation set for the V6 salinity model, and assessment of saxaul label recovery from the 2012–2014 campaign.

**Date of audit:** 2026-06-19  
**Auditor role:** research / data-mining (read-only)  
**Primary deliverable:** this file.

---

## EXECUTIVE SUMMARY

**Verdict: 2020/2021 data CANNOT be immediately wired into the V6 validation pipeline; however, it has modest potential as an out-of-period validation set if coordinates are recovered.**

- ✅ **Can be used as OUT-OF-PERIOD validation** — the three 2020/2021 CSVs contain soil chemistry and granulometry measurements (EC/TDS/pH/humus + granulometry) that are conceptually comparable to the 2012–2014 basis (*different* campaign, *same* measurement types).
- ⚠️ **MAJOR BLOCKER: No coordinates.** None of the three 2020/2021 files contain latitude/longitude or location identifiers that can be matched to the mapped AOI or existing pit records. Without coordinates, the data cannot be spatially validated.
- ⚠️ **Sample count is modest.** The 2020/2021 dataset spans ~21 pits (with per-layer depth profiles), which is sufficient for a weak out-of-period check but underpowered for strong validation (n=21 << target n≥30 for model stability).
- ❌ **Saxaul labels: NO new recoverable evidence.** All saxaul mentions in the 2012–2014 morphology/narrative are already captured in `saxaul_labels_v6.csv`. The 2020/2021 data has no vegetation/saxaul description — it is purely chemistry + granulometry.

**If coordinates are recovered (e.g. via geocoding locality text), the 2020/2021 data would support a weak, regional out-of-period check on salinity calibration; this requires explicit provenance flagging and does NOT expand the saxaul-label ceiling.**

---

## 1. Data Inventory & Structure

### 1.1 2020/2021 Field Files (Repo Root)

| File | Rows | Columns | Key measurements | Coverage |
|------|------|---------|------------------|----------|
| `Результаты EC,TDS, pH, в.в., гумус(в.в.csv` | 111 | ~30 | Per-layer water-extract salts (EC,TDS,HCO₃,Cl,SO₄,Ca,Mg,Na,K), salinity%, humus | ~10 pits, multi-layer (5–12 layers per pit) |
| `результаты Арал(Лист2).csv` | 104 | 4 | Humus%, CO₂, pH per layer | ~11 pits, multi-layer (4–10 layers per pit) |
| `мехсостав 2020, 2021 по типам(Лист4).csv` | 42 | ~7 | Sand%, Silt%, Clay%, Soil texture class | ~21 pits (single-line summary) + 11 soil-type reference rows |

**Total unique pits:** Approximately 21 pit samples (labelled "Разрез N" or "разреза N" in Russian, meaning "pit" or "profile").

### 1.2 Pit Identification

Pit labelling in 2020/2021 files:
- **Результаты EC/TDS/pH:** Pits labelled as "Разрез 1" through "Разрез 9", with per-layer depth sampling (0–150+ cm depths).
- **результаты Арал(Лист2):** Pits labelled as "01/20А", "02/20А", …, "11/20А", "20/21" style nomenclature — **21 distinct pits**.
- **мехсостав granulometry:** Pits 1–21 (top section), matching the 11–21 naming convention.

**Cross-file matching:** The pit IDs do NOT directly align across the three files. "Разрез 1" from the EC/TDS file has no labeled identifier in the other files. The granulometry table lists pits 1–21 but does not use the "XX/20А" nomenclature.

### 1.3 Coordinate Availability

**❌ CRITICAL FINDING: NO coordinates in any of the three 2020/2021 files.**

- `Результаты EC,TDS, pH`: No latitude/longitude columns. Pit ID only ("Разрез 1"–"Разрез 9").
- `результаты Арал(Лист2)`: No coordinate columns. Pit ID only ("01/20А"–"20/21").
- `мехсостав 2020,2021`: No coordinate columns. Pit number only (1–21).

**Comparison to V6 training data (profiles_v6.csv):**
- All 76 V6 pits have `lat_dd`, `lon_dd`, and a `coord_source_file` / `coord_provenance` column.
- 70 of 76 pits have valid coordinates; 6 are marked `missing` (pits 03/14, 04/14, 10/13, 13/14, 15/14, and one other).

**Consequence:** Without coordinates, the 2020/2021 data cannot be:
1. Spatially matched to the mapped AOI or existing pit locations.
2. Validated against RS indices (NDMI, NDVI, etc.) that depend on pixel location.
3. Used for spatial leave-out-block (LOO) cross-validation.

---

## 2. Measurement Comparability: 2020/2021 vs. 2012–2014 V6

### 2.1 Salinity & Ion Chemistry

**2012–2014 basis (in V6):**
- Water-extract ions: HCO₃, Cl, SO₄, Ca, Mg, Na, K (from Appendix Б.4–Б.6).
- Salinity: "sum_salts" (derived from the above).
- Aggregated to `top_salt_*` (topsoil) and `prof_salt_*` (profile mean).

**2020/2021 (Результаты EC,TDS, pH file):**
- Per-layer water-extract ions: **identical list** (HCO₃, Cl, SO₄, Ca, Mg, Na, K).
- Salinity: Explicitly labelled "Соленость, %" (Salinity, %) — **directly comparable**.
- Depth-resolved: Per-layer measurements down to 150+ cm, allowing depth-to-salt analysis.

**Comparability verdict:** ✅ **STRONG.** The ion suite and salinity derivation match the 2012–2014 methodology exactly. The per-layer structure is richer than the V6 top/prof aggregates.

### 2.2 Humus & pH

**2012–2014 basis:**
- Humus: `top_humus_pct`, `prof_humus_pct` (from Appendix Б.1–Б.3).
- pH: `top_ph_water`, `prof_ph_water` (aqueous pH from the same appendices).

**2020/2021 (результаты Арал file):**
- Humus: "гумус, %" (humus, %) per layer — **identical units and definition**.
- pH: "рН" (pH) per layer — **identical measurement**.

**Comparability verdict:** ✅ **STRONG.** Exact match to V6 top/prof structure.

### 2.3 Granulometry

**2012–2014 basis:**
- Sand%, silt%, clay%, physical-clay%, hygroscopic-water% (from Appendix Б.7–Б.9).

**2020/2021 (мехсостав granulometry file):**
- Sand (0.25–0.05 mm), Silt (0.05–0.01 mm), Clay (<0.01 mm) per pit.
- Soil texture class derived (e.g., "супесчаный", "легко суглинистый").

**Comparability verdict:** ✅ **STRONG for primary fractions.** The three standard fractions match V6 definitions. Missing: physical-clay and hygroscopic-water (more specialised; OK for bulk validation).

### 2.4 Temporal Assumption

**2012–2014 V6 training data** vs. **2020/2021 out-of-period validation data** spans **6–9 years**. This violates the quasi-stationary assumption (SCIENCE_REDTEAM_AUDIT, W10):
- Soil chemistry (salinity, carbonates) on the active seabed may shift with weather, groundwater, and salt transport.
- ρ(NDMI, salt) measured on decade-old soils may not hold for current vegetation patterns.

**Recommendation:** Treat 2020/2021 as a **conservative** out-of-period check; any agreement shows robustness across time; disagreement is expected and should be interpreted cautiously.

---

## 3. Overlap with Mapped AOI & Training Pits

### 3.1 Geographic Extent

**V6 training pits:**
- 70 of 76 have valid coordinates.
- Located in the Priaralye region (broader area); 54 pits **lie OUTSIDE the mapped AOI** (1960 Aral seabed footprint), per SCIENCE_REDTEAM_AUDIT (W4).
- 16 pits lie **INSIDE the AOI** (the intended validation domain).

**2020/2021 pits:**
- **No coordinates available** — cannot determine if they fall inside or outside the AOI.
- Nomenclature hints ("Результаты Арал", "Арал Лист2") suggest they may be from the Aral region, but this is speculative without geocoding.

### 3.2 Pit Matching

**Can any 2020/2021 pit be linked to an existing V6 pit?**

**❌ No direct match found.** The three 2020/2021 files use distinct ID schemes ("Разрез N", "XX/20А", pit number 1–21) that do not overlap with the V6 pit nomenclature ("XX/YY" where YY = 12, 13, 14, or "XX А/YY" for specific campaigns). The 2020/2021 campaign appears to have established a *new* set of sampling points, not re-occupied existing 2012–2014 pits.

**Implication:** Even if coordinates were available, the 2020/2021 data would be **independent test data**, not a re-measurement of the same locations.

---

## 4. Saxaul Label Ceiling: Can 2020/2021 Expand It?

### 4.1 Saxaul Labels in 2012–2014 Data (REPORT_MINING_AUDIT Review)

From REPORT_MINING_AUDIT (§3):
> "Every `саксаул`/`посадка саксаула` occurrence in the morph files (pits `8А/12`, `12`, `13/14`, `23/14`, `24/14`, plus narrative pits `4А/12`, `12/14`, `14/13`, `25/14`, `28/13`, `29/13`) maps to an existing row in `saxaul_labels_v6.csv`."

**Saxaul-positive labels in `saxaul_labels_v6.csv` — 8 total, 6 trainable:**

| pit_id | src_year | status | strength | coords? | in ML set (trainable)? |
|--------|----------|--------|----------|---------|------------------------|
| `4А/12` | 2012 | present | strong | ✅ | ✅ |
| `8А/12` | 2012 | sparse_suppressed | weak | ✅ | ✅ |
| `14/13` | 2013 | sparse_suppressed | weak | ✅ | ✅ |
| `24/13` | 2013 | sparse_suppressed | weak | ✅ | ✅ |
| `12/14` | 2014 | sparse_suppressed | weak | ✅ | ✅ |
| `23/14` | 2014 | present_dominant | strong | ✅ | ✅ |
| `13/14` | 2014 | present_dominant | strong | ❌ (no coords) | ❌ excluded |
| `25/14` | 2014 | sparse_suppressed | weak | ❌ (no coords) | ❌ excluded |

**The "6-positive ceiling" = the 6 georeferenced positives that survive the coordinate/AOI filter and reach the 70-row ML set.** Two further positives (`13/14`, `25/14`) carry real saxaul evidence but are dropped for missing coordinates — they confirm the mining is exhausted but cannot be trained on without coordinate recovery.

**Audit conclusion (REPORT_MINING_AUDIT §3):** No *un-labelled* presence/absence evidence was found in the 2012–2014 morphology or narrative. All extractable saxaul mentions are already in the label set.

### 4.2 Saxaul Information in 2020/2021 Files

**❌ CRITICAL FINDING: 2020/2021 files contain NO vegetation or saxaul descriptions.**

- `Результаты EC,TDS, pH`: Pure chemistry/water-extract data. No vegetation, no plant species, no saxaul mentions.
- `результаты Арал(Лист2)`: Soil chemistry only (humus, CO₂, pH). No vegetation or saxaul mentions.
- `мехсостав 2020,2021`: Granulometry and soil texture class only. No vegetation or saxaul mentions.

**Comparison to AralField dataset (2018):**
- AralField (n=14 points) contains explicit vegetation fields: "Vegetation type", "Vegetation name" (e.g., "Haloxylon, herbs"), "Density of vegetation %".
- AralField records 4 Haloxylon-associated points (S124, S125, S129, others noted).
- This independent external dataset is used for V6 validation but provides no new *training* labels, only external test signals.

### 4.3 Label Ceiling: Verdict

**❌ Saxaul-label ceiling remains at 6 positives; 2020/2021 data does NOT expand it.**

**Why:**
1. The 2020/2021 CSVs have no vegetation data (measured only soil chemistry).
2. The REPORT_MINING_AUDIT already exhaustively mined the 2012–2014 morphology and narrative for saxaul mentions.
3. To raise the ceiling to, say, 12–15 positives would require:
   - **Option A (NEW FIELDWORK):** A dedicated field campaign that measures both soil chemistry (for model training) AND vegetation/saxaul presence (for labels), geo-referenced to specific coordinates, and sampled in the target AOI.
   - **Option B (2020/2021 WITH CONTEXT):** Geocode the 2020/2021 pits + add vegetation observations from field photos or external satellite/drone imagery; then label them as presence/absence. This is labour-intensive and speculative.

---

## 5. Coordinated Recovery: Is It Feasible?

### 5.1 What Locality Information Exists?

Searching the 2020/2021 files for geographic context:

**Результаты EC,TDS, pH file:**
- Column 2: "Место отбора" (Sampling location) — contains free text like "Разрез 1", "Разрез 2", etc., but **no geographic descriptors** (no place names, no distance references, no landmarks).

**результаты Арал(Лист2) file:**
- Column 1: "№ разреза" (Pit №) — only numeric IDs (01/20А, 02/20А, etc.). No location text.

**мехсостав granulometry file:**
- No location column. Only pit numbers.

### 5.2 Geocoding Feasibility

**❌ VERY LIMITED.** Unlike the 2012–2014 morphology tables (which carry locality text and field notes for each pit, e.g., "Волнистая равнина к северу от пос. Аманоткель"), the 2020/2021 CSVs have zero narrative locality information. Even geocoding requires:
1. Place names (e.g., "Казахстан, Аральское море, пос. Аманоткель") — **absent**.
2. Relative distance/direction (e.g., "5 км юго-западнее") — **absent**.
3. Pit sequence or site context that can be cross-referenced to known areas — **absent**.

**Conclusion:** Coordinates cannot be reliably recovered from the CSV contents alone. If original field notebooks or metadata exist (not in the repo), they might contain location references; but the CSVs themselves offer no pathway.

---

## 6. Summary: Out-of-Period Validation Potential

### 6.1 What 2020/2021 CAN Support

If coordinates were available, the 2020/2021 data would enable:

1. **Salinity model validation (weak, region-specific):**
   - Compare V6-predicted salinity (via NDMI) to observed 2020/2021 salinity measurements.
   - Expected pattern: if model is robust, predicted rank-order of salinity should match observed, even with temporal drift.
   - Expected n=21 pits is below target (want ≥30 for stable AUC), but non-trivial.

2. **Temporal drift assessment:**
   - Measure how much soil salinity/chemistry changed between 2012–2014 (training) and 2020/2021 (validation) at *potentially matched locations*.
   - If no pits are re-occupied, this is speculative.

3. **Regional generalization check:**
   - If 2020/2021 pits span a different sub-region (Ustyurt vs. seabed, for example), allow comparison of model transferability.
   - Currently unknown without coordinates.

### 6.2 What 2020/2021 CANNOT Support

1. **Saxaul-label expansion:** No vegetation data; the 6-positive ceiling stands.
2. **Spatial validation:** No coordinates = no AOI check, no NDMI matching, no spatial LOO.
3. **Coordinate recovery:** Extremely limited metadata; no geocoding pathway.

---

## 7. Hard Rule: Period Separation & Provenance Flagging

**RULE (from REPORT_MINING_AUDIT §2, §5c):**
> "Do NOT propose merging the 2020/2021 campaign into the 2012–2014 training labels. It is a different campaign. Document the required period separation and provenance flagging if it were used as validation only."

### 7.1 Implementation

Should a future effort attempt to use 2020/2021 pits (even with coordinates) as out-of-period validation:

1. **Strict label provenance:** All validation pits must carry a `src_campaign="2020_2021_external"` or similar flag, separate from the 2012–2014 training cohort.

2. **No label reuse:** Validation-set salinity observations (e.g., "measured EC = 1.2%") are used only to evaluate model predictions, **not** to re-fit or re-train the salinity logit.

3. **Report stratified by period:** When reporting AUC/performance, explicitly state:
   - AUC on 2012–2014 training hold-out (e.g., spatial LOO within the training period): **0.792 per-block** (current).
   - AUC on 2020/2021 out-of-period validation (if available): **[new value]** (TBD; expected to be lower due to drift).

4. **Document assumptions:** Any out-of-period validation is subject to:
   - Temporal drift (6–9 year gap; salinity distributions may shift).
   - Spatial mismatch (if new pits are not at training pit locations).
   - Methodological equivalence (lab/measurement protocols assumed unchanged).

---

## 8. Recommendations

### If Coordinates Become Available (Higher Priority)

1. **Coordinate recovery effort:**
   - Search for original 2020/2021 field notebooks, expedition reports, or metadata files (not in repo).
   - If found, geocode the pits and merge into a `2020_2021_pits.csv` with lat/lon/src_campaign.

2. **Out-of-period validation pipeline:**
   - For each 2020/2021 pit, extract V6 RS indices (NDMI, etc.) at the coordinate.
   - Fit a *separate* logit to V6-predicted vs. observed 2020/2021 salinity (no refit of training weights).
   - Report per-block AUC or slope agreement; expected to show drift/instability.
   - Publish the per-period AUC gap prominently (see W1 in SCIENCE_REDTEAM_AUDIT).

3. **Raise saxaul ceiling:**
   - Do NOT use 2020/2021 chemistry as a proxy for labelling.
   - If field photos or UAV/satellite vegetation data exist for the 2020/2021 locations, hand-label for saxaul presence/absence independently.
   - Target: recover 5–10 new presence labels to move n from 6 → 11–16 (still weak but measurable improvement).

### If Coordinates Cannot Be Recovered (Lower Priority)

1. **Archive 2020/2021 CSVs with caveat:**
   - Document that they exist, contain comparable chemistry, but are un-georeferenced.
   - Flag as "potential future validation data; metadata/coordinates required."

2. **Focus on new fieldwork:**
   - If the goal is to expand the saxaul-label ceiling, a **dedicated new campaign** is more efficient.
   - Target: 10–20 new pits within the mapped AOI, with:
     - Coordinates (GPS-surveyed).
     - Saxaul presence/absence labelled by field observers.
     - Soil chemistry measured (EC, TDS, ions, texture — same protocol as 2012–2014).
     - Regional coverage: prioritize gaps (e.g., undersampled sub-regions identified by spatial LOO).

---

## 9. Conclusion

**Verdict: 2020/2021 data is INFORMATIONALLY PRESENT but GEOGRAPHICALLY BLOCKED.**

- ✅ Measurement types (ions, salinity, humus, pH, granulometry) are directly comparable to 2012–2014 V6 training data.
- ✅ Sample size (n≈21) is modest but non-negligible for weak out-of-period validation.
- ❌ **CRITICAL BLOCKER:** No coordinates; cannot be spatially validated or matched to the AOI.
- ❌ No vegetation/saxaul observations; does NOT expand the 6-positive label ceiling.
- ⚠️ If coordinates are recovered, the data would support a **conservative, region-specific, time-windowed salinity validation check** with explicit caveats (temporal drift, spatial mismatch, methodology assumptions).

**Recommended next action:**
1. **Short term:** Search for 2020/2021 field metadata to recover coordinates (if it exists).
2. **Long term:** If coordinates are unavailable, prioritize a **dedicated new field campaign** with explicit design for saxaul labelling and AOI coverage, rather than attempting to retrofit ungeoreferenced chemistry data.

The hard rule stands: **no merging of 2020/2021 pits into 2012–2014 training labels. Strict period separation if used as validation only.**

---

## Appendix A: Per-File Data Sample

### Результаты EC,TDS, pH File (First 3 Pits)

```
Pit 1 (Разрез 1), depths 0–13, 13–50, 50–75, 75–94, 94–140 cm:
  EC (mS/cm): 0.06, 0.05, 0.04, 0.06, 0.06 (low salinity, typical aeolian sand)
  HCO₃: 0.5, 0.4, 0.3, 0.55, 0.45 mmol/100g (carbonate buffer)
  Cl, SO₄: trace (0.24–0.28 mmol/100g Cl; <0.1 mmol/100g SO₄)
  
Pit 2 (Разрез 2), depths 0–25, 25–56, 56–87, 87–100, 100–150 cm:
  EC: 0.07, 0.05, 0.05, 0.08, 0.14 (low–moderate, increasing with depth)
  Cl: 0.32, 0.36, 0.36, 0.32, 0.36 mmol/100g
  
Pit 5 (Разрез 5), depths 0–6, 6–23, 23–34, 34–60, 60–103, 103–128, 125–160 cm:
  EC: 31.3, 6.99, 7.78, 9.6, 6.03, 2.94, 1.51 (HIGHLY SALINE in topsoil; lixiviation with depth)
  Cl: 262.4, 85.6, 34.0, 31.2, 21.2, 9.6, 8.0 mmol/100g (clear salt leaching profile)
```

**Interpretation:** Pit 5 is a solonchak (saline soil) with severe surface salt accumulation; Pits 1 & 2 are freshly leached aeolian or alluvial sands. Profiles are realistic for the Aral seabed.

### результаты Арал(Лист2) File (First 10 Pits)

```
Pit 01/20А, layers:
  Depth 0–10: humus 0.45%, CO₂ 1.49%, pH 7.68
  Depth 16–26: humus —, CO₂ 0.62%, pH 7.79
  Depth 38–48: humus —, CO₂ 0.58%, pH 7.81
  (typical desert soil, low humus, weakly carbonate)

Pit 05/20А, layers:
  Depth 0–3: humus 1.31%, CO₂ 6.06%, pH 8.50
  Depth 3–11: humus 0.75%, CO₂ 6.91%, pH 8.74
  (higher humus, strong carbonate; possible seabed solonchak)
```

**Interpretation:** Consistent with 2012–2014 profiles (desert soils, pH 7.5–8.5, low humus, variable CO₂).

### мехсостав Granulometry (First 5 Pits)

```
Pit 1: Sand 86.69%, Silt 8.16%, Clay 5.12% → песок (sandy)
Pit 2: Sand 67.58%, Silt 25.7%, Clay 6.68% → супесчаный (loamy sand)
Pit 3: Sand 47.12%, Silt 39.95%, Clay 12.91% → легко суглинистый (loamy)
Pit 4: Sand 49.21%, Silt 39.25%, Clay 11.52% → легко суглинистый (loamy)
Pit 5: Sand 40.94%, Silt 48.66%, Clay 10.38% → средне суглинистая (medium loam)
```

**Interpretation:** Range from sand to loam; matches V6 diversity (sand%, silt%, clay% all present).

---

## Appendix B: Cross-Reference to Audits

- **REPORT_MINING_AUDIT §U10:** 2020/2021 re-survey field tables identified but noted as a separate campaign, not integrated.
- **SCIENCE_REDTEAM_AUDIT W1, W4, W10:** Regional calibration drift, geographic domain shift, and temporal mismatch — all risks to out-of-period validation.

---

*No code, pipeline, or canonical dataset was modified in producing this audit.*
