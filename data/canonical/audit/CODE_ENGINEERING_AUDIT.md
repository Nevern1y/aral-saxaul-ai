# CODE & PIPELINE ENGINEERING AUDIT — Aral Saxaul V6

**Scope:** read-only engineering audit of the V6 lab-data science pipeline
(`scripts/v6/*`, `scripts/run_v6_pipeline.py`) and the `app.py` data contracts it
feeds. Goal: map implementation debt and fragility that blocks safe iteration
toward higher accuracy. **No pipeline behavior or outputs were changed.** Tests
and the V6 science audit were run read-only (results in §2).

**Auditor role:** engineering / pipeline (read-only) · **Date:** 2026-06-19
**Companion docs:** `REPORT_MINING_AUDIT.md` (data-mining, sibling lane) and
`README.md` / `CLAUDE.md` (project context). This file does not repeat their
science findings; it covers code structure, contracts, reproducibility and tests.

---

## 0. How this audit was run (reproducible)

| Action | Command | Result |
|---|---|---|
| Test suite | `python -m pytest -v` (system Py 3.12.10) | **14 passed in 23.75 s** |
| V6 science Red Team | `python -X utf8 scripts/v6/qa_science_audit_v6.py` | **28 checks, 0 failed** |
| Tree / provenance inspection | `git ls-files data/ outputs/`, VRT XML read | see §3, §5 |

> ⚠️ **Environment note (not a code defect, but a reproducibility friction).** The
> Hermes-default interpreter (`...\hermes-agent\venv`) has **no pytest and none of
> the GIS stack**. The suite only runs under the system Python
> `…\Programs\Python\Python312` (pytest 9.1.0, streamlit 1.57, rasterio 1.5,
> pandas 3.0, numpy 2.4). There is **no committed lockfile or venv pin** — see
> RISK-9. A fresh contributor will pick the wrong interpreter and see
> `No module named pytest`.

---

## 1. Component map of the V6 DAG

The orchestrator `scripts/run_v6_pipeline.py` runs 9 steps (the 8 named DAG steps
plus `render_v6_map`) as **separate `subprocess` calls**, gated by two flags:
`--docx` (step 1) and `--rasters` (steps 4, 7, 8 + render). On a clean checkout
the raster/docx steps are skipped and only the CSV/JSON steps run from tracked
inputs.

```
                 (source .docx, 22 MB, OUTSIDE repo)
                          │  [--docx]
  1. extract_docx_tables ▼ ──────────────► data/interim/docx_*_long.csv (+provenance)
                          │
  2. build_canonical_db   ◄── data/raw_external/morph_*.csv
                          ▼ ──────────────► profiles_v6.csv, soil_layers_v6.csv, canonical_manifest.json
                          │
  3. extract_saxaul_labels◄── (source .docx narrative, re-opened)
                          ▼ ──────────────► saxaul_labels_v6.csv, saxaul_labels_manifest.json
                          │
  4. build_ml_dataset     ◄── 10m bands, dem_slope_30m, suitability_map_v5, feature_stack_30m tiles
       [--rasters]        ▼ ──────────────► ml_dataset_v6.csv, ml_correlations_v6.csv, ml_dataset_manifest.json
                          │
  5. calibrate_thresholds ▼ ──────────────► thresholds_v6_calibrated.json, CALIBRATION_REPORT.md
                          │
  6. train_suitability_model ──────────────► outputs/models/salinity_v6_logit.json (+ exploratory),
                          │                   model_v6_fit_report.md
                          │                   ▲ also reads aralfield_validation_v6.csv (UNPRODUCED, see RISK-1)
  6b. build_suitability_index [--rasters]
                          ◄── feature_stack_30m.vrt (geometry), 2 good tiles (pixels), aoi_mask_v5
                          ▼ ──────────────► suitability_index_v6.tif, suitability_zones_v6.tif,
                          │                   suitability_v6_stats.json, SUITABILITY_INDEX_V6_QA.md,
                          │                   suitability_v6_pit_validation.csv + _summary.json
  8b. render_v6_map [--rasters]
                          ▼ ──────────────► outputs/reports/suitability_map_v6.html (+PNGs)
                          │
  7/8. spatial_validation [--rasters]
                          ◄── ml_dataset_v6.csv, salinity model, 2 good tiles, aoi_mask
                          ▼ ──────────────► spatial_blocks_v6.csv, spatial_validation_v6.json,
                                              suitability_uncertainty_v6.tif, SPATIAL_VALIDATION_V6_QA.md
```

**Consumption boundary (the contract that matters):** `app.py::load_v6_science()`
reads exactly five tracked artifacts — `salinity_v6_logit.json`,
`suitability_v6_stats.json`, `suitability_v6_pit_validation_summary.json`,
`spatial_validation_v6.json`, `suitability_v6_pit_validation.csv` — plus the
self-contained `suitability_map_v6.html`. Everything else is intermediate.

**Tracking model:** JSON/CSV/MD under `data/canonical/`, `outputs/models/`, and the
three `*_v6*.json` under `outputs/data/` are git-tracked (dashboard inputs);
`*.tif/*.vrt` are gitignored and regenerated; `data/interim/` is gitignored
(reproducible from `--docx`).

---

## 2. Test & QA results (read-only)

**`pytest` — 14 passed** (CLAUDE.md still says "13 tests" — doc drift, RISK-10):

- `test_app_smoke.py` (4): full `app.py` AppTest run, 3 tabs, title, dataframes.
- `test_data_contracts.py` (10): 5 V5 contracts + 5 V6 contracts (loader shape,
  salinity-model keys, spatial-AUC sanity incl. CI lower-bound ≥ 0.5, zone areas,
  self-contained map HTML).

All V5 + V6 contract tests **actually executed** (did not skip) because this dev
machine has the full `outputs/` tree present. **On a clean checkout / Streamlit
Cloud, every V5 contract test and the raster-backed V6 tests skip** (guarded on
`outputs/data/v5_stats.json` or per-artifact `.exists()`); only the loader-shape
test and tracked-JSON tests remain — see RISK-8 for what that masks.

**`qa_science_audit_v6.py` — 28 checks, 0 failed.** Independence/no-circularity,
AUC∈[0,1], CI brackets + lower-bound ≥ 0.5, severity ordering of cuts, pixel-count
reconciliation (166,208,268 == grid), coverage 96.9 %, extrapolation 6.1 %,
coverage parity V6=15 ≥ V5=13, detector sens 0.7 / spec 0.8, honesty-caveat
presence in the QA markdown. **This script is the de-facto integration test of the
science layer, but it is NOT wired into pytest** (RISK-7).

---

## 3. Per-component detail: inputs · outputs · fragility · change-points

Legend: **CP** = change-point that must be touched to expand dataset/features.

### Step 1 — `extract_docx_tables.py`
- **In:** `../Закл отчет … .docx` (22 MB, **outside the repo**, hardcoded sibling path).
- **Out:** `data/interim/docx_{chem_physchem,water_extract,granulometry}_long.csv` + provenance JSON.
- **Fragility:**
  - Positional column maps keyed by **table arity** (`chem_physchem_17`,
    `_18`, `water_extract_11`, `granulometry_10`). If a continuation table has
    a different column count it silently falls into the `c{i}` positional branch
    (line 161-163) and those values **never reach the canonical schema**.
  - Caption parsing is regex on Cyrillic `Таблица Б.N` / `Продолжение` — brittle
    to OCR/spacing variants; an unmatched caption drops a whole table without error.
  - The `mass% / meq` two-row pairing is reconstructed downstream (Step 2) by
    `cumcount`, which assumes the report always emits exactly two rows per
    (pit, depth) in that order.
- **CP:** adding a new appendix-table family or a year with a different column set
  requires editing `FAMILY` + `COLS`. Adding 2015+ data means new `Б.N` letters.

### Step 2 — `build_canonical_db.py`
- **In:** `data/raw_external/morph_*.csv` (4 files, **hardcoded list** line 45),
  `data/interim/docx_*_long.csv`.
- **Out:** `profiles_v6.csv` (76 pits), `soil_layers_v6.csv` (369 layers),
  `canonical_manifest.json`.
- **Fragility:**
  - **Positional column access** `df.columns[0..3]` for id/lat/lon/elevation —
    any reordering of the morph CSVs breaks coordinates silently. Vegetation
    column is found by fuzzy substring (`"Растительност"`/`"рельеф"`) with a
    `df.columns[-2]` fallback.
  - `dms_to_dd` and `parse_float` are tolerant parsers that return `NaN` on
    anything unexpected — **malformed coordinates become "missing", not an error.**
  - **Hand-coded coordinate recovery** `OFFSET_RECOVERY` (only `24/14`): a
    single documented metric offset baked into code. Correct and flagged
    (`coord_provenance="recovered_offset"`), but it is data-as-code.
  - `MORPH_FILES` excludes `chem_2014_final.csv` / `chem_2014_karaterren.csv`,
    which are tracked in `raw_external/` but **read by nothing** (confirmed: no
    script references them). Dead inputs / latent expansion source — see
    REPORT_MINING_AUDIT §U11.
- **CP:** dataset expansion (new survey years, new regions) lands here first —
  extend `MORPH_FILES`, confirm column order, and decide AOI bbox handling.

### Step 3 — `extract_saxaul_labels.py`
- **In:** `profiles_v6.csv` + the source `.docx` re-opened for narrative.
- **Out:** `saxaul_labels_v6.csv` (76), `saxaul_labels_manifest.json`.
- **Fragility:**
  - Label target is produced by **~12 hand-tuned Cyrillic regex vocabularies**
    (`SAXAUL`, `PLANT`, `FAIL`, `SURVIVE`, `DOMINANT`, `SPARSE`, `NEARBY`,
    `RICE_AG`, `FLOODPLAIN`, `BARREN`, …) plus a `STATUS_RANK` fusion table. This
    is the single most assumption-laden code in the pipeline and **directly sets
    the model's y**. Small wording changes in the source shift labels.
  - `BLOCK_WINDOW = 12` paragraph cap to avoid EOF contamination — a magic
    number tuned to the current document layout.
  - The DOCX path constant is **duplicated** here and in Step 1 (RISK-2).
- **CP:** any new label source or re-mining of the narrative changes positives
  count (currently only 6 trainable positives — the accuracy ceiling). High-value,
  high-risk to touch.

### Step 4 — `build_ml_dataset.py`  *(needs rasters)*
- **In:** `soil_layers_v6.csv`, `saxaul_labels_v6.csv`, 10 m bands
  (`B3/B4/B8/B11/B12/SCL`), `dem_slope_30m.tif`, `suitability_map_v5.tif`,
  the two good 30 m tiles, `thresholds_v5.json`. **Imports `scripts/v5_rules`**
  (`compute_indices`, `classify_pixel`, `CLASS_NAMES`) via `sys.path.insert`.
- **Out:** `ml_dataset_v6.csv` (70 rows × 88 cols), `ml_correlations_v6.csv`,
  `ml_dataset_manifest.json`.
- **Fragility:**
  - **Cross-version coupling:** the V6 ML table depends on the V5 module staying
    import-compatible. A V5 refactor of `compute_indices`/`classify_pixel`
    signatures breaks V6 silently at sample time. `sys.path.insert(0, scripts/)`
    is fragile under different CWDs.
  - **Legacy band-order assumption** `LEGACY_BANDS = [ndmi, msavi, si, ndwi,
    slope, twi, vh]` positionally mapped from the tiles — no header validation.
    `ndwi` is band 4 (used in 6b/7 too). If a re-export changes band order, every
    downstream NDMI value is wrong **with no error**.
  - Point sampling is **nearest-pixel `src.index()` with no averaging/buffer** —
    sub-pixel coordinate error (incl. the recovered-offset pit) maps to a single
    30 m cell. NoData edges → `NaN` → row drops out of correlations.
  - The 76→70 reduction (profiles → ML rows) happens by `has_coordinates`
    filtering; it is correct but only documented in the manifest, not asserted.
- **CP:** new features (extra indices, new bands, soil covariates beyond topsoil
  salinity) are added here. The `AGG_COLS` / `BAND_PATHS` / `LEGACY_BANDS` lists
  and the topsoil window (`TOPSOIL_CM = 30`) are the knobs.

### Step 5 — `calibrate_thresholds.py`
- **In:** `ml_dataset_v6.csv`. **Out:** `thresholds_v6_calibrated.json`,
  `CALIBRATION_REPORT.md`.
- **Fragility:**
  - **The "Interpretation" section of `CALIBRATION_REPORT.md` is a hardcoded
    prose block** (lines 186-208) quoting specific numbers ("Cl ≤ 0.06 %, AUC
    0.73", "CaCO3 ≥ 5.3 %, AUC 0.86", "optimal cut ≈ -0.10"). The table above it
    is generated from `results`, but **the narrative is not** — if recalibration
    shifts these cuts, the prose silently lies. This is the clearest violation of
    the project's "every shipped number produced by a script" rule (RISK-3).
  - Thresholds emitted only when `auc≥0.62 and n≥12` — a predictor dropping below
    the bar disappears from the JSON, and any consumer keyed on its name crashes
    (see Step 6b `load_cuts`).
- **CP:** `SALINE_PCT`/`STRONG_PCT`/`MIN_N`/`MIN_AUC` and the `rs_preds`/`soil_preds`
  lists. Adding predictors is low-risk; the prose block must be hand-synced (debt).

### Step 6 — `train_suitability_model.py`
- **In:** `ml_dataset_v6.csv`, **`aralfield_validation_v6.csv`** (independent check).
- **Out:** `outputs/models/salinity_v6_logit.json` (validated core, tracked),
  `suitability_v6_logit.json` (exploratory, marked), `model_v6_fit_report.md`.
- **Fragility:**
  - **`aralfield_validation_v6.csv` is read but produced by NO script** (confirmed
    `grep`). It is a tracked hand-made artifact (11 rows, columns
    `id,lon,lat,haloxylon,veg,note,ndmi,msavi`). The `ndmi`/`msavi` were sampled
    by some unknown process; **this breaks the reproducibility rule** and means
    the "independent AUC" number cannot be regenerated (RISK-1).
  - Custom numpy Newton-IRLS logistic fit (no sklearn, deploy-light). Correct but
    unverified against a reference implementation in any test. `fit_logit` can
    return an unconverged `beta` after `MAX_ITER` without signalling.
  - `predict_*` formulas live as **strings** in the JSON for documentation, while
    the actual inference is re-implemented in 6b and 7 — three copies of the same
    sigmoid (RISK-4).
- **CP:** `PREDICTORS`, `L2_GRID`. Promoting a multivariate model means changing
  the exported schema → cascades to 6b, 7, app.py.

### Step 6b — `build_suitability_index.py`  *(needs rasters; the corrupt-VRT step)*
- **In:** `feature_stack_30m.vrt` (**geometry/profile only**), the two good tiles,
  `aoi_mask_v5.tif`, `salinity_v6_logit.json`, `thresholds_v6_calibrated.json`,
  `ml_dataset_v6.csv` (support range), `suitability_map_v5_filtered.tif`.
- **Out:** `suitability_index_v6.tif`, `suitability_zones_v6.tif`,
  `suitability_v6_stats.json`, `SUITABILITY_INDEX_V6_QA.md`,
  `suitability_v6_pit_validation.csv` + `_summary.json`.
- **Corrupt-VRT handling (verified on disk):** the on-disk
  `feature_stack_30m.vrt` (14844×11197, 7 bands) mosaics over
  `feature_stack_30m.tif`, whose 61–62 °E column (`DstRect xOff="12544"
  xSize="2300"`) is a corrupt tile (`TIFFReadEncodedTile` failure). The code
  **never reads VRT pixels** — it opens the VRT only for `profile`/`transform`/
  `H,W`, then reads pixels from a **hardcoded two-tile list with literal column
  offsets**: `feature_stack_30m_tile1.tif` at `col0=0` (12544 wide) and
  `feature_stack_30m_tile0_redo.tif` at `col0=12544` (2300 wide). It guards row
  count and column overflow, and warns if `written_cols != W`.
  - **Fragility of this scheme:**
    - The split (`12544 / 2300`) and tile filenames are **magic constants in two
      files** (6b and 7) that must match the VRT's `DstRect` exactly. Nothing
      asserts the code's `col0` against the VRT XML; a re-mosaic that changes the
      split silently mis-places pixels (RISK-5). On a clean checkout the VRT, the
      corrupt tile, and `tile0_redo` are all gitignored, so the geometry source
      can be absent while the script still believes it can run.
    - `feature_stack_30m.tif` (the corrupt original, 675 MB) is still on disk and
      referenced by the VRT; it is a **landmine** — any future code that reads VRT
      pixels (instead of metadata) re-hits the corruption.
    - Band semantics (NDMI=1, MSAVI=2, NDWI=4) are positional literals, same
      brittleness as Step 4.
  - **`load_cuts()` (line 137-141) indexes two exact dict keys**
    `rs30_ndmi__soil_salinity>1.0%` / `>3.0%` with no `.get`/guard. If Step 5
    drops either below the AUC/n bar, this raises `KeyError` and the raster build
    dies (RISK-6).
- **Reproducibility strength:** the QA markdown prose **is** generated from the
  computed `stats`/`pv` dicts (good — matches the rule), unlike Step 5's prose.
- **CP:** `VEG_MSAVI_CUT` (tuned to reproduce V5's 12.6 % veg), the zoning cut
  logic, AOI mask choice. Expanding the mapped area = swap `aoi_mask_v5.tif`.

### Step 8b — `render_v6_map.py`  *(needs rasters)*
- **In:** `suitability_zones_v6.tif`, `suitability_index_v6.tif`.
- **Out:** `outputs/reports/suitability_map_v6.html` + 2 PNGs.
- **Fragility:** `ImageOverlay(image=str(PNG…))` passes a **file path**, but the
  test `test_v6_map_html_self_contained` asserts the HTML embeds
  `data:image/png;base64`. Folium base64-inlines local PNG paths on `save()`, so
  it currently passes — but this is an **implicit Folium behavior**, not an
  explicit encode. A Folium change would break Streamlit-Cloud rendering with a
  green test until that very assertion runs (it does run, but only with rasters
  present). Downsample `MAX_DIM=1400` and the Google-satellite tile URL are
  hardcoded.
- **CP:** palette/legend if zone scheme changes.

### Step 7/8 — `spatial_validation.py`  *(needs rasters)*
- **In:** `ml_dataset_v6.csv`, `salinity_v6_logit.json`,
  `aralfield_validation_v6.csv`, the two good tiles, `aoi_mask_v5.tif`.
- **Out:** `spatial_blocks_v6.csv`, `spatial_validation_v6.json`,
  `suitability_uncertainty_v6.tif`, `SPATIAL_VALIDATION_V6_QA.md`.
- **Fragility:**
  - **`L2 = 0.1` is hardcoded** (line 77) "to match the committed salinity
    model's lambda". It is a manual copy of Step 6's `best_lam`; if Step 6's
    lambda grid selects differently on re-fit, the spatial CV silently validates a
    **different model** than the one shipped (RISK-4). Nothing asserts the two agree.
  - Re-implements `fit_logit` / `rank_auc` / sigmoid (4th copy) and the tile-read
    + AOI-mask loop (2nd copy of 6b's, including the same magic offsets).
  - `greedy_blocks` is deterministic on input order; reordering `ml_dataset_v6.csv`
    rows changes the blocks and thus the spatial AUC. Reproducible only because the
    CSV order is stable.
  - Seed `SEED=12345`, `N_BOOT=2000`, `BLOCK_KM=20` are fixed (good for repro).
- **CP:** block radius, bootstrap N, predictor set must track Step 6.

---

## 4. Brittle data contracts (cross-cutting)

1. **Positional column/band contracts everywhere.** morph CSV columns 0-3
   (Step 2), `LEGACY_BANDS` order (Step 4/6b/7), VRT band indices (1/2/4). None
   are validated against a header or `band Description`. Adding/reordering a
   column or band is silent corruption, not an error. **Highest-leverage
   hardening target.**
2. **Exact-key JSON contracts.** `load_cuts()` (`rs30_ndmi__soil_salinity>1.0%`),
   `load_salinity_model()` (`coefficients_standardized.rs30_ndmi`), and app.py's
   `load_v6_science()` consumers index fixed keys. The pytest contract tests cover
   the app-side keys (good) but **not** the inter-script keys (`load_cuts`).
3. **Filename/offset contracts for the tile workaround.** `feature_stack_30m_tile1`
   @0 / `tile0_redo` @12544 hardcoded in 6b and 7; must equal the VRT `DstRect`.
4. **app.py ↔ artifact schema.** Solidly covered by `test_data_contracts.py`
   (loader shape, salinity keys, spatial AUC keys+sanity, zone-area codes, map
   HTML). This is the **best-tested boundary** in the project.

---

## 5. Reproducibility gaps (the project's core rule is "every shipped number is script-produced")

| # | Gap | Where | Impact |
|---|---|---|---|
| **R1** | `aralfield_validation_v6.csv` is **read but produced by no script**; its `ndmi`/`msavi` columns were sampled by an unknown/lost process. | consumed by Steps 6, 7 | The "independent AralField AUC" shipped on the dashboard **cannot be regenerated**. Direct rule violation. |
| **R2** | `data/raw_external/morph_*.csv` and `chem_2014_*.csv` are tracked hand-made extracts with **no producing script** in-repo (a `docx_lossless_extract.py` lives *outside* the repo, next to the .docx). | Step 2 inputs | The morphology/coordinate layer is not reproducible from the repo alone. |
| **R3** | `CALIBRATION_REPORT.md` "Interpretation" prose hardcodes AUC/cut numbers (Step 5 §3). | `calibrate_thresholds.py:186-208` | Recalibration desyncs prose from JSON; numbers can silently go stale. |
| **R4** | The salinity model's L2 λ is **duplicated as a literal** in `spatial_validation.py` (`L2=0.1`) instead of read from `salinity_v6_logit.json`. | Step 7 | Spatial CV can validate a different model than is shipped. |
| **R5** | Inference sigmoid re-implemented in 4 places (model JSON string, 6b, 7, and conceptually app.py metrics). | Steps 6/6b/7 | Drift risk; a fix must be made 3×. |
| **R6** | The source `.docx` is **outside the repo** and the path is hardcoded twice. | Steps 1, 3 | `--docx` / `--all` rebuild only works on the one machine that has the sibling file. |

**What IS reproducible and good:** the 6b and 7 QA markdown prose is generated
from computed dicts; rasters are regenerated and gitignored; seeds/bootstrap are
fixed; `data/interim/` is documented as reproducible from `--docx`. The
discipline is real — it just has the six holes above.

---

## 6. Test coverage gaps

**What exists (strong):** `app.py` smoke (full AppTest run) + app↔artifact data
contracts incl. five V6-specific contracts. `qa_science_audit_v6.py` gives 28
adversarial science checks. This is unusually good for a project this size.

**Gaps:**

1. **No test imports or runs any `scripts/v6/*` code.** Every pipeline transform —
   docx parsing, DMS→DD, `parse_float`, soil aggregation, the label regex fusion,
   the numpy logistic fit, the tile-offset placement — is **completely untested at
   the unit level**. Tests only inspect *committed outputs*. A logic regression
   that still produces a schema-valid CSV passes green.
2. **`qa_science_audit_v6.py` is not in pytest** — it is the real integration
   test but runs only when someone invokes the script (or `run_v6_pipeline --qa`).
   CI/pytest does not enforce it.
3. **The golden snapshot is written but never asserted.** `snapshot_v5_golden.py`
   produces `tests/golden/v5_golden_snapshot.json`, and CLAUDE.md + the script
   docstring claim `tests/golden` "catches regressions" — but **no test reads it**
   (confirmed `grep`). The V5-freeze guarantee that justifies the whole V6-on-top
   design is therefore **unenforced**. (RISK-11)
4. **Clean-checkout blindness.** All raster-backed contracts skip when `outputs/`
   is absent, which is exactly the Streamlit-Cloud / CI condition. The numbers the
   dashboard shows there come entirely from tracked JSON that **no test
   cross-checks against the rasters** that produced them.
5. **No reproducibility test.** Nothing asserts "re-running step N yields
   byte-identical tracked artifact" — the rule the project leans on is unverified
   in CI.
6. **No contract test for inter-script keys** (`load_cuts`, `LEGACY_BANDS` order,
   VRT-offset vs `DstRect`).
7. **`requirements.txt` is unpinned** (all `>=`) and there is no lockfile; the
   suite passing today (pandas 3.0, numpy 2.4) is not reproducible tomorrow.

---

## 7. Places where adding features / rows will break

| You want to… | First thing that breaks | Why |
|---|---|---|
| Add survey year 2015+ | `extract_docx_tables FAMILY/COLS` + `build_canonical_db MORPH_FILES` | new `Б.N` letters, new morph file, positional columns |
| Add a spectral feature/band | `LEGACY_BANDS` order assumption + `BAND_PATHS` | positional band map, no header check; cascades to 6b/7 |
| Add rows that shift the NDMI range | `load_support` + extrapolation mask + `aralfield` (unreproducible) | support range recomputed; AralField can't be re-derived |
| Re-mine labels (raise the 6-positive ceiling) | `extract_saxaul_labels` regex vocab + `STATUS_RANK` | hand-tuned Cyrillic regex directly sets `y` |
| Recalibrate thresholds | `CALIBRATION_REPORT.md` hardcoded prose + `load_cuts` exact keys | prose desyncs; missing key → `KeyError` in 6b |
| Re-mosaic the 30 m stack | tile offsets in 6b **and** 7 | `col0=0/12544` + filenames are magic literals tied to VRT `DstRect` |
| Promote a multivariate model | `salinity_v6_logit.json` schema → 6b/7/app.py | sigmoid re-implemented 3×; λ duplicated in 7 |
| Expand the mapped AOI | `aoi_mask_v5.tif` pixel-identity assumption | mask must be byte-aligned to the 30 m grid |

---

## 8. Top risks (ranked)

1. **RISK-1 — Unreproducible independent validation.** `aralfield_validation_v6.csv`
   has no producer; the shipped AralField AUC violates the core rule and can't be
   regenerated. *Fix: add a `build_aralfield_validation.py` that samples NDMI/MSAVI
   from the stack at the AralField points, or tag the file as a frozen external
   input with documented provenance.*
2. **RISK-11 — V5-freeze is unenforced.** The golden snapshot is written but no
   test asserts it; the invariant the entire V6-companion architecture depends on
   has no guard. *Fix: a `test_golden_v5_unchanged.py` comparing live V5 stats to
   the snapshot (skip if `outputs/` absent).*
3. **RISK-2/3/4 — Reproducibility holes:** hardcoded morph extracts (no in-repo
   producer), hardcoded calibration prose, duplicated L2 λ in spatial CV. Each can
   silently ship a stale or mismatched number.
4. **RISK-5/6 — Tile-workaround fragility:** magic split `12544/2300` + filenames
   in two files with no assertion against the VRT; `load_cuts` unguarded exact-key
   access. A re-mosaic or a dropped threshold key breaks the raster build (or
   worse, mis-places pixels without error).
5. **RISK-7/8 — Test blind spots:** zero unit tests over `scripts/v6`; the real
   integration check (`qa_science_audit_v6`) is outside pytest; raster contracts
   skip in CI/cloud. Logic regressions that keep schemas valid pass green.
6. **RISK-9/10 — Environment & doc drift:** no interpreter pin / lockfile (suite
   only runs under system Py, not the default venv); CLAUDE.md says 13 tests
   (actually 14) and references `tests/golden` as if enforced.
7. **Positional column/band contracts** pervasively (morph cols, `LEGACY_BANDS`,
   VRT bands) — the dominant *silent-corruption* surface for any data expansion.

---

## 9. Proposed safe-iteration checklist

Before changing any V6 step, in order:

1. **Pin the environment.** Record the working interpreter; add a lockfile
   (`pip freeze > requirements.lock` or `pip-tools`). Run `python -m pytest`
   under it and confirm **14 passed**; run `scripts/v6/qa_science_audit_v6.py`
   and confirm **0 failed**. These two are your regression baseline.
2. **Snapshot the shipped numbers** you might perturb (copy the relevant tracked
   JSON/CSV) so you can diff after.
3. **Make the change behind the existing flags.** Re-run only the affected
   sub-DAG: `run_v6_pipeline.py` with `--rasters`/`--docx`/`--all` as needed.
   Never hand-edit a tracked artifact.
4. **Re-run `qa_science_audit_v6.py`** — it enforces no-circularity, CI sanity,
   coverage parity, severity ordering. Treat any FAIL as blocking.
5. **Diff tracked artifacts.** For a no-op change expect byte-identical output
   (the reproducibility rule). For an intended change, confirm only the intended
   files moved and the QA-markdown prose regenerated to match.
6. **Verify the V5 freeze** (until RISK-11 is fixed, do this manually): confirm
   `thresholds_v5.json`, `v5_stats.json`, and `suitability_map_v5*.tif` are
   untouched, and that the golden snapshot still matches.
7. **Check inter-script keys by hand** (until tested): if you touched Step 5,
   confirm `load_cuts` keys still exist; if Step 6, confirm Step 7's `L2`
   literal still equals the model's `l2_lambda`.
8. **Run the dashboard** (`streamlit run app.py`) and confirm the V6 panel +
   maps render; AppTest does **not** see the Folium map (use `/verify-map-render`).
9. **Update the docs you invalidated** (CLAUDE.md test count, any prose quoting a
   recalibrated number).

**Recommended low-risk hardening to do first (separately, with sign-off):** add a
producer (or provenance tag) for `aralfield_validation_v6.csv`; wire
`qa_science_audit_v6.py` and a golden-snapshot assertion into pytest; read Step 7's
`L2` from the model JSON; add band/column header validation to Step 4. None change
outputs; all shrink the blast radius of the accuracy work to come.

---

## 10. Summary

**Components mapped:** all 8 named DAG steps + the orchestrator + `render_v6_map`
+ the `app.py::load_v6_science` consumption contract (9 scripts, ~3,000 LoC).
**Tests run read-only:** pytest **14 passed**, V6 science audit **28/28 passed**.
**Top risks:** (1) unreproducible AralField independent validation;
(2) unenforced V5 golden-freeze; (3) hardcoded calibration prose + duplicated L2 λ
+ hand-made raw_external extracts (reproducibility holes); (4) magic tile-offset /
exact-key contracts around the corrupt-VRT workaround; (5) zero unit coverage of
`scripts/v6` with the real integration check outside pytest and all raster
contracts skipping in CI/cloud; (6) unpinned deps + doc drift.

The science layer is honest and well-instrumented; the **engineering debt is
concentrated in contracts that fail silently and numbers that can't be
regenerated** — exactly the surfaces that make iterating toward higher accuracy
risky. The checklist in §9 lets that iteration proceed without breaking the V5.1
freeze or the reproducibility rule.
