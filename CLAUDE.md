# CLAUDE.md — Aral Saxaul AI

Operational context for AI agents working in this repository. The human-facing
source is `README.md`; this file is the concise AI-facing rulebook.

## Product Rule

The current product is **V6-first salinity-risk screening**.

- **V6 is the main layer.** It estimates topsoil salinity risk from laboratory
  soil data. `V6 score = 1 - P(saline)` where saline means topsoil total salts
  > 1%.
- **V5.1 is a frozen helper layer.** It provides 10 m visual detail, contours,
  KML files, and road/logistics support. Do not treat V5.1 candidate classes as
  planting proof.
- **Never write as if the map proves planting suitability.** The correct claim is
  preliminary screening and field-check prioritization.

Safe wording: “V6 salinity-risk screening layer for selecting field-check sites.”

Unsafe wording: “model proves where saxaul can be planted.”

## V6 Science Layer

<!-- V6_SCOPE_AUTO -->
> **V6 scope (auto-generated — see `data/canonical/SCOPE_AND_LIMITATIONS_V6.md`):** Shipped salinity anchor = **M0 (univariate NDMI logit)**, LOO AUC 0.682 (CI [0.556, 0.802]). **Regional calibration drift (W1):** pooled spatial AUC 0.385 vs per-block 0.792 — the model ranks salinity *locally* but absolute level must be calibrated per region.
<!-- /V6_SCOPE_AUTO -->

- Shipped model: `outputs/models/salinity_v6_logit.json`.
- Model type: L2 logistic regression, pure NumPy IRLS.
- Predictor: `rs30_ndmi`.
- Target: measured topsoil total salts > 1%.
- Training data: 70 georeferenced Pachikin/Kozybaeva soil profiles, 2012-2014;
  27 saline profiles.
- Key metric: LOO AUC 0.682, benchmark CI [0.556, 0.802].
- Important scope: only about 15/70 profiles are inside the 1960 seabed AOI; the
  rest support training from wider Priaralye but are not independent seabed
  validation.
- Spatial validation: pooled spatial AUC is low because of regional calibration
  drift; per-block ranking remains positive. Prefer local ranking, not absolute
  cross-region comparison without calibration.

The direct NDMI+MSAVI -> saxaul classifier is exploratory only. It has too few
positive saxaul labels and must not become a decision model.

## V5.1 Helper Layer

V5.1 is a rule-based Sentinel-2 10 m cascade:

- water/shadow/no-data gate;
- slope obstacle gate;
- vegetation gate;
- wet-brine proxy;
- dry-salt proxy;
- residual candidate class.

Use V5.1 for:

- detail boundaries;
- helper contours;
- KML task files;
- road-distance logistics;
- context around the V6 salinity score.

Do not use V5.1 as the headline suitability result.

## Main Entry Points

- `app.py` — standalone Streamlit dashboard with 3 tabs: map+summary,
  checks+data, work planning.
- `scripts/run_v6_pipeline.py` — V6 science pipeline coordinator.
- `scripts/v6/train_suitability_model.py` — shipped salinity model producer.
- `scripts/v6/build_suitability_index.py` — wall-to-wall V6 raster producer.
- `scripts/v6/render_v6_map.py` — self-contained Folium HTML/PNG map producer.
- `scripts/v6/spatial_validation.py` — spatial CV, bootstrap CIs, uncertainty.
- `scripts/v6/qa_science_audit_v6.py` — V6 red-team science audit.
- `scripts/*v5*.py` — frozen/helper V5.1 product, stats, logistics, reports.

Legacy pre-V5 code and old XGBoost artifacts are not current product paths. Do
not revive, document, or use them for current claims.

## Commands

Dashboard:

```bash
streamlit run app.py
```

V6 full rebuild when large local inputs exist:

```bash
python scripts/run_v6_pipeline.py --all --qa
```

V6 tracked-input steps only:

```bash
python scripts/run_v6_pipeline.py
```

Tests and lint:

```bash
python -m ruff check app.py tests scripts/v6/render_v6_map.py scripts/v6/build_suitability_index.py
python -m pytest tests
```

On this workstation, use system Python 3.12:

```powershell
& "C:\Users\cynok\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests
```

## Deployment Rules

- Entry point: `app.py`.
- Maps render through `_render_map()` -> `st.iframe(...)`. Do not switch to
  `st.html` or `streamlit.components.v1.html`; Leaflet JS can fail under CSP or
  deprecation paths.
- Keep `rasterio` and `geopandas` guarded in `app.py`. The dashboard must degrade
  gracefully when local GIS wheels or generated outputs are absent.
- `environment.yml` must not be committed.
- AppTest cannot verify Folium internals; use browser/Playwright verification for
  map changes.

## Reproducibility Rules

- Do not hand-edit auto-generated canonical reports such as
  `data/canonical/SCOPE_AND_LIMITATIONS_V6.md`.
- Every shipped number in UI/docs should come from JSON/CSV/MD artifacts or from
  scripts, not from memory.
- If `scripts/v6/render_v6_map.py` changes, regenerate
  `outputs/reports/suitability_map_v6.html` and related PNGs.
- Keep `tests/test_data_contracts.py` strict around V6 map interactivity and V6
  metric sources.

## Language And UX

- Russian is the primary UI language.
- Speak to field users: where to go, what risk is present, what to download, what
  to verify next.
- Technical metrics (AUC, LOO, CI, Spearman, NDMI internals) belong in specialist
  expanders unless the user explicitly asks for science details.
- Repeated caveat: this is preliminary screening; final planting decisions require
  field soil sampling and local agronomic review.
