# Aral Saxaul AI — V6 Salinity-Risk Screening

**Aral Saxaul AI** is a Streamlit/Folium dashboard for preliminary field-trip
selection on the dried bed of the Aral Sea. The current product is **V6-first**:
it ranks places by laboratory-anchored soil-salinity risk and helps decide where
to inspect next.

It is **not** a planting plan and not a proven saxaul survival model. A green or
high-score area means “check this place first”, not “plant here without field
sampling”.

[Open the dashboard](https://aral-saxaul-ai-7skntrgzemfiwqmjd7feoo.streamlit.app/)

---

## Current Rule

| Layer | Current role | What it answers | What it must not be used for |
|---|---|---|---|
| **V6** | Main decision-support layer | Where salinity risk is lower/higher and where to field-check first | Final planting approval |
| **V5.1** | Frozen 10 m helper layer | Fine boundaries, contours, KML/logistics, road planning | Replacing the V6 salinity-risk score |

The operational rule is:

```text
V6 = current main layer: P(topsoil salinity > 1%) from lab data
V6 score = 1 - P(saline)
V5.1 = helper layer: 10 m detail, contours, KML, logistics
```

Use the result as a **screening aid**:

1. Open the V6 map.
2. Hover/click a colored location.
3. Read the salinity-risk score and suggested next step.
4. Use V5.1/KML to plan access and boundaries.
5. Go to the field and sample soil.
6. Only after field confirmation discuss planting and budget.

---

## What The Dashboard Does

The dashboard has three tabs:

| Tab | Purpose |
|---|---|
| **Map and summary** | V6 salinity-risk map, headline metrics, practical interpretation, V5.1 detail layer |
| **Checks and data** | What the result can/cannot be used for, validation numbers, specialist expanders |
| **Work planning** | Road/access filters, practical defaults, KML ZIP export, resource estimate with warnings |

The V6 map includes an interactive decision panel. On hover/click it shows:

- conclusion for the point;
- low-salt score from 0 to 100;
- approximate salinity risk from 0 to 100;
- V6 zone class;
- what to do next;
- coordinates.

The panel deliberately uses field language. It does not ask users to interpret ML
internals before making a reconnaissance decision.

---

## V6 Model

<!-- V6_SCOPE_AUTO -->
> **V6 scope (auto-generated — see `data/canonical/SCOPE_AND_LIMITATIONS_V6.md`):** Shipped salinity anchor = **M0 (univariate NDMI logit)**, LOO AUC 0.682 (CI [0.556, 0.802]). **Regional calibration drift (W1):** pooled spatial AUC 0.385 vs per-block 0.792 — the model ranks salinity *locally* but absolute level must be calibrated per region.
<!-- /V6_SCOPE_AUTO -->

V6 is the scientifically anchored layer. It is trained on measured soil salinity,
not on visual map classes.

| Item | Value |
|---|---|
| Training source | Pachikin/Kozybaeva soil report, 2012-2014 |
| Georeferenced soil profiles | 70 |
| Soil layers | 369 |
| Target | topsoil total salts > 1% |
| Positive saline profiles | 27 / 70 |
| Predictor used in shipped model | `rs30_ndmi` |
| Model | L2 logistic regression, pure NumPy IRLS |
| Output | `P(saline)` and `1 - P(saline)` |

The shipped model is intentionally simple:

```text
P(saline) = sigmoid(intercept + coef * standardized(NDMI))
V6 score = 1 - P(saline)
```

Current fitted values from `outputs/models/salinity_v6_logit.json`:

| Parameter | Value |
|---|---:|
| NDMI mean | -0.076845981 |
| NDMI std | 0.089559679 |
| intercept | -0.469524595 |
| NDMI coefficient | +0.324947379 |
| L2 lambda | 0.1 |

The positive NDMI coefficient is expected for this environment: high NDMI on the
dry seabed is interpreted as capillary saline moisture/brine, not favorable water.

---

## V6 Validation

| Metric | Value | Interpretation |
|---|---:|---|
| LOO AUC | 0.682 | useful signal, not final operational accuracy |
| 95% CI | [0.556, 0.802] | interval is above chance but still wide |
| In-sample AUC | 0.767 | not used as the main claim |
| In-AOI pits | 15 | small target-area validation subset |
| Out-of-AOI pits | 55 | training support from wider Priaralye, not independent seabed validation |
| Pooled spatial AUC | 0.385 | poor cross-region absolute calibration |
| Mean per-block spatial AUC | 0.792 | good local ranking signal |
| NDMI->salt sign in blocks | 5/5 positive | salinity signal is directionally stable |

The honest scientific conclusion is:

> V6 is suitable for local salinity-risk ranking and field-check prioritization.
> Comparing distant regions by one absolute score requires local calibration.

---

## Why There Is No Saxaul-Survival Model Yet

The project tested a direct saxaul classifier, but it is not decision-grade.

| Attempt | Result |
|---|---:|
| Direct NDMI+MSAVI -> saxaul classifier | LOO AUC about 0.48 |
| Positive saxaul labels | 6 |
| Status | exploratory only, not for decisions |

This is why the current science is anchored on salinity. The project does **not**
claim that V6 predicts survival or planting success. It claims that V6 estimates a
key limiting factor: topsoil salinity risk.

**A note on the saxaul field-evidence pattern.** Beyond the direct classifier above, the project also holds a small set of field pits where soil chemists recorded whether saxaul was actually present or absent, alongside soil chemistry. Only 6 pits in the whole dataset carry a confirmed saxaul-present label. Correlations between salinity-related soil properties (chloride, exchangeable sodium, carbonate, texture) and saxaul presence in these pits run in ecologically sensible directions (saxaul favors lower chloride, lower sodium, more carbonate, sandier soil) but every one falls below the project's own stability gates (MIN_N=12, MIN_N_POS=8, MIN_AUC=0.62) and is flagged `indicative_only` in the data.

Field notes also include documented plantation failures and barren, vegetation-free pits with no saxaul, plus at least one pit where saxaul was growing successfully alongside other salt-tolerant shrubs on salt-affected ground. Per the labeling guardrails in `SAXAUL_LABELS_QA.md`, these negative labels are assigned from documented outcomes (failure to establish, genuine bareness), not from measured salinity itself, so not every "no saxaul" record is a high-salinity site. Treat all of this as a screening hint about a limiting factor, not as a validated ecological model, and never as proof that any specific site will or will not support saxaul.

---

## V6 Raster Products

| Artifact | Meaning |
|---|---|
| `outputs/data/suitability_index_v6.tif` | continuous `1 - P(saline)` score, 0..1 |
| `outputs/data/suitability_zones_v6.tif` | zone view using V5.1-compatible class codes |
| `outputs/data/suitability_uncertainty_v6.tif` | uncertainty of `P(saline)` |
| `outputs/reports/suitability_map_v6.html` | self-contained Folium map used by Streamlit |
| `outputs/reports/suitability_map_v6_lookup.png` | hidden lookup layer for hover/click decisions |

V6 zones:

| Code | V6 meaning |
|---:|---|
| 1 | Candidate / lower salinity risk |
| 3 | Moderate salinity risk |
| 4 | Strong salinity risk |
| 10 | Existing vegetation |
| 0 | Water / NoData |

Important caveat: zones 3/4 are a **severity split on one validated NDMI axis**.
They do not reproduce every V5.1 wet-brine vs dry-salt physical distinction.

---

## V5.1 Helper Layer

V5.1 is frozen and remains useful, but it is not the current main decision layer.

V5.1 uses Sentinel-2 10 m rules:

| Class | Rule |
|---|---|
| Water/NoData/Shadow | SCL water/cloud/shadow, NDWI > 0, invalid data, dark pixels |
| Obstacle | slope > 5 degrees |
| Vegetation | NDVI > 0.08 |
| Wet brine risk | NDMI > P85 and B8/B12 > P85 |
| Dry salt risk | NDSI Green/SWIR2 > P85 and NDMI < P15 |
| Candidate | residual class after exclusions |

V5.1 outputs support logistics:

| Artifact | Use |
|---|---|
| `outputs/data/operational_zones_v5.geojson` | helper contours |
| `outputs/data/operational_zones_v5_kazakhstan.geojson` | Kazakhstan-clipped helper contours |
| `outputs/logistics/tasks_index_v5_enriched.csv` | KML task index with distances |
| `outputs/logistics/tractor_tasks_v5/*.kml` | GPS/Google Earth field files |

V5.1 candidate area is a screening estimate, not a planting area.

---

## End-To-End Workflow

### V6 science pipeline

```bash
python scripts/run_v6_pipeline.py --all --qa
```

Pipeline order:

| Step | Script | Output |
|---:|---|---|
| 1 | `scripts/v6/extract_docx_tables.py` | extracted source tables |
| 2 | `scripts/v6/build_canonical_db.py` | `profiles_v6.csv`, `soil_layers_v6.csv` |
| 3 | `scripts/v6/morph_features.py` | `morph_features_v6.csv` |
| 4 | `scripts/v6/extract_saxaul_labels.py` | `saxaul_labels_v6.csv` |
| 5 | `scripts/v6/build_ml_dataset.py` | `ml_dataset_v6.csv` |
| 6 | `scripts/v6/calibrate_thresholds.py` | `thresholds_v6_calibrated.json` |
| 7 | `scripts/v6/train_suitability_model.py` | `salinity_v6_logit.json` |
| 8 | `scripts/v6/build_suitability_index.py` | V6 rasters and stats |
| 9 | `scripts/v6/render_v6_map.py` | V6 Folium HTML/PNG map |
| 10 | `scripts/v6/spatial_validation.py` | spatial CV, CI, uncertainty raster |
| 11 | `scripts/v6/qa_science_audit_v6.py` | adversarial science QA |

Some steps need local large inputs: source DOCX, GeoTIFF rasters, and GEE-derived
stacks. JSON/CSV/MD science artifacts are tracked; large rasters are regenerated.

### V5.1 helper/logistics pipeline

```bash
python scripts/fetch_gee_raw_v5.py
python scripts/prepare_slope_data.py
python scripts/run_inference_v5.py
python scripts/v5_finalize_viz.py
python scripts/v5_extract_stats.py
python scripts/v5_kazakhstan_boundary_prep.py
python scripts/v5_roads_prep.py
python scripts/v5_logistics_prep.py
```

Or for the V5.1 product/science suite:

```bash
python scripts/run_v5_science_suite.py --products --refresh-roads --qa
```

---

## Running The Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Development checks:

```bash
pip install -r requirements-dev.txt
python -m ruff check app.py tests scripts/v6/render_v6_map.py scripts/v6/build_suitability_index.py
python -m pytest tests
```

On this Windows workstation, use the system Python 3.12 interpreter for tests:

```powershell
& "C:\Users\cynok\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests
```

---

## Scientific And Commercial Readiness

Current defensible use:

- research screening prototype;
- field reconnaissance prioritization;
- salinity-risk decision support;
- KML/logistics generation for field teams;
- paid pilot only with explicit disclaimers.

Not yet defensible:

- final planting plan;
- planting success guarantee;
- regulatory-grade land suitability decision;
- insurance/investment-grade automated decision;
- direct saxaul survival prediction.

Before stronger commercial use, collect fresh stratified field samples inside the
1960 seabed AOI, calibrate regions separately, add groundwater/root-zone data, and
link predictions to real saxaul survival outcomes.

---

## Known Limitations

- V6 labels are 2012-2014 soil measurements; recent imagery assumes the
  NDMI-salinity relation is quasi-stationary.
- Only about 15 of 70 training profiles are inside the 1960 seabed AOI.
- Cross-region absolute scores drift; local ranking is more defensible.
- The 30 m stack has no Sentinel-2 SCL water mask.
- The 30 m slope band is finite for only about 33% of the AOI (flat seabed terrain, not a missing-data gap); this is too sparse for a slope gate.
- V5.1 10 m validation has only 14 valid production pixels at the V6 soil points (10 m NDMI vs. measured salinity correlation is not significant, rho approx 0.42, p approx 0.13); recalibration is deferred until a future campaign yields n>=30 valid 10 m points (trigger defined in scripts/v6/calibrate_thresholds.py).
- Direct saxaul labels are too few for a reliable survival model.
- Remote sensing cannot replace root-zone salinity, groundwater depth, field EC,
  or agronomic inspection.

---

## What To Say About The Project

Safe wording:

> This is a V6 salinity-risk screening dashboard. It uses a lab-anchored NDMI
> logistic model to rank places for field checking. V5.1 provides 10 m detail and
> KML logistics. It supports reconnaissance; it does not authorize planting.

Unsafe wording:

> The map proves where saxaul can be planted.

---

## Current Files To Know

| Path | Role |
|---|---|
| `app.py` | Streamlit dashboard |
| `scripts/run_v6_pipeline.py` | V6 end-to-end science pipeline |
| `scripts/v6/train_suitability_model.py` | shipped salinity model producer |
| `scripts/v6/build_suitability_index.py` | V6 raster producer |
| `scripts/v6/render_v6_map.py` | interactive V6 map producer |
| `scripts/v6/qa_science_audit_v6.py` | V6 red-team QA |
| `data/canonical/SCOPE_AND_LIMITATIONS_V6.md` | auto-generated V6 scope statement |
| `data/canonical/model_v6_benchmark.json` | honest benchmark table |
| `outputs/models/salinity_v6_logit.json` | shipped model coefficients |

Legacy pre-V5 code and old XGBoost-style artifacts are not current product paths
and should not be used for claims or UI decisions.

---

## Data Licenses

Data sources include OpenStreetMap, Copernicus/Sentinel, NASA/SRTM/JRC-derived
products, and local soil report tables. Before selling a hosted commercial product,
review each source license and attribution requirement, especially OSM ODbL and
Google Earth Engine terms.
