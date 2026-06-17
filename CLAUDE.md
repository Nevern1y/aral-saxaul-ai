# CLAUDE.md — Aral Saxaul AI

Project context for Claude Code. Current as of V5.1 (2026-06). The authoritative
human-facing doc is `README.md`; this file is the operational summary for agents.

## What this is

A **rule-based remote-sensing screening pipeline** that flags candidate zones for
saxaul (Haloxylon) planting on the dried bed of the Aral Sea (Aralkum). It ingests
Sentinel-2 L2A (10 m) + Copernicus DEM via Google Earth Engine, applies adaptive
spectral/topographic rules, and serves the result through a Streamlit dashboard
with logistics (KML field tasks, OSM road distances).

It is **not** a trained habitat model. `Candidate suitable` is a residual class
left after excluding water/shadow, slope obstacles, vegetation, dry-salt, and
wet-brine proxies. It marks places to field-check, not proven planting sites.

## Current version: V5.1 only

V1.0–V4.1 are historical and frozen. Do NOT add fallbacks to old versions or
revive old artifacts. The dashboard (`app.py`) reads V5 outputs strictly with no
V4 fallback. The science layer (provenance, dual-coordinate validation,
uncertainty diagnostics) is what distinguishes V5.1 from plain V5.

Version history lives in `README.md` ("История версий"). Three obsolete design
docs (SCRUM.md, PROJECT_CONTEXT.md, V3_ARCHITECTURE.md) were removed in this
cleanup — don't recreate them; update `README.md` instead.

## Layout

- `app.py` — **standalone** Streamlit dashboard (3 tabs: map+summary, checks+data,
  work-planning). Imports nothing from `src/`. ~1000 lines.
- `scripts/*v5*.py` — the current V5.1 pipeline (fetch → inference → viz → stats →
  logistics → science). See README "Полный пайплайн V5.1".
- `scripts/v5_rules.py`, `scripts/v5_config.py` — class scheme + config.
- `scripts/qa_sanity_check.py`, `scripts/qa_forensic_audit_v5.py` — Red Team audits.
- **Legacy V1–V4 code (`src/`, `main.py`, the old XGBoost pipeline, all
  `run_inference_v2/v3/v4`, `phase5_v3/v4_*`, etc.) has been removed.** Only the
  V5.1 set remains. If you need it, recover from git history; don't re-add it.
- **Producers that look old but are LIVE V5.1 inputs — do not delete:**
  - `scripts/prepare_slope_data.py` → `dem_slope_30m.tif`. Slope is part of the V5
    classification rules; `fetch_gee_raw_v5.py` does **not** export the DEM.
  - `scripts/build_ground_truth.py` / `validate_ground_truth.py` → `ground_truth_v2.csv`,
    read by `build_v5_science_dataset.py` (the "v2" name is historical, the data is current).
  - `scripts/download_aoi.py` / `fetch_historic_aral_gee.py` / `fetch_historic_aral_jrc.py`
    → `outputs/aoi/aral_sea_1960.geojson`, read by `app.py` and `fetch_gee_raw_v5.py`.
- `outputs/` — generated data (gitignored: *.tif/*.vrt, large GeoJSON). The
  dashboard degrades gracefully when absent. Committed V1–V4 artifacts
  (`xgb_classifier.pkl`, `optimal_zones_v3/v4.geojson`, etc.) are unused by V5.1
  but retained; `app.py` reads only V5 artifacts + the shared 1960 coastline.
- `tests/` — pytest + Streamlit AppTest suite (added in tooling setup).
- `.claude/` — hooks, subagents, skills (see below).

## Running

```bash
pip install -r requirements.txt        # app/runtime deps (Python 3.12)
streamlit run app.py                    # dashboard, port 8501

pip install -r requirements-dev.txt     # pytest + ruff (NOT for Streamlit Cloud)
python -m pytest                         # 9 tests; smoke + data contracts
```

Full pipeline regeneration (needs GEE auth + local GIS libs): see README
"Полный пайплайн V5.1", or one-shot: `python scripts/run_v5_science_suite.py
--products --refresh-roads --qa`.

## Deployment (Streamlit Cloud)

- Entry point `app.py`, branch `main`, deps from `requirements.txt`.
- **`environment.yml` must never be committed** — Streamlit Cloud switches to
  conda and crashes (it is gitignored; multiple past commits had to remove it).
- Maps render via `_render_map()` → `streamlit.components.v1.html` (app.py:65-67).
  This path is load-bearing: earlier `st.html` / `data:` URI / srcdoc attempts all
  failed (CSP blocks JS / strips it). The git history has 7+ commits chasing this.
- **Known time-bomb:** `st.components.v1.html` is deprecated ("removed after
  2026-06-01") in Streamlit ≥1.56 (installed: 1.57). Migrating to `st.iframe`
  must be verified against the live site with the `/verify-map-render` skill —
  AppTest cannot see the map.
- `app.py` guards `import rasterio` with try/except (None when unavailable). Keep
  every rasterio/geopandas use behind an existence/None check so the cloud app
  doesn't crash if a GIS wheel is missing.

## Conventions

- Russian is the primary language for UI strings and docs; keep it. Code
  identifiers and commit messages are English.
- Tests skip cleanly when `outputs/` data is absent (it's gitignored) — never
  hard-require generated artifacts in tests.
- NDMI is the primary salinity marker (Spearman r≈+0.69); high NDMI on the dry bed
  means capillary saline moisture/brine, NOT favorable moisture. Don't invert this.
- Thresholds are adaptive P15/P85 per scene; preserve ordering invariants
  (P15 ≤ P85).

## Tooling in .claude/

- **Hook** `PostToolUse` → `.claude/hooks/ruff_on_edit.py`: formats + lints edited
  `.py` files (no-op if ruff absent; never blocks).
- **Subagent** `qa-red-team-auditor`: audits regenerated pipeline outputs (raster
  integrity, class distribution, cross-projection area, threshold plausibility).
- **Subagent** `streamlit-deploy-reviewer`: pre-flights diffs against the deploy
  traps above before pushing.
- **Skill** `/verify-map-render`: Playwright check that the Folium map actually
  renders on the deployed/local site (pierces the components iframe, checks
  Leaflet + tiles + no CSP errors).
