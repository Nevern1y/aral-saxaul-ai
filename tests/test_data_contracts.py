"""Contract tests: pipeline outputs must match what the dashboard reads.

These call app.py's own @st.cache_data loaders directly (no Streamlit runtime
needed) and assert the schema the UI depends on. When a script regenerates an
output with a renamed key or dropped column, the dashboard would otherwise show
an empty panel; these tests turn that into a clear failure.

All tests skip when the local V5 data tree is absent (it is gitignored), so the
suite stays green on a clean checkout / Streamlit Cloud.
"""

from __future__ import annotations

import pandas as pd
import pytest

import app

V5_DATA_PRESENT = (app.BASE_DIR / "outputs" / "data" / "v5_stats.json").exists()
v5_only = pytest.mark.skipif(
    not V5_DATA_PRESENT,
    reason="local V5 data tree not present (outputs/ is gitignored)",
)


@v5_only
def test_v5_stats_has_keys_used_by_metrics_panel() -> None:
    stats = app.load_v5_stats()
    assert isinstance(stats, dict)
    # Keys consumed by the top metrics panel + audit figure in tab_analytics.
    for key in ("clusters", "class_pixels_10m", "class_total_pixels_10m"):
        assert key in stats, f"v5_stats.json missing '{key}'"


@v5_only
def test_class_pixels_loader_shape() -> None:
    pixels, total_px, pixel_area_ha = app.load_v5_class_pixels()
    assert isinstance(pixels, dict)
    assert isinstance(total_px, int)
    assert pixel_area_ha > 0
    # Candidate class (1) must be represented in the class histogram.
    assert 1 in pixels or "1" in pixels


@v5_only
def test_thresholds_have_percentile_bounds() -> None:
    th = app.load_v5_thresholds()
    assert isinstance(th, dict)
    # The map legend explains NDMI percentile gates; both bounds must exist.
    for key in ("NDMI_P15", "NDMI_P85"):
        assert key in th, f"thresholds_v5.json missing '{key}'"
    assert th["NDMI_P15"] <= th["NDMI_P85"]


@v5_only
def test_tasks_index_columns() -> None:
    df = app.load_tasks()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Columns the logistics tab filters/sorts on.
    expected = {
        "filename",
        "centroid_lat",
        "centroid_lon",
        "area_ha",
        "distance_to_road_km",
    }
    missing = expected - set(df.columns)
    assert not missing, f"tasks index missing columns: {missing}"


@v5_only
def test_audit_figure_builds_from_real_pixels() -> None:
    import json

    pixels, total_px, _ = app.load_v5_class_pixels()
    fig = app.make_audit_fig(json.dumps({str(k): v for k, v in pixels.items()}), total_px)
    # With real data present the audit donut must build (not None).
    assert fig is not None


# ── V6 lab-data science layer contracts ────────────────────────────────
# These artifacts ARE tracked in git (JSON/CSV), so they should exist on a clean
# checkout; guard anyway so the suite is robust if a script has not been run.

def test_v6_science_loader_shape() -> None:
    v6 = app.load_v6_science()
    assert set(v6) >= {"salinity", "suit_stats", "pit_validation", "spatial", "benchmark", "pit_table"}


def test_v6_salinity_model_keys() -> None:
    sal = app.load_v6_science()["salinity"]
    if not sal:
        pytest.skip("salinity_v6_logit.json not present")
    # Keys the V6 metrics panel + the numpy inference path depend on.
    assert sal["predictor"] == "rs30_ndmi"
    assert {"mean", "std"} <= set(sal["scaler"])
    assert {"intercept", "rs30_ndmi"} <= set(sal["coefficients_standardized"])
    assert sal["training"]["n"] == 70


def test_v6_model_metrics_prefer_benchmark() -> None:
    v6 = app.load_v6_science()
    if not v6.get("benchmark"):
        pytest.skip("model_v6_benchmark.json not present")
    metrics = app.v6_model_metrics(v6)
    rec = v6["benchmark"]["recommendation"]
    assert metrics["auc"] == rec["baseline_loo_auc"]
    assert metrics["ci"] == rec["baseline_loo_auc_ci95"]


def test_v6_spatial_validation_auc_sane() -> None:
    sp = app.load_v6_science()["spatial"]
    if not sp:
        pytest.skip("spatial_validation_v6.json not present")
    sm = sp["salinity_model"]
    # LOO AUC and its bootstrap CI must be well-formed (lower bound >= 0.5 after the fix).
    assert 0.0 <= sm["loo_auc"] <= 1.0
    lo, hi = sm["loo_auc_ci95"]
    assert lo <= sm["loo_auc"] <= hi
    assert lo >= 0.5, "bootstrap AUC lower bound below 0.5 signals the in-sample-refit bug"
    # per-block spatial AUC must be reported (the honest spatial metric).
    assert "spatial_lbo_perblock_auc" in sm


def test_v6_suitability_zone_areas_present() -> None:
    stats = app.load_v6_science()["suit_stats"]
    if not stats:
        pytest.skip("suitability_v6_stats.json not present")
    zone_ha = stats["zone_area_ha"]
    # The V5.1 ZoneClass codes the dashboard table renders.
    for code in ("0", "1", "3", "4", "10"):
        assert code in zone_ha
    assert 0.0 <= stats["valid_fraction_of_aoi"] <= 1.0


def test_v6_map_html_self_contained() -> None:
    # The primary V6 map must be tracked HTML with a base64-embedded image so it
    # renders on Streamlit Cloud (the source .tif is gitignored).
    p = app.V6_MAP_PATH
    if not p.exists():
        pytest.skip("suitability_map_v6.html not present (run render_v6_map.py)")
    txt = p.read_text(encoding="utf-8")
    assert "data:image/png;base64" in txt, "V6 map must embed image as base64 (not a file path)"
    assert "leaflet" in txt.lower(), "V6 map must be a Leaflet/Folium map"
    assert "What the selected spot shows" in txt, "V6 map must explain hovered locations"
    assert "Low-salinity-risk score" in txt, "V6 map must show a human-readable low-salt score"
    assert "Estimated salinity risk" in txt, "V6 map must translate score into salinity risk"
    assert "map.on('mousemove'" in txt, "V6 map must keep hover/click decision panel"
    assert "map.on('click'" in txt, "V6 map must update decision panel on click"
    assert "V6 salinity risk zones" in txt
    assert "V6 score: higher = lower salinity risk (0..1)" in txt


def test_v6_calibrated_thresholds_omit_10m() -> None:
    # W12 policy (see scripts/v6/calibrate_thresholds.py): the V5.1 10 m
    # cascade stays frozen. This must never silently start publishing 10 m
    # cuts here.
    path = app.BASE_DIR / "data" / "canonical" / "thresholds_v6_calibrated.json"
    if not path.exists():
        pytest.skip("thresholds_v6_calibrated.json not present")
    import json

    thresholds = json.loads(path.read_text(encoding="utf-8"))
    keys = thresholds.get("calibrated_thresholds", {})
    ten_m_keys = [
        k for k in keys
        if "_10m" in k or (k.startswith("rs_") and not k.startswith("rs30_"))
    ]
    assert not ten_m_keys, f"10 m thresholds must not be published (frozen policy): {ten_m_keys}"
