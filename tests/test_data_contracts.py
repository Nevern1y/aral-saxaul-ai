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

pytestmark = pytest.mark.skipif(
    not (app.BASE_DIR / "outputs" / "data" / "v5_stats.json").exists(),
    reason="local V5 data tree not present (outputs/ is gitignored)",
)


def test_v5_stats_has_keys_used_by_metrics_panel() -> None:
    stats = app.load_v5_stats()
    assert isinstance(stats, dict)
    # Keys consumed by the top metrics panel + audit figure in tab_analytics.
    for key in ("clusters", "class_pixels_10m", "class_total_pixels_10m"):
        assert key in stats, f"v5_stats.json missing '{key}'"


def test_class_pixels_loader_shape() -> None:
    pixels, total_px, pixel_area_ha = app.load_v5_class_pixels()
    assert isinstance(pixels, dict)
    assert isinstance(total_px, int)
    assert pixel_area_ha > 0
    # Candidate class (1) must be represented in the class histogram.
    assert 1 in pixels or "1" in pixels


def test_thresholds_have_percentile_bounds() -> None:
    th = app.load_v5_thresholds()
    assert isinstance(th, dict)
    # The map legend explains NDMI percentile gates; both bounds must exist.
    for key in ("NDMI_P15", "NDMI_P85"):
        assert key in th, f"thresholds_v5.json missing '{key}'"
    assert th["NDMI_P15"] <= th["NDMI_P85"]


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


def test_audit_figure_builds_from_real_pixels() -> None:
    import json

    pixels, total_px, _ = app.load_v5_class_pixels()
    fig = app.make_audit_fig(json.dumps({str(k): v for k, v in pixels.items()}), total_px)
    # With real data present the audit donut must build (not None).
    assert fig is not None
