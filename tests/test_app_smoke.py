"""Smoke tests: the whole dashboard script must execute without raising.

Uses Streamlit's AppTest harness (headless, no browser). This is the single
highest-value guard for app.py — it re-runs the entire 1000-line script and
fails on any uncaught exception, which is exactly the class of regression that
a refactor of the data loaders or layout tends to introduce.

NOTE: AppTest cannot see the Folium maps. They are injected via Streamlit's
iframe API (see app.py _render_map) and are opaque to the harness. Visual map
verification lives in the `verify-map-render` skill (Playwright against the
deployed site), not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"

# app.py does heavy CSV/JSON/raster I/O at import+run time; 3s default is too low.
TIMEOUT = 90


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception(app: AppTest) -> None:
    # `at.exception` collects uncaught exceptions from the script run.
    assert not app.exception, [repr(e.value) for e in app.exception]


def test_three_tabs_present(app: AppTest) -> None:
    # app.py builds st.tabs([...]) with exactly three sections.
    assert len(app.tabs) == 3


def test_title_rendered(app: AppTest) -> None:
    titles = [t.value for t in app.title]
    assert any("Карта предварительной оценки засоленности почв" in t for t in titles)


def test_renders_data_tables(app: AppTest) -> None:
    # The dashboard surfaces multiple dataframes (tasks, validation, etc.).
    # Don't pin an exact count (it shifts with content), just require presence.
    assert len(app.dataframe) > 0
