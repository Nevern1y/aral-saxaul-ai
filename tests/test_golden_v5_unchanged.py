"""Enforce the V5.1 freeze against the golden snapshot (RISK-11).

The entire V6-companion design rests on one invariant: **the frozen V5.1 product
never changes**. ``scripts/v6/snapshot_v5_golden.py`` captured that invariant into
``tests/golden/v5_golden_snapshot.json`` (thresholds, class distribution, headline
stats) — but until now *no test read it*, so the freeze was unenforced and a V6
change could silently perturb the shipped V5 map with a green suite.

This test compares the live V5 outputs against that snapshot:

- ``thresholds_v5.json``                     — exact match (the adaptive P15/P85 gates).
- ``v5_stats.json`` invariants               — class pixel histogram, totals, cluster
                                               count, candidate area (the numbers the
                                               dashboard's metrics panel renders).
- ``suitability_map_v5_filtered.tif``        — content fingerprint: dimensions/CRS plus
                                               the per-class pixel histogram, which must
                                               equal the snapshot's ``class_pixels_10m``
                                               (this is the raster the histogram was taken
                                               from; the unfiltered map differs by design).

Everything skips cleanly when the V5 data tree / rasters are absent (gitignored,
so this is the Streamlit-Cloud / clean-checkout condition), per project convention.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "outputs" / "data"
GOLDEN = ROOT / "tests" / "golden" / "v5_golden_snapshot.json"

THRESHOLDS = DATA / "thresholds_v5.json"
STATS = DATA / "v5_stats.json"
V5_MAP_FILTERED = DATA / "suitability_map_v5_filtered.tif"


def _load_snapshot() -> dict:
    if not GOLDEN.exists():
        pytest.skip("golden snapshot missing (run scripts/v6/snapshot_v5_golden.py)")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_v5_thresholds_match_golden() -> None:
    snap = _load_snapshot()
    if "thresholds_v5" not in snap or not THRESHOLDS.exists():
        pytest.skip("V5 thresholds snapshot or live file absent")
    live = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    assert live == snap["thresholds_v5"], (
        "thresholds_v5.json drifted from the frozen V5.1 golden snapshot — "
        "the V5 freeze is broken."
    )


def test_v5_stats_invariants_match_golden() -> None:
    snap = _load_snapshot()
    inv = snap.get("v5_stats_invariants")
    if inv is None or not STATS.exists():
        pytest.skip("V5 stats snapshot or live file absent")
    live = json.loads(STATS.read_text(encoding="utf-8"))
    for key, want in inv.items():
        got = live.get(key)
        # class_pixels_10m keys are strings in JSON on both sides; compare directly.
        assert got == want, (
            f"v5_stats.json['{key}'] drifted from the golden snapshot "
            f"(V5 freeze broken): snapshot={want!r} live={got!r}"
        )


def test_v5_filtered_map_fingerprint_matches_golden() -> None:
    """The frozen 10 m product's class histogram + geometry must match the snapshot."""
    snap = _load_snapshot()
    inv = snap.get("v5_stats_invariants") or {}
    ref_hist = inv.get("class_pixels_10m")
    ref_total = inv.get("class_total_pixels_10m")
    if ref_hist is None or not V5_MAP_FILTERED.exists():
        pytest.skip("V5 filtered map or class-pixel snapshot absent (raster gitignored)")
    try:
        import numpy as np
        import rasterio
    except ImportError:
        pytest.skip("rasterio/numpy not available")

    with rasterio.open(V5_MAP_FILTERED) as src:
        # Geometry fingerprint: the frozen product is EPSG:32641 at a fixed size.
        assert str(src.crs) == "EPSG:32641", f"V5 map CRS changed: {src.crs}"
        assert (src.width, src.height) == (26786, 34033), (
            f"V5 map dimensions changed: {src.width}x{src.height}"
        )
        hist: Counter[int] = Counter()
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            vals, counts = np.unique(arr, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                hist[int(v)] += int(c)

    total = sum(hist.values())
    assert total == ref_total, (
        f"V5 filtered map total pixels changed: live={total:,} snapshot={ref_total:,}"
    )
    # Every class count in the snapshot must reproduce exactly from the live raster.
    for code, want in ref_hist.items():
        got = hist.get(int(code), 0)
        assert got == want, (
            f"V5 filtered map class {code} pixel count drifted from golden snapshot "
            f"(V5 freeze broken): snapshot={want:,} live={got:,}"
        )
