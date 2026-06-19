# -*- coding: utf-8 -*-
"""Capture a golden snapshot of current V5.1 outputs before V6 changes.

This freezes the invariants the dashboard depends on (class distribution,
thresholds, headline stats) so any V6 change that silently alters the
production V5 map is caught by ``tests/golden`` as a regression.

Run once at the start of V6 work. Re-running overwrites the snapshot, so do
NOT re-run after V5 outputs have been touched.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "outputs" / "data"
GOLDEN = BASE / "tests" / "golden"


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    snapshot: dict = {"description": "V5.1 production invariants frozen before V6 science layer."}

    thresholds_path = DATA / "thresholds_v5.json"
    stats_path = DATA / "v5_stats.json"

    if thresholds_path.exists():
        snapshot["thresholds_v5"] = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        # Only the invariants the contract tests + UI metrics depend on.
        snapshot["v5_stats_invariants"] = {
            "class_pixels_10m": stats.get("class_pixels_10m"),
            "class_total_pixels_10m": stats.get("class_total_pixels_10m"),
            "candidate_10m_area_ha": stats.get("candidate_10m_area_ha"),
            "clusters": stats.get("clusters"),
            "area_ha": stats.get("area_ha"),
        }

    out = GOLDEN / "v5_golden_snapshot.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Golden snapshot written: {out}")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
