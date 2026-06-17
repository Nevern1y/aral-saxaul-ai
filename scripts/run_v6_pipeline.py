"""Run the V6 lab-data science pipeline end-to-end (reproducible one-shot).

Dependency order (each step consumes the previous step's outputs):

  1. v6/extract_docx_tables.py     docx -> interim long tables       [--docx]
  2. v6/build_canonical_db.py      -> profiles_v6.csv, soil_layers_v6.csv
  3. v6/extract_saxaul_labels.py   -> saxaul_labels_v6.csv
  4. v6/build_ml_dataset.py        + RS rasters -> ml_dataset_v6.csv  [needs rasters]
  5. v6/calibrate_thresholds.py    -> thresholds_v6_calibrated.json
  6. v6/train_suitability_model.py -> salinity_v6_logit.json (+ exploratory)
  7. v6/build_suitability_index.py -> suitability rasters, stats, pit validation [needs rasters]
  8. v6/spatial_validation.py      -> spatial CV, bootstrap CIs, uncertainty raster [needs rasters]

Steps 1, 4, 7, 8 need large local inputs (the 22 MB docx and the gitignored
GeoTIFF stack). On a clean checkout those are absent, so the raster-dependent
steps are SKIPPED by default and only the CSV/JSON science steps that can run
from tracked inputs are executed. Use the flags to opt in:

  --docx     also run step 1 (requires the source .docx next to the repo)
  --rasters  also run steps 4, 7, 8 (require the local GeoTIFF stack)
  --all      run everything (--docx + --rasters)
  --qa       run the QA red-team after the pipeline (sanity + forensic + science)

Examples:
  python scripts/run_v6_pipeline.py --all --qa     # full rebuild + QA
  python scripts/run_v6_pipeline.py                # tracked-input steps only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
V6 = BASE / "v6"

# (script, needs_rasters, needs_docx)
STEPS = [
    ("v6/extract_docx_tables.py", False, True),
    ("v6/build_canonical_db.py", False, False),
    ("v6/extract_saxaul_labels.py", False, False),
    ("v6/build_ml_dataset.py", True, False),
    ("v6/calibrate_thresholds.py", False, False),
    ("v6/train_suitability_model.py", False, False),
    ("v6/build_suitability_index.py", True, False),
    ("v6/spatial_validation.py", True, False),
]

QA_STEPS = [
    "qa_sanity_check.py",
    "qa_forensic_audit_v5.py",
    "v6/qa_science_audit_v6.py",
]


def run_step(rel: str) -> None:
    path = BASE / rel
    if not path.exists():
        print(f"!!! missing script: {rel} (skipping)", flush=True)
        return
    cmd = [sys.executable, "-X", "utf8", str(path)]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False, env=_utf8_env())
    if result.returncode != 0:
        raise SystemExit(f"step failed ({result.returncode}): {rel}")


def _utf8_env():
    import os
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the V6 lab-data science pipeline")
    ap.add_argument("--docx", action="store_true", help="also extract from the source .docx (step 1)")
    ap.add_argument("--rasters", action="store_true", help="also run raster-dependent steps (4,7,8)")
    ap.add_argument("--all", action="store_true", help="run everything (--docx + --rasters)")
    ap.add_argument("--qa", action="store_true", help="run the QA red-team afterwards")
    args = ap.parse_args()

    do_docx = args.docx or args.all
    do_rasters = args.rasters or args.all

    planned, skipped = [], []
    for rel, needs_rasters, needs_docx in STEPS:
        if needs_docx and not do_docx:
            skipped.append((rel, "needs --docx"))
            continue
        if needs_rasters and not do_rasters:
            skipped.append((rel, "needs --rasters"))
            continue
        planned.append(rel)

    print("V6 pipeline", flush=True)
    print("  steps to run:", flush=True)
    for s in planned:
        print(f"    - {s}", flush=True)
    if skipped:
        print("  skipped (missing large inputs / not opted in):", flush=True)
        for s, why in skipped:
            print(f"    - {s}  [{why}]", flush=True)

    for rel in planned:
        run_step(rel)

    if args.qa:
        print("\n=== QA red-team ===", flush=True)
        for rel in QA_STEPS:
            run_step(rel)

    print("\nV6 pipeline complete", flush=True)


if __name__ == "__main__":
    main()
