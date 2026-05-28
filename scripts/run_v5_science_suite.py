"""Run the V5.1 scientific hardening suite end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    script_path = BASE / script_name
    cmd = [sys.executable, str(script_path)]
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5.1 science/provenance/validation workflow")
    parser.add_argument(
        "--products",
        action="store_true",
        help="Regenerate filtered map, stats, and logistics products before science reports",
    )
    parser.add_argument(
        "--refresh-roads",
        action="store_true",
        help="Refresh OSM road layers, including Kazakhstan access roads, before logistics products",
    )
    parser.add_argument("--qa", action="store_true", help="Also run sanity and forensic QA after science reports")
    args = parser.parse_args()

    steps = []
    if args.products:
        if args.refresh_roads:
            steps.append("v5_kazakhstan_boundary_prep.py")
            steps.append("v5_roads_prep.py")
        steps.extend([
            "v5_finalize_viz.py",
            "v5_extract_stats.py",
            "v5_logistics_prep.py",
        ])

    steps.extend([
        "build_v5_science_dataset.py",
        "v5_validation_report.py",
        "v5_coordinate_adjudication_report.py",
        "v5_uncertainty_report.py",
    ])
    if args.qa:
        steps.extend([
            "qa_sanity_check.py",
            "qa_forensic_audit_v5.py",
        ])

    print("V5.1 science suite", flush=True)
    for step in steps:
        run_step(step)
    print("\nV5.1 science suite complete", flush=True)


if __name__ == "__main__":
    main()
