#!/usr/bin/env python
"""
Aral Saxaul AI - Full-Pipeline Orchestrator.

Usage
-----
    conda activate aral-saxaul
    python main.py --all                          # Run every phase end-to-end
    python main.py --phase 1                      # Build AOI + Feature Stack
    python main.py --phase 2 --features <path>    # Generate synthetic labels
    python main.py --phase 3 --csv <path>         # Train XGBoost + Optuna + SHAP
    python main.py --phase 4 --tile-dir <path>    # Distributed inference
    python main.py --phase 5 --raster <path>      # Visualise suitability map
    python main.py --phase 1-3                    # Run Phases 1 through 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.utils import initialize_gee, logger


# ============================================================================
# Internal helpers
# ============================================================================


def _header(phase: str) -> None:
    print()
    print("=" * 62)
    print(f"  ARAL SAXAUL AI  -  PHASE {phase}")
    print("=" * 62)
    print()


def _save_checkpoint(artifacts: dict, path: str) -> None:
    """Persist a checkpoint JSON with paths to key artifacts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, default=str)
    logger.info("Checkpoint saved: %s", path)


def _load_checkpoint(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Phase runners
# ============================================================================


def _run_phase1(gcp_project: Optional[str]) -> Tuple:
    """Phase 1: AOI + Feature Stack."""
    _header("1 - DATA INGESTION & FEATURE ENGINEERING")
    from src.phase1_ingestion import run_phase1

    aoi, feature_stack = run_phase1(project=gcp_project)

    checkpoint = {
        "phase": 1,
        "aoi_area_km2": round(aoi.area(1).divide(1e6).getInfo(), 1),
        "feature_stack_bands": config.FEATURE_COLUMNS,
    }
    _save_checkpoint(checkpoint, str(config.output_dir / "phase1_checkpoint.json"))

    return aoi, feature_stack


def _run_phase2_local(aoi, scale_m=100) -> str:
    """Phase 2: Export raw bands from GEE -> compute indices locally -> sample -> label."""
    _header("2 - SYNTHETIC DATA GENERATION (LOCAL)")
    from src.phase2_local import run_phase2_local

    csv_path = run_phase2_local(aoi, scale_m=scale_m)

    import pandas as pd
    df = pd.read_csv(csv_path)
    checkpoint = {
        "phase": 2,
        "csv_path": csv_path,
        "n_rows": len(df),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
    }
    _save_checkpoint(checkpoint, str(config.output_dir / "phase2_checkpoint.json"))
    return csv_path


def _run_phase3(csv_path: str, n_trials: int) -> Tuple:
    """Phase 3: XGBoost training + Optuna + SHAP."""
    _header("3 - MODEL TRAINING & XAI")
    from src.phase3_training import run_phase3

    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Synthetic labels not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s.", len(df), csv_path)

    model, scaler, metrics = run_phase3(df, n_optuna_trials=n_trials)

    checkpoint = {
        "phase": 3,
        "model_path": str(config.output_dir / "models" / "xgb_classifier.pkl"),
        "scaler_path": str(config.output_dir / "models" / "scaler.pkl"),
        "feature_names_path": str(config.output_dir / "models" / "feature_names.json"),
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix_plot"},
    }
    _save_checkpoint(checkpoint, str(config.output_dir / "phase3_checkpoint.json"))

    return model, scaler


def _run_phase4(input_feature_stack: str) -> Tuple[str, dict]:
    """Phase 4: Local tile-based inference."""
    _header("4 - LOCAL INFERENCE")
    from src.phase4_local import run_phase4

    output_path = str(config.output_dir / "data" / "suitability_full.tif")
    run_phase4(input_path=input_feature_stack, output_path=output_path)

    import rasterio, numpy as np
    with rasterio.open(output_path) as src:
        proba = src.read(1)
        valid = proba != -1.0
        n_high = int((proba[valid] > 0.5).sum())
        stats = {
            "n_valid": int(valid.sum()),
            "n_high_suitability": n_high,
            "pct_high": round(100 * n_high / valid.sum(), 1) if valid.sum() > 0 else 0,
        }

    checkpoint = {"phase": 4, "probability_map": output_path, "inference_stats": stats}
    _save_checkpoint(checkpoint, str(config.output_dir / "phase4_checkpoint.json"))
    return output_path, stats


def _run_phase5(raster_path: str, threshold: float) -> Tuple[str, str, dict]:
    """Phase 5: Decision boundaries + visualisation."""
    _header("5 - DECISION BOUNDARIES & VISUALISATION")
    from src.phase5_viz import run_phase5

    map_path, geojson_path, stats = run_phase5(
        prob_raster_path=raster_path,
        prob_threshold=threshold,
    )

    checkpoint = {
        "phase": 5,
        "html_map": map_path,
        "geojson": geojson_path,
        "statistics": stats,
    }
    _save_checkpoint(checkpoint, str(config.output_dir / "phase5_checkpoint.json"))

    return map_path, geojson_path, stats


# ============================================================================
# Full pipeline
# ============================================================================


def run_all(
    gcp_project: Optional[str] = None,
    csv_path: Optional[str] = None,
    n_optuna_trials: int = config.OPTUNA_N_TRIALS,
    feature_stack_path: Optional[str] = None,
    prob_threshold: float = config.PROB_THRESHOLD,
    skip_phase4: bool = False,
) -> dict:
    """Execute the complete pipeline end-to-end."""
    start = time.time()

    # ── Phase 1 ────────────────────────────────────────────────────────
    logger.info("PHASE 1: AOI")
    initialize_gee(project=gcp_project)
    from src.phase1_ingestion import AOIBuilder
    aoi_builder = AOIBuilder(vectorize=False)
    aoi = aoi_builder.run()

    # ── Phase 2 ────────────────────────────────────────────────────────
    logger.info("PHASE 2: Synthetic Labels (local)")
    if csv_path:
        logger.info("Using pre-existing CSV: %s", csv_path)
    else:
        csv_path = _run_phase2_local(aoi)

    # ── Phase 3 ────────────────────────────────────────────────────────
    logger.info("PHASE 3: Training + Optuna + SHAP")
    import pandas as pd
    df = pd.read_csv(csv_path)
    model, scaler, metrics = _run_phase3(df, n_optuna_trials)

    # ── Phase 4 ────────────────────────────────────────────────────────
    if skip_phase4:
        logger.info("PHASE 4: SKIPPED")
        merged_path = None
        inf_stats = {}
    elif feature_stack_path and os.path.exists(feature_stack_path):
        logger.info("PHASE 4: Local inference from %s", feature_stack_path)
        merged_path, inf_stats = _run_phase4(feature_stack_path)
    else:
        logger.info("PHASE 4: Building feature stack then inferring")
        from src.phase2_local import export_s2_mosaic, export_dem, export_s1_vh
        scale = 100
        out_dir = str(config.output_dir / "data")
        export_s2_mosaic(aoi, scale, out_dir)
        export_dem(aoi, scale, out_dir)
        export_s1_vh(aoi, scale, out_dir)
        from src.phase2_local import _compute_feature_stack
        fs_path = _compute_feature_stack(out_dir, scale)
        merged_path, inf_stats = _run_phase4(fs_path)

    # ── Phase 5 ────────────────────────────────────────────────────────
    if merged_path is None or not os.path.exists(merged_path):
        logger.warning("PHASE 5: SKIPPED - Phase 4 output missing.")
        map_path, geojson_path, viz_stats = None, None, {}
    else:
        logger.info("PHASE 5: Visualisation")
        map_path, geojson_path, viz_stats = _run_phase5(merged_path, prob_threshold)

    elapsed = time.time() - start
    summary = {
        "pipeline": "aral_saxaul_ai",
        "elapsed_minutes": round(elapsed / 60, 1),
        "phase1": {"aoi_area_km2": round(aoi.area(1).divide(1e6).getInfo(), 1)},
        "phase2": {"synthetic_csv": csv_path},
        "phase3": {"model_pkl": model_path},
        "phase4": {"probability_map": merged_path},
        "phase5": {"html_map": map_path, "geojson": geojson_path},
    }

    summary_path = str(config.output_dir / "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 62)
    print("  PIPELINE COMPLETE")
    print("=" * 62)
    print(f"  Elapsed:   {elapsed / 60:.1f} min")
    print(f"  AOI area:  {summary['phase1']['aoi_area_km2']:.0f} km^2")
    print(f"  CSV:       {csv_path}")
    print(f"  Model:     {model_path}")
    print(f"  Raster:    {merged_path}")
    print(f"  Map:       {map_path}")
    print(f"  Summary:   {summary_path}")
    print("=" * 62)

    return summary


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Aral Saxaul AI - Full-Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                              Run full pipeline
  python main.py --phase 1                          Build AOI + Feature Stack
  python main.py --phase 2                          Run Phase 1+2
  python main.py --phase 3 --csv data/labels.csv    Train with pre-existing CSV
  python main.py --phase 4 --tile-dir tiles/        Run inference from local tiles
  python main.py --phase 5 --raster prob_map.tif    Visualise existing raster
  python main.py --phase 1-3                        Run first three phases
        """,
    )

    # ── Run mode ──────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 5 phases end-to-end.")
    group.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Run a specific phase or range: '1', '2', '1-3', '3-5'.",
    )

    # ── GEE ───────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=None, help="GCP project ID for Earth Engine.")

    # ── Phase 2 overrides ─────────────────────────────────────────
    parser.add_argument("--csv", type=str, default=None, help="Path to synthetic_labels.csv (skip Phase 2).")

    # ── Phase 3 overrides ─────────────────────────────────────────
    parser.add_argument("--trials", type=int, default=config.OPTUNA_N_TRIALS, help="Number of Optuna trials.")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip hyperparameter search.")

    # ── Phase 4 overrides ─────────────────────────────────────────
    parser.add_argument("--tile-dir", type=str, default=None, help="Directory with tile_*.tif files.")
    parser.add_argument("--workers", type=int, default=config.MAX_INFERENCE_WORKERS, help="Inference workers.")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip Phase 4 (inference).")

    # ── Phase 5 overrides ─────────────────────────────────────────
    parser.add_argument("--raster", type=str, default=None, help="Path to probability_map.tif.")
    parser.add_argument("--threshold", type=float, default=config.PROB_THRESHOLD, help="Suitability threshold.")

    return parser


def main() -> None:
    config.ensure_output_dirs()
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Aral Saxaul AI - Pipeline Orchestrator v%s", "1.0.0")

    # ── --all ─────────────────────────────────────────────────────
    if args.all:
        initialize_gee(project=args.project)
        run_all(
            gcp_project=args.project,
            csv_path=args.csv,
            n_optuna_trials=0 if args.skip_optuna else args.trials,
            tile_dir=args.tile_dir,
            max_workers=args.workers,
            prob_threshold=args.threshold,
            skip_phase4=args.skip_phase4,
        )
        return

    # ── --phase ───────────────────────────────────────────────────
    phase_spec = args.phase.strip()

    # Phase range: e.g. "1-3"
    if "-" in phase_spec:
        parts = phase_spec.split("-")
        start_p, end_p = int(parts[0]), int(parts[1])
        phases_to_run = list(range(start_p, end_p + 1))
    else:
        phases_to_run = [int(phase_spec)]

    for phase in phases_to_run:
        if phase == 1:
            initialize_gee(project=args.project)
            _run_phase1(gcp_project=args.project)

        elif phase == 2:
            if phase != phases_to_run[0]:
                # Try loading checkpoint from Phase 1
                chk = _load_checkpoint(str(config.output_dir / "phase1_checkpoint.json"))
                logger.info("Phase 2: AOI from checkpoint (area=%s km^2).", chk.get("aoi_area_km2"))
                # Re-run Phase 1 mini to get live objects
                initialize_gee(project=args.project)
                from src.phase1_ingestion import run_phase1
                aoi, fs = run_phase1(project=args.project)
            else:
                initialize_gee(project=args.project)
                from src.phase1_ingestion import run_phase1
                aoi, fs = run_phase1(project=args.project)
            _run_phase2(fs, aoi)

        elif phase == 3:
            csv_path = args.csv or str(config.output_dir / "data" / "synthetic_labels.csv")
            _run_phase3(csv_path, n_trials=0 if args.skip_optuna else args.trials)

        elif phase == 4:
            model_path = str(config.output_dir / "models" / "xgb_classifier.pkl")
            scaler_path = str(config.output_dir / "models" / "scaler.pkl")
            feature_path = str(config.output_dir / "models" / "feature_names.json")
            _run_phase4(
                model_path=model_path,
                scaler_path=scaler_path,
                feature_names_path=feature_path,
                tile_dir=args.tile_dir,
                max_workers=args.workers,
            )

        elif phase == 5:
            raster_path = args.raster or str(config.output_dir / "probability_map.tif")
            _run_phase5(raster_path, threshold=args.threshold)

        else:
            parser.error(f"Unknown phase: {phase}. Valid: 1–5.")

    logger.info("Requested phases complete.")


if __name__ == "__main__":
    main()
