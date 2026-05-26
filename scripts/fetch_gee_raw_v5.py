"""
fetch_gee_raw_v5.py — Export native-resolution Sentinel-2 bands from GEE.

Exports six single-band GeoTIFFs to Google Drive, preserving native resolution:
  10m:  B3, B4, B8, SCL (median/mode composite)
  20m:  B11, B12        (median composite)

Usage:
    python scripts/fetch_gee_raw_v5.py [--drive-folder <name>] [--crs <EPSG>]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path for V5.0 standalone imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ee

from scripts.v5_config import ARAL_BBOX, EXPORT_CRS, DRIVE_FOLDER, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_gee_v5")

# ── Defaults ────────────────────────────────────────────────────────────
SEASON_START: str = "2024-05-01"
SEASON_END: str = "2024-05-15"
MAX_CLOUD_PCT: int = 10
MAX_PIXELS: int = 1_000_000_000_000  # 1e12 — sufficient for the full AOI


# =========================================================================
# 1. BUILD AOI
# =========================================================================


def _build_aoi() -> ee.Geometry:
    """AOI from GeoJSON or fallback to BBOX."""
    aoi_path = Path(PROJECT_ROOT) / "outputs" / "aoi" / "aral_sea_1960.geojson"
    if aoi_path.exists():
        import json
        with open(aoi_path, encoding="utf-8") as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            geom = geojson["features"][0]["geometry"]
        elif geojson.get("type") == "Feature":
            geom = geojson["geometry"]
        else:
            geom = geojson
        log.info("AOI loaded from GeoJSON: %s", aoi_path.name)
        return ee.Geometry(geom)
    min_lon, min_lat, max_lon, max_lat = ARAL_BBOX
    geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    log.info("AOI from BBOX: %.1f°E–%.1f°E / %.1f°N–%.1f°N",
             min_lon, max_lon, min_lat, max_lat)
    return geom


# =========================================================================
# 2. BUILD SENTINEL-2 COLLECTION & COMPOSITES
# =========================================================================


def _build_composites(aoi: ee.Geometry):
    """Return dict ``{band_name: ee.Image}`` ready for export."""
    log.info("Filtering Sentinel-2 SR HARMONIZED …")
    log.info("  Window: %s – %s    Cloud < %d%%",
             SEASON_START, SEASON_END, MAX_CLOUD_PCT)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(SEASON_START, SEASON_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
    )

    count = col.size().getInfo()
    if count == 0:
        log.error("No scenes found. Try widening the time/cloud filter.")
        sys.exit(1)
    log.info("  Scenes found: %d", count)

    # ── 10 m spectral bands (median) ──────────────────────────────────
    bands_10m = col.select(["B3", "B4", "B8"]).median()
    # clamps to valid reflectance range
    bands_10m = bands_10m.uint16()

    # ── SCL (mode for categorical data) ───────────────────────────────
    scl = (col
           .select("SCL")
           .reduce(ee.Reducer.mode())
           .toByte()
           .rename("SCL"))

    # ── 20 m spectral bands (median) ──────────────────────────────────
    bands_20m = col.select(["B11", "B12"]).median().uint16()

    composites: Dict[str, ee.Image] = {
        "B3":  bands_10m.select("B3"),
        "B4":  bands_10m.select("B4"),
        "B8":  bands_10m.select("B8"),
        "SCL": scl,
        "B11": bands_20m.select("B11"),
        "B12": bands_20m.select("B12"),
    }

    for name, img in composites.items():
        meta = img.getInfo()["bands"][0]
        log.info("  %-4s  crs=%-10s  dtype=%s", name, meta["crs"], meta["data_type"]["precision"])

    return composites


# =========================================================================
# 3. LAUNCH EXPORT TASKS
# =========================================================================


def _export_single(
    band_name: str,
    image: ee.Image,
    aoi: ee.Geometry,
    scale: int,
    folder: str,
    crs: str,
) -> ee.batch.Task:
    """Export a single-band image to Google Drive."""
    task_label = f"Aral_{band_name}_{scale}m_Spring2024"
    task = ee.batch.Export.image.toDrive(
        image=image.clip(aoi).unmask(0),
        description=task_label,
        folder=folder,
        fileNamePrefix=task_label,
        scale=scale,
        crs=crs,
        crsTransform=None,
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    log.info("  Task launched: %-35s  (id: %s)", task_label, task.id)
    return task


# =========================================================================
# 4. TASK MONITORING
# =========================================================================


POLL_INTERVAL_S: int = 30


def _poll_tasks(tasks: List[ee.batch.Task], timeout_s: int = 7200) -> None:
    """Poll task states until all complete or timeout."""
    pending = {t.id: t for t in tasks}
    n_total = len(pending)
    log.info("Monitoring %d export tasks (timeout: %ds) …", n_total, timeout_s)

    t_start = time.time()
    while pending and (time.time() - t_start) < timeout_s:
        completed = 0
        failed = 0
        running = 0
        ready = 0

        for tid, task in list(pending.items()):
            try:
                status = task.status()
                state = status.get("state", "UNKNOWN")
            except Exception:
                state = "ERROR"

            if state == "COMPLETED":
                completed += 1
                del pending[tid]
            elif state in ("FAILED", "CANCELLED"):
                failed += 1
                log.warning("  %s FAILED: %s", tid, status.get("error_message", ""))
                del pending[tid]
            elif state == "RUNNING":
                running += 1
            else:
                ready += 1  # READY / UNSUBMITTED

        n_remain = len(pending)
        pct = (n_total - n_remain) / n_total * 100
        elapsed = int(time.time() - t_start)

        log.info(
            "  [%ds]  %d/%d done (%.0f%%)  ready=%d  running=%d  fail=%d",
            elapsed, n_total - n_remain, n_total, pct,
            ready, running, failed,
        )

        if n_remain > 0:
            time.sleep(POLL_INTERVAL_S)

    # Final report
    elapsed = time.time() - t_start
    n_completed = n_total - len(pending)
    log.info("=" * 50)
    log.info("Export summary after %.0fs:", elapsed)
    log.info("  Completed:  %d / %d", n_completed, n_total)
    if pending:
        log.warning("  Still pending: %d (timed out)", len(pending))
    log.info("=" * 50)


# =========================================================================
# 5. MAIN
# =========================================================================


def main(
    drive_folder: Optional[str] = None,
    crs: Optional[str] = None,
) -> None:
    """Full pipeline: initialise GEE → build composites → launch → monitor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export native-resolution Sentinel-2 bands for V5.0"
    )
    parser.add_argument(
        "--drive-folder", default=DRIVE_FOLDER,
        help="Google Drive folder for exported GeoTIFFs",
    )
    parser.add_argument(
        "--crs", default=EXPORT_CRS,
        help=f"Export CRS (default: {EXPORT_CRS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be exported without launching tasks",
    )
    args = parser.parse_args()

    folder = args.drive_folder
    export_crs = args.crs
    dry_run = args.dry_run

    log.info("=" * 50)
    log.info("GEE V5.0 RAW BAND EXPORT")
    log.info("  Folder:    %s", folder)
    log.info("  CRS:       %s", export_crs)
    log.info("  Dry run:   %s", dry_run)
    log.info("=" * 50)

    scales = {
        "B3": 10, "B4": 10, "B8": 10, "SCL": 10,
        "B11": 20, "B12": 20,
    }
    band_names = list(scales.keys())

    if dry_run:
        log.info("Dry run — GEE not required. Tasks that would be launched:")
        for bn in band_names:
            s = scales[bn]
            log.info("  Aral_%s_%dm_Spring2024  ->  %s/", bn, s, folder)
        return

    # Initialise GEE
    ee.Initialize()
    log.info("GEE initialised.")

    # Build AOI & composites
    aoi = _build_aoi()
    composites = _build_composites(aoi)

    # Launch tasks
    tasks: List[ee.batch.Task] = []
    for band_name in band_names:
        scale = scales[band_name]
        task = _export_single(band_name, composites[band_name], aoi, scale, folder, export_crs)
        tasks.append(task)

    # Monitor
    _poll_tasks(tasks)

    log.info("Done. Download files from Google Drive folder: %s", folder)
    log.info("Expected files:")
    for bn in band_names:
        s = scales[bn]
        log.info("  Aral_%s_%dm_Spring2024.tif", bn, s)


if __name__ == "__main__":
    main()
