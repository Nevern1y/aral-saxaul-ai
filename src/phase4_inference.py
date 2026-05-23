"""
Phase 4: Distributed Tile-Based Inference.

Problem: 60 000 km^2 at 10 m resolution -> ~600 million pixels x 7 bands
         -> 16.8 GB raw.  Impossible to load into RAM at once.

Solution: Tile-based streaming.
  1. Split AOI into 20x20 km tiles (~150 tiles).
  2. Export Feature Stack as GeoTIFF tiles from GEE to Google Drive.
  3. Process each tile independently (read -> predict -> write).
  4. Merge prediction tiles into a single GeoTIFF via GDAL VRT.

Classes
-------
TileGridGenerator       - Partition AOI into a regular grid of tiles.
GEEExporter             - Batch-export tiles from Earth Engine to Drive.
TileProcessor           - Load a tile, run XGBoost inference, save result.
TileMerger              - GDAL VRT-based mosaic (O(1) memory).
DistributedInference    - Orchestrator for the full Phase 4 pipeline.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ee
import numpy as np
import rasterio
import xgboost as xgb
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from src.config import config
from src.utils import initialize_gee, logger


# ============================================================================
# TileGridGenerator
# ============================================================================


@dataclass
class Tile:
    """Metadata for a single tile in the inference grid."""
    row: int
    col: int
    geometry: ee.Geometry
    tile_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.tile_id = f"{self.row:03d}_{self.col:03d}"


class TileGridGenerator:
    """Partition the AOI into a regular grid of square tiles.

    Each tile is ``tile_size_m x tile_size_m`` metres with a small overlap
    to ensure seamless merging.
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        tile_size_km: int = config.TILE_SIZE_KM,
        overlap_m: int = config.TILE_OVERLAP_M,
    ) -> None:
        """
        Args:
            aoi: Area-of-interest geometry.
            tile_size_km: Tile edge length in kilometres.
            overlap_m: Overlap between adjacent tiles in metres.
        """
        self.aoi = aoi
        self.tile_size_m = float(tile_size_km * 1000)
        self.overlap_m = float(overlap_m)

        # Tile CRS: use a local UTM projection for equal-area tiling
        # Aral Sea spans longitudes 58E-62E -> UTM zone 41N (EPSG:32641)
        self.tile_crs = "EPSG:32641"

        logger.info(
            "TileGridGenerator: %d x %d km tiles, %d m overlap, CRS=%s",
            tile_size_km,
            tile_size_km,
            overlap_m,
            self.tile_crs,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def generate(self, min_overlap_pct: float = 5.0) -> List[Tile]:
        """Generate the list of tiles covering the AOI.

        Only tiles whose intersection with the AOI exceeds *min_overlap_pct*
        of the tile area are retained.

        Args:
            min_overlap_pct: Minimum AOI overlap (percentage) to keep a tile.

        Returns:
            Sorted list of ``Tile`` objects.
        """
        logger.info("Generating tile grid …")
        t0 = time.time()

        # Project AOI to UTM for metric measurements
        aoi_utm = self.aoi.transform(self.tile_crs, 1.0)
        coords = aoi_utm.bounds().getInfo()["coordinates"][0]
        min_x, min_y = coords[0][0], coords[0][1]
        max_x, max_y = coords[2][0], coords[2][1]

        # Effective tile stride (accounting for overlap)
        stride = self.tile_size_m - self.overlap_m

        n_cols = max(1, int(np.ceil((max_x - min_x) / stride)))
        n_rows = max(1, int(np.ceil((max_y - min_y) / stride)))

        logger.info("Grid: %d cols x %d rows ~ %d potential tiles.", n_cols, n_rows, n_cols * n_rows)

        tiles: List[Tile] = []
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = min_x + col * stride
                y0 = min_y + row * stride
                x1 = x0 + self.tile_size_m
                y1 = y0 + self.tile_size_m

                tile_utm = ee.Geometry.Rectangle(
                    [x0, y0, x1, y1],
                    proj=self.tile_crs,
                    evenOdd=False,
                )

                # Check overlap with AOI
                try:
                    intersection = tile_utm.intersection(aoi_utm, 1.0)
                    inter_area = intersection.area(1.0).getInfo()
                    tile_area = self.tile_size_m**2
                    overlap_pct = 100.0 * inter_area / tile_area
                except Exception:
                    overlap_pct = 0.0

                if overlap_pct >= min_overlap_pct:
                    # Convert back to EPSG:4326 for GEE export
                    tile_wgs84 = tile_utm.transform("EPSG:4326", 1.0)
                    tiles.append(Tile(row=row, col=col, geometry=tile_wgs84))

        elapsed = time.time() - t0
        logger.info(
            "Generated %d tiles covering AOI (%.1f s, overlap ≥ %.0f%%).",
            len(tiles),
            elapsed,
            min_overlap_pct,
        )
        return sorted(tiles, key=lambda t: (t.row, t.col))


# ============================================================================
# GEEExporter
# ============================================================================


class GEEExporter:
    """Batch-export Feature Stack tiles from GEE to Google Drive.

    Exports are launched serially with a small inter-task delay to respect
    GEE rate limits.  This method returns immediately - the actual export
    runs asynchronously on Google's servers.
    """

    def __init__(
        self,
        feature_stack: ee.Image,
        drive_folder: str = config.GEE_EXPORT_FOLDER,
        scale_m: int = config.TARGET_SCALE_M,
        crs: str = config.TARGET_CRS,
        max_concurrent: int = config.MAX_CONCURRENT_GEE_EXPORTS,
    ) -> None:
        """
        Args:
            feature_stack: Multi-band image from Phase 1.
            drive_folder: Google Drive folder name for exported tiles.
            scale_m: Pixel resolution in metres.
            crs: Output coordinate reference system.
            max_concurrent: Limit on concurrent GEE tasks.
        """
        self.feature_stack = feature_stack
        self.drive_folder = drive_folder
        self.scale_m = scale_m
        self.crs = crs
        self.max_concurrent = max_concurrent

        self._task_ids: List[str] = []

        logger.info(
            "GEEExporter: -> Drive folder '%s', scale=%dm, max_concurrent=%d.",
            drive_folder,
            scale_m,
            max_concurrent,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def export_all(self, tiles: List[Tile]) -> List[str]:
        """Launch tile exports and return task IDs.

        Args:
            tiles: List of tiles from ``TileGridGenerator``.

        Returns:
            List of GEE task IDs for status monitoring.
        """
        logger.info("Launching %d GEE tile exports …", len(tiles))
        t0 = time.time()

        task_ids: List[str] = []
        for idx, tile in enumerate(tiles):
            if idx > 0 and idx % self.max_concurrent == 0:
                cooldown = 30
                logger.debug(
                    "Throttling: %d/%d tiles submitted, cooling down %ds …",
                    idx, len(tiles), cooldown,
                )
                time.sleep(cooldown)

            task_id = self._export_single_tile(tile)
            task_ids.append(task_id)
            time.sleep(2)  # Inter-task spacing

        self._task_ids = task_ids
        elapsed = time.time() - t0
        logger.info(
            "%d export tasks launched (%.1f s). Monitor with monitor_tasks().",
            len(task_ids),
            elapsed,
        )
        return task_ids

    def monitor_tasks(
        self,
        task_ids: Optional[List[str]] = None,
        poll_interval_sec: int = 30,
        max_wait_sec: int = 3600,
    ) -> Dict[str, str]:
        """Poll GEE task statuses until all complete or timeout.

        Args:
            task_ids: Task IDs to monitor (default: from last export_all()).
            poll_interval_sec: Seconds between status checks.
            max_wait_sec: Maximum total wait time.

        Returns:
            ``{task_id: "COMPLETED"|"FAILED"|"RUNNING"}``
        """
        task_ids = task_ids or self._task_ids
        if not task_ids:
            raise ValueError("No task IDs to monitor. Call export_all() first.")

        logger.info("Monitoring %d GEE tasks (timeout %ds) …", len(task_ids), max_wait_sec)

        statuses: Dict[str, str] = {}
        start = time.time()
        n_completed = 0
        n_failed = 0

        while time.time() - start < max_wait_sec:
            for tid in task_ids:
                if tid in statuses and statuses[tid] in ("COMPLETED", "FAILED"):
                    continue
                try:
                    task = ee.data.getTaskStatus(tid)[0]
                    state = task.get("state", "UNKNOWN")
                    statuses[tid] = state
                except Exception:
                    statuses[tid] = "ERROR"

            n_completed = sum(1 for s in statuses.values() if s == "COMPLETED")
            n_failed = sum(1 for s in statuses.values() if s in ("FAILED", "CANCELLED", "ERROR"))

            if n_completed + n_failed >= len(task_ids):
                break

            logger.debug(
                "Tasks: %d/%d completed, %d failed. Sleeping %ds …",
                n_completed,
                len(task_ids),
                n_failed,
                poll_interval_sec,
            )
            time.sleep(poll_interval_sec)

        logger.info(
            "Monitor finished: %d completed, %d failed, %d still running.",
            n_completed,
            n_failed,
            len(task_ids) - n_completed - n_failed,
        )

        if n_failed > 0:
            failed_tasks = [tid for tid, s in statuses.items() if s in ("FAILED", "CANCELLED", "ERROR")]
            logger.error("Failed tasks: %s", failed_tasks)

        return statuses

    # ── Internal ───────────────────────────────────────────────────────

    def _export_single_tile(self, tile: Tile) -> str:
        """Launch a single tile export to Google Drive."""
        task = ee.batch.Export.image.toDrive(
            image=self.feature_stack.clip(tile.geometry),
            description=f"aral_tile_{tile.tile_id}",
            folder=self.drive_folder,
            fileNamePrefix=f"tile_{tile.tile_id}",
            scale=self.scale_m,
            crs=self.crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        logger.debug("Export launched: tile_%s (task: %s)", tile.tile_id, task.id)
        return task.id


# ============================================================================
# TileProcessor
# ============================================================================


class TileProcessor:
    """Process a single tile: read GeoTIFF -> scale -> predict -> write GeoTIFF.

    Memory footprint per tile: ~112 MB for a 20x20 km tile at 10 m resolution
    with 7 float32 bands.  Easily fits within 32 GB RAM even with 4 concurrent workers.
    """

    def __init__(
        self,
        model: xgb.XGBClassifier,
        scaler: StandardScaler,
        feature_order: List[str],
    ) -> None:
        """
        Args:
            model: Trained XGBoost classifier.
            scaler: Fitted StandardScaler.
            feature_order: Ordered list of feature names matching the export bands.
        """
        self.model = model
        self.scaler = scaler
        self.feature_order = feature_order
        self.n_features = len(feature_order)

        logger.debug("TileProcessor init: %d features.", self.n_features)

    # ── Public API ─────────────────────────────────────────────────────

    def process(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 1_000_000,
    ) -> Dict[str, Any]:
        """Run inference on a single tile.

        Pipeline:
            1. Read all bands with ``rasterio``.
            2. Reshape to ``(n_pixels, n_features)``.
            3. Apply ``StandardScaler.transform()``.
            4. Run ``model.predict_proba()`` in batches.
            5. Reshape back to ``(height, width)``.
            6. Write single-band probability GeoTIFF.

        Args:
            input_path: Path to the input Feature Stack GeoTIFF tile.
            output_path: Destination path for the prediction GeoTIFF.
            batch_size: Number of pixels to predict in one call (GPU VRAM management).

        Returns:
            Statistics dict: ``{pixels_total, pixels_valid, prob_mean, prob_std, ...}``.
        """
        t0 = time.time()

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input tile not found: {input_path}")

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            n_bands = src.count
            height, width = src.height, src.width

            descriptions = list(src.descriptions) if src.descriptions else []
            if descriptions and any(d is not None for d in descriptions):
                name_to_index = {
                    name.upper(): idx + 1
                    for idx, name in enumerate(descriptions)
                    if name is not None
                }
                missing = [
                    name for name in self.feature_order if name.upper() not in name_to_index
                ]
                if missing:
                    raise ValueError(
                        "Missing expected bands in raster: "
                        f"{missing}. Available: {descriptions}"
                    )
                band_indices = [name_to_index[name.upper()] for name in self.feature_order]
            else:
                if n_bands != self.n_features:
                    raise ValueError(
                        "Raster band count does not match feature order length "
                        f"({n_bands} vs {self.n_features})."
                    )
                band_indices = list(range(1, self.n_features + 1))
                logger.warning("No band descriptions found; assuming raster band order.")

            nodata = src.nodata

            # Read all bands -> (bands, height, width)
            data = src.read(indexes=band_indices)

        # ── Reshape to (pixels, features) ──────────────────────────
        n_pixels = height * width
        pixels = data.reshape(self.n_features, -1).T.astype(np.float64)

        # Free the (bands, H, W) array
        del data
        gc.collect()

        # ── Handle NaN / nodata ─────────────────────────────────────
        valid_mask = ~np.isnan(pixels).any(axis=1)
        if nodata is not None and not np.isnan(nodata):
            valid_mask &= ~np.any(pixels == nodata, axis=1)

        n_valid = int(valid_mask.sum())
        n_nan = n_pixels - n_valid

        # ── Scale ───────────────────────────────────────────────────
        if n_valid > 0:
            pixels_valid = pixels[valid_mask]
            pixels_scaled = self.scaler.transform(pixels_valid)
        else:
            pixels_scaled = np.empty((0, self.n_features))

        del pixels
        gc.collect()

        # ── Predict (batch for VRAM) ────────────────────────────────
        probs = np.full(n_pixels, np.nan, dtype=np.float32)

        if n_valid > 0:
            n_batches = int(np.ceil(n_valid / batch_size))
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, n_valid)
                batch = pixels_scaled[start_idx:end_idx]

                proba_batch = self.model.predict_proba(batch)[:, 1]
                # Map back to global positions
                global_indices = np.where(valid_mask)[0][start_idx:end_idx]
                probs[global_indices] = proba_batch.astype(np.float32)

        del pixels_scaled
        gc.collect()

        # ── Reshape to (height, width) ──────────────────────────────
        prob_2d = probs.reshape(height, width)

        # ── Update profile for single-band output ───────────────────
        profile.update(
            dtype="float32",
            count=1,
            compress="lzw",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=np.nan,
            bigtiff="IF_NEEDED",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prob_2d, 1)
            dst.update_tags(
                model="XGBoost",
                phase="4-inference",
                feature_order=",".join(self.feature_order),
            )

        prob_mean = float(np.nanmean(probs)) if n_valid > 0 else np.nan
        prob_std = float(np.nanstd(probs)) if n_valid > 0 else np.nan

        elapsed = time.time() - t0

        stats = {
            "tile": os.path.basename(input_path),
            "pixels_total": n_pixels,
            "pixels_valid": n_valid,
            "pixels_nan": n_nan,
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "elapsed_sec": round(elapsed, 2),
        }

        logger.debug(
            "Tile %s: %d px (%.1f%% NaN) mean=%.3f ± %.3f (%.1fs).",
            stats["tile"],
            n_pixels,
            100 * n_nan / n_pixels if n_pixels > 0 else 0,
            prob_mean,
            prob_std,
            elapsed,
        )

        return stats


# ============================================================================
# TileMerger
# ============================================================================


class TileMerger:
    """Merge prediction tiles into a single GeoTIFF via GDAL VRT.

    Why VRT (Virtual Raster) instead of rasterio.merge?
        ``rasterio.merge`` loads all source windows into RAM before writing -
        guaranteed OOM with 150 tiles x 112 MB.
        GDAL VRT is a *virtual* format - pixels are read on-the-fly during
        ``gdal.Translate``, keeping memory footprint constant O(tile_size).
    """

    def __init__(self, pred_dir: str) -> None:
        """
        Args:
            pred_dir: Directory containing ``pred_*.tif`` files.
        """
        self.pred_dir = Path(pred_dir)

        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for TileMerger. "
                "Install it via conda: conda install -c conda-forge gdal"
            )

        tif_files = list(self.pred_dir.glob("pred_*.tif"))
        logger.info(
            "TileMerger: %d prediction tiles found in %s.", len(tif_files), pred_dir
        )

    # ── Public API ─────────────────────────────────────────────────────

    def merge(
        self,
        output_path: str,
        delete_tiles: bool = False,
        resampling: str = "nearest",
        num_threads: str = "ALL_CPUS",
    ) -> str:
        """Merge all prediction tiles into a single GeoTIFF.

        Steps
        -----
        1. ``gdal.BuildVRT`` - virtual mosaic (no pixel copying).
        2. ``gdal.Translate`` - materialise VRT into a compressed GeoTIFF.

        Args:
            output_path: Destination path (e.g. ``outputs/probability_map.tif``).
            delete_tiles: If True, delete individual tile files after merging.
            resampling: GDAL resampling algorithm.
            num_threads: GDAL NUM_THREADS creation option.

        Returns:
            Path to the merged GeoTIFF.
        """
        tif_files = sorted(self.pred_dir.glob("pred_*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No prediction tiles found in {self.pred_dir}")

        logger.info("Merging %d tiles -> %s ...", len(tif_files), output_path)

        t0 = time.time()

        vrt_path = str(self.pred_dir / "_merged.vrt")

        # Step 1: Build VRT
        # Use gdal.BuildVRT with a temporary .vrt file
        gdal.BuildVRT(
            vrt_path,
            [str(f) for f in tif_files],
            VRTNodata=np.nan,
            resampleAlg=resampling,
        )

        logger.info("VRT built: %s", vrt_path)

        # Step 2: Translate VRT -> compressed GeoTIFF
        creation_options = [
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "BIGTIFF=YES",
            f"NUM_THREADS={num_threads}",
        ]

        gdal.Translate(
            output_path,
            vrt_path,
            format="GTiff",
            creationOptions=creation_options,
            noData=np.nan,
            stats=True,
        )

        # Clean up VRT (temporary file)
        os.remove(vrt_path)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            "Merge complete: %s (%.1f MB, %.1f s).",
            output_path,
            size_mb,
            elapsed,
        )

        # Optionally delete individual tiles
        if delete_tiles:
            for f in tif_files:
                f.unlink()
            logger.info("Deleted %d individual prediction tiles.", len(tif_files))

        return output_path


# ============================================================================
# DistributedInference (Orchestrator)
# ============================================================================


class DistributedInference:
    """Orchestrate the full distributed-inference pipeline.

    Modes
    -----
    * **colab**: Mount Google Drive -> copy tiles -> infer -> merge.
    * **local**: Download tiles via Google Drive API -> infer -> merge
      (requires a pre-authenticated ``ee`` session and download handler).
    * **from_disk**: Tiles are already on local disk - skip export/download.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        feature_names_path: str,
        feature_stack: Optional[ee.Image] = None,
        aoi: Optional[ee.Geometry] = None,
        mode: str = "from_disk",
        tile_dir: Optional[str] = None,
        pred_dir: Optional[str] = None,
        max_workers: int = config.MAX_INFERENCE_WORKERS,
    ) -> None:
        """
        Args:
            model_path: Path to ``xgb_classifier.pkl``.
            scaler_path: Path to ``scaler.pkl``.
            feature_names_path: Path to ``feature_names.json``.
            feature_stack: GEE Feature Stack image (needed for ``colab`` / ``local`` modes).
            aoi: AOI geometry (needed for tile generation).
            mode: "colab", "local", or "from_disk".
            tile_dir: Directory containing input tiles (for "from_disk" mode).
            pred_dir: Directory for prediction output tiles.
            max_workers: Max parallel workers for inference.
        """
        self.mode = mode
        self.max_workers = max_workers

        # Load model & scaler
        logger.info("Loading model from %s …", model_path)
        self.model: xgb.XGBClassifier = joblib_load(model_path)

        logger.info("Loading scaler from %s …", scaler_path)
        self.scaler: StandardScaler = joblib_load(scaler_path)

        with open(feature_names_path, "r") as f:
            self.feature_order: List[str] = json.load(f)
        logger.info("Feature order: %s", self.feature_order)

        self.feature_stack = feature_stack
        self.aoi = aoi

        # Directories
        if tile_dir is None:
            tile_dir = str(config.output_dir / "tiles")
        if pred_dir is None:
            pred_dir = str(config.output_dir / "tiles" / "predictions")

        self.tile_dir = Path(tile_dir)
        self.pred_dir = Path(pred_dir)
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        self._stats: List[Dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────

    def run(self, output_raster: Optional[str] = None, delete_tiles: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Execute the full inference pipeline according to ``self.mode``.

        Args:
            output_raster: Path for the merged probability_map.tif.
            delete_tiles: If True, delete intermediate tiles post-merge.

        Returns:
            ``(merged_raster_path, stats_dict)``.
        """
        if output_raster is None:
            output_raster = str(config.output_dir / "probability_map.tif")

        t0 = time.time()

        if self.mode in ("colab", "local"):
            self._export_and_download()

        # Run parallel inference
        self._run_inference_parallel()

        # Merge
        merger = TileMerger(str(self.pred_dir))
        merged_path = merger.merge(
            output_path=output_raster,
            delete_tiles=delete_tiles,
        )

        elapsed = time.time() - t0
        summary = self._build_summary(elapsed)
        logger.info("Phase 4 complete (%.1f s). Raster: %s", elapsed, merged_path)

        return merged_path, summary

    def run_from_disk(
        self,
        tile_dir: Optional[str] = None,
        pred_dir: Optional[str] = None,
        output_raster: Optional[str] = None,
        delete_tiles: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Shorthand for ``mode='from_disk'`` with explicit directories."""
        if tile_dir is not None:
            self.tile_dir = Path(tile_dir)
        if pred_dir is not None:
            self.pred_dir = Path(pred_dir)
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        return self.run(output_raster=output_raster, delete_tiles=delete_tiles)

    # ── Internal ───────────────────────────────────────────────────────

    def _export_and_download(self) -> None:
        """Phase A: Export tiles from GEE, then stage them locally."""
        if self.feature_stack is None or self.aoi is None:
            raise ValueError("feature_stack and aoi are required for GEE export mode.")

        # 1. Generate grid
        grid_gen = TileGridGenerator(aoi=self.aoi)
        tiles = grid_gen.generate()

        # 2. Export
        exporter = GEEExporter(feature_stack=self.feature_stack)
        task_ids = exporter.export_all(tiles)

        # 3. Wait
        statuses = exporter.monitor_tasks(task_ids, max_wait_sec=7200)
        n_ok = sum(1 for s in statuses.values() if s == "COMPLETED")
        logger.info("Export complete: %d/%d tiles ready.", n_ok, len(tiles))

        # 4. Download / copy from Drive
        logger.info("Tiles exported to Google Drive. Download them manually "
                     "or via googleapiclient to: %s", self.tile_dir)

    def _run_inference_parallel(self) -> None:
        """Process all input tiles in parallel using a thread pool."""
        input_files = sorted(self.tile_dir.glob("tile_*.tif"))
        if not input_files:
            raise FileNotFoundError(
                f"No tile_*.tif files found in {self.tile_dir}. "
                f"Run GEE export first or place tiles manually."
            )

        logger.info(
            "Starting parallel inference: %d tiles, %d workers.",
            len(input_files),
            self.max_workers,
        )

        processor = TileProcessor(
            model=self.model,
            scaler=self.scaler,
            feature_order=self.feature_order,
        )

        # Prepare input->output mapping
        tasks: Dict[str, str] = {}
        for infile in input_files:
            outfile = self.pred_dir / f"pred_{infile.name}"
            tasks[str(infile)] = str(outfile)

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(processor.process, inp, out): inp
                for inp, out in tasks.items()
            }

            with tqdm(total=len(future_to_path), desc="Inference", unit="tile") as pbar:
                for future in as_completed(future_to_path):
                    inp = future_to_path[future]
                    try:
                        stats = future.result()
                        results.append(stats)
                    except Exception as exc:
                        logger.error("FAILED tile %s: %s", Path(inp).name, exc, exc_info=True)
                        results.append({
                            "tile": Path(inp).name,
                            "error": str(exc),
                        })
                    pbar.update(1)

        self._stats = results

        n_ok = sum(1 for r in results if "error" not in r)
        n_fail = len(results) - n_ok
        logger.info("Inference complete: %d OK, %d failed.", n_ok, n_fail)

    def _build_summary(self, total_elapsed: float) -> Dict[str, Any]:
        """Aggregate statistics across all processed tiles."""
        valid_stats = [s for s in self._stats if "error" not in s]

        if not valid_stats:
            return {"error": "No tiles processed successfully."}

        total_pixels = sum(s["pixels_total"] for s in valid_stats)
        total_valid = sum(s["pixels_valid"] for s in valid_stats)
        total_nan = sum(s["pixels_nan"] for s in valid_stats)

        prob_means = [s["prob_mean"] for s in valid_stats if not np.isnan(s["prob_mean"])]

        return {
            "n_tiles_ok": len(valid_stats),
            "n_tiles_failed": len(self._stats) - len(valid_stats),
            "pixels_total": total_pixels,
            "pixels_valid": total_valid,
            "pixels_nan": total_nan,
            "prob_grand_mean": float(np.mean(prob_means)) if prob_means else np.nan,
            "prob_grand_std": float(np.std(prob_means)) if prob_means else np.nan,
            "total_elapsed_sec": round(total_elapsed, 1),
        }


# ============================================================================
# Phase 4 entrypoint
# ============================================================================


def run_phase4(
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    feature_names_path: Optional[str] = None,
    tile_dir: Optional[str] = None,
    output_raster: Optional[str] = None,
    max_workers: int = config.MAX_INFERENCE_WORKERS,
) -> Tuple[str, Dict[str, Any]]:
    """Convenience function: run Phase 4 in 'from_disk' mode.

    Assumes tiles have already been exported from GEE and downloaded locally.

    Args:
        model_path: Path to xgb_classifier.pkl.
        scaler_path: Path to scaler.pkl.
        feature_names_path: Path to feature_names.json.
        tile_dir: Directory containing tile_*.tif files.
        output_raster: Destination for merged probability_map.tif.
        max_workers: Parallel workers for inference.

    Returns:
        ``(merged_raster_path, stats_dict)``.
    """
    out_dir = config.output_dir
    model_path = model_path or str(out_dir / "models" / "xgb_classifier.pkl")
    scaler_path = scaler_path or str(out_dir / "models" / "scaler.pkl")
    feature_names_path = feature_names_path or str(out_dir / "models" / "feature_names.json")
    tile_dir = tile_dir or str(out_dir / "tiles")
    output_raster = output_raster or str(out_dir / "probability_map.tif")

    runner = DistributedInference(
        model_path=model_path,
        scaler_path=scaler_path,
        feature_names_path=feature_names_path,
        mode="from_disk",
        tile_dir=tile_dir,
        max_workers=max_workers,
    )

    return runner.run_from_disk(
        tile_dir=tile_dir,
        output_raster=output_raster,
    )


# ============================================================================
# CLI entrypoint
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4: Distributed Tile-Based Inference"
    )
    parser.add_argument("--tile-dir", type=str, default=None, help="Directory with tile_*.tif files.")
    parser.add_argument("--model", type=str, default=None, help="Path to xgb_classifier.pkl.")
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler.pkl.")
    parser.add_argument("--features", type=str, default=None, help="Path to feature_names.json.")
    parser.add_argument("--output", type=str, default=None, help="Output GeoTIFF path.")
    parser.add_argument("--workers", type=int, default=config.MAX_INFERENCE_WORKERS)
    parser.add_argument("--delete-tiles", action="store_true", help="Delete tile files after merge.")
    args = parser.parse_args()

    merged, summary = run_phase4(
        model_path=args.model,
        scaler_path=args.scaler,
        feature_names_path=args.features,
        tile_dir=args.tile_dir,
        output_raster=args.output,
        max_workers=args.workers,
    )

    logger.info("Phase 4 done. Merged raster: %s", merged)
    print(json.dumps(summary, indent=2))
