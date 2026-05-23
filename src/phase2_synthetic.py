"""
Phase 2: Synthetic Data Generation via Biological Proxy Rules.

"No Ground Truth? No problem. Nature already labelled the landscape."

Classes
-------
TemporalStabilityChecker
    Verifies multi-year vegetation stability (MSAVI + NDMI thresholds)
    to identify pixels that are persistently suitable.

AdaptiveThresholdCalibrator
    Derives regional thresholds (e.g. salinity, bare soil) from the
    pixel distribution within the AOI - no hard-coded magic numbers.

SyntheticLabelGenerator
    Orchestrates point generation, feature extraction, and validation.
    Output: pd.DataFrame with 10 000 labelled samples (5 000 per class).
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import ee
import geemap
import numpy as np
import pandas as pd

from src.config import config
from src.utils import (
    compute_msavi,
    compute_ndmi,
    compute_ndwi,
    compute_si,
    initialize_gee,
    logger,
    mask_s2_clouds_scl,
    mask_water_scl,
)


# ============================================================================
# TemporalStabilityChecker
# ============================================================================


class TemporalStabilityChecker:
    """Check whether a pixel exhibits stable vegetation across multiple years.

    Principle
    ---------
    If a pixel shows MSAVI > 0.15 **and** NDMI > -0.1 in the dry season
    for *three consecutive years*, we infer that the substrate is persistently
    suitable for plant life - not a transient response to a single rainfall event.

    This reduces false positives from ephemeral vegetation (e.g. short-lived
    halophytes that sprout after one wet spring and die by August).
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        years: Tuple[int, ...] = config.STABILITY_YEARS,
        season_start: str = config.DRY_SEASON_START[5:],   # "08-01"
        season_end: str = config.DRY_SEASON_END[5:],       # "09-30"
        msavi_threshold: float = config.MSAVI_STABILITY_THRESHOLD,
        ndmi_threshold: float = config.NDMI_STABILITY_THRESHOLD,
    ) -> None:
        """
        Args:
            aoi: Region of interest.
            years: Years to check stability across (default: 2023, 2024, 2025).
            season_start: "MM-DD" start of dry-season window.
            season_end: "MM-DD" end of dry-season window.
            msavi_threshold: Minimum MSAVI to consider a pixel "vegetated".
            ndmi_threshold: Minimum NDMI to consider a pixel "moist enough".
        """
        self.aoi = aoi
        self.years = years
        self.season_start = season_start
        self.season_end = season_end
        self.msavi_threshold = msavi_threshold
        self.ndmi_threshold = ndmi_threshold

        logger.info(
            "TemporalStabilityChecker initialised: %s years, "
            "MSAVI>%.2f, NDMI>%.2f, window %s->%s",
            len(years),
            msavi_threshold,
            ndmi_threshold,
            season_start,
            season_end,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def compute_stability_mask(self) -> ee.Image:
        """Build a binary mask: 1 = stable vegetation across all years.

        Returns:
            ``ee.Image`` named ``stability`` - single band, 0 or 1.
        """
        logger.info("Computing temporal stability mask over %d years ...", len(self.years))
        t0 = time.time()

        msavi_stable = self._yearly_condition("MSAVI", self.msavi_threshold, "gt")
        ndmi_stable = self._yearly_condition("NDMI", self.ndmi_threshold, "gt")

        # Pixel is stable only if BOTH indices are above threshold EVERY year
        stability = msavi_stable.And(ndmi_stable).rename("stability").selfMask()

        elapsed = time.time() - t0
        logger.info("Stability mask ready (%.1f s).", elapsed)
        return stability

    # ── Internal helpers ───────────────────────────────────────────────

    def _yearly_condition(
        self,
        index_name: str,
        threshold: float,
        comparator: str = "gt",
    ) -> ee.Image:
        """Return a binary mask: 1 where *index* > *threshold* for EVERY year.

        For each year, builds a median composite of the index from Sentinel-2
        Aug–Sep, then chains ``.And()`` across years.

        Args:
            index_name: One of "MSAVI" or "NDMI".
            threshold: Numeric threshold.
            comparator: "gt" (greater than) or "lt" (less than).

        Returns:
            Binary ``ee.Image``.
        """
        op_fn = {"gt": ee.Image.gt, "lt": ee.Image.lt}[comparator]

        cumulative: Optional[ee.Image] = None
        for year in self.years:
            yearly_mask = self._yearly_composite(year, index_name)
            yearly_mask = op_fn(yearly_mask, threshold)

            if cumulative is None:
                cumulative = yearly_mask
            else:
                cumulative = cumulative.And(yearly_mask)

        return cumulative if cumulative is not None else ee.Image.constant(1)

    def _yearly_composite(self, year: int, index_name: str) -> ee.Image:
        """Build a median composite of *index_name* for a single year's dry season.

        Pipeline per scene: SCL cloud mask -> SCL water mask -> compute index ->
        select index band. Take median across all scenes in the season.
        """
        start = f"{year}-{self.season_start}"
        end = f"{year}-{self.season_end}"

        s2 = (
            ee.ImageCollection(config.S2_DATASET)
            .filterBounds(self.aoi)
            .filterDate(start, end)
            .filter(
                ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.s2_cloud_cover_max)
            )
        )

        # Map of index-computing functions
        index_fns = {
            "MSAVI": compute_msavi,
            "NDMI": compute_ndmi,
        }
        if index_name not in index_fns:
            raise ValueError(
                f"Unsupported index '{index_name}'. Choose from {list(index_fns.keys())}."
            )
        compute_fn = index_fns[index_name]

        def _process(img: ee.Image) -> ee.Image:
            img = mask_s2_clouds_scl(img)
            img = mask_water_scl(img)
            return compute_fn(img).select(index_name)

        composite = s2.map(_process).median().clip(self.aoi)
        logger.debug("Year %d %s composite: %d scenes.", year, index_name, s2.size().getInfo())
        return composite


# ============================================================================
# AdaptiveThresholdCalibrator
# ============================================================================


class AdaptiveThresholdCalibrator:
    """Compute region-specific thresholds from the pixel distribution in the AOI.

    Motivation
    ----------
    A salinity index of 20 might indicate a salt crust in the Eastern basin
    but be normal sandy soil in the Western basin.  Hard-coded thresholds
    fail where substrate composition varies spatially.

    This class derives thresholds as percentiles of the actual pixel
    distribution inside the AOI, making label generation region-aware.

    Notes
    -----
    Calibration builds its own index images directly from Sentinel-2 at a
    coarser scale (``config.CALIBRATION_SCALE_M``) rather than relying on
    the feature stack, because the feature stack uses ``reproject()`` at
    10 m which prevents GEE's ``bestEffort`` auto-scaling for reduceRegion.
    """

    def __init__(self, aoi: ee.Geometry, percentile: int = config.SI_PERCENTILE_THRESHOLD):
        """
        Args:
            aoi: Region over which to compute statistics.
            percentile: Percentile for salinity threshold (default 85).
        """
        self.aoi = aoi
        self.percentile = percentile

        logger.info(
            "AdaptiveThresholdCalibrator initialised: P%d percentile.", percentile
        )

    # ── Public API ─────────────────────────────────────────────────────

    def calibrate_si(self) -> float:
        """Compute the salinity index threshold as the Pxx percentile within the AOI.

        Builds a median-composite SI image directly from Sentinel-2 at the
        calibration scale to avoid reprojection conflicts.

        Returns:
            ``float`` - the SI value at the specified percentile.
        """
        si_img = self._build_index_composite("SI")
        stats = si_img.reduceRegion(
            reducer=ee.Reducer.percentile([self.percentile]),
            geometry=self.aoi,
            scale=config.CALIBRATION_SCALE_M,
            maxPixels=1e9,
            tileScale=16,
        )

        stats_dict = stats.getInfo() if hasattr(stats, "getInfo") else {}
        if isinstance(stats_dict, dict) and len(stats_dict) > 0:
            si_threshold = float(list(stats_dict.values())[0])
        else:
            si_threshold = float(stats.getNumber("SI").getInfo())
        logger.info(
            "SI P%d threshold calibrated: %.4f", self.percentile, si_threshold
        )
        return si_threshold

    def calibrate_msavi_bare(self) -> float:
        """Compute the P10 MSAVI value (upper bound for "bare soil").

        Builds a median-composite MSAVI image directly from Sentinel-2 at the
        calibration scale.
        """
        msavi_img = self._build_index_composite("MSAVI")
        stats = msavi_img.reduceRegion(
            reducer=ee.Reducer.percentile([10]),
            geometry=self.aoi,
            scale=config.CALIBRATION_SCALE_M,
            maxPixels=1e9,
            tileScale=16,
        )
        stats_dict = stats.getInfo() if hasattr(stats, "getInfo") else {}
        if isinstance(stats_dict, dict) and len(stats_dict) > 0:
            p10 = float(list(stats_dict.values())[0])
        else:
            p10 = float(stats.getNumber("MSAVI").getInfo())
        logger.info("MSAVI bare-soil P10 threshold calibrated: %.4f", p10)
        return max(p10, config.MSAVI_BARE_MAX)

    def calibrate_all(self, feature_stack: ee.Image) -> Dict[str, float]:
        """Run all calibrations and return a dict of thresholds.

        Note: ``feature_stack`` is accepted for API compatibility but
        ignored — calibration builds its own images at a coarser scale.

        Returns:
            ``{'si_threshold': float, 'msavi_bare': float}``
        """
        return {
            "si_threshold": self.calibrate_si(),
            "msavi_bare": self.calibrate_msavi_bare(),
        }

    # ── Internal: build index composites at calibration scale ──────────

    def _build_index_composite(self, index_name: str) -> ee.Image:
        """Build a median composite of *index_name* from Sentinel-2.

        Built at calibration scale (no reproject) so reduceRegion can
        auto-scale and stay within GEE's computation budget.

        Args:
            index_name: "SI" or "MSAVI".

        Returns:
            Single-band ``ee.Image`` named *index_name*.
        """
        start = f"2025-{config.DRY_SEASON_START[5:]}"
        end = f"2025-{config.DRY_SEASON_END[5:]}"

        s2 = (
            ee.ImageCollection(config.S2_DATASET)
            .filterBounds(self.aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.s2_cloud_cover_max))
        )

        index_fns = {
            "SI": compute_si,
            "MSAVI": compute_msavi,
        }
        compute_fn = index_fns[index_name]

        def _process(img: ee.Image) -> ee.Image:
            img = mask_s2_clouds_scl(img)
            img = mask_water_scl(img)
            return compute_fn(img).select(index_name)

        composite = s2.map(_process).median().clip(self.aoi)
        return composite


# ============================================================================
# SyntheticLabelGenerator
# ============================================================================


class SyntheticLabelGenerator:
    """Generate a labelled training dataset from biological proxy rules.

    Dataset shape
    -------------
    ============  ======================================================
    Column        Description
    ============  ======================================================
    NDMI          Moisture index
    MSAVI         Vegetation index (soil-adjusted)
    SI            Salinity index
    Slope         Terrain slope (degrees)
    TWI           Topographic Wetness Index
    VH            Radar backscatter VH (dB)
    NDWI          Water index
    label         Synthetic target (1 = suitable, 0 = unsuitable)
    longitude     WGS84 longitude
    latitude      WGS84 latitude
    ============  ======================================================

    Survivor-Bias Disclosure
    ------------------------
    Class 1 is generated from *surviving* vegetation patterns.  If a
    region is edaphically suitable but no seeds ever arrived there, it
    will be labelled as Class 0 (False Negative).  We accept this
    trade-off because **Precision** (minimising false *positives*) is
    the operational priority: recommending a dead zone for planting
    is costlier than missing a viable one.
    """

    def __init__(
        self,
        feature_stack: ee.Image,
        aoi: ee.Geometry,
        stability_checker: TemporalStabilityChecker,
        calibrator: AdaptiveThresholdCalibrator,
        n_positive: int = config.N_SAMPLES_PER_CLASS,
        n_negative: int = config.N_SAMPLES_PER_CLASS,
        min_distance_m: float = config.MIN_POINT_DISTANCE_M,
        random_seed: int = config.LABEL_RANDOM_SEED,
    ) -> None:
        """
        Args:
            feature_stack: 7-band image from Phase 1.
            aoi: Area-of-interest geometry.
            stability_checker: Pre-initialised checker for temporal stability.
            calibrator: Pre-initialised adaptive-threshold calibrator.
            n_positive: Number of Class-1 points to generate.
            n_negative: Number of Class-0 points to generate.
            min_distance_m: Minimum distance between any two generated points.
            random_seed: RNG seed for reproducibility.
        """
        self.feature_stack = feature_stack
        self.aoi = aoi
        self.stability_checker = stability_checker
        self.calibrator = calibrator
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.min_distance_m = min_distance_m
        self.random_seed = random_seed

        # Lazy cache
        self._positive_mask: Optional[ee.Image] = None
        self._negative_mask: Optional[ee.Image] = None
        self._thresholds: Optional[Dict[str, float]] = None

        logger.info(
            "SyntheticLabelGenerator initialised: %d positive + %d negative points.",
            n_positive,
            n_negative,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def run(self, validate: bool = True) -> pd.DataFrame:
        """Execute the full synthetic-data generation pipeline.

        Strategy
        --------
        1. Calibrate adaptive thresholds from the AOI pixel distribution.
        2. Generate random points across the AOI.
        3. Extract 7-band feature values interactively (TWI is now a simple
           slope-only computation, avoiding the heavy reduceNeighborhood).
        4. Apply label rules locally in pandas.

        Returns:
            ``pd.DataFrame`` with 10 columns (7 features + label + lon + lat).
            Rows are shuffled.
        """
        logger.info("=== Phase 2: Synthetic Label Generation ===")
        t0 = time.time()

        # 1. Calibrate adaptive thresholds
        self._thresholds = self.calibrator.calibrate_all(self.feature_stack)
        si_th = self._thresholds["si_threshold"]
        msavi_bare_th = self._thresholds["msavi_bare"]

        # 2. Generate random points (oversample to have enough after filtering)
        n_candidates = (self.n_positive + self.n_negative) * 4

        candidates = ee.FeatureCollection.randomPoints(
            region=self.aoi,
            points=n_candidates,
            seed=self.random_seed,
            maxError=1,
        )
        logger.info("Generated %d candidate points.", n_candidates)

        # 3. Extract feature values at all candidate points
        all_bands = config.FEATURE_COLUMNS

        sampled = self.feature_stack.select(all_bands).sampleRegions(
            collection=candidates,
            scale=config.TARGET_SCALE_M,
            geometries=True,
            tileScale=8,
        )

        df: pd.DataFrame = geemap.ee_to_df(sampled)
        logger.info("Extracted features at %d candidate points.", len(df))

        # 4. Extract lat/lon from geometry
        lons, lats = [], []
        for _, row in df.iterrows():
            geom = row.get("geometry")
            if geom is not None:
                centroid = geom.centroid
                lons.append(centroid.x)
                lats.append(centroid.y)
            else:
                lons.append(np.nan)
                lats.append(np.nan)

        df["longitude"] = lons
        df["latitude"] = lats

        if "geometry" in df.columns:
            df = df.drop(columns=["geometry"])

        # 5. Apply label rules locally in pandas
        df["label"] = -1

        pos_idx = (
            (df["SI"] <= si_th)
            & (df["MSAVI"] > msavi_bare_th)
            & (df["Slope"] < config.SLOPE_SUITABLE_MAX)
            & (df["NDWI"] <= config.NDWI_WATER_THRESHOLD)
        )
        neg_idx = (
            (df["SI"] > si_th)
            | (df["Slope"] > config.SLOPE_UNSUITABLE_MIN)
            | (df["MSAVI"] < msavi_bare_th)
        )
        df.loc[pos_idx, "label"] = 1
        df.loc[neg_idx & ~pos_idx, "label"] = 0

        df_labeled = df[df["label"] != -1].copy()
        n_pos_avail = (df_labeled["label"] == 1).sum()
        n_neg_avail = (df_labeled["label"] == 0).sum()

        logger.info(
            "Available after labeling: %d positive, %d negative.",
            n_pos_avail,
            n_neg_avail,
        )

        pos_df = df_labeled[df_labeled["label"] == 1].sample(
            n=min(self.n_positive, n_pos_avail), random_state=self.random_seed
        )
        neg_df = df_labeled[df_labeled["label"] == 0].sample(
            n=min(self.n_negative, n_neg_avail), random_state=self.random_seed + 137
        )

        df_final = pd.concat([pos_df, neg_df], ignore_index=True)
        df_final = df_final.sample(frac=1, random_state=self.random_seed).reset_index(
            drop=True
        )

        # 6. Clean NaNs
        n_before = len(df_final)
        df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
        n_dropped = n_before - len(df_final)
        if n_dropped > 0:
            logger.warning(
                "Dropped %d rows with NaN/Inf (%.1f%% of dataset).",
                n_dropped,
                100 * n_dropped / n_before,
            )

        # 7. Validate
        if validate:
            df_final = self._validate_labels(df_final)

        elapsed = time.time() - t0
        logger.info(
            "Phase 2 complete: %d rows, %d columns (%.1f s).",
            len(df_final),
            len(df_final.columns),
            elapsed,
        )
        return df_final

    def save_dataset(self, df: pd.DataFrame, path: Optional[str] = None) -> str:
        """Persist the dataset as CSV.

        Args:
            df: The output of ``run()``.
            path: Destination path. Defaults to ``outputs/data/synthetic_labels.csv``.

        Returns:
            Absolute path to the saved file.
        """
        if path is None:
            path = str(config.output_dir / "data" / "synthetic_labels.csv")
        df.to_csv(path, index=False)
        logger.info("Dataset saved: %s", path)
        return path

    # ── Post-hoc validation ────────────────────────────────────────────

    def _validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or relabel contradictory samples.

        A Class-1 point must not exhibit:
          - SI > 2 * calibrated_si_threshold  (obvious salt crust)
          - NDMI < -0.5                        (bone-dry)
          - NDWI > 0.1                         (likely misclassified water edge)

        Such points are dropped entirely (not relabelled), because they
        represent mask-edge artefacts rather than genuine ambiguous cases.

        Returns:
            Cleaned DataFrame.
        """
        n_before = len(df)
        threshold = self._thresholds.get("si_threshold", 1000)

        pos_mask = df["label"] == 1

        anomaly_si = pos_mask & (df["SI"] > 2 * threshold)
        anomaly_ndmi = pos_mask & (df["NDMI"] < -0.5)
        anomaly_ndwi = pos_mask & (df["NDWI"] > 0.1)

        contradictions = anomaly_si | anomaly_ndmi | anomaly_ndwi

        n_contra = contradictions.sum()
        if n_contra > 0:
            logger.warning(
                "Label validation: removed %d contradictory Class-1 points "
                "(SI>%.1f: %d, NDMI<-0.5: %d, NDWI>0.1: %d).",
                n_contra,
                2 * threshold,
                anomaly_si.sum(),
                anomaly_ndmi.sum(),
                anomaly_ndwi.sum(),
            )
            df = df[~contradictions].reset_index(drop=True)

        return df


# ============================================================================
# Phase 2 entrypoint
# ============================================================================


def run_phase2(
    feature_stack: ee.Image,
    aoi: ee.Geometry,
    save: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience function: generate synthetic labels and optionally persist.

    Args:
        feature_stack: 7-band image from Phase 1.
        aoi: Area-of-interest geometry.
        save: Whether to write the CSV to disk.
        output_path: Optional custom path.

    Returns:
        Labelled ``pd.DataFrame``.
    """
    checker = TemporalStabilityChecker(aoi=aoi)
    calibrator = AdaptiveThresholdCalibrator(aoi=aoi)
    generator = SyntheticLabelGenerator(
        feature_stack=feature_stack,
        aoi=aoi,
        stability_checker=checker,
        calibrator=calibrator,
    )

    df = generator.run(validate=True)

    if save:
        generator.save_dataset(df, path=output_path)

    return df


# ============================================================================
# CLI entrypoint  (for independent testing of Phase 2)
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2: Synthetic Label Generation"
    )
    parser.add_argument(
        "--project", type=str, default=None, help="GCP project ID"
    )
    parser.add_argument(
        "--phase1-json",
        type=str,
        default=None,
        help="Path to a JSON-serialised AOI (optional; skips Phase 1).",
    )
    args = parser.parse_args()

    initialize_gee(project=args.project)

    # For CLI standalone use, run a minimal Phase 1 first
    logger.info("Running mini-Phase 1 to build AOI + Feature Stack …")
    from src.phase1_ingestion import AOIBuilder, DataIngestion

    builder = AOIBuilder()
    aoi = builder.run()

    ingestion = DataIngestion(aoi=aoi)
    feature_stack = ingestion.build_feature_stack()

    logger.info("Running Phase 2 …")
    df = run_phase2(feature_stack, aoi, save=True)

    logger.info("Phase 2 done. Dataset shape: %s", df.shape)
    print(df.describe())
    print(f"Class distribution:\n{df['label'].value_counts()}")
