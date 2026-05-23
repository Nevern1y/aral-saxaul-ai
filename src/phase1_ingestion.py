"""
Phase 1: Data Ingestion & Feature Engineering via Google Earth Engine.

Classes:
    AOIBuilder     - Builds the Aral Sea dried-bed AOI autonomously
                      (JRC historical water minus current NDWI water mask).
    DataIngestion   - Computes a 7-band Feature Stack:
                      [NDMI, MSAVI, SI, Slope, TWI, VH, NDWI]
                      over the AOI for Aug–Sep 2025.
"""

from __future__ import annotations

import gc
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import ee
import geemap
import numpy as np

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
# AOIBuilder
# ============================================================================


class AOIBuilder:
    """Build the Area-of-Interest polygon for the dried Aral Sea bed.

    Algorithm
    ---------
    1. Load *JRC Global Surface Water* ``max_extent`` (1984-2021) -
       every pixel that was ever water.
    2. Load *Sentinel-2 2025 NDWI median composite* and apply SCL-water
       mask -> current water mask.
    3. AOI = ``historical_water AND NOT current_water``, cleaned with
       morphological operations (focal_min / focal_max).
    4. Vectorise -> filter small slivers -> buffer -> return ``ee.Geometry``.

    No external shapefiles required - the AOI is built programmatically.
    """

    def __init__(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        current_year: int = 2025,
        buffer_m: int = config.AOI_BUFFER_M,
        min_water_patch_ha: float = config.MIN_WATER_PATCH_HA,
        min_aoi_polygon_ha: float = config.MIN_AOI_POLYGON_HA,
        vectorize: bool = False,
    ) -> None:
        """
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]. Defaults to config.ARAL_BBOX.
            current_year: Year used for the current-water NDWI mask.
            buffer_m: Buffer distance (m) applied to the final AOI.
            min_water_patch_ha: Patches smaller than this are not treated
                as persistent water bodies.
            min_aoi_polygon_ha: AOI sub-polygons below this threshold are discarded.
            vectorize: If True, vectorize the AOI mask (slow). If False, return
                the bbox geometry and keep a raster AOI mask for masking.
        """
        self.bbox = bbox or config.ARAL_BBOX
        self.current_year = current_year
        self.buffer_m = buffer_m
        self.min_water_patch_ha = min_water_patch_ha
        self.min_aoi_polygon_ha = min_aoi_polygon_ha
        self.vectorize = vectorize
        self._aoi_mask: Optional[ee.Image] = None

        min_lon, min_lat, max_lon, max_lat = self.bbox
        self._bbox_geom = ee.Geometry.Rectangle(
            [min_lon, min_lat, max_lon, max_lat], proj="EPSG:4326", evenOdd=False
        )

        logger.info(
            "AOIBuilder initialised - bbox: %s, buffer: %dm, year: %d",
            self.bbox,
            self.buffer_m,
            self.current_year,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def run(self, max_retries: int = 3) -> ee.Geometry:
        """Execute the full AOI-building pipeline.

        Returns:
            An ``ee.Geometry`` (possibly MultiPolygon) of the dried Aral bed.
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Building AOI (attempt %d/%d) ...", attempt, max_retries)
                aoi, mask = self._compute_aoi()
                self._aoi_mask = mask

                # If raster-only, AOI geometry is bbox (mask holds the true shape)
                area_km2 = aoi.area(1).divide(1e6).getInfo()
                logger.info("AOI built successfully - area ~ %.0f km^2", area_km2)

                if not self.vectorize:
                    logger.warning(
                        "AOI vectorization disabled: using bbox geometry + raster mask."
                    )

                return aoi
            except ee.EEException as exc:
                logger.error("GEE error on attempt %d: %s", attempt, exc)
                if attempt == max_retries:
                    raise
                time.sleep(5 * attempt)  # Exponential-ish backoff
        raise RuntimeError("AOI computation exhausted retries - unreachable.")

    # ── Internal methods ───────────────────────────────────────────────

    def _get_historical_water_mask(self) -> ee.Image:
        """JRC ``max_extent`` -> binary water mask clipped to bbox.

        Returns:
            Binary ``ee.Image``: 1 = was water at any point 1984–2021.
        """
        # JRC Global Surface Water is an Image (not a collection) in recent GEE
        try:
            jrc = ee.Image(config.JRC_DATASET)
        except Exception:
            # Fallback for legacy collection form
            jrc = ee.ImageCollection(config.JRC_DATASET).first()

        hist_water = (
            jrc.select(config.JRC_MAX_EXTENT_BAND)
            .eq(1)
            .clip(self._bbox_geom)
            .rename("historical_water")
        )
        logger.info("Historical water mask loaded (JRC GSW max_extent).")
        return hist_water

    def _get_current_water_mask(self) -> ee.Image:
        """Sentinel-2 NDWI median composite + SCL water check -> binary water mask.

        Uses both NDWI > config.NDWI_WATER_THRESHOLD (default 0) AND SCL == 6
        for double-confirmation of open water.

        Returns:
            Binary ``ee.Image``: 1 = current persistent surface water.
        """
        start = f"{self.current_year}-{config.DRY_SEASON_START[5:]}"
        end = f"{self.current_year}-{config.DRY_SEASON_END[5:]}"

        s2 = (
            ee.ImageCollection(config.S2_DATASET)
            .filterBounds(self._bbox_geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.s2_cloud_cover_max))
        )

        # Apply cloud masking per scene, then compute NDWI
        def _ndwi_scene(img: ee.Image) -> ee.Image:
            img = mask_s2_clouds_scl(img)
            return compute_ndwi(img).select("NDWI")

        ndwi_median = s2.map(_ndwi_scene).median().clip(self._bbox_geom)

        # SCL water mask (median SCL class over the same period)
        def _scl_water(img: ee.Image) -> ee.Image:
            return img.select("SCL").eq(config.SCL_WATER).rename("scl_water")

        scl_water = s2.map(_scl_water).max().clip(self._bbox_geom)

        # Combine: NDWI > threshold AND (SCL water OR deep NDWI)
        ndwi_water = ndwi_median.gt(config.NDWI_WATER_THRESHOLD)
        deep_water = ndwi_median.gt(config.NDWI_DEEP_WATER_THRESHOLD)

        current_water = ndwi_water.And(scl_water.Or(deep_water)).rename("current_water")

        logger.info("Current water mask computed (NDWI + SCL, %s-%s).", start, end)
        return current_water

    def _filter_small_patches(
        self, mask: ee.Image, min_ha: float, label: str
    ) -> ee.Image:
        """Remove contiguous patches smaller than *min_ha* hectares.

        Uses ``connectedPixelCount`` with 8-connectivity.

        Args:
            mask: Binary ``ee.Image``.
            min_ha: Minimum patch area in hectares.
            label: Log label.

        Returns:
            Filtered binary image.
        """
        min_pixels = int(min_ha * 10_000 / (config.TARGET_SCALE_M**2))
        conn = mask.connectedPixelCount(maxSize=1024, eightConnected=True)
        filtered = mask.updateMask(conn.gte(min_pixels))
        logger.info(
            "Patch filter '%s': removed patches < %d px (%.0f ha).",
            label,
            min_pixels,
            min_ha,
        )
        return filtered

    def _compute_aoi(self) -> ee.Geometry:
        """Core AOI computation pipeline.

        Returns:
            (aoi_geometry, aoi_mask)
        """
        # 1. Historical water mask
        hist_water = self._get_historical_water_mask()

        # 2. Clean narrow channels (Syr Darya delta, irrigation canals)
        hist_clean = hist_water.focalMin(
            radius=config.HISTORICAL_WATER_FOCAL_RADIUS_M,
            kernelType="circle",
            units="meters",
        )

        # 3. Current water mask -> filter small patches
        curr_water_raw = self._get_current_water_mask()
        curr_water = self._filter_small_patches(
            curr_water_raw, self.min_water_patch_ha, "current_water"
        )

        # 4. AOI raster: historical water AND NOT current water
        aoi_raster = hist_clean.And(curr_water.Not()).rename("aoi_raster")

        # 5. Fill small gaps with focal_max
        aoi_filled = aoi_raster.focalMax(
            radius=config.HISTORICAL_WATER_FOCAL_RADIUS_M,
            kernelType="circle",
            units="meters",
        )

        # 6. Vectorise if requested (slow), otherwise return bbox geometry + mask
        if self.vectorize:
            vectors = aoi_filled.selfMask().reduceToVectors(
                geometryType="polygon",
                reducer=ee.Reducer.countEvery(),
                scale=config.TARGET_SCALE_M,
                maxPixels=1e10,
                geometry=self._bbox_geom,
                eightConnected=True,
                bestEffort=True,
            )

            min_area_m2 = self.min_aoi_polygon_ha * 10_000
            vectors = vectors.filter(ee.Filter.gte("count", 1))

            # Filter by area and buffer
            def _filter_and_buffer(feature: ee.Feature) -> ee.Feature:
                area = feature.geometry().area(1)
                return feature.set("area_m2", area)

            vectors_with_area = vectors.map(_filter_and_buffer)
            filtered = vectors_with_area.filter(
                ee.Filter.gte("area_m2", min_area_m2)
            )

            aoi_geom = filtered.geometry().buffer(self.buffer_m, 1)

            logger.info(
                "AOI geometry created: %d polygons kept (>=%.0f ha).",
                filtered.size().getInfo() if hasattr(filtered, "size") else "?",
                self.min_aoi_polygon_ha,
            )
            return aoi_geom, aoi_filled.selfMask()

        # Raster-only AOI path: skip vectorization for speed/stability
        return self._bbox_geom, aoi_filled.selfMask()

    def get_mask(self) -> Optional[ee.Image]:
        """Return the raster AOI mask if available."""
        return self._aoi_mask


# ============================================================================
# DataIngestion
# ============================================================================


class DataIngestion:
    """Build a 7-band Feature Stack over the Aral Sea dried bed.

    Output bands
    ------------
    ========  ====================================  ================
    Band      Description                           Source
    ========  ====================================  ================
    NDMI      Normalised Difference Moisture Index  Sentinel-2 L2A
    MSAVI     Modified Soil-Adjusted Vegetation     Sentinel-2 L2A
    SI        Salinity Index                        Sentinel-2 L2A
    Slope     Terrain slope (degrees)               Copernicus DEM 30m
    TWI       Topographic Wetness Index (approx.)   Copernicus DEM 30m
    VH        Radar backscatter (VH, dB)            Sentinel-1 GRD
    NDWI      Normalised Difference Water Index     Sentinel-2 L2A
    ========  ====================================  ================

    All bands are resampled to the target resolution (10 m) in EPSG:4326.
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        aoi_mask: Optional[ee.Image] = None,
        start_date: str = config.DRY_SEASON_START,
        end_date: str = config.DRY_SEASON_END,
    ) -> None:
        """Initialise the ingestion pipeline.

        Args:
            aoi: Area of interest ``ee.Geometry`` (usually from ``AOIBuilder.run()``).
            start_date: "YYYY-MM-DD".
            end_date: "YYYY-MM-DD".
        """
        self.aoi = aoi
        self.aoi_mask = aoi_mask
        self.start_date = start_date
        self.end_date = end_date

        # Lazy-initialised collections
        self._s2_collection: Optional[ee.ImageCollection] = None
        self._s1_collection: Optional[ee.ImageCollection] = None
        self._dem: Optional[ee.Image] = None

        logger.info(
            "DataIngestion initialised - window: %s -> %s (mask=%s)",
            start_date,
            end_date,
            "yes" if aoi_mask is not None else "no",
        )

    # ── Public API ─────────────────────────────────────────────────────

    def build_feature_stack(self) -> ee.Image:
        """Assemble the complete 7-band Feature Stack.

        Returns:
            ``ee.Image`` with bands [NDMI, MSAVI, SI, Slope, TWI, VH, NDWI].
        """
        logger.info("Building Feature Stack ...")
        t0 = time.time()

        s2_composite = self._get_s2_composite()
        dem_stack = self._compute_terrain_indices()
        s1_composite = self._get_s1_composite()

        # Merge all bands
        feature_stack = s2_composite.addBands(dem_stack).addBands(s1_composite)

        # Prefer raster mask to avoid heavy vector AOI operations
        if self.aoi_mask is not None:
            feature_stack = feature_stack.updateMask(self.aoi_mask)
        else:
            feature_stack = feature_stack.clip(self.aoi)

        elapsed = time.time() - t0
        logger.info(
            "Feature Stack assembled (%d bands, %.1fs).",
            len(config.FEATURE_COLUMNS),
            elapsed,
        )
        return feature_stack

    def run(self) -> ee.Image:
        """Alias for ``build_feature_stack()``."""
        return self.build_feature_stack()

    def get_band_names(self) -> List[str]:
        """Return the ordered list of band names in the Feature Stack."""
        return list(config.FEATURE_COLUMNS)

    # ── Sentinel-2 composite ───────────────────────────────────────────

    def _get_s2_collection(self) -> ee.ImageCollection:
        """Lazy-load filtered Sentinel-2 collection."""
        if self._s2_collection is None:
            self._s2_collection = (
                ee.ImageCollection(config.S2_DATASET)
                .filterBounds(self.aoi)
                .filterDate(self.start_date, self.end_date)
                .filter(
                    ee.Filter.lt(
                        "CLOUDY_PIXEL_PERCENTAGE", config.s2_cloud_cover_max
                    )
                )
            )
            count = self._s2_collection.size().getInfo()
            logger.info("Sentinel-2 collection filtered: %d scenes.", count)
        return self._s2_collection

    def _mask_clouds_and_water(self, image: ee.Image) -> ee.Image:
        """Apply SCL cloud mask + SCL water mask to a single S2 scene."""
        image = mask_s2_clouds_scl(image)
        image = mask_water_scl(image)
        return image

    def _add_s2_indices(self, image: ee.Image) -> ee.Image:
        """Chain: add NDMI, MSAVI, SI, NDWI bands to a single S2 scene."""
        image = compute_ndmi(image)
        image = compute_msavi(image)
        image = compute_si(image)
        image = compute_ndwi(image)
        return image

    def _get_s2_composite(self) -> ee.Image:
        """Build the median composite of Sentinel-2 derived indices.

        Pipeline per scene:
            1. Mask clouds (SCL) + water (SCL)
            2. Compute NDMI, MSAVI, SI, NDWI
            3. Select only the index bands
            4. Take median across the collection
        """
        s2 = self._get_s2_collection()

        def _process_scene(img: ee.Image) -> ee.Image:
            img = self._mask_clouds_and_water(img)
            img = self._add_s2_indices(img)
            return img.select(["NDMI", "MSAVI", "SI", "NDWI"])

        composite = s2.map(_process_scene).median().clip(self.aoi)

        logger.info(
            "Sentinel-2 median composite created (%d index bands).",
            4,
        )
        return composite

    # ── Terrain indices from Copernicus DEM ────────────────────────────

    def _get_dem(self) -> ee.Image:
        """Lazy-load Copernicus DEM (GLO-30)."""
        if self._dem is None:
            self._dem = (
                ee.ImageCollection(config.DEM_DATASET)
                .filterBounds(self.aoi)
                .mosaic()
                .select("DEM")
                .clip(self.aoi)
                .rename("elevation")
            )
            logger.info("Copernicus DEM (GLO-30) loaded.")
        return self._dem

    def _compute_terrain_indices(self) -> ee.Image:
        """Compute Slope and TWI from Copernicus DEM.

        Returns:
            Two-band ``ee.Image``: ``Slope`` (degrees), ``TWI`` (dimensionless).
        """
        dem = self._get_dem()

        # Slope in degrees
        slope = ee.Terrain.slope(dem).rename("Slope")

        # ── TWI approximation ──────────────────────────────────────────
        # TWI = ln(α / tan(β))
        #   α = specific catchment area (m² per unit contour)
        #   β = local slope angle (radians)
        #
        # For the predominantly flat Aral basin we approximate α using
        # a large-radius focal-mean on the filled DEM as a flow-proxy.
        # On sloped terrain this captures relative topographic position;
        # on flat terrain TWI becomes uniform - which is physically correct
        # (water doesn't drain, it ponds).

        # Slope in radians - avoid /0
        slope_rad = slope.multiply(math.pi / 180.0)
        tan_slope = slope_rad.tan().max(config.TWI_EPSILON)

        # Simplified TWI = ln(1 / tan(slope))
        # In the flat Aral basin the flow-accumulation proxy adds little
        # value but forces a heavy reduceNeighborhood kernel that causes
        # GEE timeouts during point-based sampling.
        twi = tan_slope.pow(-1).max(config.TWI_EPSILON).log().rename("TWI")

        terrain = slope.addBands(twi)
        logger.info("Terrain indices computed (Slope, TWI).")
        return terrain

    # ── Sentinel-1 (SAR) composite ─────────────────────────────────────

    def _get_s1_collection(self) -> ee.ImageCollection:
        """Lazy-load and preprocess Sentinel-1 GRD collection (VH only)."""
        if self._s1_collection is None:
            raw = (
                ee.ImageCollection(config.S1_DATASET)
                .filterBounds(self.aoi)
                .filterDate(self.start_date, self.end_date)
                .filter(
                    ee.Filter.eq("instrumentMode", config.S1_INSTRUMENT_MODE)
                )
                .filter(
                    ee.Filter.listContains(
                        "transmitterReceiverPolarisation", config.S1_POLARIZATION
                    )
                )
            )

            # Apply per-scene preprocessing
            self._s1_collection = raw.map(self._preprocess_s1_scene)

            count = self._s1_collection.size().getInfo()
            logger.info("Sentinel-1 collection filtered: %d scenes (VH, IW).", count)
        return self._s1_collection

    def _preprocess_s1_scene(self, image: ee.Image) -> ee.Image:
        """Preprocess a single Sentinel-1 GRD scene.

        Steps:
            1. Select VH band
            2. Convert linear sigma0 to dB: 10*log10(sigma0)
            3. Apply border noise mask (GEE metadata)
            4. Light speckle filtering (focal-mean)
        """
        vh = image.select(config.S1_POLARIZATION)

        # Convert to dB (clamp floor to avoid -inf)
        vh_db = vh.max(1e-6).log10().multiply(10.0)

        # Border-noise mask from GEE scene metadata
        # S1 GRD images include an 'angle' band; edges with extreme angles are noisy
        angle = image.select("angle")
        border_mask = angle.gt(10).And(angle.lt(50))

        vh_masked = vh_db.updateMask(border_mask)

        # Speckle reduction: light focal-mean (complemented by temporal median)
        if config.s1_speckle_kernel > 0:
            vh_filtered = vh_masked.focalMean(
                radius=config.s1_speckle_kernel,
                kernelType="square",
                units="pixels",
            )
        else:
            vh_filtered = vh_masked

        return vh_filtered.rename(config.S1_POLARIZATION)

    def _get_s1_composite(self) -> ee.Image:
        """Build median composite of Sentinel-1 VH backscatter.

        Returns:
            Single-band ``ee.Image``: ``VH`` in dB.
        """
        s1 = self._get_s1_collection()
        composite = s1.median().clip(self.aoi).rename(config.S1_POLARIZATION)
        logger.info("Sentinel-1 VH median composite created (dB).")
        return composite


# ============================================================================
# Phase 1 entrypoint
# ============================================================================


def run_phase1(
    project: Optional[str] = None,
    skip_aoi: bool = False,
    aoi_geom: Optional[ee.Geometry] = None,
) -> Tuple[ee.Geometry, ee.Image]:
    """Convenience function: initialise GEE, build AOI, compute Feature Stack.

    Args:
        project: Optional GCP project ID for GEE initialisation.
        skip_aoi: If True, use ``aoi_geom`` directly instead of computing it.
        aoi_geom: Pre-built AOI geometry (required if ``skip_aoi=True``).

    Returns:
        ``(aoi_geometry, feature_stack_image)``.
    """
    initialize_gee(project=project)

    aoi_mask = None

    if skip_aoi and aoi_geom is not None:
        aoi = aoi_geom
        logger.info("Using pre-supplied AOI geometry.")
    else:
        builder = AOIBuilder(vectorize=False)
        aoi = builder.run()
        aoi_mask = builder.get_mask()

    ingestion = DataIngestion(aoi=aoi, aoi_mask=aoi_mask)
    feature_stack = ingestion.build_feature_stack()

    logger.info("Phase 1 complete - AOI + Feature Stack ready.")
    return aoi, feature_stack


# ============================================================================
# CLI entrypoint
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: AOI Builder + Feature Engineering (GEE)"
    )
    parser.add_argument(
        "--project", type=str, default=None, help="GCP project ID"
    )
    parser.add_argument(
        "--aoi-only",
        action="store_true",
        help="Only build and display the AOI; skip Feature Stack.",
    )
    args = parser.parse_args()

    initialize_gee(project=args.project)

    if args.aoi_only:
        builder = AOIBuilder()
        aoi = builder.run()
        area_km2 = aoi.area(1).divide(1e6).getInfo()
        logger.info("AOI area: %.0f km²", area_km2)
        print(f"AOI area: {area_km2:.0f} km²")
    else:
        aoi, stack = run_phase1(project=args.project)
        logger.info("Feature Stack ready for Phase 2.")
