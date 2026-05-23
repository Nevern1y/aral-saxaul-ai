"""Central configuration for the Aral Saxaul AI pipeline (all 5 phases)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final, List, Tuple


@dataclass
class AralSaxaulConfig:
    """
    Immutable configuration dataclass covering every tunable parameter
    across the full Phases 1–5 pipeline.

    Usage:
        cfg = AralSaxaulConfig()
        # Optional overrides:
        cfg = AralSaxaulConfig(s2_cloud_cover_max=30)
    """

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs"
    )

    # ── AOI: Aral Sea bounding box  ────────────────────────────────────
    ARAL_BBOX: Tuple[float, float, float, float] = (58.0, 43.5, 62.0, 46.5)
    AOI_BUFFER_M: int = 500
    MIN_WATER_PATCH_HA: float = 100.0
    MIN_AOI_POLYGON_HA: float = 1_000.0

    # ── AOI: JRC Global Surface Water  ─────────────────────────────────
    JRC_DATASET: str = "JRC/GSW1_4/GlobalSurfaceWater"
    JRC_MAX_EXTENT_BAND: str = "max_extent"
    HISTORICAL_WATER_FOCAL_RADIUS_M: int = 300

    # ── Dry-season window  ─────────────────────────────────────────────
    DRY_SEASON_START: str = "2025-08-01"
    DRY_SEASON_END: str = "2025-09-30"

    # ── Sentinel-2 (optical)  ──────────────────────────────────────────
    S2_DATASET: str = "COPERNICUS/S2_SR_HARMONIZED"
    S2_BANDS: ClassVar[List[str]] = [
        "B2",      # Blue   (490 nm)
        "B3",      # Green  (560 nm)
        "B4",      # Red    (665 nm)
        "B8",      # NIR    (842 nm)
        "B11",     # SWIR 1 (1610 nm)
        "B12",     # SWIR 2 (2190 nm)
    ]
    s2_cloud_cover_max: int = 40        # Max CLOUDY_PIXEL_PERCENTAGE per scene

    # SCL (Scene Classification Layer) cloud classes
    SCL_CLOUD_SHADOW: int = 3
    SCL_CLOUD_MEDIUM: int = 8
    SCL_CLOUD_HIGH: int = 9
    SCL_CIRRUS: int = 10
    SCL_CLOUD_CLASSES: Tuple[int, ...] = (3, 8, 9, 10)

    # SCL water class
    SCL_WATER: int = 6

    # QA60 bitmask
    QA60_CIRRUS_BIT: int = 11     # Bit 11 = cirrus mask

    # ── Water mask thresholds  ─────────────────────────────────────────
    NDWI_WATER_THRESHOLD: float = 0.0       # NDWI > 0  -> open water
    NDWI_DEEP_WATER_THRESHOLD: float = 0.3  # NDWI > 0.3 -> stable deep water

    # ── Sentinel-1 (SAR radar)  ────────────────────────────────────────
    S1_DATASET: str = "COPERNICUS/S1_GRD"
    S1_INSTRUMENT_MODE: str = "IW"
    S1_POLARIZATION: str = "VH"             # Cross-pol -> surface structure
    s1_speckle_kernel: int = 3              # Focal-mean radius for speckle reduction

    # ── Copernicus DEM  ────────────────────────────────────────────────
    DEM_DATASET: str = "COPERNICUS/DEM/GLO30"
    DEM_SCALE_M: int = 30

    # ── TWI computation  ───────────────────────────────────────────────
    TWI_EPSILON: float = 0.001              # min tan(slope) to avoid /0
    TWI_CATCHMENT_RADIUS_M: int = 300      # Focal-mean kernel for flow proxy

    # ── Spectral indices (formula parameters)  ─────────────────────────
    # Sentinel-2 band indices:
    #   B2=Blue  B3=Green  B4=Red  B8=NIR  B11=SWIR1  B12=SWIR2

    # ── Target resolution  ─────────────────────────────────────────────
    TARGET_SCALE_M: int = 10
    TARGET_CRS: str = "EPSG:4326"

    # ── Phase 2: Synthetic label generation  ───────────────────────────
    CALIBRATION_SCALE_M: int = 500         # Coarser scale for reduceRegion (regional stats)
    STABILITY_YEARS: Tuple[int, ...] = (2023, 2024, 2025)
    MSAVI_STABILITY_THRESHOLD: float = 0.15
    NDMI_STABILITY_THRESHOLD: float = -0.1
    SLOPE_SUITABLE_MAX: float = 15.0        # degrees
    SLOPE_UNSUITABLE_MIN: float = 25.0      # degrees
    MSAVI_BARE_MAX: float = 0.02
    SI_PERCENTILE_THRESHOLD: int = 85
    N_SAMPLES_PER_CLASS: int = 5_000
    MIN_POINT_DISTANCE_M: float = 100.0
    LABEL_RANDOM_SEED: int = 42

    # ── Phase 3: Training  ─────────────────────────────────────────────
    TEST_SIZE: float = 0.2
    OPTUNA_N_TRIALS: int = 100
    OPTUNA_TIMEOUT_SEC: int = 600
    CV_FOLDS: int = 5
    XGB_MAX_DEPTH_MIN: int = 2
    XGB_MAX_DEPTH_MAX: int = 8
    XGB_LEARNING_RATE_MIN: float = 0.01
    XGB_LEARNING_RATE_MAX: float = 0.3
    XGB_N_ESTIMATORS_MIN: int = 50
    XGB_N_ESTIMATORS_MAX: int = 500
    XGB_RANDOM_STATE: int = 42
    XGB_TREE_METHOD: str = "hist"
    XGB_DEVICE: str = "cuda"
    SHAP_SAMPLE_SIZE: int = 2_000

    # ── Phase 4: Distributed inference  ────────────────────────────────
    TILE_SIZE_KM: int = 20
    TILE_OVERLAP_M: int = 200
    MAX_INFERENCE_WORKERS: int = 4
    GEE_EXPORT_FOLDER: str = "aral_saxaul_tiles"
    MAX_CONCURRENT_GEE_EXPORTS: int = 10
    BIGTIFF: bool = True

    # ── Phase 5: Visualization  ────────────────────────────────────────
    PROB_THRESHOLD: float = 0.85
    MIN_CLUSTER_HA: float = 1.0
    MORPH_RADIUS_PX: int = 5
    VECTOR_SIMPLIFY_TOLERANCE_M: float = 15.0
    MAP_CENTER: Tuple[float, float] = (45.0, 60.0)
    MAP_DEFAULT_ZOOM: int = 9

    # ── Feature column order (must match model training order) ─────────
    FEATURE_COLUMNS: ClassVar[List[str]] = [
        "NDMI", "MSAVI", "SI", "Slope", "TWI", "VH", "NDWI"
    ]
    META_COLUMNS: ClassVar[List[str]] = ["longitude", "latitude"]

    def ensure_output_dirs(self) -> None:
        """Create all output subdirectories if they don't exist."""
        for sub in ["aoi", "models", "data", "reports", "tiles", "tiles/predictions"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def aoi_bbox_geom(self) -> str:
        """Return the AOI bounding box as a WKT polygon string."""
        min_lon, min_lat, max_lon, max_lat = self.ARAL_BBOX
        return (
            f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
            f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        )


# Singleton convenience
config = AralSaxaulConfig()
