"""Shared utilities: logging setup, GEE initialisation, reusable image functions."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import os
import io

import ee

from src.config import config

# ── Logger ────────────────────────────────────────────────────────────────

def _safe_stdout() -> object:
    """Return a stdout stream that won't crash on unicode in Windows consoles."""
    try:
        if hasattr(sys.stdout, "buffer"):
            return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    return sys.stdout


def _setup_logger(name: str = "aral_saxaul") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(_safe_stdout())
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger

logger: logging.Logger = _setup_logger()


# ── GEE Initialisation ────────────────────────────────────────────────────

def initialize_gee(project: Optional[str] = None) -> None:
    """Authenticate and initialise Google Earth Engine.

    Args:
        project: Optional GCP project ID. Uses default if not provided.

    Raises:
        RuntimeError: If initialisation fails.
    """
    # Allow project to be supplied via env vars
    project = project or os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")

    try:
        ee.Initialize(project=project)
        logger.info("GEE initialised successfully.")
        return
    except ee.EEException as exc:
        logger.warning("GEE init failed: %s", exc)

    # Attempt interactive authentication (CLI/browser flow)
    try:
        ee.Authenticate()
        ee.Initialize(project=project)
        logger.info("GEE initialised after authentication.")
        return
    except Exception as auth_exc:
        logger.warning("GEE auth failed: %s", auth_exc)

    # Fallback: try geemap helper if available (older versions)
    try:
        import geemap

        if hasattr(geemap, "ee_initialize"):
            geemap.ee_initialize()
            logger.info("GEE initialised via geemap fallback.")
            return
    except Exception as geemap_exc:
        logger.warning("GEE geemap fallback failed: %s", geemap_exc)

    raise RuntimeError(
        "GEE authentication failed. Run `earthengine authenticate` and retry."
    )


# ── Sentinel-2 cloud masking (SCL-based) ──────────────────────────────────

def mask_s2_clouds_scl(image: ee.Image) -> ee.Image:
    """Mask clouds and cloud shadows using the SCL band (Scene Classification Layer).

    Masks:
      - SCL_CLOUD_SHADOW  (3)
      - SCL_CLOUD_MEDIUM  (8)
      - SCL_CLOUD_HIGH    (9)
      - SCL_CIRRUS        (10)
      - QA60 bit-11 cirrus (supplementary)

    Args:
        image: A Sentinel-2 SR Harmonised image.

    Returns:
        The input image with cloud pixels set to 0 (masked).
    """
    scl = image.select("SCL")

    # Build SCL cloud mask
    cloud_mask = ee.Image.constant(0).byte()
    for cls in config.SCL_CLOUD_CLASSES:
        cloud_mask = cloud_mask.Or(scl.eq(cls))

    # Supplementary cirrus from QA60 (bit 11)
    qa60 = image.select("QA60")
    cirrus = qa60.bitwiseAnd(1 << config.QA60_CIRRUS_BIT).neq(0)
    cloud_mask = cloud_mask.Or(cirrus)

    return image.updateMask(cloud_mask.Not())


def mask_water_scl(image: ee.Image) -> ee.Image:
    """Mask open-water pixels using SCL water class (6).

    Args:
        image: A Sentinel-2 SR Harmonised image.

    Returns:
        Image with water pixels set to 0 (masked).
    """
    scl = image.select("SCL")
    water = scl.eq(config.SCL_WATER)
    return image.updateMask(water.Not())


# ── Sentinel-2 spectral indices (pure functions) ──────────────────────────

def compute_ndmi(image: ee.Image) -> ee.Image:
    """NDMI = (B8 - B11) / (B8 + B11)  - Normalised Difference Moisture Index.

    Sentinel-2 bands: B8=NIR (842nm), B11=SWIR1 (1610nm).
    """
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return image.addBands(ndmi)


def compute_ndwi(image: ee.Image) -> ee.Image:
    """NDWI = (B3 - B8) / (B3 + B8)  - Normalised Difference Water Index.

    Sentinel-2 bands: B3=Green (560nm), B8=NIR (842nm).
    """
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands(ndwi)


def compute_msavi(image: ee.Image) -> ee.Image:
    """MSAVI = (2*B8 + 1 - sqrt((2*B8+1)^2 - 8*(B8-B4))) / 2

    Modified Soil-Adjusted Vegetation Index - compensates for bare-soil background.
    Sentinel-2 bands: B8=NIR, B4=Red.
    """
    nir = image.select("B8")
    red = image.select("B4")
    msavi = (
        nir.multiply(2).add(1)
        .subtract(
            nir.multiply(2).add(1).pow(2).subtract(
                nir.subtract(red).multiply(8)
            ).sqrt()
        )
        .divide(2)
        .rename("MSAVI")
    )
    return image.addBands(msavi)


def compute_si(image: ee.Image) -> ee.Image:
    """SI = sqrt(B2 * B4)  - Salinity Index.

    Sentinel-2 bands: B2=Blue, B4=Red.
    Higher values indicate higher soil salinity (bright salt crusts).
    """
    si = image.select("B2").multiply(image.select("B4")).sqrt().rename("SI")
    return image.addBands(si)
