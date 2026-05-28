"""V5.0 standalone configuration — zero imports from src/ (legacy)."""

from enum import IntEnum
from pathlib import Path


class ZoneClass(IntEnum):
    WATER_NODATA = 0
    OPTIMAL = 1
    RISK_DRY_SALT = 3
    DEAD_WET_TOXIC = 4
    OBSTACLE_TOPO = 5
    VEGETATION = 10


ARAL_BBOX: tuple = (57.5, 43.3, 62.0, 46.7)
EXPORT_CRS: str = "EPSG:32641"
DRIVE_FOLDER: str = "aral_saxaul_v5_raw"
PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)
