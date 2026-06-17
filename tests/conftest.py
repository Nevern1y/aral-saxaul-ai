"""Shared pytest fixtures for the Aral Saxaul dashboard test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable so `import app` works from tests/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def has_v5_data() -> bool:
    """True when the locally generated V5 data tree is present.

    The pipeline rasters/CSVs are gitignored and only exist on a dev machine
    that has run scripts/. Contract tests skip cleanly when they are absent.
    """
    return (OUTPUTS / "data" / "v5_stats.json").exists()
