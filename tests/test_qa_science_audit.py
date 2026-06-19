"""Wire the V6 science Red Team audit into pytest (RISK-7).

``scripts/v6/qa_science_audit_v6.py`` is the de-facto *integration* test of the
V6 science layer — 28 adversarial checks (no label circularity, AUC/CI sanity,
zone reconciliation, coverage parity, honesty caveats). It was previously only
runnable by hand (``python scripts/v6/qa_science_audit_v6.py`` or
``run_v6_pipeline --qa``) and never enforced by the test suite.

This test runs that script as a subprocess (it executes at import time and calls
``sys.exit``) and asserts it reports **0 failed**. It skips cleanly when the V6
science artifacts the audit hard-requires are absent (a bare checkout that has
not run the pipeline) so the suite stays green on Streamlit Cloud / CI, matching
the project's "tests skip when generated data is absent" convention.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "scripts" / "v6" / "qa_science_audit_v6.py"

# Artifacts the audit turns into hard failures (check(False)) when missing. These
# are git-tracked, so they exist on a clean checkout and the audit runs there too;
# guard anyway so a bare tree (pipeline never run) skips instead of failing.
REQUIRED = [
    ROOT / "outputs" / "models" / "salinity_v6_logit.json",
    ROOT / "outputs" / "data" / "spatial_validation_v6.json",
    ROOT / "data" / "canonical" / "SUITABILITY_INDEX_V6_QA.md",
    ROOT / "data" / "canonical" / "ml_dataset_v6.csv",
]


@pytest.mark.skipif(
    not AUDIT.exists() or any(not p.exists() for p in REQUIRED),
    reason="V6 science artifacts not present (pipeline not run on this checkout)",
)
def test_qa_science_audit_v6_passes() -> None:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(AUDIT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    # exit 0 == "0 failed"; on failure surface the audit's own summary for triage.
    assert proc.returncode == 0, (
        "qa_science_audit_v6.py reported failures:\n" + "\n".join(out.splitlines()[-25:])
    )
    assert "0 failed" in out, f"audit did not report '0 failed':\n{out[-1500:]}"
