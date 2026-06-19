# -*- coding: utf-8 -*-
"""Producer / provenance guard for ``data/canonical/aralfield_validation_v6.csv``.

RISK-1 (CODE_ENGINEERING_AUDIT R1): the independent AralField 2018 validation set
is READ by Phase 6 (``train_suitability_model.py``) and Phase 7
(``spatial_validation.py``) to report the shipped "independent AralField AUC", but
its ``ndmi``/``msavi`` columns were sampled by a lost process and produced by NO
script. That violated the project's core rule — *every shipped number is
script-produced* — and made the AralField AUC unreproducible.

This script closes that hole **without changing any shipped number**:

- The FIELD columns (``id, lon, lat, haloxylon, veg, note``) are the frozen 2018
  AralField survey — independent field observations, not derivable from rasters.
  They live in ``aralfield_validation_v6.provenance.json`` (the committed external
  input) and are written verbatim.
- The REMOTE-SENSING columns (``ndmi`` = band 1, ``msavi`` = band 2) are re-sampled
  from the two good 30 m tiles at the AralField points, nearest-pixel
  (``rasterio.src.index``), first-valid-tile-wins, over tile order
  ``[feature_stack_30m_tile0_redo.tif, feature_stack_30m_tile1.tif]``. This
  reproduces the committed file **byte-for-byte** (verified: SHA256 in the
  provenance JSON).

Modes
-----
``--rasters``  Regenerate the CSV from the provenance field records + raster
               sampling. Writes the file and asserts the SHA256 matches the
               provenance (fail loudly on drift). This is the real producer.
(no flag)      Frozen-input verify mode: do NOT touch rasters; assert the
               committed CSV's SHA256 equals the provenance's ``expected_output``.
               Runs on a clean checkout where the rasters are gitignored/absent.

Either mode makes the AralField number *regenerable and pinned*: the file can no
longer silently drift from what produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
ODATA = BASE / "outputs" / "data"

AF = CANON / "aralfield_validation_v6.csv"
PROV = CANON / "aralfield_validation_v6.provenance.json"

# RS sampling: ndmi = band 1, msavi = band 2 of the 30 m stack. Tile order and
# first-valid-wins are part of the reproducibility contract (see provenance JSON).
RS_BANDS = {"ndmi": 1, "msavi": 2}
TILE_ORDER = ["feature_stack_30m_tile0_redo.tif", "feature_stack_30m_tile1.tif"]
FIELD_COLS = ["id", "lon", "lat", "haloxylon", "veg", "note"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_provenance() -> dict:
    if not PROV.exists():
        raise SystemExit(f"missing provenance file: {PROV}")
    return json.loads(PROV.read_text(encoding="utf-8"))


def verify_only(prov: dict) -> None:
    """Frozen-input mode: assert the committed CSV matches its pinned SHA256."""
    if not AF.exists():
        raise SystemExit(f"missing AralField validation CSV: {AF}")
    want = prov["expected_output"]["sha256"]
    got = sha256_of(AF)
    if got != want:
        raise SystemExit(
            f"AralField CSV SHA256 mismatch (frozen-input drift!):\n"
            f"  expected {want}\n  found    {got}\n"
            f"Regenerate with --rasters or restore the committed file."
        )
    print(f"OK: {AF.name} matches pinned SHA256 ({got[:16]}…, {AF.stat().st_size} bytes).")


def _verify_tiles(prov: dict) -> None:
    """If the good tiles are present, assert their SHA256 match the pins (no silent
    re-mosaic). Skip a tile that is absent (gitignored on a clean checkout)."""
    pins = prov["remote_sensing_sampling"]["tiles"]
    for name, meta in pins.items():
        p = ODATA / name
        if not p.exists():
            print(f"  (tile absent, skipping pin check: {name})")
            continue
        got = sha256_of(p)
        if got != meta["sha256"]:
            raise SystemExit(
                f"tile SHA256 mismatch for {name}:\n  expected {meta['sha256']}\n  found    {got}\n"
                f"The 30 m stack changed; the AralField sampling cannot be trusted to reproduce."
            )
        print(f"  tile pin OK: {name} ({got[:16]}…)")


def regenerate(prov: dict) -> None:
    """Producer mode: re-sample ndmi/msavi from the rasters and write the CSV,
    then assert it reproduces the pinned SHA256 byte-for-byte."""
    try:
        import numpy as np
        import pandas as pd
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - builder needs the GIS stack
        raise SystemExit(
            f"--rasters needs the GIS stack (numpy/pandas/rasterio): {exc}. "
            f"Run without --rasters to verify the frozen input instead."
        )

    tiles = [ODATA / n for n in TILE_ORDER]
    for tp in tiles:
        if not tp.exists():
            raise SystemExit(
                f"missing good tile: {tp}. The 30 m stack is gitignored; "
                f"run without --rasters to verify the committed CSV instead."
            )
    _verify_tiles(prov)

    records = prov["field_survey"]["records"]

    def sample(lon: float, lat: float, band: int) -> float:
        for tp in tiles:
            with rasterio.open(tp) as src:
                try:
                    r, c = src.index(lon, lat)
                except Exception:
                    continue
                if 0 <= r < src.height and 0 <= c < src.width:
                    v = src.read(band, window=Window(c, r, 1, 1))[0, 0]
                    if np.isfinite(v):
                        return float(v)
        return float("nan")

    df = pd.DataFrame(records, columns=FIELD_COLS)
    for col, band in RS_BANDS.items():
        df[col] = [sample(float(rec["lon"]), float(rec["lat"]), band) for rec in records]

    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\r\n")
    data = buf.getvalue().encode("utf-8")

    want = prov["expected_output"]["sha256"]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        # Write to a sidecar for inspection and fail loudly — never silently ship a
        # different number than what is pinned.
        bad = AF.with_suffix(".regenerated.csv")
        bad.write_bytes(data)
        raise SystemExit(
            f"regenerated AralField CSV does NOT match the pinned SHA256:\n"
            f"  expected {want}\n  found    {got}\n  wrote {bad} for inspection.\n"
            f"The sampling no longer reproduces the shipped numbers — investigate before committing."
        )

    AF.write_bytes(data)
    print(f"OK: regenerated {AF.name} byte-identically (SHA256 {got[:16]}…, {len(data)} bytes).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rasters", action="store_true",
                    help="re-sample ndmi/msavi from the 30 m tiles and rewrite the CSV "
                         "(asserts byte-identical to the pinned SHA256)")
    args = ap.parse_args()

    prov = load_provenance()
    if args.rasters:
        regenerate(prov)
    else:
        verify_only(prov)


if __name__ == "__main__":
    sys.exit(main())
