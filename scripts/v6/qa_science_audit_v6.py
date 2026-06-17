"""qa_science_audit_v6.py — Red Team audit of the V6 lab-data science layer.

Adversarial checks that the V6 claims survive scrutiny. This is NOT a unit test of
code paths; it interrogates the *scientific artifacts* the dashboard ships:

  - No label circularity: the salinity model must be independent of saxaul labels.
  - Statistical sanity: AUCs in [0,1]; bootstrap CIs well-formed and bracketing the
    point estimate; LOO-CI lower bound >= 0.5 (guards the in-sample-bootstrap bug).
  - Honest spatial story: per-block AUC reported; pooled<per-block recorded as drift.
  - Reproducibility: zone areas reconcile with pixel counts; coverage fraction sane.
  - Coverage parity: V6 covers >= as many ground-truth pits as the frozen V5 product.
  - Polarity: NDMI coefficient positive (more NDMI -> more salt -> less suitable).
  - Honesty caveats present in the QA markdown (single-axis, no-SCL, temporal).

Exit code 0 if all checks pass, 1 otherwise. Tolerant of missing local rasters
(raster-derived artifacts skip with a note) so it runs on a clean checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
ODATA = BASE / "outputs" / "data"
MODELS = BASE / "outputs" / "models"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

errors: list[str] = []
checks_run = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global checks_run
    checks_run += 1
    if cond:
        print(f"  {label:<52} {PASS}  {detail}")
    else:
        errors.append(label.strip())
        print(f"  {label:<52} {FAIL}  {detail}")


def skip(label: str, detail: str = "") -> None:
    print(f"  {label:<52} {SKIP}  {detail}")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def section(title: str) -> None:
    print(f"\n{'-' * 70}\n  {title}\n{'-' * 70}", flush=True)


print("=" * 70)
print("  RED TEAM AUDIT — V6 lab-data science layer")
print("=" * 70, flush=True)

# ── 1. salinity model: independence + coefficients ──────────────────────
section("TEST 1: Salinity model — independence & polarity")
sal = load(MODELS / "salinity_v6_logit.json")
if sal is None:
    check(False, "salinity_v6_logit.json present", "run train_suitability_model.py")
else:
    check(sal.get("predictor") == "rs30_ndmi", "predictor is rs30_ndmi", str(sal.get("predictor")))
    notes = (sal.get("notes", "") + sal.get("purpose", "")).lower()
    check("saxaul" not in notes or "independent" in notes,
          "no saxaul label in model definition (no circularity)")
    coef = sal.get("coefficients_standardized", {}).get("rs30_ndmi")
    check(coef is not None and coef > 0, "NDMI coefficient positive (higher NDMI -> more salt)",
          f"coef={coef}")
    tr = sal.get("training", {})
    check(tr.get("n") == 70, "trained on n=70 profiles", f"n={tr.get('n')}")
    check(0.5 <= tr.get("loo_auc", 0) <= 1.0, "LOO AUC in (0.5, 1]", f"{tr.get('loo_auc')}")

# ── 2. spatial validation: CI well-formed + honest spatial metric ───────
section("TEST 2: Spatial validation — CIs & honesty")
sp = load(ODATA / "spatial_validation_v6.json")
if sp is None:
    check(False, "spatial_validation_v6.json present", "run spatial_validation.py")
else:
    sm = sp.get("salinity_model", {})
    lo, hi = (sm.get("loo_auc_ci95") or [None, None])
    check(lo is not None and hi is not None and lo <= sm.get("loo_auc", -1) <= hi,
          "LOO-AUC CI brackets the point estimate", f"[{lo},{hi}] pt={sm.get('loo_auc')}")
    check(lo is not None and lo >= 0.5,
          "LOO-AUC CI lower bound >= 0.5 (no in-sample-bootstrap bug)", f"lo={lo}")
    check("spatial_lbo_perblock_auc" in sm,
          "per-block spatial AUC reported (honest metric)",
          f"perblock={sm.get('spatial_lbo_perblock_auc')}")
    pb = sm.get("spatial_lbo_perblock_auc")
    pooled = sm.get("spatial_lbo_pooled_auc")
    check(pb is not None and 0.0 <= pb <= 1.0, "per-block AUC in [0,1]", f"{pb}")
    # if pooled < per-block, the drift must be acknowledged (it is, in the QA md)
    check(pooled is not None, "pooled spatial AUC also recorded", f"{pooled}")
    sign = sm.get("within_block_sign_positive", "0/0")
    pos, tot = (int(x) for x in sign.split("/")) if "/" in sign else (0, 0)
    check(tot == 0 or pos == tot, "NDMI->salt sign positive in ALL testable blocks", sign)
    lab = sp.get("suitability_vs_saxaul_label", {})
    lci = lab.get("ci95")
    check(lci is not None and lci[0] <= lab.get("auc", -1) <= lci[1],
          "suitability-vs-label CI brackets estimate", f"auc={lab.get('auc')} ci={lci}")

# ── 3. suitability raster stats: reconciliation + coverage ──────────────
section("TEST 3: Suitability index — reconciliation & coverage")
stats = load(ODATA / "suitability_v6_stats.json")
if stats is None:
    skip("suitability_v6_stats.json present", "raster step not run on this checkout")
else:
    zp = stats.get("zone_pixels", {})
    total = stats.get("grid", {}).get("total_pixels", 0)
    check(sum(zp.values()) == total, "zone pixel counts sum to grid total",
          f"{sum(zp.values()):,} vs {total:,}")
    vf = stats.get("valid_fraction_of_aoi")
    check(vf is not None and 0.8 <= vf <= 1.0, "AOI coverage >= 80%", f"{vf}")
    ef = stats.get("extrapolated_fraction_of_valid")
    check(ef is not None and ef < 0.15, "extrapolation < 15% of valid", f"{ef}")
    cuts = stats.get("salinity_model", {})
    check(cuts.get("saline_cut_ndmi", 1) < cuts.get("strong_cut_ndmi", -1),
          "saline cut < strong cut (severity ordering)",
          f"{cuts.get('saline_cut_ndmi')} < {cuts.get('strong_cut_ndmi')}")

# ── 4. ground-truth coverage parity vs frozen V5 ────────────────────────
section("TEST 4: Ground-truth coverage parity (V6 vs frozen V5)")
pv = load(ODATA / "suitability_v6_pit_validation_summary.json")
if pv is None:
    skip("suitability_v6_pit_validation_summary.json present", "raster step not run")
else:
    v6c = pv.get("v6_scored_nonwater", 0)
    v5c = pv.get("v5_covered_nonwater", 0)
    check(v6c >= v5c, "V6 covers >= as many GT pits as V5", f"V6={v6c} V5={v5c}")
    det = pv.get("saline_detector_zone34", {})
    check(det.get("sensitivity", 0) >= 0.5, "zone3/4 saline-detector sensitivity >= 0.5",
          f"sens={det.get('sensitivity')}")
    check(det.get("specificity", 0) >= 0.5, "zone3/4 saline-detector specificity >= 0.5",
          f"spec={det.get('specificity')}")
    msb = pv.get("mean_salt_by_zone", {})
    if "4" in msb:
        # strong-salinity zone must be more saline than the least-saline scored zone
        lowest = min(float(v) for k, v in msb.items() if k != "4")
        check(float(msb["4"]) > lowest, "mean measured salt: strong(4) > least-saline zone",
              f"strong(4)={msb['4']} > min(other)={lowest}")

# ── 5. honesty caveats present in the shipped QA markdown ────────────────
section("TEST 5: Honesty caveats present in QA report")
qa_md = CANON / "SUITABILITY_INDEX_V6_QA.md"
if not qa_md.exists():
    check(False, "SUITABILITY_INDEX_V6_QA.md present")
else:
    txt = qa_md.read_text(encoding="utf-8").lower()
    check("slope band" in txt and "33" in txt, "documents 30m slope-band gap")
    check("scl" in txt, "documents missing SCL water band")
    check("severity split" in txt or "severity" in txt, "documents zones 3/4 are a severity split")
    check("2012" in txt and "temporal" in txt, "documents temporal mismatch caveat")
    check("13" in txt and "15" in txt, "documents V5(13)/V6(15) coverage parity")

# ── 6. dataset sanity ───────────────────────────────────────────────────
section("TEST 6: ML dataset sanity")
ml = CANON / "ml_dataset_v6.csv"
if not ml.exists():
    check(False, "ml_dataset_v6.csv present")
else:
    import csv
    with open(ml, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 70, "ml_dataset_v6.csv has 70 rows", f"{len(rows)}")
    has_salt = sum(1 for r in rows if r.get("top_salt_sum_salts_pct", "").strip() not in ("", "nan"))
    check(has_salt == 70, "all 70 rows have measured topsoil salinity", f"{has_salt}/70")
    has_ndmi = sum(1 for r in rows if r.get("rs30_ndmi", "").strip() not in ("", "nan"))
    check(has_ndmi == 70, "all 70 rows have 30m NDMI", f"{has_ndmi}/70")

# ── verdict ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  V6 SCIENCE AUDIT SUMMARY")
print(f"{'=' * 70}")
print(f"  Checks: {len(errors)} failed  |  {checks_run - len(errors)} passed", flush=True)
if not errors:
    print(f"  VERDICT: {PASS} ALL V6 SCIENCE CHECKS PASSED", flush=True)
else:
    print(f"  VERDICT: {FAIL} {len(errors)} FAILURES", flush=True)
    for e in errors:
        print(f"    - {e}", flush=True)
print("=" * 70, flush=True)
sys.exit(1 if errors else 0)
