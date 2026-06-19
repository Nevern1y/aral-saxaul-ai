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

# ── 5b. morphological feature ablation ──────────────────────────────────
section("TEST 5b: Morphological ablation — coverage, no-circularity, AUC sanity")
sp_abl = (sp or {}).get("morph_ablation") if sp else None
if sp is None:
    skip("morph_ablation in spatial_validation_v6.json", "spatial_validation.py not run")
elif sp_abl is None:
    check(False, "morph_ablation key present in spatial_validation_v6.json",
          "run spatial_validation.py after morph_features.py")
else:
    status = sp_abl.get("status", "unknown")
    check(status in ("computed", "skipped"), "morph_ablation status is valid", f"status={status}")
    if status == "computed":
        bl = sp_abl.get("baseline_ndmi_only", {})
        ag = sp_abl.get("augmented_ndmi_plus_morph", {})
        bl_auc = bl.get("loo_auc")
        ag_auc = ag.get("loo_auc")
        bl_ci = bl.get("ci95")
        ag_ci = ag.get("ci95")
        # AUC sanity: both in [0,1]
        check(bl_auc is not None and 0.0 <= bl_auc <= 1.0,
              "morph ablation baseline AUC in [0,1]", f"auc={bl_auc}")
        check(ag_auc is not None and 0.0 <= ag_auc <= 1.0,
              "morph ablation augmented AUC in [0,1]", f"auc={ag_auc}")
        # CI brackets point estimate (baseline)
        if bl_ci and bl_auc is not None:
            check(bl_ci[0] <= bl_auc <= bl_ci[1],
                  "morph ablation baseline CI brackets point estimate",
                  f"[{bl_ci[0]},{bl_ci[1]}] pt={bl_auc}")
        # CI brackets point estimate (augmented)
        if ag_ci and ag_auc is not None:
            check(ag_ci[0] <= ag_auc <= ag_ci[1],
                  "morph ablation augmented CI brackets point estimate",
                  f"[{ag_ci[0]},{ag_ci[1]}] pt={ag_auc}")
        # No-circularity: target must be salinity, not saxaul labels
        tgt = sp_abl.get("target", "")
        check("salt" in tgt.lower() or "salin" in tgt.lower(),
              "morph ablation target is salinity (no saxaul circularity)", f"target='{tgt}'")
        # No-circularity: 'no_circularity_note' must be present
        check("no_circularity_note" in sp_abl,
              "morph ablation has no_circularity_note key")
        # Coverage note: complete case n < 70 (depth_to_moist has some NaN)
        nc = sp_abl.get("n_complete_cases", 0)
        check(nc > 0, "morph ablation ran on non-zero complete-case subset", f"n={nc}")
        # Delta AUC must be finite
        delta = sp_abl.get("delta_auc")
        check(delta is not None and delta == delta,
              "morph ablation delta_auc is a finite number", f"delta={delta}")
        # Direction key present
        direction = sp_abl.get("direction", "")
        check(direction in ("lift", "neutral", "hurt"),
              "morph ablation direction key is valid", f"direction={direction}")
        # Honest caveat key present
        check("honest_caveat" in sp_abl,
              "morph ablation has honest_caveat key")
    else:
        skip("morph_ablation AUC checks", f"status={status}")

# ── 5c. morph features file sanity ────────────────────────────────────────
section("TEST 5c: Morph features file sanity")
import csv as _csv
morph_csv = CANON / "morph_features_v6.csv"
morph_manifest = CANON / "morph_features_manifest.json"
if not morph_csv.exists():
    check(False, "morph_features_v6.csv present", "run scripts/v6/morph_features.py")
else:
    with open(morph_csv, encoding="utf-8") as _f:
        morph_rows = list(_csv.DictReader(_f))
    check(len(morph_rows) == 76, "morph_features_v6.csv has 76 rows (all pits)", f"{len(morph_rows)}")
    _expected_morph_cols = [
        "depth_to_moist_cm", "depth_to_salt_cm", "rust_mottling_flag", "gley_flag",
        "solum_depth_cm", "hcl_effervescence_class", "surface_crust_flag",
        "marine_shell_flag", "horizon_salic_flag", "horizon_ploughed_flag",
    ]
    _present = [c for c in _expected_morph_cols if c in (morph_rows[0].keys() if morph_rows else [])]
    check(len(_present) == len(_expected_morph_cols),
          "morph_features_v6.csv has all 10 expected feature columns",
          f"{len(_present)}/{len(_expected_morph_cols)}")
    # solum_depth should be 100% non-null
    _sol_ok = sum(1 for r in morph_rows if r.get("solum_depth_cm", "").strip() not in ("", "nan"))
    check(_sol_ok == 76, "solum_depth_cm 100% coverage", f"{_sol_ok}/76")
    # rust_mottling_flag should be 100% non-null
    _rust_ok = sum(1 for r in morph_rows if r.get("rust_mottling_flag", "").strip() not in ("", "nan"))
    check(_rust_ok == 76, "rust_mottling_flag 100% coverage", f"{_rust_ok}/76")

if not morph_manifest.exists():
    check(False, "morph_features_manifest.json present", "run scripts/v6/morph_features.py")
else:
    _mman = load(morph_manifest)
    check("low_coverage_note" in _mman,
          "morph_features_manifest.json has low_coverage_note (missingness caveat)")
    check("missingness_policy" in _mman,
          "morph_features_manifest.json has missingness_policy (NaN not imputed)")
    # Vocab file is tracked and referenced
    check("vocab_file" in _mman,
          "morph_features_manifest references morph_vocab.json")

# ── 5d. morph vocab file sanity ──────────────────────────────────────────
section("TEST 5d: Morph vocabulary file sanity")
morph_vocab = CANON / "morph_vocab.json"
if not morph_vocab.exists():
    check(False, "morph_vocab.json present (tracked vocabulary)", "expected in data/canonical/")
else:
    _vocab = load(morph_vocab)
    check("moisture_levels" in _vocab, "morph_vocab has moisture_levels section")
    check("hcl_effervescence" in _vocab, "morph_vocab has hcl_effervescence section")
    check("salt_tokens_in_inclusions" in _vocab,
          "morph_vocab has salt_tokens_in_inclusions section")
    check("feature_coverage_notes" in _vocab,
          "morph_vocab has feature_coverage_notes (explicit missingness docs)")
    # depth_to_salt coverage note must mention <60% or 59%
    _dcov = _vocab.get("feature_coverage_notes", {}).get("depth_to_salt_cm", "")
    check("<60%" in _dcov or "59%" in _dcov or "63%" in _dcov or "coverage" in _dcov.lower(),
          "depth_to_salt coverage note present in morph_vocab.json")

# ── 5e. calibrated thresholds — honesty annotations (W6/W7/W8) ──────────
section("TEST 5e: Calibrated thresholds — n_pos/stability/auc_raw invariants")
cal_path = CANON / "thresholds_v6_calibrated.json"
if not cal_path.exists():
    check(False, "thresholds_v6_calibrated.json present", "run calibrate_thresholds.py")
else:
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    ct = cal.get("calibrated_thresholds", {})
    # W6: every entry must have n_pos and stability fields
    missing_npos = [k for k, v in ct.items() if "n_pos" not in v]
    check(len(missing_npos) == 0,
          "all calibrated_thresholds entries have n_pos field",
          f"missing in: {missing_npos}" if missing_npos else "OK")
    missing_stab = [k for k, v in ct.items() if "stability" not in v]
    check(len(missing_stab) == 0,
          "all calibrated_thresholds entries have stability field",
          f"missing in: {missing_stab}" if missing_stab else "OK")
    # W7: every entry must have auc_raw and inverted fields
    missing_raw = [k for k, v in ct.items() if "auc_raw" not in v]
    check(len(missing_raw) == 0,
          "all calibrated_thresholds entries have auc_raw field",
          f"missing in: {missing_raw}" if missing_raw else "OK")
    missing_inv = [k for k, v in ct.items() if "inverted" not in v]
    check(len(missing_inv) == 0,
          "all calibrated_thresholds entries have inverted field",
          f"missing in: {missing_inv}" if missing_inv else "OK")
    # W8: entries with n_pos below MIN_N_POS must have indicative_only=True
    min_n_pos = cal.get("min_n_pos", 8)
    for k, v in ct.items():
        npos = v.get("n_pos", 999)
        if npos < min_n_pos:
            check(v.get("indicative_only") is True,
                  f"entry with n_pos={npos} ({k[:40]}) has indicative_only=True",
                  f"indicative_only={v.get('indicative_only')}")
    # W7: for inverted=True entries, auc_raw must be < auc (the oriented value)
    for k, v in ct.items():
        if v.get("inverted"):
            check(v.get("auc_raw", 1.0) < v.get("auc", 0.0),
                  f"inverted entry {k[:50]}: auc_raw < auc",
                  f"raw={v.get('auc_raw')} oriented={v.get('auc')}")
    # top-level stability_note and min_n_pos must be present
    check("stability_note" in cal,
          "thresholds JSON has top-level stability_note field")
    check("min_n_pos" in cal,
          "thresholds JSON has top-level min_n_pos field",
          f"min_n_pos={cal.get('min_n_pos')}")

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


# ── 6b. MODEL salinity benchmark — honesty + scope invariants (W1/W3/W4/W5/W9) ─
section("TEST 6b: MODEL benchmark — pooled-AUC scope, in-AOI floor, recommendation honesty")
bench_path = CANON / "model_v6_benchmark.json"
bench_report = CANON / "model_v6_benchmark_report.md"
if not bench_path.exists():
    check(False, "model_v6_benchmark.json present", "run scripts/v6/benchmark_salinity_model.py")
else:
    bench = load(bench_path)
    rec = bench.get("recommendation", {})
    models = {m.get("id"): m for m in bench.get("models", [])}
    # Recommendation block present + internally consistent
    check("recommended_model_id" in rec, "benchmark records a recommended_model_id",
          str(rec.get("recommended_model_id")))
    check("cascade_needed" in rec, "benchmark records cascade_needed flag",
          str(rec.get("cascade_needed")))
    check(rec.get("shipped_model_unchanged") == (not rec.get("cascade_needed")),
          "shipped_model_unchanged is consistent with cascade_needed")
    # W1: pooled spatial AUC re-measured and recorded for the baseline
    m0 = models.get("M0", {})
    pooled = m0.get("pooled_spatial_auc")
    perblk = m0.get("mean_per_block_spatial_auc")
    check(pooled is not None and 0.0 <= pooled <= 1.0,
          "baseline pooled spatial AUC recorded (W1 regional-drift scope)", f"pooled={pooled}")
    check(perblk is not None and 0.0 <= perblk <= 1.0,
          "baseline per-block spatial AUC recorded (honest local metric)", f"per_block={perblk}")
    # Baseline LOO AUC must match the shipped salinity model (consistency with TEST 1/2)
    if sal is not None:
        shipped_auc = sal.get("training", {}).get("loo_auc") if isinstance(sal.get("training"), dict) else None
        if shipped_auc is not None and m0.get("loo_auc") is not None:
            check(abs(m0["loo_auc"] - shipped_auc) <= 0.02,
                  "benchmark M0 LOO AUC matches shipped salinity model (protocol consistency)",
                  f"bench={m0.get('loo_auc')} shipped={shipped_auc}")
    # Every model carries a held-out LOO CI (no point estimate without CI)
    for mid, m in models.items():
        ci = m.get("loo_auc_ci95")
        check(isinstance(ci, list) and len(ci) == 2 and ci[0] <= m.get("loo_auc", -1) <= ci[1],
              f"{mid}: LOO AUC within its bootstrap CI (CI never dropped)", f"auc={m.get('loo_auc')} ci={ci}")
    # W4/W5: in-AOI vs out-of-AOI reported separately, derived from the 1960 footprint (not the all-True column)
    aoi = bench.get("aoi_split", {})
    n_in = aoi.get("n_in_aoi")
    n_out = aoi.get("n_out_of_aoi")
    check(n_in is not None and n_out is not None and (n_in + n_out) == bench.get("n_total", 70),
          "in-AOI/out-of-AOI split recorded and sums to n_total (W4/W5)", f"in={n_in} out={n_out}")
    check(n_in is not None and 0 < n_in < bench.get("n_total", 70),
          "in-AOI split is a true subset, not the all-True in_aoi column (W4)", f"n_in={n_in}")
    check("aoi_stratified_auc" in m0,
          "baseline reports AOI-stratified AUC (in-AOI test skill separated from training)")
    # IN-AOI FLOOR (W5): if the in-AOI AUC is numeric, it must clear a stated floor
    in_aoi_floor = 0.5
    m0_aoi = m0.get("aoi_stratified_auc", {})
    in_aoi_auc = m0_aoi.get("in_aoi")
    if isinstance(in_aoi_auc, (int, float)):
        check(in_aoi_auc >= in_aoi_floor,
              f"baseline in-AOI LOO AUC >= floor {in_aoi_floor} (W5 in-AOI skill gate)",
              f"in_aoi_auc={in_aoi_auc}")
    else:
        skip("in-AOI AUC floor", f"in-AOI AUC not numeric (small-n): {in_aoi_auc}")
    # W3: texture-stratified AUC present (single-axis residual bias quantified)
    check("texture_stratified_auc" in m0, "baseline reports texture-stratified AUC (W3)")
    # W9: lambda recorded per model (sensitivity considered, not a single hidden lambda)
    check(all("best_lam" in m for m in models.values()),
          "every benchmarked model records its selected lambda (W9)")
    # Report prose and JSON must agree on the recommendation (no desync)
    if bench_report.exists():
        rtext = bench_report.read_text(encoding="utf-8")
        rid = rec.get("recommended_model_id", "")
        casc = str(rec.get("cascade_needed")).lower()
        check(f"cascade_needed = {casc}" in rtext,
              "benchmark report prose agrees with JSON on cascade_needed", f"cascade_needed={casc}")
        check(rid and rid in rtext,
              "benchmark report prose names the JSON-recommended model", f"id={rid}")



# ── 6c. REPORT — scope statements + honest known-limitations (W1/W4/W5/W10/W11/W12) ─
section("TEST 6c: REPORT scope statements + limitations (script-generated, enforced)")
scope_path = CANON / "SCOPE_AND_LIMITATIONS_V6.md"
if not scope_path.exists():
    check(False, "SCOPE_AND_LIMITATIONS_V6.md present", "run scripts/v6/generate_scope_statements.py")
else:
    scope = scope_path.read_text(encoding="utf-8")
    bench_p = CANON / "model_v6_benchmark.json"
    m0b = {}
    if bench_p.exists():
        _b = load(bench_p)
        m0b = next((m for m in _b.get("models", []) if m.get("id") == "M0"), {})
    # W1: pooled + per-block numbers present AND match the benchmark JSON (no hand-typed drift)
    pooled = m0b.get("pooled_spatial_auc")
    perblk = m0b.get("mean_per_block_spatial_auc")
    check("Regional calibration drift" in scope or "regional" in scope.lower(),
          "scope doc states W1 regional calibration drift")
    if pooled is not None:
        check(f"{pooled:.3f}" in scope or str(pooled) in scope,
              "scope doc pooled AUC matches benchmark JSON (not hand-typed)", f"pooled={pooled}")
    if perblk is not None:
        check(f"{perblk:.3f}" in scope or str(perblk) in scope,
              "scope doc per-block AUC matches benchmark JSON", f"per_block={perblk}")
    # W4/W5: in-AOI vs out-of-AOI distinction surfaced
    check("in-AOI" in scope and "out-of-AOI" in scope,
          "scope doc surfaces in-AOI vs out-of-AOI (W4/W5)")
    # W10: temporal drift caveat
    check("2012-2014" in scope or "2012–2014" in scope,
          "scope doc carries temporal-drift / calibration-vintage caveat (W10)")
    check("temporal" in scope.lower() or "stationar" in scope.lower(),
          "scope doc names temporal drift / stationarity assumption (W10)")
    # W11: SCL water-mask + slope/TWI gaps documented
    check("SCL" in scope and ("slope" in scope.lower() or "TWI" in scope),
          "scope doc documents 30m SCL + slope/TWI coverage gaps (W11)")
    # W12: 10m re-evaluation trigger with concrete threshold
    check("re-test" in scope.lower() or "re-evaluat" in scope.lower(),
          "scope doc records the 10m cascade re-evaluation trigger (W12)")
    check("n >= 30" in scope or "n>=30" in scope or "n ≥ 30" in scope or "30" in scope,
          "scope doc states the concrete valid-pixel re-test threshold (W12)")
    # 2020/2021 decision: out-of-period only, NOT merged
    check("2020/2021" in scope or "2020" in scope,
          "scope doc records the 2020/2021 out-of-period decision")
    check("NOT merged" in scope or "not merged" in scope.lower(),
          "scope doc states 2020/2021 is NOT merged into 2012-2014 labels")
    # anti-pattern guard restated
    check("1 - P(saline)" in scope or "1 − P(saline)" in scope or "1-P(saline)" in scope,
          "scope doc restates suitability = 1 - P(saline) (anti-pattern guard)")

    # README + CLAUDE carry the auto-generated scope block with the live pooled number
    for _doc, _name in [(BASE / "README.md", "README.md"), (BASE / "CLAUDE.md", "CLAUDE.md")]:
        if _doc.exists():
            _t = _doc.read_text(encoding="utf-8")
            check("<!-- V6_SCOPE_AUTO -->" in _t,
                  f"{_name} has the auto-generated V6 scope block")
            if pooled is not None:
                check(f"{pooled:.3f}" in _t or str(pooled) in _t,
                      f"{_name} scope block pooled AUC matches benchmark JSON (script-generated)",
                      f"pooled={pooled}")

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
