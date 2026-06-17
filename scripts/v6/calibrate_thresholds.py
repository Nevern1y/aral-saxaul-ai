# -*- coding: utf-8 -*-
"""Phase 5 — calibrate RS thresholds against measured soil + saxaul labels.

The V5.1 production thresholds are blind percentiles (P15/P85) with no
ground-truth calibration — the documented scientific weakness. This module
replaces "blind" with "measured" wherever the data support it.

Two decisions are recorded here, both evidence-based:

1. **The 10 m production cascade is NOT recalibrated.** Only 14/70 ground-truth
   points fall on valid 10 m pixels, the 10 m NDMI↔salinity correlation there is
   non-significant (rho≈+0.42, p≈0.13, n=14), and the 10 m NDMI even
   anti-correlates with the 30 m NDMI at the overlap points (r≈-0.35) — the 14
   valid points sit on NoData edges and are low quality. Recalibrating at n=14
   would be statistically hollow, and it would perturb the frozen V5.1 map
   (protected by the golden snapshot). So the V5.1 screening map is kept exactly.

2. **The wall-to-wall 30 m stack IS calibrated**, because its NDMI tracks
   measured topsoil salinity at rho≈+0.66 (p<1e-9, n=70) and it is 99 % valid
   across the AOI. These calibrated thresholds feed the Phase 6 suitability
   layer, not the V5.1 cascade.

For each candidate predictor we report, against two targets (soil salinity bands
and the saxaul label), a rank-AUC, a Youden-optimal cut, and n. Thresholds are
emitted only where AUC is meaningfully above chance and n is adequate; each
carries a human-readable justification. Nothing blind survives silently.

Output: data/canonical/thresholds_v6_calibrated.json + CALIBRATION_REPORT.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"

# salinity class breakpoints (% total salts, agronomic convention)
SALINE_PCT = 1.0     # > 1 %: saline soil, stressful for establishment
STRONG_PCT = 3.0     # > 3 %: strongly saline / solonchak
MIN_N = 12           # minimum sample for a calibrated threshold
MIN_AUC = 0.62       # minimum discrimination to publish a cut


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def rank_auc(score: np.ndarray, y: np.ndarray) -> float | None:
    """Mann-Whitney rank AUC (numpy only). Higher score -> higher P(y=1)."""
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return None
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def youden_cut(score: np.ndarray, y: np.ndarray) -> dict | None:
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return None
    auc = rank_auc(score, y)
    # if AUC<0.5 the predictor is inversely related; evaluate ">=" direction and
    # report the better-oriented cut.
    invert = auc is not None and auc < 0.5
    s = -score if invert else score
    best = None
    for t in np.unique(s):
        pred = s >= t
        tpr = (pred & pos).sum() / npos
        fpr = (pred & neg).sum() / nneg
        j = tpr - fpr
        if best is None or j > best["youden_j"]:
            cut = -t if invert else t
            best = {"cut": float(cut), "direction": ("<=" if invert else ">="),
                    "sensitivity": round(float(tpr), 2), "specificity": round(float(1 - fpr), 2),
                    "youden_j": round(float(j), 3)}
    if best:
        # report the ORIENTED discrimination (how well the cut separates in its
        # chosen direction), not the raw rank-AUC. For an inversely-related
        # predictor the raw AUC is <0.5; oriented AUC = 1 - raw. The direction
        # field already encodes the sign, so this is the honest "strength".
        best["auc"] = round(float(1.0 - auc if invert else auc), 3)
        best["auc_raw"] = round(float(auc), 3)
        best["n"] = int(npos + nneg)
        best["n_pos"] = npos
    return best


def calibrate_predictor(df, col, target_mask, target_y, label):
    v = num(df[col])
    m = v.notna() & target_mask
    if m.sum() < MIN_N:
        return None
    res = youden_cut(v[m].to_numpy(), target_y[m].to_numpy().astype(int))
    if not res:
        return None
    res.update({"predictor": col, "target": label})
    return res


def main() -> None:
    df = pd.read_csv(CANON / "ml_dataset_v6.csv")

    salt_top = num(df["top_salt_sum_salts_pct"])
    has_salt = salt_top.notna()
    y_saline = (salt_top > SALINE_PCT)
    y_strong = (salt_top > STRONG_PCT)

    y_label = pd.to_numeric(df["y_suitable"], errors="coerce")
    has_label = y_label.notna() & (num(df["label_weight"]) > 0)

    # candidate wall-to-wall 30 m predictors (VH is dead, excluded)
    rs_preds = ["rs30_ndmi", "rs30_ndwi", "rs30_si", "rs30_msavi", "rs30_twi", "rs30_slope"]
    soil_preds = ["top_salt_sum_salts_pct", "top_salt_cl_pct", "top_exch_na",
                  "top_caco3_pct", "top_caso4_pct", "top_sand_pct", "top_ph_water"]

    results = []
    # RS predictors vs measured salinity (the physical link the map relies on)
    for col in rs_preds:
        for y, lbl in [(y_saline, f"soil_salinity>{SALINE_PCT}%"),
                       (y_strong, f"soil_salinity>{STRONG_PCT}%")]:
            r = calibrate_predictor(df, col, has_salt, y, lbl)
            if r:
                results.append(r)
    # RS + soil predictors vs the saxaul suitability label
    for col in rs_preds + soil_preds:
        r = calibrate_predictor(df, col, has_label, y_label, "saxaul_suitable")
        if r:
            results.append(r)

    res_df = pd.DataFrame(results)

    # ---- emit calibrated thresholds where evidence is adequate ------------
    calibrated = {}
    for r in results:
        if r["auc"] >= MIN_AUC and r["n"] >= MIN_N:
            key = f'{r["predictor"]}__{r["target"]}'
            calibrated[key] = {
                "cut": round(r["cut"], 4), "direction": r["direction"],
                "auc": r["auc"], "sensitivity": r["sensitivity"],
                "specificity": r["specificity"], "n": r["n"], "n_pos": r["n_pos"],
                "justification": (
                    f'{r["predictor"]} {r["direction"]} {round(r["cut"],3)} separates '
                    f'{r["target"]} with AUC={r["auc"]} (sens={r["sensitivity"]}, '
                    f'spec={r["specificity"]}, n={r["n"]}).'),
            }

    out = {
        "version": "V6-calibrated-thresholds",
        "policy": {
            "v5_10m_cascade": "KEPT AS-IS (frozen). n=14 valid 10m points, non-significant, "
                              "low-quality NoData-edge pixels; recalibration not defensible.",
            "v6_30m_layer": "CALIBRATED below. 30m NDMI vs measured topsoil salinity "
                            "rho=+0.66 p<1e-9 n=70; stack is 99% valid across AOI.",
        },
        "salinity_breakpoints_pct": {"saline": SALINE_PCT, "strong": STRONG_PCT},
        "min_n": MIN_N, "min_auc": MIN_AUC,
        "calibrated_thresholds": calibrated,
        "all_evaluations": results,
    }
    (CANON / "thresholds_v6_calibrated.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- human report ----------------------------------------------------
    lines = ["# RS / soil threshold calibration — Phase 5", "",
             "Replaces blind P15/P85 percentiles with measured, AUC-justified cuts.",
             "", "## Policy", "",
             "- **10 m V5.1 cascade: kept frozen.** " + out["policy"]["v5_10m_cascade"],
             "- **30 m suitability layer: calibrated.** " + out["policy"]["v6_30m_layer"],
             "", "## Evaluations (rank-AUC, Youden-optimal cut)", "",
             "| predictor | target | dir | cut | AUC | sens | spec | n | n_pos |",
             "|-----------|--------|-----|-----|-----|------|------|---|-------|"]
    for r in sorted(results, key=lambda x: (-x["auc"], x["target"])):
        lines.append(f'| {r["predictor"]} | {r["target"]} | {r["direction"]} | '
                     f'{round(r["cut"],3)} | {r["auc"]} | {r["sensitivity"]} | '
                     f'{r["specificity"]} | {r["n"]} | {r["n_pos"]} |')
    lines += ["", f"Published thresholds (AUC≥{MIN_AUC}, n≥{MIN_N}): "
              f"{len(calibrated)} of {len(results)} evaluations.", ""]
    (CANON / "CALIBRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_evaluations": len(results),
                      "n_published": len(calibrated),
                      "published_keys": list(calibrated.keys())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
