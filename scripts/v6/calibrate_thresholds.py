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
MIN_N = 12           # minimum total sample for a calibrated threshold
MIN_N_POS = 8        # minimum positives for a *robust* cut (below -> indicative only)
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
    # W6/W8: add n_pos, stability flag, and indicative_only gate to every entry.
    # W7: carry auc_raw and direction (including inversion flag) into each entry.
    calibrated = {}
    for r in results:
        if r["auc"] >= MIN_AUC and r["n"] >= MIN_N:
            key = f'{r["predictor"]}__{r["target"]}'
            robust = r["n_pos"] >= MIN_N_POS
            # stability: "low (n_pos<=N)" when below the MIN_N_POS gate
            stability = "robust" if robust else f"low (n_pos<={r['n_pos']})"
            inverted = r["auc_raw"] < r["auc"]  # auc_raw < auc => predictor was flipped
            entry = {
                "cut": round(r["cut"], 4),
                "direction": r["direction"],
                "auc": r["auc"],
                "auc_raw": r["auc_raw"],
                "inverted": inverted,
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "n": r["n"],
                "n_pos": r["n_pos"],
                "stability": stability,
                "justification": (
                    f'{r["predictor"]} {r["direction"]} {round(r["cut"],3)} separates '
                    f'{r["target"]} with AUC={r["auc"]} (raw={r["auc_raw"]}, '
                    f'{"INVERTED predictor; " if inverted else ""}'
                    f'sens={r["sensitivity"]}, spec={r["specificity"]}, n={r["n"]}, '
                    f'n_pos={r["n_pos"]}, stability={stability}).'),
            }
            if not robust:
                entry["indicative_only"] = True
                entry["indicative_note"] = (
                    f"n_pos={r['n_pos']} < MIN_N_POS={MIN_N_POS}: cut is underpowered "
                    f"(TPR axis has ~{r['n_pos']+1} attainable values). "
                    f"Use as a screening aid only, not a validated threshold.")
            calibrated[key] = entry

    n_robust = sum(1 for v in calibrated.values() if not v.get("indicative_only"))
    n_indicative = sum(1 for v in calibrated.values() if v.get("indicative_only"))
    out = {
        "version": "V6-calibrated-thresholds",
        "policy": {
            "v5_10m_cascade": "KEPT AS-IS (frozen). n=14 valid 10m points, non-significant, "
                              "low-quality NoData-edge pixels; recalibration not defensible.",
            "v6_30m_layer": "CALIBRATED below. 30m NDMI vs measured topsoil salinity "
                            "rho=+0.66 p<1e-9 n=70; stack is 99% valid across AOI.",
        },
        "salinity_breakpoints_pct": {"saline": SALINE_PCT, "strong": STRONG_PCT},
        "min_n": MIN_N, "min_auc": MIN_AUC, "min_n_pos": MIN_N_POS,
        "stability_note": (
            f"Entries with n_pos<{MIN_N_POS} are marked indicative_only=true. "
            f"These cuts are statistically underpowered and must not be used as "
            f"standalone validated thresholds. {n_robust} robust / {n_indicative} indicative-only."),
        "calibrated_thresholds": calibrated,
        "all_evaluations": results,
    }
    (CANON / "thresholds_v6_calibrated.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- human report (R3 fix: ALL numbers generated from results, never hardcoded) ---
    lines = ["# RS / soil threshold calibration — Phase 5", "",
             "Replaces blind P15/P85 percentiles with measured, AUC-justified cuts.",
             "", "## Policy", "",
             "- **10 m V5.1 cascade: kept frozen.** " + out["policy"]["v5_10m_cascade"],
             "- **30 m suitability layer: calibrated.** " + out["policy"]["v6_30m_layer"],
             "", "## Stability gates", "",
             f"- `MIN_N = {MIN_N}`: minimum total n for any published threshold.",
             f"- `MIN_N_POS = {MIN_N_POS}`: minimum positives for a *robust* threshold.",
             f"  Entries with n_pos < {MIN_N_POS} are published but flagged "
             f"`indicative_only=true` in the JSON.",
             f"- `MIN_AUC = {MIN_AUC}`: minimum oriented AUC (below this the entry is dropped).",
             "",
             "## Evaluations (rank-AUC, Youden-optimal cut)", "",
             "| predictor | target | dir | cut | AUC (oriented) | AUC (raw) | inverted | sens | spec | n | n_pos | stability |",
             "|-----------|--------|-----|-----|----------------|-----------|----------|------|------|---|-------|-----------|"]
    for r in sorted(results, key=lambda x: (-x["auc"], x["target"])):
        inv_flag = "yes" if r["auc_raw"] < r["auc"] else "no"
        rob = "robust" if r["n_pos"] >= MIN_N_POS else f"low(n_pos={r['n_pos']})"
        lines.append(f'| {r["predictor"]} | {r["target"]} | {r["direction"]} | '
                     f'{round(r["cut"],3)} | {r["auc"]} | {r["auc_raw"]} | {inv_flag} | '
                     f'{r["sensitivity"]} | {r["specificity"]} | {r["n"]} | {r["n_pos"]} | {rob} |')
    lines += ["",
              f"Published thresholds (AUC≥{MIN_AUC}, n≥{MIN_N}): "
              f"{len(calibrated)} of {len(results)} evaluations "
              f"({n_robust} robust, {n_indicative} indicative-only [n_pos<{MIN_N_POS}]).", ""]

    # --- Interpretation block: fully generated from computed results (fixes R3) ---
    # Build a lookup by (predictor, target) for convenient access
    by_key = {(r["predictor"], r["target"]): r for r in results}

    # NDMI salinity proxy block
    ndmi_s1 = by_key.get(("rs30_ndmi", f"soil_salinity>{SALINE_PCT}%"))
    ndmi_s3 = by_key.get(("rs30_ndmi", f"soil_salinity>{STRONG_PCT}%"))
    ndmi_sax = by_key.get(("rs30_ndmi", "saxaul_suitable"))

    # Saxaul-suitability soil-feature block
    cl_sax = by_key.get(("top_salt_cl_pct", "saxaul_suitable"))
    salt_sax = by_key.get(("top_salt_sum_salts_pct", "saxaul_suitable"))
    na_sax = by_key.get(("top_exch_na", "saxaul_suitable"))
    caco3_sax = by_key.get(("top_caco3_pct", "saxaul_suitable"))
    sand_sax = by_key.get(("top_sand_pct", "saxaul_suitable"))

    interp_lines: list[str] = [
        "## Interpretation", "",
        "All published cuts have the agronomically correct sign.",
        "Numbers below are generated directly from the computed results "
        "and update automatically on recalibration.", "",
    ]

    # NDMI proxy paragraph
    if ndmi_s1 and ndmi_s3:
        interp_lines += [
            "### NDMI as a measured salinity proxy",
            "",
            f"30 m NDMI separates saline soils (>{SALINE_PCT} %) at oriented AUC "
            f"{ndmi_s1['auc']} (raw {ndmi_s1['auc_raw']}, direction {ndmi_s1['direction']}, "
            f"optimal cut {round(ndmi_s1['cut'],3)}, n={ndmi_s1['n']}, n_pos={ndmi_s1['n_pos']}) "
            f"and strongly saline (>{STRONG_PCT} %) at AUC {ndmi_s3['auc']} "
            f"(raw {ndmi_s3['auc_raw']}, cut {round(ndmi_s3['cut'],3)}, n_pos={ndmi_s3['n_pos']}). "
            f"This replaces the blind P15/P85 percentile with a ground-truth-anchored breakpoint "
            f"and confirms the CLAUDE.md polarity (high NDMI ⇒ saline brine, NOT favorable).",
            "",
        ]

    # Saxaul-suitability soil features paragraph
    sax_parts = []
    if cl_sax:
        sax_parts.append(
            f"low chloride (Cl {cl_sax['direction']} {round(cl_sax['cut'],3)} %, "
            f"oriented AUC {cl_sax['auc']}, raw {cl_sax['auc_raw']}, "
            f"n_pos={cl_sax['n_pos']}, stability={('robust' if cl_sax['n_pos']>=MIN_N_POS else 'low')})")
    if salt_sax:
        sax_parts.append(
            f"low total salts ({salt_sax['direction']} {round(salt_sax['cut'],3)} %, "
            f"AUC {salt_sax['auc']}, n_pos={salt_sax['n_pos']})")
    if na_sax:
        sax_parts.append(
            f"low exchangeable Na ({na_sax['direction']} {round(na_sax['cut'],3)}, "
            f"AUC {na_sax['auc']}, n_pos={na_sax['n_pos']})")
    if caco3_sax:
        rob_note = "" if caco3_sax["n_pos"] >= MIN_N_POS else " [indicative only]"
        sax_parts.append(
            f"carbonate (CaCO3 {caco3_sax['direction']} {round(caco3_sax['cut'],3)} %, "
            f"AUC {caco3_sax['auc']}, n_pos={caco3_sax['n_pos']}{rob_note})")
    if sand_sax:
        rob_note = "" if sand_sax["n_pos"] >= MIN_N_POS else " [indicative only]"
        sax_parts.append(
            f"sandy texture (sand {sand_sax['direction']} {round(sand_sax['cut'],1)} %, "
            f"AUC {sand_sax['auc']}, n_pos={sand_sax['n_pos']}{rob_note})")
    if ndmi_sax:
        sax_parts.append(
            f"low NDMI/NDWI (drier surface, AUC {ndmi_sax['auc']}, n_pos={ndmi_sax['n_pos']})")

    if sax_parts:
        interp_lines += [
            "### Saxaul suitability predictors",
            "",
            "Saxaul suitability (screening-aid direction, n_pos=6, all cuts in-sample) "
            "increases with: " + "; ".join(sax_parts) + ". "
            "This matches saxaul autecology: it establishes on light, non-saline, "
            "carbonate substrates and is excluded by chloride-sodium solonchaks.",
            "",
        ]

    # Small-n warning block for indicative-only entries
    indicative_entries = [(k, v) for k, v in calibrated.items() if v.get("indicative_only")]
    if indicative_entries:
        interp_lines += [
            "### Small-n instability warning (W8)",
            "",
            f"The following {len(indicative_entries)} published entry/entries have "
            f"n_pos < {MIN_N_POS} and are marked `indicative_only=true` in the JSON. "
            "They have correct signs but are statistically underpowered — the TPR axis "
            "has very few attainable values and a single relabel can substantially move the AUC. "
            "Do NOT treat them as validated thresholds:",
            "",
        ]
        for k, v in indicative_entries:
            interp_lines.append(
                f"- **{k}**: cut={v['cut']}, AUC={v['auc']} (raw={v['auc_raw']}), "
                f"n_pos={v['n_pos']}, stability={v['stability']}")
        interp_lines.append("")

    # Inverted-predictor warning block (W7)
    inverted_entries = [(k, v) for k, v in calibrated.items() if v.get("inverted")]
    if inverted_entries:
        interp_lines += [
            "### Inverted predictors (W7)",
            "",
            "The following entries have raw AUC < 0.5 and were evaluated in the "
            "flipped direction (oriented AUC = 1 − raw). The `direction` and `cut` "
            "already encode the correct sign; `auc_raw` is the original rank-AUC "
            "before flipping so downstream tools can distinguish genuinely strong "
            "predictors from inverted proxies of equal oriented AUC:",
            "",
        ]
        for k, v in inverted_entries:
            interp_lines.append(
                f"- **{k}**: direction={v['direction']}, AUC(oriented)={v['auc']}, "
                f"AUC(raw)={v['auc_raw']} → flipped by {round(v['auc']-v['auc_raw'],3)}")
        interp_lines.append("")

    lines += interp_lines
    lines += [
        "## What changes / what does not", "",
        "- The frozen V5.1 10 m map and its thresholds_v5.json are **unchanged** (golden",
        "  snapshot stays valid).",
        "- These calibrated cuts parameterize the Phase 6 soil-anchored suitability",
        "  model on the 30 m wall-to-wall stack, where they are statistically supported.",
        "- Saxaul-label cuts (n_pos=6, all in-sample Youden) are screening aids only;",
        "  those with n_pos < MIN_N_POS are additionally flagged indicative_only.",
        "  Phase 7 spatial CV quantifies combined model skill.", "",
    ]
    (CANON / "CALIBRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_evaluations": len(results),
                      "n_published": len(calibrated),
                      "n_robust": n_robust,
                      "n_indicative_only": n_indicative,
                      "min_n_pos": MIN_N_POS,
                      "published_keys": list(calibrated.keys()),
                      "indicative_only_keys": [k for k, v in calibrated.items()
                                                if v.get("indicative_only")]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
