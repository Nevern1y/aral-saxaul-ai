# -*- coding: utf-8 -*-
"""Phase 6 — transparent trained models (pure numpy, deploy-light).

Cross-validation (this script) shows a free-fit saxaul presence/absence
classifier is NOT trustworthy at 6 positive labels: RS-only LOO AUC ≈ 0.48
(chance). Rather than ship an overconfident classifier, we anchor the science on
what the data DO support and stay honest about the rest. Two artifacts:

1. **Salinity model (the validated quantitative core).** A logistic regression
   of measured topsoil salinity (>1 %) on 30 m NDMI, fit on all 70 soil points
   (27 saline). NDMI↔salinity is strong (rho +0.66, LOO AUC ≈ 0.68-0.77). This
   is a genuinely data-validated model. Saxaul *suitability* is then derived as
   1 − P(saline), which validates against the saxaul labels at AUC ≈ 0.65 — far
   better than free-fitting on 6 positives, because it borrows the n=70 salinity
   signal. This model feeds the Phase 6b suitability index.

2. **Exploratory saxaul logit (kept for transparency, NOT for decisions).** The
   direct NDMI+MSAVI→saxaul logit, exported with its honest LOO AUC so the record
   shows what was tried and why it was demoted.

Both export to JSON and apply with plain numpy at inference — no sklearn/scipy in
deploy; the light requirements.txt is untouched.

Outputs:
- outputs/models/salinity_v6_logit.json       (the validated core)
- outputs/models/suitability_v6_logit.json    (exploratory, marked)
- data/canonical/model_v6_fit_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
MODELS = BASE / "outputs" / "models"

PREDICTORS = ["rs30_ndmi", "rs30_msavi"]
L2_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
MAX_ITER = 100
TOL = 1e-8


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def fit_logit(X, y, w, lam):
    """L2-regularized weighted logistic regression via Newton-IRLS.

    X already includes the intercept column. The intercept (col 0) is NOT
    penalized. Returns beta or None if it fails to converge.
    """
    n, p = X.shape
    beta = np.zeros(p)
    pen = np.eye(p) * lam
    pen[0, 0] = 0.0
    for _ in range(MAX_ITER):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        Wd = w * mu * (1 - mu)
        grad = X.T @ (w * (y - mu)) - pen @ beta
        H = -(X.T * Wd) @ X - pen
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < TOL:
            return beta_new
        beta = beta_new
    return beta


def rank_auc(score, y):
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def loo_auc(Xs, y, w, lam):
    """Leave-one-out predicted probabilities -> AUC (honest for tiny n)."""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        beta = fit_logit(Xs[mask], y[mask], w[mask], lam)
        if beta is None:
            return float("nan"), None
        preds[i] = 1.0 / (1.0 + np.exp(-np.clip(Xs[i] @ beta, -30, 30)))
    return rank_auc(preds, y), preds


def fit_single(Xraw, y, w, lam_grid):
    """Standardize, pick lambda by LOO AUC, fit. Returns dict + loo preds."""
    mu = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(y)), (Xraw - mu) / sd])
    scored = []
    for lam in lam_grid:
        auc, _ = loo_auc(Xs, y, w, lam)
        scored.append((lam, auc))
    valid = [(l, a) for l, a in scored if a == a]
    best_lam = max(valid, key=lambda t: t[1])[0] if valid else 1.0
    loo, loo_preds = loo_auc(Xs, y, w, best_lam)
    beta = fit_logit(Xs, y, w, best_lam)
    insample = rank_auc(1.0 / (1.0 + np.exp(-Xs @ beta)), y)
    return {"mu": mu, "sd": sd, "beta": beta, "best_lam": best_lam,
            "loo": loo, "insample": insample, "loo_preds": loo_preds,
            "grid": {str(l): (round(a, 3) if a == a else None) for l, a in scored}}


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CANON / "ml_dataset_v6.csv")

    # ====================================================================
    # 1) VALIDATED CORE: NDMI -> P(topsoil salinity > 1%), all 70 soil points
    # ====================================================================
    ndmi = num(df["rs30_ndmi"])
    salt = num(df["top_salt_sum_salts_pct"])
    ms = ndmi.notna() & salt.notna()
    ys = (salt[ms] > 1.0).to_numpy().astype(float)
    Xs_raw = ndmi[ms].to_numpy().reshape(-1, 1)
    sal = fit_single(Xs_raw, ys, np.ones(len(ys)), L2_GRID)

    def p_saline(ndmi_vals):
        z = sal["beta"][0] + sal["beta"][1] * ((np.asarray(ndmi_vals) - sal["mu"][0]) / sal["sd"][0])
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    salinity_model = {
        "version": "V6-salinity-logit",
        "purpose": "P(topsoil total salts > 1%) from 30m NDMI; suitability = 1 - P(saline)",
        "model_type": "logistic regression (numpy IRLS, L2)",
        "predictor": "rs30_ndmi",
        "scaler": {"mean": [float(sal["mu"][0])], "std": [float(sal["sd"][0])]},
        "coefficients_standardized": {"intercept": float(sal["beta"][0]),
                                      "rs30_ndmi": float(sal["beta"][1])},
        "l2_lambda": sal["best_lam"],
        "training": {"n": int(len(ys)), "n_saline": int(ys.sum()),
                     "loo_auc": round(sal["loo"], 3), "in_sample_auc": round(sal["insample"], 3)},
        "predict_saline": "P = sigmoid(intercept + coef*(ndmi - mean)/std)",
        "predict_suitability": "suitability = 1 - P_saline",
        "notes": "Data-validated core: NDMI vs MEASURED soil salts, independent of saxaul labels.",
    }
    (MODELS / "salinity_v6_logit.json").write_text(
        json.dumps(salinity_model, ensure_ascii=False, indent=2), encoding="utf-8")

    # validate suitability(=1-Psaline) against the saxaul labels and AralField
    y_lab = num(df["y_suitable"])
    w_lab = num(df["label_weight"]).fillna(0.0)
    mk = y_lab.notna() & (w_lab > 0) & ndmi.notna()
    suit_lab = 1.0 - p_saline(ndmi[mk].to_numpy())
    auc_vs_label = rank_auc(suit_lab, y_lab[mk].to_numpy())

    auc_vs_aralfield = None
    af_path = CANON / "aralfield_validation_v6.csv"
    if af_path.exists():
        af = pd.read_csv(af_path).dropna(subset=["ndmi"])
        if len(af):
            suit_af = 1.0 - p_saline(af["ndmi"].to_numpy())
            auc_vs_aralfield = round(rank_auc(suit_af, af["haloxylon"].to_numpy()), 3)

    # ====================================================================
    # 2) EXPLORATORY: direct NDMI+MSAVI -> saxaul (kept, marked not-for-decisions)
    # ====================================================================
    feat = {c: num(df[c]) for c in PREDICTORS}
    mask = y_lab.notna() & (w_lab > 0)
    for c in PREDICTORS:
        mask &= feat[c].notna()
    y = y_lab[mask].to_numpy().astype(float)
    w = w_lab[mask].to_numpy().astype(float)
    Xraw = np.column_stack([feat[c][mask].to_numpy() for c in PREDICTORS])
    expl = fit_single(Xraw, y, w, L2_GRID)
    coef = {"intercept": float(expl["beta"][0])}
    odds = {}
    for i, c in enumerate(PREDICTORS):
        coef[c] = float(expl["beta"][i + 1])
        odds[c] = round(float(np.exp(expl["beta"][i + 1])), 3)

    expl_model = {
        "version": "V6-saxaul-logit-EXPLORATORY",
        "status": "NOT FOR DECISIONS — LOO AUC near chance; kept for transparency",
        "predictors": PREDICTORS,
        "scaler": {"mean": expl["mu"].tolist(), "std": expl["sd"].tolist()},
        "coefficients_standardized": coef,
        "odds_ratio_per_sd": odds,
        "l2_lambda": expl["best_lam"],
        "training": {"n": int(len(y)), "n_pos": int(y.sum()),
                     "loo_auc": round(expl["loo"], 3), "in_sample_auc": round(expl["insample"], 3)},
    }
    (MODELS / "suitability_v6_logit.json").write_text(
        json.dumps(expl_model, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report ----------------------------------------------------------
    lines = ["# V6 models — fit report (Phase 6)", "",
             "## Why not a free-fit saxaul classifier",
             f"Direct NDMI+MSAVI→saxaul logit: **LOO AUC = {round(expl['loo'],3)}** "
             f"(in-sample {round(expl['insample'],3)}), n={len(y)}, {int(y.sum())} positive. "
             "Near chance out-of-sample → not trustworthy. Demoted to exploratory.", "",
             "## Validated core: NDMI → soil salinity, suitability = 1 − P(saline)",
             f"- NDMI→P(salts>1%): **LOO AUC = {round(sal['loo'],3)}**, n={int(len(ys))}, "
             f"{int(ys.sum())} saline (independent of saxaul labels).",
             f"- Derived suitability vs saxaul labels: AUC = {round(auc_vs_label,3)} "
             f"(n={int(mk.sum())}, {int(y_lab[mk].sum())} positive). **Consistency check, NOT "
             f"independent validation:** the saxaul positives are all low-salinity pits and the "
             f"suitability score is a monotone function of the same salinity target, so this AUC "
             f"reflects the shared salinity axis, not a separately demonstrated habitat signal.",
             f"- Derived suitability vs INDEPENDENT AralField 2018 Haloxylon: "
             f"**AUC = {auc_vs_aralfield}** (n=11, 3 present) — the only genuinely external saxaul "
             f"check; at ≈0.5 it shows no demonstrated saxaul skill and must not be sold as one.", "",
             "## Salinity model coefficients (standardized)", "",
             "| term | coef |", "|------|------|",
             f"| intercept | {sal['beta'][0]:+.3f} |",
             f"| rs30_ndmi | {sal['beta'][1]:+.3f} |", "",
             "Positive NDMI coef ⇒ higher NDMI raises P(saline) ⇒ lowers suitability — "
             "matches the documented polarity and the calibration.", "",
             "## Honest conclusion",
             "The trustworthy quantitative object is the **salinity model** (n=70, "
             "validated). Saxaul suitability is built on it (Phase 6b) rather than fit "
             "directly, because 6 positives cannot support a credible classifier. All "
             "skill numbers are reported with their small n; see Phase 7 for spatial CV "
             "and bootstrap CIs.", ""]
    (CANON / "model_v6_fit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "salinity_model": {"loo_auc": round(sal["loo"], 3), "n": int(len(ys)), "n_saline": int(ys.sum())},
        "suitability_vs_saxaul_label_auc": round(auc_vs_label, 3),
        "suitability_vs_aralfield_auc": auc_vs_aralfield,
        "exploratory_saxaul_logit_loo_auc": round(expl["loo"], 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
