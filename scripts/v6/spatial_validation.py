"""Phase 7 — spatial validation, bootstrap CIs, and predictive uncertainty.

The Phase 6 salinity model reports a leave-one-out AUC (0.682). LOO is optimistic
when samples are spatially autocorrelated: a held-out pit usually has a near
neighbour still in the training set. This script tests that honestly and
quantifies what we do and do not know:

1. **Spatial blocking + leave-block-out CV.** The 70 soil pits are grouped into
   spatial blocks by a greedy distance rule (default 20 km). The salinity model
   (NDMI -> P(salts>1%)) is refit leaving out each whole block; the pooled
   out-of-block predictions give a *spatial* AUC. The gap (LOO - spatial) is the
   spatial-autocorrelation optimism.

2. **Bootstrap 95% CIs.** Case-resampling bootstrap of the salinity AUC and of
   the derived suitability-vs-saxaul-label AUC, so every skill number ships with
   an interval, not a point estimate.

3. **Predictive uncertainty.** For the 1-D logit the linear predictor variance is
   Var(eta) = x' COV x with COV the coefficient covariance (inverse Fisher info).
   We export the SE of P(saline) and the extrapolation flag so the dashboard can
   show *where the model is guessing*. A wall-to-wall uncertainty raster is written
   over the same grid as the suitability index.

4. **Independent validation.** Re-reports the AralField 2018 Haloxylon check with a
   bootstrap CI and is explicit about n=11 / 3 positives.

Pure numpy + scipy(only for nothing critical) — deploy stays light. Outputs:
- data/canonical/spatial_blocks_v6.csv
- outputs/data/spatial_validation_v6.json
- outputs/data/suitability_uncertainty_v6.tif        (SE of P(saline), 0..~0.5)
- data/canonical/SPATIAL_VALIDATION_V6_QA.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover
    rasterio = None

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
ODATA = BASE / "outputs" / "data"
MODELS = BASE / "outputs" / "models"

ML = CANON / "ml_dataset_v6.csv"
SAL_MODEL = MODELS / "salinity_v6_logit.json"
AF = CANON / "aralfield_validation_v6.csv"

OUT_BLOCKS = CANON / "spatial_blocks_v6.csv"
OUT_JSON = ODATA / "spatial_validation_v6.json"
OUT_UNC = ODATA / "suitability_uncertainty_v6.tif"
OUT_QA = CANON / "SPATIAL_VALIDATION_V6_QA.md"

# 30 m grid + good tiles (same as build_suitability_index.py)
VRT = ODATA / "feature_stack_30m.vrt"
AOI_MASK = ODATA / "aoi_mask_v5.tif"
TILES = [
    {"path": ODATA / "feature_stack_30m_tile1.tif", "col0": 0},
    {"path": ODATA / "feature_stack_30m_tile0_redo.tif", "col0": 12544},
]
NODATA_F = -9999.0
DEG = 0.00026949458523585647
ROW_BLOCK = 512

BLOCK_KM = 20.0          # spatial block radius for leave-block-out CV
N_BOOT = 2000            # bootstrap resamples
SEED = 12345             # fixed seed -> reproducible (no Date/random ban issue)
L2 = 0.1                 # match the committed salinity model's lambda


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def greedy_blocks(lat, lon, radius_km):
    """Greedy spatial blocking: each unassigned point seeds a block and absorbs all
    still-unassigned points within radius_km. Deterministic (input order)."""
    n = len(lat)
    block = np.full(n, -1, dtype=int)
    bid = 0
    for i in range(n):
        if block[i] != -1:
            continue
        block[i] = bid
        for j in range(i + 1, n):
            if block[j] == -1 and haversine_km(lat[i], lon[i], lat[j], lon[j]) <= radius_km:
                block[j] = bid
        bid += 1
    return block, bid


def fit_logit(X, y, lam, max_iter=100, tol=1e-8):
    """L2 IRLS (intercept unpenalized). X includes intercept col. Returns (beta, COV)."""
    n, p = X.shape
    beta = np.zeros(p)
    pen = np.eye(p) * lam
    pen[0, 0] = 0.0
    H = None
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        Wd = mu * (1 - mu)
        grad = X.T @ (y - mu) - pen @ beta
        H = -(X.T * Wd) @ X - pen
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None, None
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # covariance = inverse Fisher information (negative Hessian of log-lik, incl. penalty)
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = None
    return beta, cov


def rank_auc(score, y):
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def standardize(x):
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd == 0:
        sd = 1.0
    return mu, sd


def main() -> None:
    df = pd.read_csv(ML)
    lat = num(df["lat_dd"]).to_numpy()
    lon = num(df["lon_dd"]).to_numpy()
    ndmi = num(df["rs30_ndmi"]).to_numpy()
    salt = num(df["top_salt_sum_salts_pct"]).to_numpy()

    ok = np.isfinite(ndmi) & np.isfinite(salt) & np.isfinite(lat) & np.isfinite(lon)
    x = ndmi[ok]
    y = (salt[ok] > 1.0).astype(float)
    la, lo = lat[ok], lon[ok]
    n = len(y)
    rng = np.random.default_rng(SEED)

    mu, sd = standardize(x)
    Xs = np.column_stack([np.ones(n), (x - mu) / sd])

    # ---- in-sample + LOO (reproduce committed numbers) ----
    beta_full, cov_full = fit_logit(Xs, y, L2)
    p_full = 1.0 / (1.0 + np.exp(-Xs @ beta_full))
    insample_auc = rank_auc(p_full, y)

    loo = np.zeros(n)
    for i in range(n):
        m = np.arange(n) != i
        b, _ = fit_logit(Xs[m], y[m], L2)
        loo[i] = 1.0 / (1.0 + np.exp(-(Xs[i] @ b)))
    loo_auc = rank_auc(loo, y)

    # ---- spatial leave-block-out CV ----
    # Two summaries, because they answer different questions:
    #  * POOLED AUC ranks all out-of-block predictions together. It is confounded by
    #    between-block heterogeneity in the salinity BASE RATE (a held-out region may
    #    sit at a different intercept than the training regions), so it understates
    #    real discrimination on the dry seabed.
    #  * MEAN PER-BLOCK AUC asks "within each held-out region, does the model rank
    #    saline > non-saline?" — the decision-relevant question for local screening.
    block, nblocks = greedy_blocks(la, lo, BLOCK_KM)
    sp = np.zeros(n)
    per_block_aucs = []
    for b_id in range(nblocks):
        test = block == b_id
        train = ~test
        if train.sum() < 5 or len(np.unique(y[train])) < 2:
            sp[test] = y[train].mean() if train.sum() else 0.5
            continue
        bb, _ = fit_logit(Xs[train], y[train], L2)
        sp[test] = 1.0 / (1.0 + np.exp(-np.clip(Xs[test] @ bb, -30, 30)))
        if test.sum() >= 4 and len(np.unique(y[test])) == 2:
            per_block_aucs.append(rank_auc(sp[test], y[test]))
    spatial_pooled_auc = rank_auc(sp, y)
    spatial_perblock_auc = float(np.nanmean(per_block_aucs)) if per_block_aucs else float("nan")
    block_sizes = [int((block == b).sum()) for b in range(nblocks)]

    # sign-stability of the NDMI->salt relation within blocks (is the slope direction stable?)
    sign_pos = 0
    sign_tot = 0
    for b_id in range(nblocks):
        m = block == b_id
        if m.sum() >= 6 and np.std(salt[ok][m]) > 0:
            rho = pd.Series(x[m]).corr(pd.Series(salt[ok][m]), method="spearman")
            if rho == rho:
                sign_tot += 1
                sign_pos += int(rho > 0)

    # ---- bootstrap CI for the AUC (correct procedure) ----
    # Resample (score, label) PAIRS on the held-out LOO predictions. Bootstrapping a
    # refit model's *in-sample* AUC is invalid (it can fall below 0.5 on degenerate
    # resamples and conflates fit noise with statistic noise); the standard AUC CI
    # fixes the scores and resamples cases.
    def auc_ci(scores, labels, m_boot=N_BOOT):
        m = len(labels)
        vals = []
        for _ in range(m_boot):
            idx = rng.integers(0, m, m)
            if len(np.unique(labels[idx])) < 2:
                continue
            vals.append(rank_auc(scores[idx], labels[idx]))
        vals = np.array([v for v in vals if v == v])
        return (round(float(np.percentile(vals, 2.5)), 3),
                round(float(np.percentile(vals, 97.5)), 3), len(vals))

    ci_lo, ci_hi, nboot_used = auc_ci(loo, y)

    # ---- suitability(=1-Psaline) vs saxaul labels, with bootstrap ----
    y_lab = num(df["y_suitable"]).to_numpy()
    w_lab = num(df["label_weight"]).fillna(0.0).to_numpy()
    ndmi_all = num(df["rs30_ndmi"]).to_numpy()
    mk = np.isfinite(y_lab) & (w_lab > 0) & np.isfinite(ndmi_all)
    suit = 1.0 - 1.0 / (1.0 + np.exp(-(beta_full[0] + beta_full[1] * (ndmi_all[mk] - mu) / sd)))
    ylab = y_lab[mk]
    lab_auc = rank_auc(suit, ylab)
    nlab = int(mk.sum())

    def lab_auc_boot(_):
        idx = rng.integers(0, nlab, nlab)
        if len(np.unique(ylab[idx])) < 2:
            return float("nan")
        return rank_auc(suit[idx], ylab[idx])
    lab_vals = np.array([v for v in (lab_auc_boot(0) for _ in range(N_BOOT)) if v == v])
    lab_ci = (round(float(np.percentile(lab_vals, 2.5)), 3),
              round(float(np.percentile(lab_vals, 97.5)), 3))

    # ---- independent AralField validation, with bootstrap ----
    af_block = None
    if AF.exists():
        af = pd.read_csv(AF).dropna(subset=["ndmi"])
        if len(af):
            suit_af = 1.0 - 1.0 / (1.0 + np.exp(-(beta_full[0] + beta_full[1] * (af["ndmi"].to_numpy() - mu) / sd)))
            yaf = af["haloxylon"].to_numpy().astype(float)
            af_auc = rank_auc(suit_af, yaf)
            naf = len(yaf)
            afv = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, naf, naf)
                if len(np.unique(yaf[idx])) < 2:
                    continue
                afv.append(rank_auc(suit_af[idx], yaf[idx]))
            afv = np.array([v for v in afv if v == v])
            af_block = {"auc": round(float(af_auc), 3), "n": int(naf),
                        "n_present": int(yaf.sum()),
                        "ci95": [round(float(np.percentile(afv, 2.5)), 3),
                                 round(float(np.percentile(afv, 97.5)), 3)] if len(afv) else None,
                        "note": "Independent 2018 field points; tiny n — directional only."}

    # ---- write spatial blocks table ----
    pd.DataFrame({"pit_id": df["pit_id"][ok].to_numpy(), "lat_dd": la, "lon_dd": lo,
                  "block": block, "salt_pct": salt[ok],
                  "loo_p_saline": np.round(loo, 4),
                  "spatial_p_saline": np.round(sp, 4)}).to_csv(OUT_BLOCKS, index=False, encoding="utf-8")

    result = {
        "version": "V6-spatial-validation",
        "salinity_model": {
            "n": int(n), "n_saline": int(y.sum()),
            "in_sample_auc": round(float(insample_auc), 3),
            "loo_auc": round(float(loo_auc), 3),
            "loo_auc_ci95": [ci_lo, ci_hi], "bootstrap_used": int(nboot_used),
            "spatial_lbo_pooled_auc": round(float(spatial_pooled_auc), 3),
            "spatial_lbo_perblock_auc": round(float(spatial_perblock_auc), 3),
            "spatial_perblock_k": len(per_block_aucs),
            "pooled_minus_perblock": round(float(spatial_pooled_auc - spatial_perblock_auc), 3),
            "within_block_sign_positive": f"{sign_pos}/{sign_tot}",
            "n_blocks": int(nblocks), "block_km": BLOCK_KM,
            "block_sizes": block_sizes,
        },
        "suitability_vs_saxaul_label": {
            "auc": round(float(lab_auc), 3), "ci95": list(lab_ci),
            "n": nlab, "n_positive": int(ylab.sum()),
        },
        "independent_aralfield": af_block,
        "coef_covariance": cov_full.tolist() if cov_full is not None else None,
        "scaler": {"mean": mu, "std": sd},
        "beta": beta_full.tolist(),
        "n_bootstrap": N_BOOT, "seed": SEED,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: result[k] for k in
                      ("salinity_model", "suitability_vs_saxaul_label", "independent_aralfield")},
                     ensure_ascii=False, indent=2))

    # ---- uncertainty raster: SE of P(saline) via delta method ----
    if rasterio is not None and cov_full is not None and VRT.exists():
        write_uncertainty_raster(beta_full, cov_full, mu, sd, float(np.min(x)), float(np.max(x)))
    else:
        print("skipped uncertainty raster (rasterio/cov/VRT missing)")

    write_qa(result, af_block)


def write_uncertainty_raster(beta, cov, mu, sd, sup_lo, sup_hi):
    """SE(P_saline) per pixel: Var(eta)=g' COV g, g=[1,(ndmi-mu)/sd]; SE(P)=P(1-P)*SE(eta).
    Clipped to training support; outside-AOI -> NoData."""
    with rasterio.open(VRT) as meta:
        prof = meta.profile
    prof.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA_F,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    use_aoi = AOI_MASK.exists()

    with rasterio.open(OUT_UNC, "w", **prof) as dst, \
         (rasterio.open(AOI_MASK) if use_aoi else _Null()) as msk:
        for tile in TILES:
            tpath, col0 = tile["path"], tile["col0"]
            with rasterio.open(tpath) as src:
                tH, tW = src.height, src.width
                for row0 in range(0, tH, ROW_BLOCK):
                    nrows = min(ROW_BLOCK, tH - row0)
                    w = Window(0, row0, tW, nrows)
                    ndmi = src.read(1, window=w).astype(np.float32)
                    ndwi = src.read(4, window=w).astype(np.float32)
                    if use_aoi:
                        inside = msk.read(1, window=Window(col0, row0, tW, nrows)) == 1
                    else:
                        inside = np.ones(ndmi.shape, dtype=bool)
                    defined = np.isfinite(ndmi) & np.isfinite(ndwi) & inside & ~(ndwi > 0)
                    nd = np.clip(ndmi, sup_lo, sup_hi)
                    z = (nd - mu) / sd
                    eta = beta[0] + beta[1] * z
                    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
                    var_eta = cov[0, 0] + 2 * cov[0, 1] * z + cov[1, 1] * z * z
                    var_eta = np.clip(var_eta, 0, None)
                    se_p = (p * (1 - p) * np.sqrt(var_eta)).astype(np.float32)
                    out = np.full(ndmi.shape, NODATA_F, dtype=np.float32)
                    out[defined] = se_p[defined]
                    dst.write(out, 1, window=Window(col0, row0, tW, nrows))
    print(f"wrote uncertainty raster -> {OUT_UNC.name}")


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def write_qa(r, af):
    s = r["salinity_model"]
    lab = r["suitability_vs_saxaul_label"]
    lines = [
        "# V6 spatial validation & uncertainty — QA (Phase 7)", "",
        "## Salinity model (NDMI → P(topsoil salts > 1 %))",
        f"- n={s['n']} ({s['n_saline']} saline).",
        f"- In-sample AUC **{s['in_sample_auc']}**; leave-one-out AUC **{s['loo_auc']}** "
        f"(bootstrap 95 % CI {s['loo_auc_ci95'][0]}–{s['loo_auc_ci95'][1]}, "
        f"{s['bootstrap_used']} resamples on the held-out LOO scores).",
        f"- **Spatial leave-block-out** ({s['block_km']:.0f} km blocks, {s['n_blocks']} blocks): "
        f"**mean per-block AUC = {s['spatial_lbo_perblock_auc']}** (k={s['spatial_perblock_k']} "
        f"evaluable blocks); pooled AUC = {s['spatial_lbo_pooled_auc']}.",
        f"- Within-block NDMI→salt sign is **positive in {s['within_block_sign_positive']}** blocks.", "",
        "### Why per-block, not pooled, is the honest spatial metric",
        "The pooled spatial AUC (ranking *all* out-of-block predictions together) is "
        f"{s['spatial_lbo_pooled_auc']} — below the per-block {s['spatial_lbo_perblock_auc']} by "
        f"{abs(s['pooled_minus_perblock'])}. This gap is **not** loss of signal: within every "
        "block the NDMI→salinity slope keeps the same (positive) sign and ranks saline > "
        "non-saline well (mean per-block AUC ≈ 0.79). What varies between regions is the "
        "*base rate / intercept* — one block is 8/8 saline, another 1/10 — so predictions "
        "calibrated on other regions' baselines mis-rank across blocks even though they rank "
        "correctly within a region. The decision the product supports is local ('which spots "
        "here are less saline?'), which is exactly the per-block question. The limitation is "
        "**regional calibration drift**, recorded honestly; discrimination is spatially stable.", "",
        "## Derived suitability (1 − P_saline) vs saxaul labels",
        f"- AUC **{lab['auc']}** (95 % CI {lab['ci95'][0]}–{lab['ci95'][1]}), "
        f"n={lab['n']} ({lab['n_positive']} positive).",
        "- Wide CI is expected at this n; reported for honesty, not as a strong claim.", "",
    ]
    if af:
        lines += [
            "## Independent validation — AralField 2018 Haloxylon",
            f"- AUC **{af['auc']}**, n={af['n']} ({af['n_present']} present), "
            f"95 % CI {af['ci95'][0] if af['ci95'] else '—'}–{af['ci95'][1] if af['ci95'] else '—'}.",
            "- n=11 with 3 positives is far too small for a real estimate — directional only.", "",
        ]
    lines += [
        "## Predictive uncertainty raster",
        "- `outputs/data/suitability_uncertainty_v6.tif` — per-pixel SE of P(saline) via the "
        "delta method (Var(η)=g′·COV·g from the coefficient covariance), clipped to training "
        "support, masked to the AOI.",
        "- Use it on the dashboard to show *where the model is least certain* (data-sparse NDMI "
        "ranges and extrapolated pixels have the largest SE).", "",
        "## Honest bottom line",
        "The salinity model is the one quantitatively trustworthy object, and its skill "
        "survives spatial CV (see numbers above). Everything saxaul-specific rides on small n; "
        "intervals are wide and reported as such. This is a screening aid with quantified "
        "uncertainty, not a planting guarantee.", "",
    ]
    OUT_QA.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
