"""
train_v2_light.py — LOOCV Ridge/Lasso regression on 7 golden points
Target: Salinity_pct (TOP LAYER ONLY), Predictors: satellite features
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")

df = pd.read_csv(BASE / "outputs/data/ground_truth_v2.csv")
print("=" * 60)
print("V2.0 LIGHT REGRESSION -- LOOCV Ridge / Lasso")
print("Target: TOP LAYER Salinity_pct (CLIPPED at 5%)")
print("=" * 60)
print(f"\nLoaded {len(df)} points")

FEATURES = ["SI", "NDMI", "Slope", "TWI", "VH"]
TARGET = "Salinity_pct"
CLIP_MAX = 5.0

X = df[FEATURES].values
y = np.clip(df[TARGET].values, a_min=None, a_max=CLIP_MAX)

print(f"Target raw range: {df[TARGET].min():.2f} – {df[TARGET].max():.2f}%")
print(f"After clip at {CLIP_MAX}%: {y.min():.2f} – {y.max():.2f}%  (mean={y.mean():.2f}%)")
print(f"Clipped {int((df[TARGET] > CLIP_MAX).sum())} points down to {CLIP_MAX}%")

loo = LeaveOneOut()
ridge_preds = np.full(len(df), np.nan)
lasso_preds = np.full(len(df), np.nan)
ridge_models, lasso_models = [], []

for train_idx, test_idx in loo.split(X):
    X_t, X_v = X[train_idx], X[test_idx]
    y_t = y[train_idx]

    scaler = StandardScaler()
    X_ts = scaler.fit_transform(X_t)
    X_vs = scaler.transform(X_v)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), fit_intercept=True)
    ridge.fit(X_ts, y_t)
    ridge_preds[test_idx] = ridge.predict(X_vs)
    ridge_models.append((ridge, scaler))

    lasso = LassoCV(alphas=np.logspace(-3, 3, 50), fit_intercept=True, max_iter=10000, cv=3)
    lasso.fit(X_ts, y_t)
    lasso_preds[test_idx] = lasso.predict(X_vs)
    lasso_models.append((lasso, scaler))

print(f"\n{'Metric':<20} {'Ridge':>12} {'Lasso':>12}")
print("-" * 45)
for name, r, l in [
    ("LOOCV RMSE (%):", np.sqrt(mean_squared_error(y, ridge_preds)), np.sqrt(mean_squared_error(y, lasso_preds))),
    ("LOOCV MAE (%):", mean_absolute_error(y, ridge_preds), mean_absolute_error(y, lasso_preds)),
]:
    print(f"{name:<20} {r:>12.4f} {l:>12.4f}")

r2_r = 1 - np.sum((y - ridge_preds)**2) / np.sum((y - y.mean())**2)
r2_l = 1 - np.sum((y - lasso_preds)**2) / np.sum((y - y.mean())**2)
print(f"{'LOOCV R2:':<20} {r2_r:>12.4f} {r2_l:>12.4f}")

print(f"\n{'Point':>8} {'True%':>8} {'Ridge%':>8} {'Lasso%':>8} {'Err_R':>10} {'Err_L':>10}")
print("-" * 55)
for i in range(len(df)):
    print(f"{df['S_Point'].iloc[i]:>8} {y[i]:>8.2f} {ridge_preds[i]:>8.2f} {lasso_preds[i]:>8.2f} "
          f"{(y[i]-ridge_preds[i]):>10.2f} {(y[i]-lasso_preds[i]):>10.2f}")

# Feature coefficients (Ridge)
print(f"\n{'Feature':<10} {'Ridge_w':>10} {'Ridge_stderr':>12}")
print("-" * 35)
ridge_coefs = np.array([m[0].coef_ for m in ridge_models])
avg_c, std_c = ridge_coefs.mean(axis=0), ridge_coefs.std(axis=0)
for i, f in enumerate(FEATURES):
    print(f"{f:<10} {avg_c[i]:>+10.4f} {std_c[i]:>12.4f}")
print("  (standardized features: DeltaSalinity% / 1sigma)")

# Lasso sparsity
lasso_coefs = np.array([m[0].coef_ for m in lasso_models])
nonzero = (np.abs(lasso_coefs) > 1e-6).sum(axis=0)
print(f"\nLasso non-zero count (out of {len(lasso_models)} folds):")
for i, f in enumerate(FEATURES):
    print(f"  {f:<10}: {nonzero[i]:.0f}/{len(lasso_models)}")

# Final model
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
y_clip = np.clip(df[TARGET].values, a_min=None, a_max=CLIP_MAX)
final = RidgeCV(alphas=np.logspace(-3, 3, 50), fit_intercept=True)
final.fit(X_s, y_clip)
print(f"\nFinal Ridge alpha: {final.alpha_:.4f}")
print("Coefficients (standardized):", ", ".join(f"{c:+.4f}" for c in final.coef_))
for i, f in enumerate(FEATURES):
    print(f"  {f:<10}: {final.coef_[i]:+.4f}")
print(f"  Intercept: {final.intercept_:.4f}")

# Correlations
y_orig = df[TARGET].values
r_sp, p_sp = spearmanr(df["SI"], y_orig)
r_p, p_p = pearsonr(df["SI"], y_orig)
print(f"\nSI vs Salinity_pct (raw):  Spearman r={r_sp:.4f} (p={p_sp:.4f}),  Pearson r={r_p:.4f} (p={p_p:.4f})")
r_spc, p_spc = spearmanr(df["SI"], y_clip)
r_pc, p_pc = pearsonr(df["SI"], y_clip)
print(f"SI vs Salinity_pct (clipped):  Spearman r={r_spc:.4f} (p={p_spc:.4f}),  Pearson r={r_pc:.4f} (p={p_pc:.4f})")

print(f"\nDone. Clipped at {CLIP_MAX}%.")
