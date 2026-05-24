"""
run_inference_v2.py — Apply Ridge model (7 pts, clipped 5%) to full raster
Manual closed-form Ridge (avoids sklearn hang on this system)
Output: salinity_map_v2_alpha.tif (Float32, 0-5% salinity)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from pathlib import Path
import pandas as pd

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
FEATURES = ["SI", "NDMI", "Slope", "TWI", "VH"]
TARGET = "Salinity_pct"
CLIP_MAX = 5.0
ALPHA = 1000.0

print("=" * 60)
print("V2.0 INFERENCE - Ridge on stack_30m (closed-form)")
print("=" * 60, flush=True)

# ── 1. Train Ridge via closed form ────────────────────────────────────
print("\n1. Training Ridge (manual closed-form)...", flush=True)
df = pd.read_csv(BASE / "outputs/data/ground_truth_v2.csv")
X = df[FEATURES].values.astype(np.float64)
y = np.clip(df[TARGET].values, None, CLIP_MAX).astype(np.float64)

# Manual StandardScaler
x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
x_std[x_std == 0] = 1.0
X_s = (X - x_mean) / x_std

# Manual Ridge: beta = (X^T X + alpha*I)^{-1} X^T y
I = np.eye(X_s.shape[1])
beta = np.linalg.inv(X_s.T @ X_s + ALPHA * I) @ X_s.T @ y
intercept = y.mean() - x_mean @ beta / x_std  # adjusted for unscaled prediction

# For unscaled prediction: y = (x - mean)/std @ beta + intercept_scaled
# We can also do: y = x @ (beta / std) + (intercept_scaled - sum(mean * beta / std))
beta_unscaled = beta / x_std
intercept_unscaled = y.mean() - (x_mean * beta_unscaled).sum()

print(f"  Coefficients (unscaled): {dict(zip(FEATURES, beta_unscaled))}", flush=True)
print(f"  Intercept: {intercept_unscaled:.4f}", flush=True)
print(f"  R2 on train: {1 - ((X @ beta_unscaled + intercept_unscaled - y)**2).sum() / ((y - y.mean())**2).sum():.4f}", flush=True)

# ── 2. Block-wise raster prediction ───────────────────────────────────
print("\n2. Reading raster...", flush=True)
src_path = BASE / "outputs/data/feature_stack_30m.vrt"
out_path = BASE / "outputs/data/salinity_map_v2_alpha.tif"

src = rasterio.open(src_path)
profile = src.profile.copy()
profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=-9999.0)

total_pixels = src.width * src.height
print(f"  Size: {src.width}x{src.height} = {total_pixels:,} pixels", flush=True)

windows = list(src.block_windows(1))
print(f"  Processing {len(windows)} blocks...", flush=True)

# Raster band index (0-based): 0=NDMI, 1=MSAVI, 2=SI, 3=NDWI, 4=Slope, 5=TWI, 6=VH
# FEATURES order: [SI, NDMI, Slope, TWI, VH]
FEAT_IDX = [2, 0, 4, 5, 6]  # block indices for [SI, NDMI, Slope, TWI, VH]

with rasterio.open(out_path, "w", **profile) as dst:
    for idx, (_, window) in enumerate(windows):
        if (idx + 1) % 200 == 0 or idx == 0:
            pct = (idx + 1) / len(windows) * 100
            print(f"  [{idx+1:5d}/{len(windows)}] {pct:.0f}%", flush=True)

        try:
            block = src.read(window=window)
            rows, cols = block.shape[1], block.shape[2]
        except Exception:
            rows, cols = window.height, window.width
            dst.write(np.full((rows, cols), -9999.0, dtype=np.float32), 1, window=window)
            continue

        # Assemble features in FEATURES order
        feat_stack = np.column_stack([block[bi].ravel() for bi in FEAT_IDX])

        valid = ~np.any(np.isnan(feat_stack) | np.isinf(feat_stack), axis=1)

        pred = np.full(feat_stack.shape[0], -9999.0, dtype=np.float32)
        if valid.sum() > 0:
            pred_v = feat_stack[valid] @ beta_unscaled + intercept_unscaled
            pred_v = np.clip(pred_v, 0, CLIP_MAX)
            pred[valid] = pred_v.astype(np.float32)

        dst.write(pred.reshape(rows, cols), 1, window=window)

src.close()
print(f"  [{len(windows):5d}/{len(windows)}] 100% - Done!", flush=True)

# ── 3. Statistics ─────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("OUTPUT STATS", flush=True)
print(f"{'='*60}", flush=True)
print(f"Saved to: {out_path}", flush=True)

with rasterio.open(out_path) as res:
    data = res.read(1)
    valid = data[data > -9990]
    print(f"  Valid pixels: {len(valid):,} / {total_pixels:,} ({100*len(valid)/total_pixels:.1f}%)", flush=True)
    print(f"  Min salinity: {valid.min():.4f}%", flush=True)
    print(f"  Max salinity: {valid.max():.4f}%", flush=True)
    print(f"  Mean salinity: {valid.mean():.4f}%", flush=True)
    print(f"  Std salinity: {valid.std():.4f}%", flush=True)
    print(f"  Median salinity: {np.median(valid):.4f}%", flush=True)
    print(f"  Pixels >{CLIP_MAX}%: {(valid>=CLIP_MAX).sum():,} ({(valid>=CLIP_MAX).sum()/len(valid)*100:.1f}%)", flush=True)
    print(f"  Pixels <0.5% (low): {(valid<0.5).sum():,} ({(valid<0.5).sum()/len(valid)*100:.1f}%)", flush=True)
    print(f"  Pixels 0.5-3% (moderate): {((valid>=0.5)&(valid<3.0)).sum():,} ({((valid>=0.5)&(valid<3.0)).sum()/len(valid)*100:.1f}%)", flush=True)
    print(f"\n  CRS: {res.crs}", flush=True)
    print(f"  Bounds: {res.bounds}", flush=True)

print(f"\nDone. V2.0 Alpha salinity map ready.", flush=True)
