"""
validate_ground_truth.py — Golden Test: compare satellite predictions vs field EC
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from scipy.stats import spearmanr

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")

# ── 1. Load AralField ──────────────────────────────────────────────────
df = pd.read_csv(
    BASE / "AralField(Sheet1).csv",
    encoding="utf-8-sig", sep=";", header=None, skiprows=3, low_memory=False,
)

# Rename columns for clarity
df.columns = [f"col{i}" for i in range(df.shape[1])]

# Extract point ID, coordinates, EC (col24=East, col25=South, col26=West, col27=North)
points = df["col0"].tolist()
e_raw = df["col11"].astype(str).str.replace(",", ".").astype(float)
n_raw = df["col12"].astype(str).str.replace(",", ".").astype(float)

# EC: cols 24-27 (East/South/West/North) - comma → dot conversion
ec_cols = ["col24", "col25", "col26", "col27"]
for c in ec_cols:
    df[c] = df[c].astype(str).str.replace(",", ".")
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["mean_EC"] = df[ec_cols].mean(axis=1, skipna=True)
df["lon"] = e_raw
df["lat"] = n_raw

print(f"Loaded {len(df)} field points")
for i in range(len(df)):
    print(f"  {df['col0'].iloc[i]}: lon={df['lon'].iloc[i]:.6f}, lat={df['lat'].iloc[i]:.6f}, mean_EC={df['mean_EC'].iloc[i]:.4f}")

# ── 2. Sample satellite rasters ────────────────────────────────────────
coords = list(zip(df["lon"], df["lat"]))

suit_path = BASE / "outputs/data/suitability_full.tif"
fs_path = BASE / "outputs/data/feature_stack_30m.vrt"  # VRT covers full AOI

# Try VRT first, fallback to tile1, then tile0
for raster_path in [fs_path]:
    if raster_path.exists():
        rp = raster_path
        break
# Also check tiles
tile1 = BASE / "outputs/data/feature_stack_30m_tile1.tif"
tile0 = BASE / "outputs/data/feature_stack_30m_tile0_redo.tif"

print(f"\nSampling rasters...")
print(f"  Suitability: {suit_path}")
print(f"  Feature stack: {fs_path} (exists={fs_path.exists()})")

# Sample suitability
prob_vals = []
with rasterio.open(suit_path) as src:
    for x, y in coords:
        row, col = src.index(x, y)
        val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
        prob_vals.append(val)
df["Prob"] = prob_vals

# Sample feature stack - Band 3 is SI
# Try VRT first, then tiles
si_vals = []
all_feat = {f: [] for f in ["NDMI", "MSAVI", "SI", "NDWI", "Slope", "TWI", "VH"]}

for raster_path in [fs_path, tile1, tile0]:
    if raster_path.exists():
        try:
            with rasterio.open(raster_path) as src:
                for x, y in coords:
                    try:
                        row, col = src.index(x, y)
                        window = ((row, row+1), (col, col+1))
                        bands = {i: src.read(i, window=window)[0, 0] for i in range(1, 8)}
                        for idx, name in enumerate(["NDMI", "MSAVI", "SI", "NDWI", "Slope", "TWI", "VH"], 1):
                            all_feat[name].append(bands[idx])
                    except Exception:
                        for name in all_feat:
                            all_feat[name].append(np.nan)
            print(f"  Sampled from: {raster_path.name}")
            break
        except Exception:
            continue

for name in all_feat:
    df[name] = all_feat[name]

# ── 3. Results ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print("GOLDEN TEST RESULTS — Field EC vs Satellite Predictions")
print("="*80)

result = df[["col0", "lon", "lat", "mean_EC", "Prob", "SI"]].copy()
result.columns = ["Point", "Lon", "Lat", "Mean_EC", "Prob", "SI"]
result.to_csv(BASE / "outputs/data/golden_test_results.csv", index=False)

print(result.to_string(index=False))

# Correlation
valid_pred = result.dropna(subset=["Mean_EC", "Prob"])
if len(valid_pred) > 2:
    r_prob, p_prob = spearmanr(valid_pred["Mean_EC"], valid_pred["Prob"])
    print(f"\nSpearman correlation: Mean_EC vs Prob")
    print(f"  r = {r_prob:.4f}, p = {p_prob:.6f}, n = {len(valid_pred)}")
else:
    print("\nNot enough valid pairs for EC vs Prob correlation")

valid_si = result.dropna(subset=["Mean_EC", "SI"])
if len(valid_si) > 2:
    r_si, p_si = spearmanr(valid_si["Mean_EC"], valid_si["SI"])
    print(f"\nSpearman correlation: Mean_EC vs SI")
    print(f"  r = {r_si:.4f}, p = {p_si:.6f}, n = {len(valid_si)}")
else:
    print("\nNot enough valid pairs for EC vs SI correlation")

# Pearson too
from scipy.stats import pearsonr
if len(valid_pred) > 2:
    r_p, p_p = pearsonr(valid_pred["Mean_EC"], valid_pred["Prob"])
    print(f"\nPearson correlation: Mean_EC vs Prob")
    print(f"  r = {r_p:.4f}, p = {p_p:.6f}")

if len(valid_si) > 2:
    r_sp, p_sp = pearsonr(valid_si["Mean_EC"], valid_si["SI"])
    print(f"\nPearson correlation: Mean_EC vs SI")
    print(f"  r = {r_sp:.4f}, p = {p_sp:.6f}")

print(f"\nResults saved to: {BASE / 'outputs/data/golden_test_results.csv'}")
