"""
temp_threshold_analysis.py — Correlation & threshold analysis for V3.0
Loads ground_truth_v2.csv, prints table + Spearman correlations + thresholds.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")
df = pd.read_csv(BASE / "outputs/data/ground_truth_v2.csv")

print("=" * 85)
print("V3.0 THRESHOLD ANALYSIS — 11 Ground Truth Points")
print("=" * 85)

table = df[["pit_code", "S_Point", "pit_number", "Salinity_pct", "SI", "NDMI"]].copy()
table["pit_code"] = table["pit_code"].str.replace("А", "A")
print(f"\n{'pit_code':>8} {'S_Point':>7} {'pit#':>4} {'Salinity%':>10} {'SI':>10} {'NDMI':>10}")
print("-" * 52)
for _, r in table.iterrows():
    print(f"{r['pit_code']:>8} {r['S_Point']:>7} {r['pit_number']:>4} "
          f"{r['Salinity_pct']:>10.2f} {r['SI']:>10.2f} {r['NDMI']:>10.6f}")

print()

# ── Spearman correlations ──────────────────────────────────────────
r_si, p_si = spearmanr(df["Salinity_pct"], df["SI"])
r_ndmi, p_ndmi = spearmanr(df["Salinity_pct"], df["NDMI"])

print(f"Spearman r(Salinity_pct vs SI):    {r_si:>+.4f}  (p={p_si:.4f})"
      f"  {'STRONG' if abs(r_si) > 0.5 else 'WEAK' if abs(r_si) > 0.3 else 'NONE'}")
print(f"Spearman r(Salinity_pct vs NDMI):  {r_ndmi:>+.4f}  (p={p_ndmi:.4f})"
      f"  {'STRONG' if abs(r_ndmi) > 0.5 else 'WEAK' if abs(r_ndmi) > 0.3 else 'NONE'}")

# ── Threshold candidates ────────────────────────────────────────────
dead = df[df["Salinity_pct"] > 5.0]
optimal = df[df["Salinity_pct"] < 2.0]

print(f"\n{'=' * 85}")
print("THRESHOLD CANDIDATES FOR V3.0 RULE-BASED")
print(f"{'=' * 85}")

if len(dead) > 0:
    si_dead_min = dead["SI"].min()
    print(f"\nDEAD ZONE  (Salinity > 5.0%, n={len(dead)}):")
    print(f"  Min SI among dead points: {si_dead_min:.2f}")
    print(f"  Max SI among dead points: {dead['SI'].max():.2f}")
    pt_str = ", ".join(f"{c.replace(chr(1040),'A')}(SI={s:.0f})" for c, s in zip(dead["pit_code"], dead["SI"]))
    print(f"  Points: {pt_str}")
else:
    print("\nDEAD ZONE: No points with Salinity > 5.0%")

if len(optimal) > 0:
    si_opt_max = optimal["SI"].max()
    print(f"\nOPTIMAL ZONE  (Salinity < 2.0%, n={len(optimal)}):")
    print(f"  Max SI among optimal points: {si_opt_max:.2f}")
    print(f"  Min SI among optimal points: {optimal['SI'].min():.2f}")
    pt_str = ", ".join(f"{c.replace(chr(1040),'A')}(SI={s:.0f})" for c, s in zip(optimal["pit_code"], optimal["SI"]))
    print(f"  Points: {pt_str}")
else:
    print("\nOPTIMAL ZONE: No points with Salinity < 2.0%")

# ── Zone ambiguity check ────────────────────────────────────────────
print(f"\n{'=' * 85}")
print("ZONE OVERLAP CHECK")
print(f"{'=' * 85}")
if len(dead) > 0 and len(optimal) > 0:
    overlap_low = max(optimal["SI"].min(), dead["SI"].min())
    overlap_high = min(optimal["SI"].max(), dead["SI"].max())
    if overlap_low <= overlap_high:
        print(f"  WARNING: SI zones OVERLAP in range [{overlap_low:.0f}, {overlap_high:.0f}]")
        print(f"  SI alone cannot separate optimal from dead.")
        print(f"  NDMI needed as secondary discriminator.")
    else:
        print(f"  SI zones CLEARLY SEPARATED: optimal SI ≤ {optimal['SI'].max():.0f} < dead SI ≥ {dead['SI'].min():.0f}")
        print(f"  SI alone is sufficient for binary separation.")

# ── NDMI zone stats ────────────────────────────────────────────────
print(f"\n{'=' * 85}")
print("NDMI ZONE STATS")
print(f"{'=' * 85}")
for label, subset in [("DEAD (>5%)", dead), ("OPTIMAL (<2%)", optimal), ("RISK (2-5%)", df[(df["Salinity_pct"] >= 2.0) & (df["Salinity_pct"] <= 5.0)])]:
    if len(subset) > 0:
        print(f"  {label:.<20} n={len(subset):>2}  NDMI: [{subset['NDMI'].min():.4f}, {subset['NDMI'].max():.4f}]  "
              f"mean={subset['NDMI'].mean():.4f}")
    else:
        print(f"  {label:.<20} n= 0")

print(f"\n{'=' * 85}")
print("RAW DATA (for plotting)")
print(f"{'=' * 85}")
print(f"SI   = {df['SI'].values.tolist()}")
print(f"NDMI = {df['NDMI'].values.tolist()}")
print(f"Salt = {df['Salinity_pct'].values.tolist()}")
print()
