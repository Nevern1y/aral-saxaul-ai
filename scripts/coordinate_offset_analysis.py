"""
coordinate_offset_analysis.py — Calculate spatial offset vector
between ODT DMM (System A) and AralField DD (System B) coordinates.

Outputs statistical diagnostics and a corrected golden_mapping dict.
"""

import sys
import numpy as np
from collections import OrderedDict

sys.stdout.reconfigure(encoding='utf-8')

# ── System A: ODT DMM → DD (hardcoded from build_ground_truth.py GOLDEN dict) ──
odt_points = OrderedDict({
    '01/20А': (59.775683, 44.704500),
    '02/20А': (59.793467, 44.711050),
    '03/20А': (59.813067, 44.729433),
    '04/20А': (59.863583, 44.755567),
    '05/20А': (59.868800, 44.762467),
    '06/20А': (59.892250, 44.763633),
    '07/20А': (60.065867, 44.820933),
})

# ── System B: AralField DD (from golden_test_results.csv, matched by pit_code) ──
aralfield_points = OrderedDict({
    '01/20А': (60.893604, 45.588539),   # S124
    '02/20А': (60.912059, 45.608637),   # S125
    '03/20А': (60.928608, 45.624813),   # S126
    '04/20А': (60.983600, 45.648156),   # S127
    '05/20А': (60.987777, 45.656316),   # S128
    '06/20А': (61.018802, 45.657462),   # S129
    '07/20А': (61.213133, 45.725247),   # S130
})

# ── Fallback: AralField DD for the 4 points without ODT coordinates ──
fallback_points = OrderedDict({
    '08/20А': (61.221228, 45.754821),   # S131
    '09/20А': (61.257944, 45.815325),   # S132
    '10/20А': (61.267256, 45.847397),   # S133
    '11/20А': (61.268779, 45.945191),   # S134
})

# ── 1. Compute deltas for the 7 paired points ──────────────────────────
keys = list(odt_points.keys())
deltas_lon = np.array([aralfield_points[k][0] - odt_points[k][0] for k in keys])
deltas_lat = np.array([aralfield_points[k][1] - odt_points[k][1] for k in keys])

# ── 2. Statistics ──────────────────────────────────────────────────────
mean_lon = deltas_lon.mean()
mean_lat = deltas_lat.mean()
std_lon = deltas_lon.std(ddof=1)
std_lat = deltas_lat.std(ddof=1)

print("=" * 60)
print("COORDINATE OFFSET ANALYSIS")
print("=" * 60)

print(f"\n{'Point':>10} {'dLon(deg)':>12} {'dLat(deg)':>12}")
print("-" * 34)
for k in keys:
    print(f"{k:>10} {deltas_lon[keys.index(k)]:>+12.6f} {deltas_lat[keys.index(k)]:>+12.6f}")

# ── 3. Decision ───────────────────────────────────────────────────────
mean_applied = bool(std_lon < 0.01 and std_lat < 0.01)

print(f"\n{'Statistic':<25} {'dLon(deg)':>12} {'dLat(deg)':>12}")
print("-" * 49)
print(f"{'Mean':<25} {mean_lon:>+12.6f} {mean_lat:>+12.6f}")
print(f"{'Std (ddof=1)':<25} {std_lon:>12.6f} {std_lat:>12.6f}")
print(f"{'Std threshold (<0.01°)':<25} {'OK' if std_lon < 0.01 else 'FAIL':>12} {'OK' if std_lat < 0.01 else 'FAIL':>12}")
print(f"\nResult: std_lat={std_lat:.4f}°, std_lon={std_lon:.4f}° — MEAN applied: {mean_applied}")

# ── 4. Build final dictionary ──────────────────────────────────────────
golden_mapping = OrderedDict()

for k in keys:
    golden_mapping[k] = {
        'Lon': round(odt_points[k][0], 6),
        'Lat': round(odt_points[k][1], 6),
    }

if mean_applied:
    print(f"\nVector MEAN applied to 4 fallback points (formula: AralField - mean_delta)")
    for k, (lon_b, lat_b) in fallback_points.items():
        corrected_lon = lon_b - mean_lon
        corrected_lat = lat_b - mean_lat
        golden_mapping[k] = {
            'Lon': round(corrected_lon, 6),
            'Lat': round(corrected_lat, 6),
        }
        print(f"  {k}: AralField({lon_b:.6f}, {lat_b:.6f}) - ({mean_lon:.6f}, {mean_lat:.6f}) "
              f"-> ODT_approx({corrected_lon:.6f}, {corrected_lat:.6f})")
else:
    print(f"\nstd >= 0.01° — using AralField DD as-is for 4 fallback points (UNCERTAIN)")
    for k, (lon_b, lat_b) in fallback_points.items():
        golden_mapping[k] = {
            'Lon': lon_b,
            'Lat': lat_b,
        }
        print(f"  {k}: AralField DD ({lon_b:.6f}, {lat_b:.6f}) [UNCORRECTED]")

# ── 5. Output final dict ───────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("FINAL golden_mapping DICT (copy this into build_ground_truth.py)")
print(f"{'=' * 60}")
print()
print("golden_mapping = {")
for i, (k, v) in enumerate(golden_mapping.items()):
    comma = "," if i < len(golden_mapping) - 1 else ""
    print(f"    '{k}': {{'Lon': {v['Lon']}, 'Lat': {v['Lat']}}}{comma}")
print("}")
print()

# ── 6. Summary for Planner ─────────────────────────────────────────────
print(f"{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
mean_dist_deg = np.sqrt(mean_lon**2 + mean_lat**2)
print(f"  Mean offset vector: dLon={mean_lon:.4f} deg, dLat={mean_lat:.4f} deg")
print(f"  Mean distance:      ~{mean_dist_deg:.2f} deg ({mean_dist_deg * 111:.0f} km)")
print(f"  Std of offset:      dLon std={std_lon:.4f} deg, dLat std={std_lat:.4f} deg")
print(f"  Vector applied:     {mean_applied} ({'CONSTANT SHIFT' if mean_applied else 'HETEROGENEOUS'})")
print(f"  Total points:       {len(golden_mapping)}")
