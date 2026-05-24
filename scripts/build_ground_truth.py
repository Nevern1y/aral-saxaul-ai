"""
build_ground_truth.py — V2.0 Master Dataset from Golden Dictionary
Source of truth: DMM coordinates from cross-reference document (hardcoded)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

BASE = Path(r"F:\OPENCODE PROJECTS\aral-saxaul-ai")

def safe_float(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    if s in ("-", "", "None", "nan", "NaN"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

# ── 1. Golden Dictionary — V3.0 expanded to 11 points ──────────────────
#  01/20А–07/20А: ODT DMM coordinates (verified via coordinate offset analysis)
#  08/20А–11/20А: AralField DD corrected by mean offset vector
#  Format: {pit_code: {S_Point, Lon (DD), Lat (DD)}}
GOLDEN = {
    "01/20А": {"S_Point": "S124", "Lon": 59.775683, "Lat": 44.704500},
    "02/20А": {"S_Point": "S125", "Lon": 59.793467, "Lat": 44.711050},
    "03/20А": {"S_Point": "S126", "Lon": 59.813067, "Lat": 44.729433},
    "04/20А": {"S_Point": "S127", "Lon": 59.863583, "Lat": 44.755567},
    "05/20А": {"S_Point": "S128", "Lon": 59.868800, "Lat": 44.762467},
    "06/20А": {"S_Point": "S129", "Lon": 59.892250, "Lat": 44.763633},
    "07/20А": {"S_Point": "S130", "Lon": 60.065867, "Lat": 44.820933},
    "08/20А": {"S_Point": "S131", "Lon": 60.097676, "Lat": 44.860309},
    "09/20А": {"S_Point": "S132", "Lon": 60.134392, "Lat": 44.920813},
    "10/20А": {"S_Point": "S133", "Lon": 60.143704, "Lat": 44.952885},
    "11/20А": {"S_Point": "S134", "Lon": 60.145227, "Lat": 45.050679},
}

# Pit number lookup
PIT_NUM = {f"{i:02d}/20\u0410": i for i in range(1, 22)}

# ── 2. Load lab chemistry (EC,TDS,pH) — TOP LAYER ONLY ────────────────
print("=" * 60)
print("GROUND TRUTH V2.0 — Golden Dictionary Pipeline")
print("=" * 60)

ectds = pd.read_csv(
    BASE / "\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b EC,TDS, pH, \u0432.\u0432., \u0433\u0443\u043c\u0443\u0441(\u0432.\u0432.csv",
    encoding="utf-8-sig", sep=";", header=None, low_memory=False,
)

# Parse: extract (pit_number, salinity_pct, ph, co3, hco3, cl, so4, ca, mg, na, k) per row
lab_rows = []
current_pit = None
for i in range(2, len(ectds)):
    v1 = ectds.iloc[i, 1]
    if pd.notna(v1):
        name = str(v1).strip()
        if name.startswith("-"):
            continue
        # "Разрез N" → N
        try:
            current_pit = int(name.split()[-1])
        except (ValueError, IndexError):
            continue

    if current_pit is None:
        continue

    row = {"pit": current_pit}
    for col, key in [(19, "Salinity_pct"), (3, "CO3"), (5, "HCO3"), (7, "Cl"),
                      (9, "SO4"), (11, "Ca"), (13, "Mg"), (15, "Na"), (17, "K")]:
        row[key] = safe_float(ectds.iloc[i, col])
    lab_rows.append(row)

df_lab = pd.DataFrame(lab_rows)

# Take TOP LAYER only (first row = shallowest depth per pit)
lab_agg = df_lab.groupby("pit").first(numeric_only=True).reset_index()
lab_agg.rename(columns={"pit": "pit_number"}, inplace=True)
print(f"Lab chemistry: {len(lab_agg)} pits, {len(lab_agg.columns)} columns (top layer only)")

# ── 3. Load sand/silt/clay (мехсостав 2020) ───────────────────────────
mex = pd.read_csv(
    BASE / "\u043c\u0435\u0445\u0441\u043e\u0441\u0442\u0430\u0432 2020, 2021 \u043f\u043e \u0442\u0438\u043f\u0430\u043c(\u041b\u0438\u0441\u04424).csv",
    encoding="utf-8-sig", sep=";", header=None, low_memory=False,
)

mex_data = []
for i in range(4, 25):
    v0 = mex.iloc[i, 0]
    try:
        pn = int(str(v0).strip())
    except (ValueError, AttributeError):
        continue
    mex_data.append({
        "pit_number": pn,
        "Sand_pct": float(str(mex.iloc[i, 1]).replace(",", ".")) if pd.notna(mex.iloc[i, 1]) else np.nan,
        "Silt_pct": float(str(mex.iloc[i, 3]).replace(",", ".")) if pd.notna(mex.iloc[i, 3]) else np.nan,
        "Clay_pct": float(str(mex.iloc[i, 6]).replace(",", ".")) if pd.notna(mex.iloc[i, 6]) else np.nan,
    })

df_mex = pd.DataFrame(mex_data)
print(f"Sand/Silt/Clay: {len(df_mex)} pits")

# ── 4. Assemble master table from Golden Dictionary ───────────────────
rows = []
for code, info in GOLDEN.items():
    pit_num = PIT_NUM.get(code)
    if pit_num is None:
        print(f"  WARNING: No pit number for {code}")
        continue

    lat_dd = info["Lat"]
    lon_dd = info["Lon"]

    # Lab chemistry for this pit
    lab_row = lab_agg[lab_agg["pit_number"] == pit_num]
    lab_vals = lab_row.iloc[0].to_dict() if not lab_row.empty else {}

    # Sand/silt/clay for this pit
    mex_row = df_mex[df_mex["pit_number"] == pit_num]
    mex_vals = mex_row.iloc[0].to_dict() if not mex_row.empty else {}

    row = {
        "pit_code": code,
        "S_Point": info["S_Point"],
        "pit_number": pit_num,
        "Lat_DD": round(lat_dd, 6),
        "Lon_DD": round(lon_dd, 6),
    }
    row.update({k: v for k, v in lab_vals.items() if k != "pit_number"})
    row.update({k: v for k, v in mex_vals.items() if k != "pit_number"})
    rows.append(row)

df = pd.DataFrame(rows).sort_values("pit_number").reset_index(drop=True)

print(f"\nMaster table: {len(df)} points")
# Print table (replace Cyrillic for terminal safety)
out_table = df[["pit_code", "S_Point", "pit_number", "Lat_DD", "Lon_DD", "Salinity_pct"]].copy()
out_table["pit_code"] = out_table["pit_code"].str.replace("\u0410", "A")
print(out_table.to_string(index=False))

# ── 5. Sample satellite features ──────────────────────────────────────
coords = list(zip(df["Lon_DD"], df["Lat_DD"]))

fs_path = BASE / "outputs/data/feature_stack_30m.vrt"
tile1 = BASE / "outputs/data/feature_stack_30m_tile1.tif"
tile0 = BASE / "outputs/data/feature_stack_30m_tile0_redo.tif"

band_names = ["NDMI", "MSAVI", "SI", "NDWI", "Slope", "TWI", "VH"]
band_vals = {name: [] for name in band_names}

for raster_path in [fs_path, tile1, tile0]:
    if raster_path.exists():
        try:
            with rasterio.open(raster_path) as src:
                for x, y in coords:
                    try:
                        row_idx, col_idx = src.index(x, y)
                        window = ((row_idx, row_idx + 1), (col_idx, col_idx + 1))
                        for idx, name in enumerate(band_names, 1):
                            val = src.read(idx, window=window)[0, 0]
                            band_vals[name].append(val)
                    except Exception:
                        for name in band_names:
                            band_vals[name].append(np.nan)
            print(f"\nSampled features from: {raster_path.name}")
            break
        except Exception as e:
            print(f"  Failed: {raster_path.name} — {e}")

for name in band_names:
    df[name] = band_vals[name]

# Also sample V1.0 suitability
suit_path = BASE / "outputs/data/suitability_full.tif"
prob_vals = []
try:
    with rasterio.open(suit_path) as src:
        for x, y in coords:
            try:
                r, c = src.index(x, y)
                prob_vals.append(src.read(1, window=((r, r+1), (c, c+1)))[0, 0])
            except Exception:
                prob_vals.append(np.nan)
    df["Prob_V1"] = prob_vals
except Exception as e:
    print(f"  Failed to sample suitability: {e}")
    df["Prob_V1"] = np.nan

# ── 6. Output ─────────────────────────────────────────────────────────
out_path = BASE / "outputs/data/ground_truth_v2.csv"
df.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print(f"SAVED: {out_path}")
print(f"{'='*60}")
print(f"{len(df)} points, {len(df.columns)} columns")
print(f"\nColumns: {df.columns.tolist()}")

# Print full table
SEP = "-" * 100
print(f"\n{SEP}")
print(f"{'pit_code':>10} {'S_Point':>8} {'pit#':>5} {'Lat_DD':>12} {'Lon_DD':>12} "
      f"{'Salinity':>9} {'Na':>8} {'Cl':>8} {'Sand':>6} {'Silt':>6} {'Clay':>6} "
      f"{'SI':>8} {'MSAVI':>8} {'Prob':>8}")
print(SEP)
for _, r in df.iterrows():
    pc = r['pit_code'].replace("\u0410", "A")
    print(f"{pc:>10} {r['S_Point']:>8} {r['pit_number']:>5} "
          f"{r['Lat_DD']:>12.6f} {r['Lon_DD']:>12.6f} "
          f"{r['Salinity_pct'] if pd.notna(r['Salinity_pct']) else 'NaN':>9} "
          f"{r['Na'] if pd.notna(r['Na']) else 'NaN':>8} "
          f"{r['Cl'] if pd.notna(r['Cl']) else 'NaN':>8} "
          f"{r['Sand_pct'] if pd.notna(r['Sand_pct']) else 'NaN':>6} "
          f"{r['Silt_pct'] if pd.notna(r['Silt_pct']) else 'NaN':>6} "
          f"{r['Clay_pct'] if pd.notna(r['Clay_pct']) else 'NaN':>6} "
          f"{r['SI'] if pd.notna(r['SI']) else 'NaN':>8.2f} "
          f"{r['MSAVI'] if pd.notna(r['MSAVI']) else 'NaN':>8.4f} "
          f"{r['Prob_V1'] if pd.notna(r['Prob_V1']) else 'NaN':>8.4f}")

# ── 7. Quick cross-check: distance to nearest AralField point ─────────
SEP2 = "-" * 80
print(f"\n{SEP2}")
print("CROSS-CHECK: Distance from Golden pit to nearest AralField S-point")
print(SEP2)

# Load AralField for reference
af = pd.read_csv(BASE / "AralField(Sheet1).csv",
                 encoding="utf-8-sig", sep=";", header=None, skiprows=3, low_memory=False)
af_pts = []
for i in range(len(af)):
    pid = str(af.iloc[i, 0]).strip()
    elon = float(str(af.iloc[i, 11]).replace(",", "."))
    nlat = float(str(af.iloc[i, 12]).replace(",", "."))
    af_pts.append({"id": pid, "lon": elon, "lat": nlat})
df_af = pd.DataFrame(af_pts)

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

for _, r in df.iterrows():
    dists = df_af.apply(
        lambda row: haversine(r["Lon_DD"], r["Lat_DD"], row["lon"], row["lat"]),
        axis=1,
    )
    nearest = df_af.iloc[dists.idxmin()]
    d_m = dists.min()
    status = "OK" if d_m < 50 else ("WARNING" if d_m < 500 else "MISMATCH")
    pc = r['pit_code'].replace("\u0410", "A")
    print(f"  {pc} (->{r['S_Point']}): {d_m:>8.1f}m from nearest {nearest['id']} [{status}]")

print(f"\nDone. Ready for V2.0 regression training.")
