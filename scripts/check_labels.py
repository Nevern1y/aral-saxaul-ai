import pandas as pd
from src.config import config

df = pd.read_csv(str(config.output_dir / "data" / "training_test.csv"))
print("Columns:", df.columns.tolist())
print("\nNDWI percentiles:")
for p in [5, 10, 25, 50, 75, 85, 90, 95]:
    print(f"  P{p}: {df.NDWI.quantile(p/100):.4f}")

print("\nSlope percentiles:")
for p in [5, 10, 25, 50, 75, 85, 90, 95]:
    print(f"  P{p}: {df.Slope.quantile(p/100):.4f}")

print("\nIndividual conditions met:")
si_t = df.SI.quantile(0.85)
msavi_t = df.MSAVI.quantile(0.10)

for col, cond, thresh in [
    ("SI", ">=", si_t),
    ("MSAVI", "<=", msavi_t),
    ("Slope", "<=", 5),
    ("NDWI", "<=", -0.15),
]:
    mask = df[col] >= thresh if cond == ">=" else df[col] <= thresh
    print(f"  {col} {cond} {thresh:.4f}: {mask.sum()}/{len(df)} ({mask.mean()*100:.1f}%)")

# Combined labels with different NDWI thresholds
for ndwi_t in [-0.15, -0.10, -0.05, 0, 0.05]:
    mask = (
        (df.SI >= si_t)
        & (df.MSAVI <= msavi_t)
        & (df.Slope <= 5)
        & (df.NDWI <= ndwi_t)
    )
    print(f"  NDWI<={ndwi_t}: {mask.sum()}/{len(df)} ({mask.mean()*100:.1f}%)")
