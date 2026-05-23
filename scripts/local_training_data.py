import rasterio
import numpy as np
import os
import pandas as pd
from scipy.ndimage import sobel
from src.config import config

out_dir = os.path.join(str(config.output_dir / "data"))

with rasterio.open(os.path.join(out_dir, "s2_mosaic_100m.tif")) as src:
    b2, b3, b4, b8, b11, b12 = [src.read(i).astype("float32") for i in range(1, 7)]
    profile = src.profile
    height, width = src.height, src.width
    transform = src.transform

with rasterio.open(os.path.join(out_dir, "dem_raw_100m.tif")) as src:
    dem = src.read(1).astype("float32")

# Convert degree cell size to meters at ~44.6N
deg_to_m_y = 111320  # 1 degree latitude = 111.32 km
deg_to_m_x = 111320 * np.cos(np.deg2rad(44.6))  # 1 deg longitude at 44.6N

cellsize_x_m = abs(transform[0]) * deg_to_m_x
cellsize_y_m = abs(transform[4]) * deg_to_m_y

dzdx = sobel(dem, axis=1) / (8 * cellsize_x_m)
dzdy = sobel(dem, axis=0) / (8 * cellsize_y_m)

slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
slope = np.where(np.isfinite(slope), slope, 0.0).astype("float32")

with rasterio.open(os.path.join(out_dir, "s1_vh_100m.tif")) as src:
    vh = src.read(1).astype("float32")

eps = 1e-10

ndmi = (b8 - b11) / (b8 + b11 + eps)

msavi = (2 * b8 + 1 - np.sqrt(np.clip((2 * b8 + 1) ** 2 - 8 * (b8 - b4), 0, None))) / 2

si = (b2 - b8) / (b2 + b8 + eps)

ndwi = (b3 - b8) / (b3 + b8 + eps)

slope_rad = slope * np.pi / 180.0
tan_slope = np.maximum(np.tan(slope_rad), 1e-6)
twi = np.log(1.0 / tan_slope)

stack = np.stack([ndmi, msavi, si, slope, twi, vh, ndwi], axis=0)

profile.update(count=7, dtype="float32", compress="lzw")
out_path = os.path.join(out_dir, "feature_stack_100m.tif")
with rasterio.open(out_path, "w", **profile) as dst:
    for i in range(7):
        dst.write(stack[i], i + 1)
print(f"Feature stack saved: {os.path.getsize(out_path)/1e6:.2f} MB ({height}x{width}x7)")

np.random.seed(42)
valid = (dem > -1000) & np.isfinite(ndmi)
valid_indices = np.where(valid)
n_valid = len(valid_indices[0])
print(f"Valid pixels: {n_valid}")

n_samples = min(5000, n_valid)
sample_idx = np.random.choice(n_valid, n_samples, replace=False)
rows, cols = valid_indices[0][sample_idx], valid_indices[1][sample_idx]

data = []
for r, c in zip(rows, cols):
    data.append(
        {
            "NDMI": ndmi[r, c],
            "MSAVI": msavi[r, c],
            "SI": si[r, c],
            "Slope": slope[r, c],
            "TWI": twi[r, c],
            "VH": vh[r, c],
            "NDWI": ndwi[r, c],
            "row": int(r),
            "col": int(c),
        }
    )

df = pd.DataFrame(data)
print(f"Samples: {len(df)}")
print(df.describe())

# Primary saxaul signal: bare saline soil (high SI) on flat terrain
# In the Aral Sea dried bed, high SI (>P85) indicates exposed soil with
# enough salinity to limit competing vegetation but suitable for saxaul.
# Low MSAVI is implicitly captured (SI-MSAVI correlation r = -0.69).
si_p85 = df["SI"].quantile(0.85)
msavi_p10 = df["MSAVI"].quantile(0.10)

print(f"SI P85: {si_p85:.4f}, MSAVI P10: {msavi_p10:.4f}")

positive = (df["SI"] >= si_p85) & (df["Slope"] <= 5)
df["label"] = positive.astype(int)

print(f"Label counts: {df['label'].value_counts().to_dict()}")
pos_pct = df["label"].mean() * 100
print(f"{pos_pct:.1f}% suitable")

df.to_csv(os.path.join(out_dir, "training_test.csv"), index=False)
print("Training data saved.")
