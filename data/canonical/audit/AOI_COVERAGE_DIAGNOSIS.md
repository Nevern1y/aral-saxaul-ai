# AOI Coverage Diagnosis — Root Causes of "Empty" / Misaligned Predictions

**Scope:** read-only forensic audit answering: does the V6 model cover the entire 1960 Aral
seabed AOI, and if not, why? No pipeline code or output artifact was modified while producing
this document; two throwaway diagnostic scripts (`scripts/diag_aoi_vectorization_loss.py`,
`scripts/diag_aoi_true_area.py`) were written to reproduce and verify the arithmetic below and
are not part of the pipeline.

**Date:** 2026-07-04
**Method:** git history (`git log -p --all`, `git show <sha>:<path>`) to recover deleted/
gitignored scripts and trace config changes commit-by-commit; direct `rasterio`/`shapely`/
`pyproj` inspection of every raster/vector artifact under `outputs/`; an independent
ground-truth cross-check against raw SRTM `.hgt` tiles (not derived from this repo's pipeline)
to test the DEM/elevation raster for correctness.

**Companion audits:** `SCIENCE_REDTEAM_AUDIT.md` (model logic/statistics), `COVERAGE_FEASIBILITY.md`
(sensor-band/water-mask/10m-cascade questions). This document is strictly about *geographic*
coverage and alignment — whether pixels claiming to represent a given lat/lon actually do, and
whether the mapped study area matches the intended 1960 footprint.

---

## Summary of Findings

| # | Question | Verdict | Severity |
|---|----------|---------|----------|
| F1 | Does the AOI raster mask's area disagree with the AOI vector polygon's area by ~30%, implying non-equivalent geometry? | **No.** A flat 900 m²/px area assumption was used in an earlier pass of this audit and was wrong; corrected for latitude, the two agree to ~2%. **Retracted.** | — (non-issue) |
| F2 | Does part of the intended 1960 study rectangle have elevation/AOI data but no Sentinel-2 imagery? | **Yes, confirmed.** A ~28,900 km² strip (≈21.6% of the intended rectangle) along the west/south/north edges has zero NDMI coverage because the satellite feature stack was never rebuilt after the bounding box was widened. Only ≈0.6% of elevation-qualifying "seabed" pixels fall in this strip. | Medium (real, but small practical footprint) |
| F3 | Is the DEM elevation/slope raster that gates "is this seabed at all" (≤54 m ASL) correctly geo-registered? | **Was no — now FIXED.** Two independent bugs in `scripts/prepare_slope_data.py`: (F3) a coordinate-blind `scipy.ndimage.zoom()` resampling step causing an offset growing smoothly from 0 km at the east edge to ~40 km at the west edge; (F3b, found during the fix) a separate north/south row-flip in the mosaic-assembly loop, entangled with F3's effect until isolated. Both fixed; `dem_elevation_30m.tif`/`dem_slope_30m.tif` regenerated and verified against 9 ground-truth points at mean 0.278 m / max 1.366 m error (previously ~24 m mean). See "F3b" subsection below for the fix detail. | **RESOLVED** (was High) |

**Bottom line on the user's original question ("does the model cover the whole 1960 AOI, or are
parts empty"):** the dominant, previously-undiagnosed cause of incorrect/missing coverage was
**F3** (now fixed — see F3b subsection), not F1 (which turned out to be a false alarm from this
audit's own earlier arithmetic error) or F2 (real, but affecting under 1% of qualifying seabed).
F3 meant the seabed/non-seabed classification itself — the very definition of the AOI — was wrong
for a large, unpredictable fraction of pixels, because the elevation used to decide "is this pixel
below 54 m ASL" was not sampled from that pixel's actual coordinates. **Next step: regenerate
`aoi_mask_v5.tif` and `aral_sea_1960.geojson` from the corrected elevation raster (recommended fix
step 3, below), then propagate through the V6 pipeline (Task #4).**

---

## F1 — Raster-vs-vector AOI area: RETRACTED false alarm (self-correction)

### What was claimed and why it was wrong

An earlier pass compared `outputs/data/aoi_mask_v5.tif` (raster mask, 88,302,423 pixels flagged
AOI=1) against `outputs/aoi/aral_sea_1960.geojson` (vector polygon) and found:

- Raster area (flat assumption: 88,302,423 px × 900 m²/px) = **79,472.2 km²**
- Vector polygon area (Albers Equal-Area, `+proj=aea +lat_1=43 +lat_2=47 +lat_0=45 +lon_0=60`) = **55,061.8 km²**
- Apparent gap: ~30%, taken as evidence the two artifacts describe non-equivalent geometry.

**The 900 m²/px figure is wrong.** `aoi_mask_v5.tif` is stored unprojected, in EPSG:4326
degrees, at pixel size 0.00026949458523585647° in both axes. At this latitude (~45°N), one
degree of longitude spans only cos(45°) ≈ 0.7071 times as much ground distance as one degree of
latitude. Treating both axes as "30 m" (i.e., assuming the pixel is a flat 30×30 m square)
overstates true pixel area by a factor of ≈1/cos(45°) ≈ 1.414.

### Correction (`scripts/diag_aoi_true_area.py`)

Computed true per-row pixel area accounting for latitude compression:
`pixel_area_m2(row) = (Δlon_deg × 111,320 m × cos(lat_row)) × (Δlat_deg × 111,320 m)`, summed
across all 88,302,423 AOI=1 pixels:

```
Total AOI=1 pixels:                    88,302,423
OLD nominal estimate (flat 900 m2/px): 79,472.2 km2   <- WRONG, ignores cos(lat)
CORRECTED geodesic estimate:           56,112.0 km2
Correction factor:                     1.4163x  (expected ≈ 1/cos(45°) = 1.4142 — matches)
Relative diff, corrected raster vs. vector polygon (55,061.8 km2): +1.9%
```

A ~2% residual gap is fully explained by ordinary vectorization losses, independently
reconciled in `scripts/diag_aoi_vectorization_loss.py`:

```
Vectorized (raw, before any filter):        56,110.4 km2   (matches corrected raster area)
Small-fragment filter (>0.01 deg2 kept):    54,965.5 km2   (-1,144.9 km2 dropped, noise specks)
After simplify(tolerance=0.001):            55,061.8 km2   (+96.3 km2, simplification inflation)
= committed aral_sea_1960.geojson area, exactly.
```

**Conclusion: the raster mask and vector polygon describe the same geometry to within normal
vectorization tolerance.** The claim that `benchmark_salinity_model.py`'s docstring is wrong
about the two sharing "the same AOI geometry" is **withdrawn**. This F1 line item is preserved
in this document only so the retraction is on the record, per this project's evidentiary
standard — no further action is needed on it.

---

## F2 — Never-imaged strip: intended AOI outgrew the satellite feature stack

### Evidence chain (git history + file timestamps + direct measurement)

`src/config.py`'s `ARAL_BBOX` was widened mid-project, same day, by one commit:

| Commit | Timestamp | `ARAL_BBOX` value |
|---|---|---|
| `58b79e7` | 2026-05-24 02:36 | `(58.0, 43.5, 62.0, 46.5)` — "narrow" |
| `9df6a8e` | 2026-05-24 17:18 | `(57.5, 43.3, 62.0, 46.7)` — "wide" |

The Sentinel-2 feature-stack files on disk (`feature_stack_30m.tif` / `.vrt` /
`_tile1.tif` / `_tile0_redo.tif`) are timestamped **00:53–01:04, 2026-05-24 — before even the
narrow-bbox commit**. Their measured raster bounds (57.9998–62.0002°E, 43.4999–46.5175°N) match
the OLD narrow bbox almost exactly.

The DEM/elevation/AOI-mask rasters, by contrast, **were** rebuilt to the new wide extent: the
SRTM `.hgt` tiles cached locally for the full wide range (E057–E062, N43–N46) were downloaded
~16:55 on 2026-05-24 — after the widening commit. `dem_elevation_30m.tif`, `dem_slope_30m.tif`,
and `aoi_mask_v5.tif` all share identical grid dimensions (14,844 × 11,197) and bounds
(57.99982–62.00019°E / 43.49993–46.51746°N).

**The Sentinel-2 feature stack — the sole source of every V6 NDMI prediction — was never
regenerated after the bbox was widened.** It is quietly narrower than the elevation/AOI-mask
grid it is paired with pixel-for-pixel in `scripts/build_aoi_mask.py`.

### Quantified impact

- Old (actually-imaged) bbox area ≈ 105,093.7 km²
- New (intended) bbox area ≈ 133,975.5 km²
- **Gap: ≈28,881.9 km² (≈21.6% of the intended study rectangle)** — a strip along the west,
  south, and north edges with elevation data but **zero Sentinel-2 imagery**.
- Of 89,019,914 pixels satisfying the elevation-only seabed criterion (≤54 m ASL, >−50 m;
  computed on the *current, mis-registered* elevation raster — see F3 for why this count itself
  needs revision), only **514,086 (0.6%)** fall inside the never-imaged strip.

**Practical read:** this is a real, previously-undocumented data gap, and it should be closed by
re-running the Sentinel-2 extraction for the full intended bbox (`57.5, 43.3, 62.0, 46.7`) so the
feature stack matches the elevation/AOI-mask grid it is consumed alongside. But on its own, it
explains a small fraction of any "empty" areas inside the actual seabed — F3 below is the larger
contributor.

**Re-confirmed after the F3/F3b fix** (elevation raster now correctly geo-registered): re-measuring
directly against the corrected wide-bbox SRTM mosaic, the never-imaged strip (57.5–58.0°E) contains
25,499 seabed-qualifying px out of 12,005,966 total (**0.21%** of qualifying seabed, 155 km²) —
consistent with the original 0.6% estimate (same order of magnitude; the small numeric difference
is expected since the original figure used the pre-fix mis-registered raster). F2's severity
assessment stands: **real but minor**, not a driver of "empty" predictions. It should still be
closed for completeness/correctness, but it is not urgent relative to F3.

---

## F3 — Systematic DEM/slope geo-registration bug (root cause, HIGH severity)

### How it was found

Independent ground-truth check: sampled `outputs/data/dem_elevation_30m.tif` at 200 random
coordinates restricted to the old/actually-imaged bbox interior (to avoid confounding with F2),
and compared against the same coordinates read directly from raw SRTM3 `.hgt` tiles
(`~/.cache/srtm/*.hgt`, big-endian int16, independent of this repo's pipeline).

```
Valid pairs: 200/200
Mean diff: 14.57   Median diff: 0.00   Std: 46.73
Correlation coefficient: 0.3844          <- should be ~1.0 for the same physical quantity
Fraction with |diff| > 10 m: 45.5%
Fraction with |diff| > 30 m: 22.5%
```

A pattern of *some* exact 0.0 m matches alongside *large* mismatches (−75.4 m, +55.6 m, −22.0 m,
+21.3 m, …) at other points ruled out simple measurement noise — noise doesn't produce exact
zeroes next to 75 m errors. This pointed to a systematic, position-dependent registration fault.

### Root cause, confirmed

`scripts/prepare_slope_data.py` (git-history only — the produced rasters are gitignored, see
`.gitignore:32`) builds its own SRTM mosaic on a private grid:

```python
LON_MIN, LON_MAX = 57.5, 62.0      # prepare_slope_data.py:21  (wide bbox)
LAT_MIN, LAT_MAX = 43.3, 46.7      # prepare_slope_data.py:22
cells_per_deg = 1200
nrows = int(round((LAT_MAX - LAT_MIN) * cells_per_deg)) + 1   # :92
ncols = int(round((LON_MAX - LON_MIN) * cells_per_deg)) + 1   # :93
```

It then resamples this mosaic onto the pipeline's actual reference grid
(`feature_stack_30m.vrt`, read at `prepare_slope_data.py:195-198`) using **plain array
rescaling, not coordinate-aware reprojection**:

```python
scale_y = ref_height / nrows        # prepare_slope_data.py:206
scale_x = ref_width / ncols         # prepare_slope_data.py:207
slope_resampled = zoom(slope_filled, (scale_y, scale_x), order=1)   # :216
...
elev_resampled = zoom(elev_for_export, (scale_y, scale_x), order=1)  # :259
```

`scipy.ndimage.zoom()` is a pure index-space rescale: it assumes the source array's full extent
and the destination array's full extent cover the **same geographic footprint**, and just
changes the sampling density. That assumption is false here. As documented under F2, the actual
`feature_stack_30m.vrt` reference grid covers the **old, narrow** bbox
(58.0–62.0°E, 43.5–46.5°N), while the SRTM mosaic being resampled onto it was built for the
**new, wide** bbox (57.5–62.0°E, 43.3–46.7°N). `zoom()` has no way to know this and silently
squeezes the wide-bbox data into the narrow-bbox grid, mislabeling every pixel's coordinates in
the process.

Both outputs of this script are affected identically — `dem_slope_30m.tif` (line 244) and
`dem_elevation_30m.tif` (`ELEV_OUT_PATH`, lines 251–264) share the exact same transform and the
exact same `zoom()` mechanism, so the slope raster carries the identical mis-registration as
elevation.

### Deterministic verification (not just statistical)

The exact coordinate offset implied by the two grids' mismatched scale factors was computed
directly and matches the empirical discrepancies:

```
Wide-bbox SRTM mosaic size:        5,401 x 4,081
Reference (output) grid size:     14,844 x 11,197
zoom() scale factors:              x=2.7484, y=2.7437

Offset field (labeled coordinate vs. coordinate the pixel's value was actually sampled from):
  East edge (col=max):    ~0 km offset               (grids happen to share this corner)
  West edge (col=0):      ~+38 to +40 km offset (E-W)
  North edge (row=0):     ~-20 km offset (N-S)
  South edge (row=max):   ~+22 km offset (N-S)
  Center of grid:         ~+20 km (E-W), ~+1 km (N-S)
```

The offset grows smoothly and linearly across the entire grid — the signature of a scale
mismatch, not random error or an edge artifact.

### Independent confirmation via correct reprojection

A properly georeferenced SRTM mosaic was built directly from the same raw `.hgt` tiles (validated
against 4 known points to exact agreement) and resampled onto the *actual* reference grid using
`rasterio.warp.reproject` (coordinate-aware, unlike `zoom()`). Cross-checked against the same 15
test points used in the original ground-truth spot-check:

```
lat       lon       ground truth   correctly-reprojected   current (buggy) raster
45.7123   60.9432   50.0           50.5                    33.8
44.8411   61.6042   47.0           46.0                    58.6
45.9324   60.0025   32.0           31.7                    53.3
43.9449   58.3474   153.0          153.1                   131.6
46.2366   59.0808   126.0          126.2                   50.6
45.7438   60.6040   37.0           37.6                    29.0
44.0331   58.7142   40.0           40.4                    94.8
45.3740   58.9792   51.0           50.5                    29.0
45.8392   61.0340   53.0           53.7                    38.8
... (15 points total)

Mean |correctly-reprojected − ground truth|: 0.33 m   (bilinear-interpolation-level noise, as expected)
Mean |current (buggy) raster − ground truth|: ~24 m   (matches the earlier 200-point spot-check)
```

This closes the loop: the correctly-reprojected version agrees with ground truth to within
normal interpolation noise, while the pipeline's actual output does not, at the same points,
using the same ground truth. The cause is confirmed, not merely hypothesized.

### F3b — a second, independent bug found during remediation (mosaic row-flip)

**Status as of the fix pass: both F3 and F3b are now fixed and verified. See below.**

When `scripts/prepare_slope_data.py` was actually rewritten to replace `zoom()` with
`rasterio.warp.reproject()` (keeping the rest of the script, including its mosaic-assembly loop,
untouched), the same 9-point ground-truth check unexpectedly showed **mean |diff| = 23.45 m,
max = 74.0 m** — not the ~0.33 m this document's "independent confirmation" script had shown above.
Since that confirmation script was a separate, standalone reimplementation (not a patch to
`prepare_slope_data.py`), the discrepancy meant the two scripts differed in some other way besides
the resampling method.

Root-caused by reading the raw `.hgt` tiles directly (bypassing the pipeline entirely) and
confirming they match ground truth exactly (0.0 m at all 9 points — the "ground truth" values
*are* direct SRTM samples). This isolated the bug to `prepare_slope_data.py`'s mosaic-assembly
loop itself (the block that writes each tile's overlap window into the mosaic array), which is
**independent of, and predates, the `zoom()` bug**:

```python
# BUGGY (original, both before and immediately after the zoom()->reproject() fix):
mos_row_start = int(round((ov_bottom - LAT_MIN) * cells_per_deg))
mos_row_end   = int(round((ov_top    - LAT_MIN) * cells_per_deg))
# This places row 0 at LAT_MIN (south-up array) ...

# ... but src_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, ncols, nrows)
# and every downstream consumer (zoom() implicitly, and the new reproject() explicitly)
# assume the rasterio-standard NORTH-UP convention: row 0 at LAT_MAX.
```

i.e. the mosaic array was being assembled upside-down relative to the geotransform used to
describe it — a **north/south vertical flip**, independent of F3's east/west scale-mismatch bug.
Under the old `zoom()` code this flip's effect was entangled with (and partially masked by) the
scale-mismatch error, so it was never isolated on its own. Confirmed by rebuilding the mosaic
in-memory with the corrected orientation and re-checking the same 9 points: **exact 0.000 m
diff on all 9**.

Fixed in `prepare_slope_data.py` by changing the mosaic-placement formula to the north-up
convention:

```python
mos_row_start = int(round((LAT_MAX - ov_top) * cells_per_deg))
mos_row_end   = int(round((LAT_MAX - ov_bottom) * cells_per_deg))
```

After this second fix, the actual pipeline output (`outputs/data/dem_elevation_30m.tif`,
regenerated end-to-end) was re-checked against the same 9 ground-truth points:

```
Mean |diff|: 0.278 m
Max |diff|:  1.366 m
```

This meets (in fact slightly beats) the ~0.33 m benchmark this document originally predicted for
a correct fix. **F3 and F3b are both now resolved as of this pass** — `dem_elevation_30m.tif` and
`dem_slope_30m.tif` have been regenerated (99.9% valid-pixel coverage over the full 58.0–62.0°E /
43.5–46.5°N reference grid; elevation range −0.6 to 254.8 m; slope range 0–37.5°, both physically
plausible for the Aral seabed).

Pre-fix versions of all touched files are preserved as `*.bak_pre_F3fix` for rollback/comparison.

### Quantified impact on the seabed mask (the ≤54 m / >−50 m criterion)

Comparing the "is this pixel seabed" boolean across the **entire** 166,208,268-pixel grid,
current (buggy) elevation vs. correctly-reprojected elevation:

| | Current (buggy) DEM | Correctly reprojected DEM |
|---|---|---|
| Pixels passing elevation criterion | 89,019,914 (53.56% of grid) | 107,125,816 (64.45% of grid) |

- **Agreement:** 70,552,684 px — 79.25% of the current pipeline's "seabed" pixels are correctly
  classified, and 65.86% of the true seabed is captured.
- **False positive** (pipeline says seabed; true elevation says it isn't): **18,467,230 px —
  20.75% of the current pipeline's seabed classification is wrong in this direction.**
- **False negative** (true elevation says seabed; pipeline misses it): **36,573,132 px — 34.14%
  of the true seabed is missing from the current pipeline's AOI.**

Elevation value statistics over all 165,783,232 jointly-valid pixels (current vs. correct):
mean diff +10.16 m, median 0.00 m, std 48.10 m, correlation 0.4436, 52.4% of pixels differ by
more than 10 m, 27.6% by more than 30 m — consistent across the full grid with the 200-point
spot-check that first surfaced this issue.

### Why this is the primary answer to "are predictions covering the whole AOI, or are parts empty"

This bug does not create "empty" (NoData) predictions — it creates **wrongly-placed ones**.
Roughly a third of pixels that should be classified as seabed (and therefore eligible for a V6
suitability prediction at all) are silently excluded, while about a fifth of pixels currently
treated as seabed shouldn't be. Because the offset is smooth and directional (worse toward the
west/north/south edges, ~0 at the east edge), the practical symptom is that entire regions of the
map can look "hollowed out" or shifted relative to where the seabed actually is — which matches
the user's original observation that motivated this whole investigation.

The NDMI/Sentinel-2 feature stack itself is **not** affected by this specific bug — it is
sourced independently via GEE export and is not touched by `prepare_slope_data.py`. F3 is
isolated to the elevation/slope pipeline (`dem_elevation_30m.tif`, `dem_slope_30m.tif`,
and — because it is derived pixel-for-pixel from the elevation raster — `aoi_mask_v5.tif` and
the vector polygon derived from it).

---

## Recommended fix (for Task #2 — AOI rebuild)

1. Rewrite `scripts/prepare_slope_data.py`'s resampling step to use `rasterio.warp.reproject`
   (or equivalent coordinate-aware method) instead of `scipy.ndimage.zoom`, sourcing directly
   from the SRTM `.hgt` tiles reprojected onto the **actual** target grid's transform/CRS — not
   a scale-factor guess between two independently-sized arrays.
2. Resolve F2 in the same pass: regenerate the Sentinel-2 feature stack for the full intended
   bbox (`57.5, 43.3, 62.0, 46.7`) so it is co-extensive with the corrected elevation grid, rather
   than leaving a ~28,900 km² imagery gap.
3. Re-run `scripts/build_aoi_mask.py` against the corrected elevation raster to produce a
   corrected `aoi_mask_v5.tif` and re-vectorize `aral_sea_1960.geojson`.
4. Regenerate every downstream V6 artifact that consumes these grids (Task #4):
   `build_suitability_index.py`, `benchmark_salinity_model.py`, `spatial_validation.py`,
   `qa_science_audit_v6.py`, `render_v6_map.py`.
5. Separately: the provenance of `outputs/aoi/coastline_raster_mask.geojson` (a third,
   structurally distinct AOI-related file, added in the same commit as the bbox widening) remains
   unresolved — no producing script was found anywhere in git history despite an exhaustive
   search across every commit's `.py` files. It should be treated as unverified/orphaned pending
   further investigation, and not relied upon, until (or unless) its origin is established.

## Open items not yet resolved

- `outputs/aoi/coastline_raster_mask.geojson` provenance (see point 5 above).
- Whether the `TIFFReadEncodedTile()` I/O errors observed when reading certain window slices of
  `feature_stack_30m.vrt` (bottom/right-edge bands) are a benign VRT tile-boundary quirk or
  indicative of a further data problem — not blocking, since other windows read cleanly, but
  flagged for completeness.
