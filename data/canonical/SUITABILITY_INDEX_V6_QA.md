# V6 suitability index — QA (Phase 6b)

Wall-to-wall suitability from the **validated salinity model** (NDMI→P(salts>1 %), LOO AUC 0.682, n=70). suitability = 1 − P(saline), applied to the 30 m NDMI band.

## Products
- `outputs/data/suitability_index_v6.tif` — float32 0..1 continuous (NoData -9999)
- `outputs/data/suitability_zones_v6.tif` — uint8, V5.1 ZoneClass codes (UX-compatible)
- `outputs/data/suitability_v6_stats.json` — areas + coverage

## Coverage
- Restricted to the V5 AOI (1960 Aral footprint): 106,500,271 of 166,208,268 grid px.
- **93.7% of the AOI is scored** (the 30 m stack is wall-to-wall; vs 54 % NoData on the 10 m S2 composite).
- Extrapolation: 1.7% of valid pixels have NDMI outside the training support [-0.174, 0.409]; their NDMI is clipped to the support before scoring (no out-of-range extrapolation).

## Zone breakdown (share of land = non-water)
| Zone | code | area (ha) | % land |
|------|------|-----------|--------|
| Candidate (low salinity) | 1 | 284,903 | 4.5 |
| Moderate salinity risk | 3 | 501,298 | 7.9 |
| Strong salinity risk | 4 | 4,135,982 | 65.5 |
| Existing vegetation | 10 | 1,396,952 | 22.1 |
| Water / NoData | 0 | 4,181,471 | — |

Mean suitability over bare land: **0.5858**.

## Ground-truth validation at the 70 measured pits
Artifacts: `data/canonical/suitability_v6_pit_validation.csv` (per pit), `outputs/data/suitability_v6_pit_validation_summary.json`.
- **Coverage parity:** the frozen 10 m V5.1 map covers **13** of the 70 lab pits as non-water; V6 covers **23**. V6 is not narrower than the shipped product.
- **Why ~15/70 are scored:** 54 of 70 pits lie OUTSIDE the 1960 Aral footprint (the Pachikin/Kozybaeva survey sampled the wider Priaralye, not just the seabed); they still train the salinity model but are not in the mapped target area.
- **Zone ↔ measured salinity** (mean measured topsoil salts per zone, scored pits): candidate (1) ≈ 0.09 %, moderate (3) ≈ 3.57 %, strong (4) ≈ 6.19 %, vegetation (10) ≈ 1.58 %. Strong-salinity zone (4) is by far the most saline — monotonic and correctly ordered.
- **Zone∈{3,4} as a saline (>1 %) detector:** sensitivity **0.89**, specificity **1.0** (TP=17, FP=0, FN=2, TN=4).

## Honesty notes
- CRS is EPSG:4326 (the 30 m stack's native grid); the frozen 10 m V5.1 map is EPSG:32641. Phase 8 overlays both on the web map (both reproject to web-mercator client-side).
- The 30 m **slope band is only ~33 % finite** and ≤0.06° where present (flat seabed; V5 found topo=0.06 %), so no slope gate is applied at 30 m — it would void 2/3 of coverage with no scientific basis.
- The two saline zones (3/4) are a **severity split** on the single validated NDMI→salinity axis, reusing V5's legend codes for UX parity — they do not reproduce V5's separate wet-brine vs dry-salt spectral physics (the 30 m stack lacks those axes). The continuous index is the quantitative object; the zoning is its legend-compatible view.
- The 30 m stack has **no Sentinel-2 SCL band**, so V5's SCL water masking is not reproduced. Open water is caught only by NDWI>0 (~0.1 %); wet saline playa that V5 hid as 'water' surfaces here as high-NDMI **strong-salinity (class 4)** — arguably more informative, but a real semantic difference from the 10 m map.
- Temporal caveat: soil labels are 2012–2014; the 30 m NDMI composite is recent. The NDMI↔salinity relation is treated as quasi-stationary on the dry seabed; see Phase 7.
