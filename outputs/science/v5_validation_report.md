# V5.1 Scientific Validation Report

This report samples the V5 map and Sentinel-derived indices at mapped field/soil profiles. It is a pilot validation report, not an accuracy certificate.

## Coordinate Policy

Coordinate status: `dual_source_unresolved`.

Both coordinate candidates are evaluated separately:

- `raw_aralfield_dd`: decimal-degree coordinates from `AralField(Sheet1).csv`.
- `shifted_odt_derived_dd`: legacy shifted/ODT-derived coordinates from `ground_truth_v2.csv`.
- `authoritative_dd`: only appears after `v5_coordinate_authority_template.csv` contains resolved reviewer decisions with evidence.

The raw and shifted coordinate systems differ by roughly `132619` m median distance. Do not merge raw and shifted validation samples. If `authoritative_dd` rows are present, use them as the evidence-backed validation subset and keep raw/shifted rows as coordinate audit diagnostics.

Authority template status counts: `{"unresolved": 11}`.

## Sample Counts

- Point samples: 22
- Unique field sites: 11
- Profiles with coordinates: 11
- Lab-only profiles without coordinates: 10
- Authoritative point samples: 0
- Authority template resolved rows: 0 / 11

## Filtered V5 Class Distribution At Points

| coordinate_source | class_filtered | count |
| --- | --- | --- |
| raw_aralfield_dd | Candidate suitable | 5 |
| raw_aralfield_dd | Existing vegetation | 4 |
| raw_aralfield_dd | Water / NoData / Shadow | 2 |
| shifted_odt_derived_dd | Water / NoData / Shadow | 11 |

## Selected Spearman Correlations

Target is top-layer lab salinity (`top_salinity_pct`). Bootstrap intervals are unstable with `n=11`; use them only as sensitivity diagnostics.

| coordinate_source | target | feature | n | spearman_r | p_value | bootstrap_ci95 | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_aralfield_dd | top_salinity_pct | ndmi | 10 | 0.2 | 0.579584 | [-0.597671, 0.849349] | ok |
| raw_aralfield_dd | top_salinity_pct | ndsi_green_swir2 | 10 | -0.066667 | 0.854813 | [-0.620313, 0.641083] | ok |
| raw_aralfield_dd | top_salinity_pct | br_nir_swir2 | 10 | 0.478788 | 0.161523 | [-0.311921, 0.838509] | ok |
| shifted_odt_derived_dd | top_salinity_pct | ndmi | 11 | 0.281818 | 0.401145 | [-0.467337, 0.8506] | ok |
| shifted_odt_derived_dd | top_salinity_pct | ndsi_green_swir2 | 11 | 0.327273 | 0.325895 | [-0.446115, 0.839314] | ok |
| shifted_odt_derived_dd | top_salinity_pct | br_nir_swir2 | 11 | 0.4 | 0.222868 | [-0.394329, 0.897208] | ok |

## Point-Level Audit

| coordinate_source | S_Point | pit_code | top_salinity_pct | field_salinity_0_20 | field_ec_0_20 | class_filtered_name | ndmi | ndsi_green_swir2 | br_nir_swir2 | authority_selected_source | authority_row_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_aralfield_dd | S124 | 01/20А | 0.06 | 0.09 | 3.14 | Existing vegetation | -0.0995 | -0.2778 | 0.8071 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S124 | 01/20А | 0.06 | 0.09 | 3.14 | Water / NoData / Shadow | -0.016 | 0.0659 | 1.3209 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S125 | 02/20А | 0.07 | 1.9 | 101.3 | Candidate suitable | -0.0956 | -0.2254 | 0.8039 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S125 | 02/20А | 0.07 | 1.9 | 101.3 | Water / NoData / Shadow | -0.0094 | 0.081 | 1.3666 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S126 | 03/20А | 0.05 | 0.97 | 46.2 | Candidate suitable | -0.0971 | -0.2033 | 0.8541 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S126 | 03/20А | 0.05 | 0.97 | 46.2 | Water / NoData / Shadow | -0.0781 | -0.06 | 1.0585 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S127 | 04/20А | 6.52 | 0.16 | 5.7 | Existing vegetation | -0.0733 | -0.2758 | 0.8734 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S127 | 04/20А | 6.52 | 0.16 | 5.7 | Water / NoData / Shadow | -0.0051 | 0.0753 | 1.3672 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S128 | 05/20А | 31.29 | 0.14 | 4.18 | Existing vegetation | -0.047 | -0.2582 | 1.018 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S128 | 05/20А | 31.29 | 0.14 | 4.18 | Water / NoData / Shadow | 0.0185 | 0.1213 | 1.5313 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S129 | 06/20А | 1.17 | 0.18 | 5.64 | Candidate suitable | -0.0371 | -0.1282 | 0.9631 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S129 | 06/20А | 1.17 | 0.18 | 5.64 | Water / NoData / Shadow | 0.0387 | 0.1156 | 1.4857 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S130 | 07/20А | 1.29 | 0.87 | 38.1 | Candidate suitable | -0.0678 | -0.1921 | 0.9233 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S130 | 07/20А | 1.29 | 0.87 | 38.1 | Water / NoData / Shadow | -0.0366 | 0.0393 | 1.326 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S131 | 08/20А | 2.42 | 0.02 | 0.7 | Existing vegetation | -0.1117 | -0.2909 | 0.825 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S131 | 08/20А | 2.42 | 0.02 | 0.7 | Water / NoData / Shadow | -0.0144 | 0.0927 | 1.4102 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S132 | 09/20А | 16.42 | 0.28 | 8.84 | Candidate suitable | -0.1253 | -0.2481 | 0.8087 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S132 | 09/20А | 16.42 | 0.28 | 8.84 | Water / NoData / Shadow | -0.0352 | -0.0223 | 1.135 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S133 | 10/20А | 7.78 | 0.1 | 3.25 | Water / NoData / Shadow | -0.0355 | 0.1984 | 1.1212 |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S133 | 10/20А | 7.78 | 0.1 | 3.25 | Water / NoData / Shadow | -0.0448 | -0.025 | 1.1532 |  | audit_source_not_authoritative |
| raw_aralfield_dd | S134 | 11/20А | 0.1 | 1.73 | 95.2 | Water / NoData / Shadow |  |  |  |  | audit_source_not_authoritative |
| shifted_odt_derived_dd | S134 | 11/20А | 0.1 | 1.73 | 95.2 | Water / NoData / Shadow | -0.0433 | -0.0319 | 1.1237 |  | audit_source_not_authoritative |

## Scientific Limitations

- V5 is a rule-based suitability screening product, not a trained habitat model.
- `Candidate suitable` is a residual class after excluding water/shadow, steep slope, vegetation, dry-salt proxy, and wet-brine proxy.
- Only 11 profiles currently have any field-site mapping; profiles 12-21 are lab-only until coordinates are provided.
- Coordinate authority is unresolved, so all reported correlations are exploratory.
- Remote-sensing indices are surface proxies and cannot replace root-zone salinity, EC, groundwater depth, or planting survival observations.
