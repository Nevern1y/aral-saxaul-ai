# V5.1 Coordinate Adjudication Appendix

This appendix prepares the coordinate-authority decision for the 11 mapped field points. It does not change V5 map products or V5 rule logic.

## Decision State

- Status: `unresolved`.
- Policy: `adjudication_required_dual_source_unresolved`.
- Field sites with coordinate candidates: 11.
- Coordinate candidates evaluated: 22.
- Authority template rows resolved: 0 / 11.
- Authority template unresolved rows: 11.
- Authority template incomplete/invalid rows: 0 / 0.
- Median raw-vs-shifted conflict: 132.6 km.
- Lab-only profiles without coordinates: 10.

## Use Policy

- Do not merge `raw_aralfield_dd` and `shifted_odt_derived_dd` validation samples.
- Do not treat V5 spectral class as coordinate proof.
- Use this appendix to record an external GPS/ODT/photo-backed decision, then rerun `python scripts/run_v5_science_suite.py --qa`.

## Source-Level Diagnostics

| coordinate_source | total_points | water_nodata_shadow_points | non_water_sampled_points | candidate_suitable_points | existing_vegetation_points | water_nodata_shadow_share_pct | diagnostic_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_aralfield_dd | 11 | 2 | 9 | 5 | 4 | 18.2 | Mixed sampled classes; usable for pilot diagnostics but still not authoritative. |
| shifted_odt_derived_dd | 11 | 11 | 0 | 0 | 0 | 100 | All sampled points fall in water/nodata/shadow; this source is suspect but not rejected without ODT/GPS authority. |

## Filtered V5 Class Distribution

| coordinate_source | class_filtered_name | count |
| --- | --- | --- |
| raw_aralfield_dd | Candidate suitable | 5 |
| raw_aralfield_dd | Existing vegetation | 4 |
| raw_aralfield_dd | Water / NoData / Shadow | 2 |
| shifted_odt_derived_dd | Water / NoData / Shadow | 11 |

## Coordinate Candidate Audit

| S_Point | pit_code | coordinate_source | lon | lat | coordinate_conflict_m | class_filtered_name | top_salinity_pct | field_salinity_0_20 | field_ec_0_20 | authority_selected_source | map_support_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S124 | 01/20А | raw_aralfield_dd | 60.8936 | 45.5885 | 131715 | Existing vegetation | 0.06 | 0.09 | 3.14 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S124 | 01/20А | shifted_odt_derived_dd | 59.7757 | 44.7045 | 131715 | Water / NoData / Shadow | 0.06 | 0.09 | 3.14 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S125 | 02/20А | raw_aralfield_dd | 60.9121 | 45.6086 | 132864 | Candidate suitable | 0.07 | 1.9 | 101.3 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S125 | 02/20А | shifted_odt_derived_dd | 59.7935 | 44.7111 | 132864 | Water / NoData / Shadow | 0.07 | 1.9 | 101.3 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S126 | 03/20А | raw_aralfield_dd | 60.9286 | 45.6248 | 132505 | Candidate suitable | 0.05 | 0.97 | 46.2 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S126 | 03/20А | shifted_odt_derived_dd | 59.8131 | 44.7294 | 132505 | Water / NoData / Shadow | 0.05 | 0.97 | 46.2 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S127 | 04/20А | raw_aralfield_dd | 60.9836 | 45.6482 | 132479 | Existing vegetation | 6.52 | 0.16 | 5.7 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S127 | 04/20А | shifted_odt_derived_dd | 59.8636 | 44.7556 | 132479 | Water / NoData / Shadow | 6.52 | 0.16 | 5.7 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S128 | 05/20А | raw_aralfield_dd | 60.9878 | 45.6563 | 132522 | Existing vegetation | 31.29 | 0.14 | 4.18 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S128 | 05/20А | shifted_odt_derived_dd | 59.8688 | 44.7625 | 132522 | Water / NoData / Shadow | 31.29 | 0.14 | 4.18 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S129 | 06/20А | raw_aralfield_dd | 61.0188 | 45.6575 | 132912 | Candidate suitable | 1.17 | 0.18 | 5.64 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S129 | 06/20А | shifted_odt_derived_dd | 59.8922 | 44.7636 | 132912 | Water / NoData / Shadow | 1.17 | 0.18 | 5.64 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S130 | 07/20А | raw_aralfield_dd | 61.2131 | 45.7252 | 134796 | Candidate suitable | 1.29 | 0.87 | 38.1 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S130 | 07/20А | shifted_odt_derived_dd | 60.0659 | 44.8209 | 134796 | Water / NoData / Shadow | 1.29 | 0.87 | 38.1 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S131 | 08/20А | raw_aralfield_dd | 61.2212 | 45.7548 | 132714 | Existing vegetation | 2.42 | 0.02 | 0.7 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S131 | 08/20А | shifted_odt_derived_dd | 60.0977 | 44.8603 | 132714 | Water / NoData / Shadow | 2.42 | 0.02 | 0.7 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S132 | 09/20А | raw_aralfield_dd | 61.2579 | 45.8153 | 132652 | Candidate suitable | 16.42 | 0.28 | 8.84 |  | V5 samples a non-water class; this is diagnostic only, not coordinate proof. |
| S132 | 09/20А | shifted_odt_derived_dd | 60.1344 | 44.9208 | 132652 | Water / NoData / Shadow | 16.42 | 0.28 | 8.84 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S133 | 10/20А | raw_aralfield_dd | 61.2673 | 45.8474 | 132619 | Water / NoData / Shadow | 7.78 | 0.1 | 3.25 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S133 | 10/20А | shifted_odt_derived_dd | 60.1437 | 44.9529 | 132619 | Water / NoData / Shadow | 7.78 | 0.1 | 3.25 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S134 | 11/20А | raw_aralfield_dd | 61.2688 | 45.9452 | 132518 | Water / NoData / Shadow | 0.1 | 1.73 | 95.2 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |
| S134 | 11/20А | shifted_odt_derived_dd | 60.1452 | 45.0507 | 132518 | Water / NoData / Shadow | 0.1 | 1.73 | 95.2 |  | V5 samples this coordinate as water/nodata/shadow; this is diagnostic only, not coordinate proof. |

## Authority Review Template

Fill `outputs/science/v5_coordinate_authority_template.csv` with the selected source or a third-party GPS coordinate for each point. Leave a reference to the original evidence in `evidence_reference`.

Manual review fields are preserved on regeneration; running the science suite will not erase reviewer decisions already present in the template.

Allowed `selected_coordinate_source` values: `raw_aralfield_dd`, `shifted_odt_derived_dd`, `third_party_gps_dd`, or `unresolved`.

| S_Point | pit_code | raw_lon_aralfield_dd | raw_lat_aralfield_dd | shifted_lon_odt_dd | shifted_lat_odt_dd | coordinate_conflict_m | selected_coordinate_source | authoritative_lon_dd | authoritative_lat_dd | evidence_reference | authority_row_status | authority_validation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S124 | 01/20А | 60.8936 | 45.5885 | 59.7757 | 44.7045 | 131715 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S125 | 02/20А | 60.9121 | 45.6086 | 59.7935 | 44.7111 | 132864 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S126 | 03/20А | 60.9286 | 45.6248 | 59.8131 | 44.7294 | 132505 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S127 | 04/20А | 60.9836 | 45.6482 | 59.8636 | 44.7556 | 132479 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S128 | 05/20А | 60.9878 | 45.6563 | 59.8688 | 44.7625 | 132522 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S129 | 06/20А | 61.0188 | 45.6575 | 59.8922 | 44.7636 | 132912 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S130 | 07/20А | 61.2131 | 45.7252 | 60.0659 | 44.8209 | 134796 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S131 | 08/20А | 61.2212 | 45.7548 | 60.0977 | 44.8603 | 132714 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S132 | 09/20А | 61.2579 | 45.8153 | 60.1344 | 44.9208 | 132652 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S133 | 10/20А | 61.2673 | 45.8474 | 60.1437 | 44.9529 | 132619 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |
| S134 | 11/20А | 61.2688 | 45.9452 | 60.1452 | 45.0507 | 132518 |  |  |  |  | unresolved | Pending external GPS/ODT authority evidence. |

## Remaining Blockers

- Authoritative coordinates for `S124-S134` / profiles `01/20A-11/20A`.
- Coordinates for lab-only profiles `12/20A-21/20A`.
- Independent survival or planting outcome observations if the product is later upgraded beyond screening.
