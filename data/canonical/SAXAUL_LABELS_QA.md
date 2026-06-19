# Saxaul habitat labels — QA notes (Phase 3)

Target variable for the V6 suitability model. Source: morphology
`landform_vegetation` column + per-pit DOCX narrative blocks. Script:
`scripts/v6/extract_saxaul_labels.py`. Output: `saxaul_labels_v6.csv`.

## Label scheme

- `saxaul_status` — honest ecological record (present_dominant, present,
  sparse_suppressed, present_nearby, planted_survived, planted, planted_failed,
  absent_recorded, unknown).
- `label_role` ∈ {positive, negative, exclude, uncertain} — modeling sign.
- `label_strength` ∈ {strong, weak} + `label_weight` ∈ [0,1] — training weight,
  so assumption-laden rows are down-weighted, not silently trusted or dropped.

## Decisive labels (georeferenced unless marked)

| pit | role | status | evidence |
|-----|------|--------|----------|
| 13/14 *(no coord)* | positive/strong | present_dominant | «саксауловая, фитобугры» |
| 23/14 | positive/strong | present_dominant | «эфемерово-саксауловой растительностью» |
| 4А/12 | positive/strong | present | shrub layer incl. saxaul at pit |
| 8А/12 | positive/weak | sparse_suppressed | «единичными экземплярами саксаула» |
| 12/14, 14/13, 24/13 | positive/weak | sparse_suppressed | natural sparse specimens at pit |
| 13/13 | negative/strong | planted_failed | «Сажали саксаул, ничего прижилось» (GOLD) |
| 24/14 | negative/strong | planted_failed | «Посадка неудачно прижилась … землю испортили» (GOLD) |
| 08/14 | negative/strong | absent_recorded | «Такыр без растительности» (barren) |
| 28/13, 29/13 | uncertain | present_nearby | saxaul on surrounding watersheds, NOT at pit |

Plus 50 weak negatives (vegetated non-saxaul communities, weight 0.4) and 14
excluded rows (rice/agriculture/floodplain — non-candidate terrain).

## Key scientific guards (audited)

1. **No silence→absence.** "Saxaul not mentioned" is a *weak* negative
   (weight 0.4), never a confident absence — the community column is a name, not
   a census.
2. **At-pit gating.** Presence is credited only when saxaul is in the pit's own
   community. 28/13 / 29/13 mention saxaul in the surrounding landscape and are
   therefore `uncertain`, not positives (prevents spatial-mismatch false
   positives that would poison a 6-positive training set).
3. **No salinity prior baked into labels.** Hard negatives are restricted to
   DOCUMENTED plantation failures and genuinely barren ground ("без
   растительности") — NOT to saline/halophyte solonchaks. The soil-chemistry
   signal must be *learned* by the model, not assumed in the target (avoids
   circularity).
4. **EOF contamination fixed.** Per-pit narrative blocks are bounded to a 12-
   paragraph window so the last profile (37/14) cannot absorb the trailing
   experimental-station section (the "76.2 % survival" figure belongs to the
   research plot, not pit 37/14).
5. **Gold negative georeferenced transparently.** 24/14 has "н/д" coordinates in
   the source; recovered as "50 m east of 23/14" (documented), flagged
   `coord_provenance="recovered_offset"` for a Phase 4/7 sensitivity check.

## Limitations (carried forward to Phase 7 reporting)

- 6 georeferenced positives is small; weak positives (sparse specimens) dominate.
  Phase 6 will train BOTH a strict variant (strong labels only) and a full
  variant (with weak, down-weighted) and report the gap.
- Labels are expert-coded from field prose, not a GPS census; confidence is
  tagged per row.
- Temporal mismatch: vegetation observed 2012-2014 vs satellite ~2024.
