# REPORT MINING AUDIT — Закл. отчёт по Приаралью 2012–2014 (Пачикин, Козыбаева)

**Scope:** data-mining audit only. No pipeline/ML code was modified. The goal is to
identify report-derived data that is **not yet used** in the V6 suitability dataset and
that could plausibly improve saxaul (*Haloxylon*) suitability prediction.

**Date of audit:** 2026-06-19
**Auditor role:** research / data-mining (read-only)
**Primary deliverable:** this file.

---

## 0. Method & sources inspected

| Source | Path | Status |
|---|---|---|
| Extraction provenance | `data/interim/docx_extraction_provenance.json` | read |
| Chem/physchem long | `data/interim/docx_chem_physchem_long.csv` (441 rows, 76 pits w/ numeric id) | read |
| Water extract long | `data/interim/docx_water_extract_long.csv` (228 rows) | read |
| Granulometry long | `data/interim/docx_granulometry_long.csv` (284 rows, 95 pit-tables) | read |
| Morphology 2012 | `data/raw_external/morph_2012.csv` (69 layers) | read |
| Morphology 2012 degr. | `data/raw_external/morph_2012_degradation.csv` (53 layers) | read |
| Morphology 2013 | `data/raw_external/morph_2013.csv` (63 layers) | read |
| Morphology 2014 | `data/raw_external/morph_2014_beloe.csv` (203 layers) | read |
| Chem 2014 final/karaterren | `data/raw_external/chem_2014_*.csv` | read |
| Canonical profiles | `data/canonical/profiles_v6.csv` (76 pits) | read |
| Canonical labels | `data/canonical/saxaul_labels_v6.csv` (76 pits) | read |
| Canonical ML dataset | `data/canonical/ml_dataset_v6.csv` (70 rows, 88 cols) | read |
| Root field tables | `AralField(Sheet1).csv`, `Результаты EC,TDS,pH…csv`, `результаты Арал(Лист2).csv`, `мехсостав 2020,2021…csv` | read |

The DOCX source (`Закл отчет по Приаралью 2012-2014, Пачикин, Козыбаева ..docx`) is **not
physically present** in the repo; only the derived extraction CSVs and provenance JSON are
available. All findings below are therefore grounded in already-extracted derivatives.

> ⚠️ **Provenance caveat.** `docx_extraction_provenance.json` reports `unique_pits: 132` for
> chem/water families, but only **76 pit IDs beginning with a digit** appear in the long
> CSVs. The extra count comes from soil-type *name* rows (e.g. "Примитивная приморская
> почва") being miscounted as pit IDs during extraction. The true unique georeferenced pit
> population is **76**, fully matched across morphology, chemistry and the canonical profiles
> (see §3). This discrepancy is a data-quality note, not a coverage gap.

---

## 1. Coverage map: what is extracted vs. what reaches V6

Appendix tables Б.1–Б.9 of the report were extracted into three families:

| Appendix | Family | Years | In `ml_dataset_v6`? | Notes |
|---|---|---|---|---|
| Б.1 / Б.2 / Б.3 | chem_physchem | 2012 / 2013 / 2014 | **Yes** | humus, N, C/N, CaCO₃, CaSO₄, exch. cations, P₂O₅, K₂O, pH → cols 31–56 |
| Б.4 / Б.5 / Б.6 | water_extract | 2012 / 2013 / 2014 | **Yes** | sum salts, HCO₃, Cl, SO₄, Ca, Mg, Na → cols 17–30 |
| Б.7 / Б.8 / Б.9 | granulometry | 2012 / 2013 / 2014 | **Yes** | sand/silt/clay/physical-clay/hygroscopic → cols 57–66 |
| (narrative морфология) | morphology | 2012–2014 | **Partial** | only `landform_vegetation` + label text reach V6; layer descriptors unused |

Every numeric chemistry/granulometry feature in the appendices is **already aggregated**
into top-layer (`top_*`) and whole-profile (`prof_*`) columns. The chemical mining of the
report is essentially complete.

The **unmined reservoir is the per-layer morphological description** (cols 6–11 of the morph
files: genetic horizon, colour, structure, HCl effervescence, inclusions, moisture). These
are currently collapsed to a single free-text `landform_vegetation` string and a layer count
(`n_layers`); none of their structured signal is featurised.

---

## 2. Un-mined / under-mined report sections

| # | Report section / column | Where it lives now | Used in V6? | Why it matters for saxaul |
|---|---|---|---|---|
| U1 | **Moisture profile** (`Влажность` per layer) | morph cols 11 | No | Depth at which the profile becomes свежий/влажный/сырой is a field proxy for capillary fringe / groundwater reach — a primary control on *Haloxylon* establishment on the dried seabed. |
| U2 | **Salt-by-depth** (`соли с N см`, jilki/druzy in `Включения`) | morph col 10 + HCl col 9 text | No (only bulk salt % in V6) | *Depth* to the salic horizon, not just bulk salinity, governs whether seedling roots escape the toxic crust. |
| U3 | **Carbonate reaction depth** (`Вскипание от HCl`) | morph col 9 | No | "с поверхности" vs "слабо/не вскипает" + horizon "к" suffix marks carbonate accumulation depth → rooting and pH context. |
| U4 | **Rust mottling / gleying** (`ржавые пятна`, horizon `g`) | morph col 7 (colour) + col 6 (horizon) | No | Mottling/gley = seasonal waterlogging & anoxia, strongly *negative* for saxaul; present in 56% of pits. |
| U5 | **Marine relict substrate** (`ракушечник`, ракушки) | morph col 10 | No | Shell-rich primitive seabed soils (14% of pits) flag the youngest, least-developed terrain. |
| U6 | **Genetic horizon sequence** (`Индекс горизонта`: A1/AB/B/BC/C, suffixes к, сн, зс, g, пах) | morph col 6 | No (only `n_layers`) | Encodes soil maturity, salinity (зс/сн), ploughing (пах) — richer than a raw layer count. |
| U7 | **Soil structure / consistence** (`Структура и сложение`: корка, плитчатый, глыбистый…) | morph col 8 | No | Surface crust ("корка") and platy/blocky density impede saxaul seedling emergence. |
| U8 | **Profile depth / horizon thickness** | morph cols 4–5 | Partial (n_layers only) | Max described depth ranges 40–110 cm (median 90); solum depth is a rooting-volume proxy. |
| U9 | **Soil colour** (`Цвет почвы`) | morph col 7 | No | Munsell-like lightness relates to humus/surface albedo; complements the optical RS bands. |
| U10 | **2020/2021 re-survey field tables** (root CSVs) | `Результаты EC…`, `результаты Арал(Лист2)`, `мехсостав 2020,2021` | No | A *later* sampling campaign (EC/TDS/pH/humus + granulometry by profile) not tied into V6 — potential out-of-period validation or extra pits. |
| U11 | **Total nitrogen `Азот общий`** in `chem_2014_final` | `data/raw_external/chem_2014_final.csv` | Partly (N in V6 cols 33–34) | Confirm 2014 N values flow through; chem_2014_final also carries per-layer salts that may differ from Б.6 aggregation. |

---

## 3. Georeferenced points / saxaul evidence — completeness check

- **Point coverage is complete.** All 76 numeric pit IDs in the morphology files are present
  in `profiles_v6.csv`; `comm -23` of morph-pits vs profile-pits returns the empty set. No
  new georeferenced pit can be recovered from the morphology tables.
- **Saxaul mentions are already captured.** Every `саксаул`/`посадка саксаула` occurrence in
  the morph files (pits `8А/12`, `24/13`, `13/14`, `23/14`, `24/14`, plus narrative pits
  `4А/12`, `12/14`, `14/13`, `25/14`, `28/13`, `29/13`) maps to an existing row in
  `saxaul_labels_v6.csv` with an appropriate status (`present_dominant`, `sparse_suppressed`,
  `planted_failed`, `present_nearby`). No un-labelled presence/absence evidence was found.
- **Residual opportunity (low):** pits `03/14`, `04/14`, `10/13`, `15/14`, `13/14`, `25/14`
  have chemistry but are dropped from the 70-row ML set (mostly missing coordinates or
  exclude-role terrain). They are correctly excluded for training but could seed a
  *coordinate-recovery* pass if the report's locality text were geocoded.

**Conclusion:** the high-value mining target is **feature enrichment of existing pits**, not
new points.

---

## 4. Candidate features for V6 (ranked)

Feasibility percentages below = share of the 76 pits for which the feature is directly
derivable from the morphology layers (measured during this audit).

| Rank | Candidate feature | Derivation | Coverage | Scientific rationale | Confidence | Provenance |
|---|---|---|---|---|---|---|
| 1 | `depth_to_moist_cm` | top depth of first layer whose `Влажность` ∈ {слабоувлажн., свежий, влажный, сырой} | **73%** (56/76) | Shallow capillary moisture is the dominant positive driver of natural saxaul on the dried Aral floor; deep/absent moisture → arid stress. | Medium-High | morph col 11; `Источник` page refs per layer |
| 2 | `depth_to_salt_cm` | top depth of first layer with salt token in `Включения`/HCl text (соли, жилки/друзы солей) | **59%** (45/76) | Depth to salic horizon separates leached, plant-available topsoils from saline crusts; complements bulk salt %. | Medium-High | morph col 10 + col 9 |
| 3 | `rust_mottling_flag` / `gley_flag` | colour contains `ржав`/`сиз` or horizon suffix `g` | **56%** (43/76) | Mottling/gleying marks seasonal waterlogging & anoxia — a strong **negative** for *Haloxylon*. | High | morph cols 6–7 |
| 4 | `hcl_effervescence_class` | map `Вскипание` → {surface, weak, none} | ~96% non-null | Carbonate at surface vs depth shapes pH and rooting; "не вскипает" sandy pits are the freshest aeolian substrates. | Medium | morph col 9 |
| 5 | `surface_crust_flag` | top layer `Структура` contains `корка` | high | A hard surface crust physically blocks saxaul seedling emergence; recurring descriptor in primitive seabed soils. | Medium | morph col 8 |
| 6 | `marine_shell_flag` | `Включения` contains `ракуш` | 14% (11/76) | Shell-rich relict seabed = youngest, least-weathered terrain; a maturity/age proxy. | Medium | morph col 10 |
| 7 | `solum_depth_cm` | max `Глубина слоя ниж` per pit | 100% | Rooting-volume proxy; ranges 40–110 cm (median 90). | Medium | morph cols 4–5 |
| 8 | `horizon_salic_flag` / `horizon_ploughed_flag` | horizon code suffix `зс`/`сн` (salic) or `пах`/`п` (ploughed) | 100% (code present) | Distinguishes naturally salic and anthropogenically disturbed profiles from candidate terrain. | Medium | morph col 6 |
| 9 | `topsoil_lightness_ord` | ordinal from `Цвет почвы` (светло-серый…бурый…) | high | Surface albedo/humus proxy; ground-truth complement to optical RS indices already in V6. | Low-Medium | morph col 7 |

All candidates are **per-pit, depth-aware summaries** of data already present in
`data/raw_external/morph_*.csv`, so they integrate at the same grain as the existing
`top_*`/`prof_*` columns without new fieldwork.

---

## 5. Provenance & confidence summary

- **Source authority:** Пачикин & Козыбаева final report, appendices Б.1–Б.9 + morphological
  pit descriptions; each morph layer row carries an explicit `Источник (Стр. / Разрез №)`
  page reference, so every proposed feature is traceable to a report page.
- **Confidence basis:**
  - *High* — features built from explicit categorical descriptors present for most pits
    (rust/gley flag, horizon codes).
  - *Medium* — depth-threshold features where the descriptor vocabulary is consistent but
    occasionally fuzzy ("влажный после дождя" vs "влажный").
  - *Low* — colour ordinalisation (subjective field wording).
- **Known risks:** (a) morphology coordinates for several 2014 "Белое"-area pits are `н/д`
  and would not survive the V6 AOI/coordinate filter; (b) free-text vocabulary needs a
  controlled mapping table before featurisation; (c) the 2020/2021 root CSVs (U10) are a
  *different* campaign and must not be silently merged with 2012–2014 labels.

---

## 6. Recommended next actions (for a future pipeline task — not done here)

1. Build a controlled-vocabulary mapping for `Влажность`, `Вскипание`, `Включения`, and
   horizon suffixes (one-time lookup table).
2. Derive per-pit `depth_to_moist_cm`, `depth_to_salt_cm`, `rust_mottling_flag`,
   `solum_depth_cm` from `morph_*.csv` and join on `pit_id` into the profile builder.
3. Ablate the new morphological block against the current RS+chem feature set to confirm lift
   on the saxaul suitability target before adopting.
4. Separately evaluate the 2020/2021 root tables (U10) as an out-of-period validation set.

*No code, pipeline, or canonical dataset was modified in producing this audit.*
