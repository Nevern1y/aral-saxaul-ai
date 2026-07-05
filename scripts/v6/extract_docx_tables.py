# -*- coding: utf-8 -*-
"""Phase 1 — lossless extraction of the appendix B lab tables from the report.

The source report (Пачикин/Козыбаева 2012-2014) stores per-profile laboratory
data across three appendix-B table families, each split over several
continuation pages:

    Б.1 / Б.2 / Б.3  — chemistry + physchem (humus, N, C:N, CO2, CaCO3, CaSO4,
                       exchangeable bases Ca/Mg/Na/K, mobile P2O5/K2O,
                       hydrolyzable N, pH)               [years 2012/2013/2014]
    Б.4 / Б.5 / Б.6  — water-extract salinity (sum of salts, HCO3, CO3, Cl,
                       SO4, Ca, Mg, Na, K; cells are "%/meq")  [2012/2013/2014]
    Б.7 / Б.8 / Б.9  — granulometry (hygroscopic water + sand/silt/clay
                       fractions, physical clay)               [2012/2013/2014]

Tables are grouped by caption ("Таблица Б.N" + "Продолжение таблицы Б.N").
Header rows and the "1 2 3 ..." enumeration row are dropped. Profile IDs in the
first column may repeat (vertical cell merges) — that is expected; each output
row is one (profile, depth) sample.

Output: three long CSVs under data/interim/ plus a provenance JSON. No values
are reinterpreted; numbers are written verbatim as text so downstream
normalization (Phase 2) is the single place that parses them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

BASE = Path(__file__).resolve().parent.parent.parent
DOCX = BASE.parent / "Закл отчет по Приаралью 2012-2014, Пачикин, Козыбаева ..docx"
INTERIM = BASE / "data" / "interim"

# Caption letter -> (family, year, output key)
FAMILY = {
    "Б.1": ("chem_physchem", 2012),
    "Б.2": ("chem_physchem", 2013),
    "Б.3": ("chem_physchem", 2014),
    "Б.4": ("water_extract", 2012),
    "Б.5": ("water_extract", 2013),
    "Б.6": ("water_extract", 2014),
    "Б.7": ("granulometry", 2012),
    "Б.8": ("granulometry", 2013),
    "Б.9": ("granulometry", 2014),
}

# Canonical column names per family (positional, matching the report layout).
COLS = {
    "chem_physchem_17": [  # Б.1 (2012): 17 columns
        "pit_id", "depth_cm", "humus_pct", "n_total_pct", "c_n_ratio", "co2_pct",
        "exch_ca", "exch_mg", "exch_na", "exch_k", "exch_sum",
        "mg_pct_of_sum", "na_pct_of_sum", "p2o5_mobile", "k2o_mobile",
        "n_hydrolyzable", "ph_water",
    ],
    "chem_physchem_18": [  # Б.2/Б.3 (2013/2014): 18 columns (adds CaCO3, CaSO4)
        "pit_id", "depth_cm", "humus_pct", "n_total_pct", "c_n_ratio", "co2_pct",
        "caco3_pct", "caso4_pct",
        "exch_ca", "exch_mg", "exch_na", "exch_k", "exch_sum",
        "na_pct_of_sum", "p2o5_mobile", "k2o_mobile", "n_hydrolyzable", "ph_water",
    ],
    "water_extract_11": [  # Б.4/Б.5/Б.6: 11 columns, ion cells are "%/meq"
        "pit_id", "depth_cm", "sum_salts", "alk_hco3", "alk_co3",
        "cl", "so4", "ca", "mg", "na", "k",
    ],
    "granulometry_10": [  # Б.7/Б.8/Б.9: 10 columns
        "pit_id", "depth_cm", "hygroscopic_water_pct",
        "sand_1_0p25", "sand_0p25_0p05", "silt_0p05_0p01", "silt_0p01_0p005",
        "silt_0p005_0p001", "clay_lt0p001", "physical_clay_lt0p01",
    ],
}


def iter_block_items(document):
    """Yield paragraphs and tables in document order."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield "p", Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield "t", Table(child, document)


def caption_letter(text: str) -> str | None:
    m = re.search(r"Таблиц[аы]?\s*(Б\.?\s*\d)", text)
    if not m:
        m = re.search(r"Продолжение таблицы\s*(Б\.?\s*\d)", text)
    if not m:
        return None
    raw = m.group(1).replace(" ", "")
    raw = raw if "." in raw else raw[0] + "." + raw[1:]
    return raw


def is_header_or_enum(cell0: str, row_cells: list[str]) -> bool:
    c = cell0.strip().lower()
    if c.startswith("№") or "разрез" in c:
        return True
    # enumeration row "1 | 2 | 3 ..."
    nonempty = [x for x in row_cells if x.strip()]
    if nonempty and all(re.fullmatch(r"\d{1,2}", x.strip()) for x in nonempty):
        return True
    return False


def normalize_split_columns(cells: list[str], family: str, ncols: int) -> tuple[list[str], int]:
    """Collapse a duplicated split cell so a mis-sized table maps to a known layout.

    One Б.2 (2013) chem_physchem table renders with 19 physical columns instead of
    18 because a single logical column (the enumeration row shows label "15" twice)
    is split into two identical adjacent cells. Without this, its data rows fall into
    the positional `cX` fallback and 4 profiles (01/13, 12/13, 16/13, 26/13) lose ALL
    chemistry downstream. When the two split cells carry identical values we drop the
    second and re-map to the canonical 18-column names; if they ever differ we leave
    the row untouched so the positional fallback still preserves the raw data for audit.

    Returns (possibly-collapsed cells, effective column count).
    """
    if family == "chem_physchem" and ncols == 19 and len(cells) == 19:
        # physical indices 14 and 15 are the duplicated column (both labelled "15")
        if cells[14].strip() == cells[15].strip():
            return cells[:15] + cells[16:], 18
    return cells, ncols


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    document = docx.Document(str(DOCX))

    # Walk blocks, tag each table with the most recent B-caption.
    tables_by_letter: dict[str, list[Table]] = {}
    current_letter: str | None = None
    recent_text: list[str] = []
    for kind, obj in iter_block_items(document):
        if kind == "p":
            t = obj.text.strip()
            if t:
                recent_text.append(t)
                let = caption_letter(t)
                if let:
                    current_letter = let
        else:
            # find caption from recent paragraphs if the immediate one set it
            let = None
            for txt in reversed(recent_text[-4:]):
                let = caption_letter(txt)
                if let:
                    break
            use = let or current_letter
            if use in FAMILY:
                tables_by_letter.setdefault(use, []).append(obj)
            recent_text = []

    # Build long records per family.
    records: dict[str, list[dict]] = {"chem_physchem": [], "water_extract": [], "granulometry": []}
    provenance: dict = {"source": DOCX.name, "tables": []}

    for letter, tabs in sorted(tables_by_letter.items()):
        family, year = FAMILY[letter]
        for t in tabs:
            ncols = len(t.columns)
            kept = 0
            for r in t.rows:
                cells = [c.text.strip().replace("\n", " ") for c in r.cells]
                if not cells:
                    continue
                if is_header_or_enum(cells[0], cells):
                    continue
                if not cells[0].strip():  # no profile id on this row
                    continue
                # Collapse a duplicated split cell (e.g. the 19-col Б.2 chem table) so
                # the row maps to the canonical layout instead of the positional fallback.
                cells, eff_cols = normalize_split_columns(cells, family, ncols)
                colnames = COLS.get(f"{family}_{eff_cols}")
                if colnames and len(cells) >= len(colnames):
                    rec = {colnames[i]: cells[i] for i in range(len(colnames))}
                else:
                    # unexpected column count: store positionally for audit
                    rec = {f"c{i}": cells[i] for i in range(len(cells))}
                    rec["pit_id"] = cells[0]
                rec["src_table"] = letter
                rec["src_year"] = year
                records[family].append(rec)
                kept += 1
            provenance["tables"].append(
                {"letter": letter, "family": family, "year": year,
                 "rows": len(t.rows), "cols": ncols, "data_rows_kept": kept}
            )

    import pandas as pd
    summary = {}
    for family, recs in records.items():
        df = pd.DataFrame(recs)
        out = INTERIM / f"docx_{family}_long.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        summary[family] = {"rows": len(df), "unique_pits": int(df["pit_id"].nunique()) if len(df) else 0,
                           "file": str(out)}
        print(f"{family}: {len(df)} rows, {summary[family]['unique_pits']} pits -> {out.name}")

    prov_path = INTERIM / "docx_extraction_provenance.json"
    provenance["summary"] = summary
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"provenance -> {prov_path.name}")


if __name__ == "__main__":
    main()
