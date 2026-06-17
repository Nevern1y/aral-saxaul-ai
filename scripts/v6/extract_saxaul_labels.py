# -*- coding: utf-8 -*-
"""Phase 3 — derive per-profile saxaul (Haloxylon) habitat labels.

This builds the *target* variable for the V6 suitability model. Label quality
caps model trustworthiness, so the design is deliberately conservative and
honest about what the field record does and does not say.

Two signals are fused:

1. Morphology ``landform_vegetation`` text (profiles_v6.csv) — a terse
   plant-COMMUNITY name. Crucial caveat: "no saxaul mentioned" does NOT mean
   "saxaul absent"; it means it was not part of the community name. We therefore
   never coerce silence into a hard negative.
2. DOCX narrative (per-profile field descriptions) — richer prose that records
   plantation attempts, failures ("сажали саксаул, ничего прижилось"), single
   specimens, and dominant stands. Each saxaul sentence is anchored to the
   nearest preceding ``Разрез N/YY`` header.

Outputs, per profile:

- ``saxaul_status`` — the honest ecological record:
    present_dominant   saxaul-dominated community AT the pit (фитобугры etc.)
    present            saxaul a recorded component AT the pit
    sparse_suppressed  single / low / suppressed natural specimens AT the pit
    present_nearby     saxaul seen in the surrounding landscape (watersheds,
                       dunes) but NOT in the pit's own community — spatial
                       mismatch, deliberately NOT treated as pit presence
    planted_survived   plantation that established
    planted            plantation attempt, survival not stated
    planted_failed     documented plantation failure (gold negative)
    absent_recorded    a community is named at the pit, saxaul not among it
    unknown            no usable vegetation statement

- ``label_role`` ∈ {positive, negative, exclude, uncertain} — the modeling sign
- ``label_strength`` ∈ {strong, weak, ""} and ``label_weight`` ∈ [0,1] — how much
  a row should count when training, so the assumption-laden rows are
  down-weighted rather than silently trusted or dropped.

Sign/strength/weight mapping (derived from status + landform):
    present_dominant/present/planted_survived → positive / strong / 1.0
    sparse_suppressed (natural, at pit)        → positive / weak   / 0.6
    planted_failed                             → negative / hard*  / 1.0
    absent_recorded + "без растительности"     → negative / hard*  / 1.0
    absent_recorded + vegetated community      → negative / weak   / 0.4
    present_nearby / planted(unknown)          → uncertain / —     / 0.0
    rice / agriculture / floodplain reed       → exclude   / —     / 0.0
    (*hard negatives are stored as strength="strong")

Design guards:
- "No saxaul mentioned" is never coerced into a confident absence.
- Hard negatives are restricted to DOCUMENTED failures and genuinely barren
  ground — NOT to saline/halophyte solonchaks, so the soil-chemistry signal is
  learned by the model rather than baked into the labels (avoids circularity).
- Presence is credited only when saxaul is recorded at the PIT, not merely in
  the surrounding scene.

Every label carries an evidence quote, source (morphology|narrative|both) and a
confidence level (high|medium|low). Nothing is imputed silently.

Output: data/canonical/saxaul_labels_v6.csv  + saxaul_labels_manifest.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
DOCX = BASE.parent / "Закл отчет по Приаралью 2012-2014, Пачикин, Козыбаева ..docx"

# --- keyword vocabularies -------------------------------------------------
SAXAUL = re.compile(r"саксаул|haloxylon", re.I)
PLANT = re.compile(r"посадк|посаж|сажал|насажден|саженц|высажен", re.I)
FAIL = re.compile(r"ничего\s+(?:не\s+)?прижил|не\s+прижил|неудачно|сгорел|погиб|испортил", re.I)
SURVIVE = re.compile(r"выжил|прижал|принял|хорошо\s+развит|в\s+фазе\s+цветени|молода\w*\s+поросл", re.I)
DOMINANT = re.compile(r"саксаулов\w*\s+растительност|растительность\s+саксаулов|"
                      r"саксаулов\w*\s+с|фитобугр", re.I)
# sparse / suppressed natural specimens (these are at-pit qualifiers)
SPARSE = re.compile(r"единичн\w*[^.]{0,30}саксаул|редким\w*\s+кустик\w*\s+саксаул|"
                    r"редкими\s+кустиками\s+саксаул|одиночн\w*\s+куст\w*\s+саксаул|"
                    r"единично[^.]{0,30}саксаул|низк\w*\s+саксаул|"
                    r"кое-где\s+сохранил\w*\s+куст\w*\s+саксаул|молод\w*\s+саксаул", re.I)
# saxaul seen only in the surrounding landscape, not at the pit
NEARBY = re.compile(r"на\s+водоразделах[^.]{0,60}саксаул|отдельные\s+саксауль|"
                    r"бугр\w*[^.]{0,40}саксаул", re.I)

# landform / community tokens for the modeling-role decision
RICE_AG = re.compile(r"рисов|\bчек\b|пашн|пахотн|распахан|залежь|посев|орошаем|борозд", re.I)
FLOODPLAIN = re.compile(r"пойма|тростник|болот|чингил", re.I)
BARREN = re.compile(r"без\s+растительност|такыр\s+без", re.I)  # genuinely bare ground only

RAZREZ = re.compile(r"Разрез\s+(\d+\s*[АA]?\s*/\s*\d{2})", re.I)
# a pit's field description is short; bound the block so the LAST profile cannot
# absorb the trailing experimental-station / abstract sections (EOF contamination)
BLOCK_WINDOW = 12
# clause describing the PIT's own vegetation (vs. scene-setting prose)
PIT_CLAUSE = re.compile(
    r"(Разрез\s+заложен[^.]*\.|Разрез\s+на\s[^.]*\.|"
    r"Растительность\s+саксаулов[^.]*\.|под\s+[^.]*растительность[ю]?[^.]*\.)",
    re.I,
)


def norm_id(value: str) -> str:
    text = str(value).strip().replace(" ", "")
    return text.replace("A", "А")  # latin A -> cyrillic А


# ---------------------------------------------------------------------------
# 1. Narrative mining: full per-pit text blocks (header -> next header)
# ---------------------------------------------------------------------------

def mine_narrative() -> dict[str, str]:
    """Return {pit_id: full_block_text} for every Разрез block in the report."""
    import docx

    doc = docx.Document(str(DOCX))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    headers = [(i, norm_id(RAZREZ.search(p).group(1)))
               for i, p in enumerate(paras) if RAZREZ.search(p)]
    blocks: dict[str, str] = {}
    for k, (i, pid) in enumerate(headers):
        nxt = headers[k + 1][0] if k + 1 < len(headers) else len(paras)
        end = min(nxt, i + BLOCK_WINDOW)  # never run past a sane pit-block length
        text = " ".join(paras[i:end])
        # keep the longest block if an id appears twice (richest description)
        if pid not in blocks or len(text) > len(blocks[pid]):
            blocks[pid] = text
    return blocks


def pit_vegetation_clause(block: str) -> str:
    """Extract the sentence(s) describing vegetation AT the pit."""
    return " ".join(m.group(0) for m in PIT_CLAUSE.finditer(block))


def classify_narrative(block: str) -> tuple[str, str, str]:
    """(status, confidence, evidence_quote) from a profile's full block text."""
    if not block:
        return "unknown", "low", ""
    pit = pit_vegetation_clause(block)
    saxaul_at_pit = bool(SAXAUL.search(pit))

    def first_sentence(pattern, scope):
        for sent in re.split(r"(?<=[.!?])\s+", scope):
            if pattern.search(sent):
                return sent.strip()[:240]
        return scope.strip()[:240]

    # plantation outcomes take precedence (most decisive evidence)
    if PLANT.search(block) and "саксаул" in block.lower():
        if FAIL.search(block):
            return "planted_failed", "high", first_sentence(FAIL, block)
        if SURVIVE.search(block):
            return "planted_survived", "high", first_sentence(SURVIVE, block)
        return "planted", "medium", first_sentence(PLANT, block)

    # natural saxaul — but only credit presence if it is AT the pit
    if saxaul_at_pit:
        if DOMINANT.search(pit):
            return "present_dominant", "high", first_sentence(DOMINANT, pit)
        if SPARSE.search(pit):
            return "sparse_suppressed", "medium", first_sentence(SPARSE, pit)
        return "present", "medium", first_sentence(SAXAUL, pit)

    # saxaul mentioned but not at the pit
    if SAXAUL.search(block):
        if NEARBY.search(block) or SPARSE.search(block):
            # decide nearby vs suppressed-at-pit by where it sits
            if NEARBY.search(block) and not saxaul_at_pit:
                return "present_nearby", "medium", first_sentence(NEARBY, block)
            return "sparse_suppressed", "low", first_sentence(SPARSE, block)
        return "present_nearby", "low", first_sentence(SAXAUL, block)
    return "unknown", "low", ""


# ---------------------------------------------------------------------------
# 2. Morphology labelling (terse community column = the pit community)
# ---------------------------------------------------------------------------

def classify_morphology(text: str) -> tuple[str, str, str]:
    """(status, confidence, evidence) from the landform_vegetation string.

    The part after '/' is the recorded community AT the pit, so any saxaul here
    is at-pit by construction (no 'nearby' case from this column).
    """
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return "unknown", "low", ""
    if PLANT.search(t) and "саксаул" in low:
        return "planted", "medium", t            # survival unknown from column
    if "саксаул" in low:
        if DOMINANT.search(low):
            return "present_dominant", "high", t
        if SPARSE.search(t):
            return "sparse_suppressed", "medium", t
        return "present", "medium", t
    return "absent_recorded", "low", t           # community named, no saxaul


# ---------------------------------------------------------------------------
# 3. Modeling sign / strength / weight from fused status + landform
# ---------------------------------------------------------------------------

STATUS_RANK = {  # which source wins when both speak; higher = more decisive
    "unknown": 0, "absent_recorded": 1, "present_nearby": 2, "present": 3,
    "sparse_suppressed": 4, "planted": 5, "planted_survived": 6,
    "present_dominant": 7, "planted_failed": 8,
}


def decide_role(status: str, landform: str) -> tuple[str, str, float, str]:
    """(label_role, label_strength, label_weight, reason)."""
    low = (landform or "").lower()

    # documented saxaul outcomes survive even on ex-farmland; only uninformative
    # agriculture/floodplain rows are excluded as non-candidate terrain.
    if status in {"absent_recorded", "unknown"}:
        if RICE_AG.search(low):
            return "exclude", "", 0.0, "agriculture/ploughed land, not candidate terrain"
        if FLOODPLAIN.search(low) and not SAXAUL.search(low):
            return "exclude", "", 0.0, "floodplain/reed/chingil wetland, not candidate terrain"

    if status in {"present_dominant", "present", "planted_survived"}:
        return "positive", "strong", 1.0, f"saxaul established at pit ({status})"
    if status == "sparse_suppressed":
        return "positive", "weak", 0.6, "natural sparse/suppressed saxaul at pit (marginal but supports it)"
    if status == "planted_failed":
        return "negative", "strong", 1.0, "documented plantation failure (gold negative)"
    if status == "absent_recorded" and BARREN.search(low):
        return "negative", "strong", 1.0, "genuinely barren ground, no vegetation"
    if status == "absent_recorded":
        return "negative", "weak", 0.4, "vegetated non-saxaul community (not confident absence)"
    # present_nearby, planted(unknown survival) -> ambiguous, do not train on
    return "uncertain", "", 0.0, f"ambiguous ({status}); excluded from training"


def main() -> None:
    profiles = pd.read_csv(CANON / "profiles_v6.csv", dtype=str, keep_default_na=False)
    profiles["pit_id"] = profiles["pit_id"].map(norm_id)

    blocks = mine_narrative()

    rows = []
    for _, p in profiles.iterrows():
        pid = p["pit_id"]
        landform = p["landform_vegetation"]

        m_status, m_conf, m_ev = classify_morphology(landform)
        block = blocks.get(pid, "")
        n_status, n_conf, n_ev = classify_narrative(block)

        # Fuse: narrative is richer prose. A confident plantation outcome
        # (failed/survived) always overrides the terse morphology guess. A
        # narrative 'present_nearby' must NOT be overridden by a morphology
        # 'present' (the column has no nearby/at-pit distinction), so when the
        # narrative says nearby we trust the spatial detail it carries.
        if n_status in {"planted_failed", "planted_survived"} and n_conf == "high":
            status, conf, source, evidence = n_status, n_conf, "narrative", n_ev
        elif n_status == "present_nearby" and m_status in {"present", "absent_recorded"}:
            status, conf, source, evidence = n_status, n_conf, "narrative", n_ev
        elif STATUS_RANK[n_status] > STATUS_RANK[m_status]:
            status, conf, source, evidence = n_status, n_conf, "narrative", n_ev
        elif STATUS_RANK[m_status] > STATUS_RANK[n_status]:
            status, conf, source, evidence = m_status, m_conf, "morphology", m_ev
        else:
            status, conf = m_status, m_conf
            source = "both" if block else "morphology"
            evidence = m_ev or n_ev

        role, strength, weight, reason = decide_role(status, landform)
        rows.append({
            "pit_id": pid,
            "src_year": p.get("src_year", ""),
            "lat_dd": p.get("lat_dd", ""),
            "lon_dd": p.get("lon_dd", ""),
            "in_aoi": p.get("in_aoi", ""),
            "has_coordinates": p.get("has_coordinates", ""),
            "saxaul_status": status,
            "label_role": role,
            "label_strength": strength,
            "label_weight": weight,
            "confidence": conf,
            "evidence_source": source,
            "role_reason": reason,
            "landform_vegetation": landform,
            "evidence_quote": evidence.replace("\n", " ")[:240],
            "has_narrative_block": bool(block),
        })

    out = pd.DataFrame(rows).sort_values("pit_id").reset_index(drop=True)
    out_path = CANON / "saxaul_labels_v6.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    def vc(col):
        return out[col].value_counts().to_dict()

    georef = out[out["has_coordinates"].isin(["True", "true", True])]
    trainable = out[out["label_weight"].astype(float) > 0]
    manifest = {
        "version": "V6-saxaul-labels",
        "n_profiles": int(len(out)),
        "status_counts": vc("saxaul_status"),
        "role_counts": vc("label_role"),
        "strength_counts": vc("label_strength"),
        "confidence_counts": vc("confidence"),
        "source_counts": vc("evidence_source"),
        "georeferenced": {
            "n": int(len(georef)),
            "role_counts": georef["label_role"].value_counts().to_dict(),
        },
        "trainable_georef": {
            "n": int(len(georef[georef["label_weight"].astype(float) > 0])),
            "positives": sorted(georef.loc[georef["label_role"] == "positive", "pit_id"]),
            "negatives_strong": sorted(
                georef.loc[(georef["label_role"] == "negative")
                           & (georef["label_strength"] == "strong"), "pit_id"]),
        },
        "positives": sorted(out.loc[out["label_role"] == "positive", "pit_id"]),
        "negatives_strong": sorted(
            out.loc[(out["label_role"] == "negative")
                    & (out["label_strength"] == "strong"), "pit_id"]),
        "uncertain": sorted(out.loc[out["label_role"] == "uncertain", "pit_id"]),
        "n_trainable": int(len(trainable)),
        "n_narrative_blocks": int(out["has_narrative_block"].sum()),
        "output": str(out_path),
    }
    (CANON / "saxaul_labels_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
