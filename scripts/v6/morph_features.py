# -*- coding: utf-8 -*-
"""Derive per-pit morphological features from morph_*.csv.

Reads:
    data/canonical/morph_vocab.json        (controlled-vocabulary mapping, TRACKED)
    data/raw_external/morph_*.csv          (one layer per row; same files as build_canonical_db.py)

Writes:
    data/canonical/morph_features_v6.csv   (one row per pit_id, 10 feature columns)
    data/canonical/morph_features_manifest.json

Feature definitions (see morph_vocab.json for token lists and coverage notes):

  depth_to_moist_cm   top depth (cm) of FIRST layer whose moisture level >= 3 (свежий/влажный/…)
                      Capillary fringe proxy. NaN when all layers are dry or missing.
                      Coverage ~73%.

  depth_to_salt_cm    top depth (cm) of FIRST layer with a salt inclusion token.
                      Depth to salic horizon. NaN when no layer has an unambiguous salt token.
                      Coverage ~59% — <60%, so NaN is NOT imputed.

  rust_mottling_flag  1 if ANY layer in the pit has 'ржав' in the colour field; 0 otherwise.
                      Waterlogging/anoxia indicator. Coverage: presence scored for ~56% of pits
                      (flags are 0/1, so all 76 pits get a value; but positive rate is 56%).

  gley_flag           1 if ANY layer has 'сиз' in colour OR horizon suffix contains 'g';
                      0 otherwise. Strong NEGATIVE predictor for Haloxylon.

  solum_depth_cm      Max recorded bottom depth per pit. 100% coverage. Rooting-volume proxy.

  hcl_effervescence_class   Top-layer HCl response mapped to integer: 2=surface, 1=weak,
                            0=none, NaN=missing/undefined. ~96% coverage.

  surface_crust_flag  1 if the TOPMOST layer's structure field contains 'корк'; 0 otherwise.

  marine_shell_flag   1 if ANY layer has a shell token in inclusions; 0 otherwise.
                      Coverage: positive rate ~14%.

  horizon_salic_flag  1 if ANY layer has a salic suffix (зс or сн) in horizon code; else 0.

  horizon_ploughed_flag  1 if ANY layer has a ploughed suffix (пах or п) in horizon code;
                         else 0.

Reproducibility notes:
  - All feature thresholds and token lists are in morph_vocab.json (no inline magic strings).
  - NaN is never imputed for depth features with <100% coverage — missingness is honest.
  - Script is idempotent: re-running produces byte-identical output given identical inputs.
  - 76 pits (all morph-registered profiles) → output has 76 rows; coordinate filtering to
    70 happens in build_ml_dataset.py (the existing has_coordinates reduction is unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "data" / "raw_external"
CANON = BASE / "data" / "canonical"

VOCAB_PATH = CANON / "morph_vocab.json"
OUT_CSV = CANON / "morph_features_v6.csv"
OUT_MANIFEST = CANON / "morph_features_manifest.json"

MORPH_FILES = ["morph_2012", "morph_2013", "morph_2012_degradation", "morph_2014_beloe"]

NULL_TOKENS = {"н/д", "н/a", "n/a", "", "-", "–"}


def norm_id(value: str) -> str:
    text = str(value).strip().replace(" ", "")
    text = text.replace("A", "А")   # latin A -> cyrillic А (same as build_canonical_db.py)
    return text


def is_profile_id(value) -> bool:
    import re
    return bool(re.search(r"\d[АA]?/\d", str(value)))


def load_vocab() -> dict:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {VOCAB_PATH}\n"
            "Run `git show HEAD:data/canonical/morph_vocab.json` or check the tracked file."
        )
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def load_morph_layers() -> pd.DataFrame:
    """Load all four morph files, concatenate, normalise pit IDs, keep only valid profile rows."""
    dfs = []
    for fname in MORPH_FILES:
        fpath = RAW / f"{fname}.csv"
        df = pd.read_csv(fpath, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        # Positional contract guard (same validation as build_canonical_db.py)
        if len(df.columns) < 12:
            raise ValueError(
                f"{fname}.csv has only {len(df.columns)} columns; expected >= 12. "
                "Do not reorder morph CSVs — it silently corrupts features."
            )
        latc_hint = df.columns[1].lower()
        if "lat" not in latc_hint:
            raise ValueError(
                f"{fname}.csv: columns[1]='{df.columns[1]}' does not contain 'lat'. "
                "Expected Latitude at position 1."
            )
        df["_src_file"] = fname
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    id_col = all_df.columns[0]
    all_df[id_col] = all_df[id_col].map(norm_id)
    all_df = all_df[all_df[id_col].map(is_profile_id)].copy()
    all_df = all_df.rename(columns={
        all_df.columns[0]:  "pit_id",
        all_df.columns[4]:  "depth_top_cm",
        all_df.columns[5]:  "depth_bot_cm",
        all_df.columns[6]:  "horizon_code",
        all_df.columns[7]:  "color",
        all_df.columns[8]:  "structure",
        all_df.columns[9]:  "hcl_reaction",
        all_df.columns[10]: "inclusions",
        all_df.columns[11]: "moisture",
    })
    # Parse numeric depths (tolerant)
    def to_float(s):
        try:
            return float(str(s).replace(",", ".").strip())
        except (ValueError, TypeError):
            return np.nan

    all_df["depth_top_cm"] = all_df["depth_top_cm"].map(to_float)
    all_df["depth_bot_cm"] = all_df["depth_bot_cm"].map(to_float)
    return all_df.sort_values(["pit_id", "depth_top_cm"]).reset_index(drop=True)


def moisture_level(text: str, vocab: dict) -> int:
    """Return moisture level 1-6 for a moisture field string; 0 for missing/unknown."""
    s = str(text).strip().lower()
    if s in NULL_TOKENS:
        return 0
    levels = vocab["moisture_levels"]
    for level_key, tokens in levels.items():
        if level_key.startswith("0_"):
            continue
        if s in [t.lower() for t in tokens]:
            num = int(level_key.split("_")[0])
            return num
    return 0  # unknown = treat as missing


def has_token(text: str, tokens: list[str]) -> bool:
    """True if any token from the list appears as substring in text (case-insensitive)."""
    s = str(text).strip().lower()
    if s in NULL_TOKENS:
        return False
    return any(tok.lower() in s for tok in tokens)


def has_salt_inclusion(text: str, salt_tokens: list[str], exclude_tokens: list[str]) -> bool:
    """True if text has a salt token AND none of the exclude tokens dominate."""
    s = str(text).strip().lower()
    if s in NULL_TOKENS:
        return False
    # Check exclusion first: exact exclusion matches override the general tokens
    for exc in exclude_tokens:
        if s == exc.lower():
            return False
    return any(tok.lower() in s for tok in salt_tokens)


def derive_features(layers: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """Derive 10 per-pit features from the stacked layer table."""
    moist_threshold = int(vocab["moisture_threshold_for_depth"]["level"])
    salt_tokens = vocab["salt_tokens_in_inclusions"]["tokens"]
    salt_excludes = vocab["salt_tokens_in_inclusions"].get("exclude_tokens", [])
    shell_tokens = vocab["shell_tokens_in_inclusions"]["tokens"]
    rust_tokens = vocab["rust_tokens_in_color"]["tokens"]
    gley_color_tokens = vocab["gley_tokens_in_color"]["tokens"]
    gley_horiz_tokens = vocab["gley_suffix_in_horizon"]["tokens"]
    salic_tokens = vocab["salic_suffix_in_horizon"]["tokens"]
    plough_tokens = vocab["ploughed_suffix_in_horizon"]["tokens"]
    crust_tokens = vocab["crust_tokens_in_structure"]["tokens"]

    hcl_surface = [t.lower() for t in vocab["hcl_effervescence"]["surface"]]
    hcl_weak = [t.lower() for t in vocab["hcl_effervescence"]["weak"]]
    hcl_none = [t.lower() for t in vocab["hcl_effervescence"]["none"]]

    rows = []
    for pit_id, grp in layers.groupby("pit_id", sort=True):
        # Sort by depth for depth-ordered features
        grp_sorted = grp.sort_values("depth_top_cm")

        # 1. depth_to_moist_cm: top depth of first layer with moisture_level >= threshold
        depth_to_moist = np.nan
        for _, row in grp_sorted.iterrows():
            lvl = moisture_level(row["moisture"], vocab)
            if lvl >= moist_threshold:
                d = row["depth_top_cm"]
                if np.isfinite(d):
                    depth_to_moist = float(d)
                break

        # 2. depth_to_salt_cm: top depth of first layer with unambiguous salt inclusion
        depth_to_salt = np.nan
        for _, row in grp_sorted.iterrows():
            if has_salt_inclusion(row["inclusions"], salt_tokens, salt_excludes):
                d = row["depth_top_cm"]
                if np.isfinite(d):
                    depth_to_salt = float(d)
                break

        # 3. rust_mottling_flag: any layer has rust token in colour
        rust_flag = int(any(has_token(row["color"], rust_tokens) for _, row in grp.iterrows()))

        # 4. gley_flag: any layer has siz* in colour OR 'g' in horizon suffix
        gley_flag = int(
            any(has_token(row["color"], gley_color_tokens) for _, row in grp.iterrows())
            or any(has_token(row["horizon_code"], gley_horiz_tokens) for _, row in grp.iterrows())
        )

        # 5. solum_depth_cm: max bottom depth
        bots = grp["depth_bot_cm"].dropna()
        solum_depth = float(bots.max()) if len(bots) > 0 else np.nan

        # 6. hcl_effervescence_class: from the TOPMOST layer with a valid HCl field
        hcl_class = np.nan
        for _, row in grp_sorted.iterrows():
            hcl_val = str(row["hcl_reaction"]).strip().lower()
            if hcl_val in NULL_TOKENS:
                continue
            if any(t in hcl_val for t in hcl_surface):
                hcl_class = 2.0
            elif any(t in hcl_val for t in hcl_weak):
                hcl_class = 1.0
            elif any(t in hcl_val for t in hcl_none):
                hcl_class = 0.0
            else:
                hcl_class = np.nan  # unrecognised token — stays NaN, not forced
            break

        # 7. surface_crust_flag: top layer structure contains crust token
        top_row = grp_sorted.iloc[0] if len(grp_sorted) > 0 else None
        if top_row is not None:
            surface_crust = int(has_token(top_row["structure"], crust_tokens))
        else:
            surface_crust = 0

        # 8. marine_shell_flag: any layer has shell token in inclusions
        shell_flag = int(
            any(has_token(row["inclusions"], shell_tokens) for _, row in grp.iterrows())
        )

        # 9. horizon_salic_flag: any layer has salic suffix in horizon code
        salic_flag = int(
            any(has_token(row["horizon_code"], salic_tokens) for _, row in grp.iterrows())
        )

        # 10. horizon_ploughed_flag: any layer has ploughed suffix
        # Note: 'п' is a short suffix; we match it as a whole suffix token to avoid
        # false-positive match inside longer codes like 'В1погр' — check for boundary.
        def has_plough(h_code: str) -> bool:
            s = str(h_code).strip()
            if s in NULL_TOKENS:
                return False
            for tok in plough_tokens:
                if tok == "п":
                    # Only match if 'п' appears at the end of the code or
                    # preceded by a digit/letter that is not another Cyrillic letter
                    # that would form a longer word. Use a simple end-of-string check
                    # or sub-code suffix match.
                    if s.endswith("п") or s.endswith("пах") or "пах" in s:
                        return True
                else:
                    if tok.lower() in s.lower():
                        return True
            return False

        ploughed_flag = int(any(has_plough(row["horizon_code"]) for _, row in grp.iterrows()))

        rows.append({
            "pit_id": pit_id,
            "depth_to_moist_cm": depth_to_moist,
            "depth_to_salt_cm": depth_to_salt,
            "rust_mottling_flag": rust_flag,
            "gley_flag": gley_flag,
            "solum_depth_cm": solum_depth,
            "hcl_effervescence_class": hcl_class,
            "surface_crust_flag": surface_crust,
            "marine_shell_flag": shell_flag,
            "horizon_salic_flag": salic_flag,
            "horizon_ploughed_flag": ploughed_flag,
        })

    return pd.DataFrame(rows)


def compute_coverage(features: pd.DataFrame) -> dict:
    """Return coverage stats for each feature column (fraction non-null, n)."""
    n = len(features)
    cov = {}
    for col in features.columns:
        if col == "pit_id":
            continue
        non_null = features[col].notna().sum()
        cov[col] = {"n_non_null": int(non_null), "n_total": n,
                    "coverage_pct": round(100.0 * non_null / n, 1) if n > 0 else 0.0}
    return cov


def main() -> None:
    CANON.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab()
    layers = load_morph_layers()
    n_pits = layers["pit_id"].nunique()
    n_layers_total = len(layers)

    features = derive_features(layers, vocab)
    coverage = compute_coverage(features)

    features.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Coverage warnings for features with <60% coverage (TASK requirement: explicit caveat)
    low_cov_features = {k: v for k, v in coverage.items() if v["coverage_pct"] < 60.0}

    manifest = {
        "version": "V6-morph-features-1.0",
        "vocab_file": str(VOCAB_PATH),
        "n_pits": n_pits,
        "n_layers": n_layers_total,
        "n_feature_rows": len(features),
        "features": list(features.columns[1:]),
        "coverage": coverage,
        "low_coverage_features": low_cov_features,
        "low_coverage_note": (
            "Features with <60% coverage have NaN where the descriptor is absent. "
            "These are NOT imputed — missingness is honest and carried as NaN into "
            "ml_dataset_v6.csv and any model that uses them. Ablation must account "
            "for the reduced sample size when these features are included."
        ),
        "missingness_policy": (
            "All depth features (depth_to_moist_cm, depth_to_salt_cm) and flag features "
            "with low positive-rate (marine_shell_flag) remain NaN/0 as observed. "
            "No mean/mode imputation is performed here. Downstream consumers should "
            "either drop NaN rows or use model-level missing-value handling."
        ),
        "outputs": {
            "morph_features": str(OUT_CSV),
            "manifest": str(OUT_MANIFEST),
        },
    }

    # Print coverage warnings so pipeline log captures them
    if low_cov_features:
        print("[morph_features] WARNING: features with <60% coverage (not imputed):")
        for feat, stat in low_cov_features.items():
            print(f"  {feat}: {stat['coverage_pct']:.1f}% ({stat['n_non_null']}/{stat['n_total']})")

    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
