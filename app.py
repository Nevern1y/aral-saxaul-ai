import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v4.html"
GT_PATH = BASE_DIR / "outputs" / "data" / "ground_truth_v2.csv"
GEOJSON_PATH = BASE_DIR / "outputs" / "data" / "optimal_zones_v4.geojson"
AOI_VECTOR_PATH = BASE_DIR / "outputs" / "aoi" / "aral_sea_1960.geojson"

NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
NDWI_WATER = 0.0
SLOPE_MAX = 5.0

st.set_page_config(page_title="Aral Saxaul V4 — Dashboard", layout="wide")

st.title("Aral Saxaul AI — V4 с полным покрытием AOI")
st.markdown(
    "Дашборд карты пригодности посадки саксаула на высохшем дне Аральского моря. "
    "V4: NDMI + AOI изоляция + топографический фильтр (уклон \u2264 5\u00b0). "
    "AOI-маска перестроена из данных feature stack (без legacy V1.0)."
)

tab1, tab2, tab3 = st.tabs([
    "\u041a\u0430\u0440\u0442\u0430 \u043f\u0440\u0438\u0433\u043e\u0434\u043d\u043e\u0441\u0442\u0438 V4",
    "V4 \u0410\u043d\u0430\u043b\u0438\u0442\u0438\u043a\u0430",
    "V2.0 Ground Truth",
])

# ── Tab 1: Map ──────────────────────────────────────────────────────────

with tab1:
    v4_stats = {}
    try:
        if GEOJSON_PATH.exists():
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
            areas_ha = [f["properties"]["area_ha"] for f in gj["features"]]
            v4_stats["clusters"] = len(areas_ha)
            v4_stats["area_km2"] = sum(areas_ha) / 100
            v4_stats["area_ha"] = sum(areas_ha)
            v4_stats["top10_km2"] = sum(sorted(areas_ha, reverse=True)[:10]) / 100
    except Exception:
        pass

    col1, col2, col3, col4 = st.columns(4)

    has_vector_aoi = AOI_VECTOR_PATH.exists()
    aoi_note = " (с контуром)" if has_vector_aoi else " (без контура)"

    col1.metric(
        "\u041e\u043f\u0442\u0438\u043c\u0430\u043b\u044c\u043d\u0430\u044f \u0437\u043e\u043d\u0430",
        f"{v4_stats.get('area_km2', 'N/A'):,} km\u00b2" if v4_stats.get('area_km2') else "N/A (ждите...)"
    )
    col2.metric(
        "\u041f\u043e\u043b\u0438\u0433\u043e\u043d\u043e\u0432 (\u22651 \u0433\u0430)",
        f"{v4_stats.get('clusters', 'N/A'):,}" if v4_stats.get('clusters') else "N/A (ждите...)"
    )
    col3.metric(
        "\u0422\u043e\u043f-\u043f\u043e\u043b\u0438\u0433\u043e\u043d\u044b (10)",
        f"{v4_stats.get('top10_km2', 0):,.0f} km\u00b2" if v4_stats.get('top10_km2') else "N/A"
    )
    col4.metric("\u0412\u0435\u0440\u0441\u0438\u044f", "V4")

    if not has_vector_aoi:
        st.info(
            "\u0414\u043b\u044f \u0442\u043e\u0447\u043d\u043e\u0439 \u0431\u0435\u0440\u0435\u0433\u043e\u0432\u043e\u0439 \u043b\u0438\u043d\u0438\u0438 "
            "\u043f\u043e\u043c\u0435\u0441\u0442\u0438\u0442\u0435 \u0432\u0435\u043a\u0442\u043e\u0440\u043d\u044b\u0439 \u0444\u0430\u0439\u043b "
            "\u0432 `outputs/aoi/aral_sea_1960.geojson`"
        )

    if MAP_PATH.exists():
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=750, scrolling=True)
    else:
        st.warning(
            f"\u0424\u0430\u0439\u043b \u043a\u0430\u0440\u0442\u044b \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d: {MAP_PATH}. "
            "\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 `python scripts/phase5_v4_export.py`"
        )

# ── Tab 2: V4 Analytics ─────────────────────────────────────────────────

with tab2:
    st.subheader("V4 \u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u0430")

    col_rules, col_stats = st.columns([1, 1])

    with col_rules:
        st.markdown("**\u041f\u0440\u0430\u0432\u0438\u043b\u0430 \u043a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438:**")
        st.markdown(
            """
            | \u041a\u043b\u0430\u0441\u0441 | \u041f\u0440\u0430\u0432\u0438\u043b\u043e |
            |---|---|
            | **1 Optimal** | NDMI < -0.055 **\u0438** Slope \u2264 5\u00b0 |
            | **2 Risk** | -0.055 \u2264 NDMI \u2264 -0.025 **\u0438** Slope \u2264 5\u00b0 |
            | **3 Dead** | NDMI > -0.025 (\u043a\u0430\u043f\u0438\u043b\u043b\u044f\u0440\u043d\u044b\u0439 \u043f\u043e\u0434\u044a\u0451\u043c) |
            | **4 Topo Obstacle** | Slope > 5\u00b0 (\u043a\u0440\u0443\u0442\u044b\u0435 \u0441\u043a\u043b\u043e\u043d\u044b) |
            | **0 Water/NoData** | NDWI > 0 \u0438\u043b\u0438 \u043d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 |

            **\u041a\u043b\u044e\u0447\u0435\u0432\u043e\u0435 \u043e\u0442\u043a\u0440\u044b\u0442\u0438\u0435:**
            SI \u043e\u0442\u0431\u0440\u0430\u043a\u043e\u0432\u0430\u043d — Spearman
            r(Salinity vs SI) = +0.41 (p=0.21).
            NDMI: **r = +0.69** (p=0.02).
            """
        )

    with col_stats:
        st.markdown("**AOI-\u043c\u0430\u0441\u043a\u0430:**")
        st.markdown(
            "\u041d\u043e\u0432\u0430\u044f \u043c\u0430\u0441\u043a\u0430 `aoi_mask_v5.tif` "
            "\u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d\u0430 \u0438\u0437 \u0432\u0430\u043b\u0438\u0434\u043d\u044b\u0445 "
            "\u0434\u0430\u043d\u043d\u044b\u0445 feature stack (NDMI \u043d\u0435 NaN) + "
            "\u0432\u043e\u0434\u043d\u0430\u044f \u043c\u0430\u0441\u043a\u0430 NDWI. "
            "Legacy \u043c\u0430\u0441\u043a\u0430 `suitability_full.tif` (V1.0) \u0431\u043e\u043b\u044c\u0448\u0435 "
            "\u043d\u0435 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f."
        )

        st.markdown("**\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 V4:**")
        stats_data = {
            "Class": ["1 Optimal", "2 Risk", "3 Dead", "4 Topo", "0 Water"],
            "px": ["100.6M", "28.8M", "24.7M", "557K", "11.5M"],
            "%": ["65.0%", "18.6%", "16.0%", "0.4%", "—"],
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)

        st.markdown("**\u041e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d\u0438\u0435:**")
        st.markdown(
            "\u0411\u0435\u0437 \u0432\u0435\u043a\u0442\u043e\u0440\u043d\u043e\u0439 "
            "\u0431\u0435\u0440\u0435\u0433\u043e\u0432\u043e\u0439 \u043b\u0438\u043d\u0438\u0438 "
            "\u0432\u0441\u0435 139K km\u00b2 \u043f\u043e\u043f\u0430\u0434\u0430\u044e\u0442 \u0432 AOI, "
            "\u0432\u043a\u043b\u044e\u0447\u0430\u044f \u043f\u0443\u0441\u0442\u044b\u043d\u0438 "
            "\u0432\u043d\u0435 \u0410\u0440\u0430\u043b\u0430. "
            "\u041f\u043e\u043c\u0435\u0441\u0442\u0438\u0442\u0435 `aral_sea_1960.geojson` "
            "\u0432 `outputs/aoi/` \u0434\u043b\u044f \u0442\u043e\u0447\u043d\u043e\u0433\u043e "
            "\u043a\u043b\u0438\u043f\u043f\u0438\u043d\u0433\u0430."
        )

    st.markdown("---")
    st.subheader("\u0421\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u0432\u0435\u0440\u0441\u0438\u0439")

    comp_data = {
        "\u0412\u0435\u0440\u0441\u0438\u044f": ["V1.0", "V2.0", "V3.0", "V3.2", "V4", "**V4 + Coast**"],
        "AOI": ["BBOX", "BBOX", "BBOX", "V1.0 mask", "data valid", "**вектор**"],
        "Optimal": ["6,558 km\u00b2", "\u2014", "75,165 km\u00b2", "14,962 km\u00b2", "~90K km\u00b2*", "**TBD**"],
        "Статус": ["Archive", "Frozen", "Archive", "Archive", "Betat", "**Goal**"],
    }
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)
    st.caption("* 90K km\u00b2 включает пустыни вне исторического Арала. "
               "Финальная цифра будет после клиппинга по береговой линии 1960 года.")

# ── Tab 3: Ground Truth ─────────────────────────────────────────────────

with tab3:
    st.subheader("V2.0 Ground Truth \u2014 11 \u0442\u043e\u0447\u0435\u043a")

    try:
        df = pd.read_csv(GT_PATH)
        df["pit_code"] = df["pit_code"].str.replace("\u0410", "A")

        cols = ["pit_code", "S_Point", "Lon_DD", "Lat_DD", "Salinity_pct",
                "SI", "NDMI", "NDWI", "Cl", "SO4", "Na", "Sand_pct", "Silt_pct"]
        available = [c for c in cols if c in df.columns]
        st.dataframe(df[available], hide_index=True, use_container_width=True)

        st.markdown("### \u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442")
        st.markdown(
            "\u2013 **01/20\u0410\u201307/20\u0410:** ODT DMM (\u0441\u043a\u0440\u0438\u043d\u0448\u043e\u0442\u044b "
            "cross-reference \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430)\n"
            "\u2013 **08/20\u0410\u201311/20\u0410:** AralField DD \u0441\u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u044b "
            "\u043d\u0430 mean offset vector (dLon=1.124\u00b0, dLat=0.895\u00b0)"
        )

        st.markdown("### Spearman \u043a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u0438")
        from scipy.stats import spearmanr
        r_si, p_si = spearmanr(df["Salinity_pct"], df["SI"])
        r_ndmi, p_ndmi = spearmanr(df["Salinity_pct"], df["NDMI"])
        corr_data = {
            "\u041f\u0430\u0440\u0430": ["Salinity vs SI", "Salinity vs NDMI"],
            "Spearman r": [f"{r_si:.4f}", f"{r_ndmi:.4f}"],
            "p-value": [f"{p_si:.4f}", f"{p_ndmi:.4f}"],
            "\u0412\u0435\u0440\u0434\u0438\u043a\u0442": ["\u0421\u043b\u0430\u0431\u0430\u044f (p>0.05)", "**\u0421\u0438\u043b\u044c\u043d\u0430\u044f (p<0.05)**"],
        }
        st.dataframe(pd.DataFrame(corr_data), hide_index=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.scatter(df["SI"], df["Salinity_pct"], c="#3498DB", s=80, alpha=0.7)
        ax1.set_xlabel("SI (Salinity Index)")
        ax1.set_ylabel("Salinity_pct (%)")
        ax1.set_title(f"SI vs Salinity (r={r_si:.2f}, p={p_si:.3f})")
        ax1.axhline(y=2.0, color="green", linestyle="--", alpha=0.5, label="<2% Optimal")
        ax1.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label=">5% Dead")
        for _, row in df.iterrows():
            ax1.annotate(row["S_Point"], (row["SI"], row["Salinity_pct"]),
                        fontsize=7, alpha=0.7)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.scatter(df["NDMI"], df["Salinity_pct"], c="#2ECC40", s=80, alpha=0.7)
        ax2.set_xlabel("NDMI (Normalized Difference Moisture Index)")
        ax2.set_ylabel("Salinity_pct (%)")
        ax2.set_title(f"NDMI vs Salinity (r={r_ndmi:.2f}, p={p_ndmi:.3f})")
        ax2.axhline(y=2.0, color="green", linestyle="--", alpha=0.5, label="<2% Optimal")
        ax2.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label=">5% Dead")
        ax2.axvline(x=-0.055, color="blue", linestyle=":", alpha=0.5, label="NDMI Optimal threshold")
        ax2.axvline(x=-0.025, color="red", linestyle=":", alpha=0.5, label="NDMI Dead threshold")
        for _, row in df.iterrows():
            ax2.annotate(row["S_Point"], (row["NDMI"], row["Salinity_pct"]),
                        fontsize=7, alpha=0.7)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SI \u043d\u0435 \u0440\u0430\u0437\u0434\u0435\u043b\u044f\u0435\u0442 \u0437\u043e\u043d\u044b. "
            "NDMI: \u0447\u0435\u043c \u0432\u043b\u0430\u0436\u043d\u0435\u0435, \u0442\u0435\u043c \u0441\u043e\u043b\u043e\u043d\u0435\u0435 "
            "(\u043a\u0430\u043f\u0438\u043b\u043b\u044f\u0440\u043d\u044b\u0439 \u043f\u043e\u0434\u044a\u0451\u043c)."
        )

    except Exception as e:
        st.warning(f"Ground truth \u043d\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d: {e}")
        st.info("\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 `python scripts/build_ground_truth.py`")
