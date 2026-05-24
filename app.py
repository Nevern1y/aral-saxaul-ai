import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
import os
import folium
from folium import plugins
from shapely.geometry import box, shape
from pathlib import Path
import plotly.express as px

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v4.html"
GT_PATH = BASE_DIR / "outputs" / "data" / "ground_truth_v2.csv"
GEOJSON_PATH = BASE_DIR / "outputs" / "data" / "optimal_zones_v4.geojson"
AOI_VECTOR_PATH = BASE_DIR / "outputs" / "aoi" / "aral_sea_1960.geojson"

TASKS_PATH = BASE_DIR / "outputs" / "logistics" / "tasks_index_enriched.csv"
ROADS_PATH = BASE_DIR / "outputs" / "logistics" / "aralkum_roads.geojson"
GRID_STEP = 0.1

NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
NDWI_WATER = 0.0
SLOPE_MAX = 5.0

st.set_page_config(page_title="Aral Saxaul: Платформа Фитомелиорации", layout="wide")


@st.cache_data
def load_tasks():
    df = pd.read_csv(TASKS_PATH)
    return df


@st.cache_data
def load_roads():
    import geopandas as gpd
    if ROADS_PATH.exists():
        return gpd.read_file(ROADS_PATH)
    return None


st.title("Aral Saxaul: Система планирования посадок")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "Система автоматического планирования безопасных зон посадки саксаула "
    "на высохшем дне Аральского моря."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1300px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_logistics, tab_analytics, tab_dev = st.tabs([
    "\U0001f4cd Карта рабочих участков",
    "\U0001f4ca Общая статистика",
    "\u2699\ufe0f Технические параметры моделей",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: 📍 Карта рабочих участков
# ══════════════════════════════════════════════════════════════════════

with tab_logistics:
    tasks_df = load_tasks()
    roads_gdf = load_roads()

    if tasks_df.empty:
        st.warning(
            f"Файл не найден: {TASKS_PATH}. "
            "Запустите `python scripts/phase7_infrastructure.py`"
        )
    else:
        max_dist = float(tasks_df["distance_to_road_km"].max())

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            dist_thresh = st.slider(
                "Максимальное удаление от грунтовых дорог (км)",
                min_value=0.0,
                max_value=min(max_dist, 100.0),
                value=5.0,
                step=0.5,
                help="Фильтрует участки по их транспортной доступности. Оставляет на карте только те зоны, которые расположены не дальше указанного расстояния от существующих грунтовых дорог, что позволяет оптимизировать логистику и затраты на топливо.",
            )
        with col_f2:
            min_area = st.slider(
                "Минимальная площадь посадочного кластера (га)",
                min_value=0,
                max_value=10000,
                value=1000,
                step=100,
                help="Исключает из плана мелкие, разрозненные участки. Позволяет сосредоточить работу лесопосадочной техники только на крупных сплошных массивах, минимизируя холостые переезды тракторов между точками.",
            )

        filtered = tasks_df[
            (tasks_df["distance_to_road_km"] <= dist_thresh)
            & (tasks_df["area_ha"] >= min_area)
        ]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(
            "Доступно гектаров",
            f"{filtered['area_ha'].sum():,.0f}" if not filtered.empty else "0",
        )
        col_m2.metric("Количество участков", f"{len(filtered):,}")
        col_m3.metric("Из общего числа участков", f"{len(tasks_df):,}")
        col_m4.metric(
            "Доля от всех",
            f"{len(filtered) / len(tasks_df) * 100:.1f}%" if not filtered.empty else "0%",
        )

        m = folium.Map(
            location=[45.0, 60.0],
            zoom_start=8,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
        )

        if roads_gdf is not None and not roads_gdf.empty:
            folium.GeoJson(
                roads_gdf,
                name="Дороги (OSM)",
                style_function=lambda f: {
                    "color": "#8B4513", "weight": 1.0, "opacity": 0.6,
                },
                tooltip=folium.GeoJsonTooltip(fields=["fclass"]),
            ).add_to(m)

        if not filtered.empty:
            task_features = []
            for _, row in filtered.iterrows():
                lat = row["centroid_lat"]
                lon = row["centroid_lon"]
                half = GRID_STEP / 2
                cell = box(lon - half, lat - half, lon + half, lat + half)
                task_features.append({
                    "type": "Feature",
                    "properties": {
                        "filename": row["filename"],
                        "area_ha": round(row["area_ha"], 1),
                        "dist_km": round(row["distance_to_road_km"], 2),
                    },
                    "geometry": cell.__geo_interface__,
                })

            task_fc = {"type": "FeatureCollection", "features": task_features}
            folium.GeoJson(
                task_fc,
                name="Отобранные участки",
                style_function=lambda f: {
                    "fillColor": "#2ecc40", "color": "#27ae60",
                    "weight": 1.0, "fillOpacity": 0.3,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["filename", "area_ha", "dist_km"],
                    aliases=["Файл:", "Площадь (га):", "До дороги (км):"],
                    localize=True,
                ),
                highlight_function=lambda f: {"weight": 2.0, "color": "#007bff"},
            ).add_to(m)

            top5 = filtered.nsmallest(5, "distance_to_road_km")
            for _, row in top5.iterrows():
                folium.Marker(
                    location=[row["centroid_lat"], row["centroid_lon"]],
                    popup=(
                        f"{row['filename']}<br>"
                        f"{row['area_ha']:.0f} ha, {row['distance_to_road_km']:.2f} km"
                    ),
                    icon=folium.Icon(color="green", icon="ok-sign", prefix="glyphicon"),
                ).add_to(m)

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MousePosition().add_to(m)

        map_html = m._repr_html_()
        st.components.v1.html(map_html, height=700, scrolling=True)

        with st.expander("\U0001f4c2 Список маршрутных файлов (KML)"):
            display_df = filtered[
                ["filename", "centroid_lat", "centroid_lon", "area_ha", "distance_to_road_km"]
            ].copy()
            display_df.columns = [
                "Файл KML", "Широта", "Долгота",
                "Площадь (га)", "До дороги (км)",
            ]
            st.dataframe(
                display_df.sort_values("До дороги (км)", ascending=True),
                hide_index=True,
                width="stretch",
            )

# ══════════════════════════════════════════════════════════════════════
# TAB 2: 📊 Общая статистика
# ══════════════════════════════════════════════════════════════════════

with tab_analytics:
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

    has_vector_aoi = AOI_VECTOR_PATH.exists()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "\u041e\u0431\u0449\u0430\u044f \u043f\u043b\u043e\u0449\u0430\u0434\u044c \u043f\u043e\u0441\u0430\u0434\u043a\u0438",
        f"{v4_stats.get('area_km2', 'N/A'):,} km\u00b2"
        if v4_stats.get('area_km2')
        else "N/A (\u0436\u0434\u0438\u0442\u0435...)",
    )
    col2.metric(
        "\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0437\u043e\u043d",
        f"{v4_stats.get('clusters', 'N/A'):,}"
        if v4_stats.get('clusters')
        else "N/A (\u0436\u0434\u0438\u0442\u0435...)",
    )
    col3.metric(
        "\u041a\u0440\u0443\u043f\u043d\u0435\u0439\u0448\u0438\u0435 \u0437\u043e\u043d\u044b (\u0442\u043e\u043f-10)",
        f"{v4_stats.get('top10_km2', 0):,.0f} km\u00b2"
        if v4_stats.get('top10_km2')
        else "N/A",
    )
    col4.metric("\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u044b\u0439 \u0440\u0430\u0437\u043c\u0435\u0440 \u0443\u0447\u0430\u0441\u0442\u043a\u0430", "\u22651 \u0433\u0430")

    if not has_vector_aoi:
        st.info(
            "\u0414\u043b\u044f \u0442\u043e\u0447\u043d\u043e\u0439 \u0431\u0435\u0440\u0435\u0433\u043e\u0432\u043e\u0439 \u043b\u0438\u043d\u0438\u0438 "
            "\u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u0410\u0440\u0430\u043b\u0430 \u043f\u043e\u043c\u0435\u0441\u0442\u0438\u0442\u0435 "
            "\u0432\u0435\u043a\u0442\u043e\u0440\u043d\u044b\u0439 \u0444\u0430\u0439\u043b "
            "\u0432 `outputs/aoi/aral_sea_1960.geojson`"
        )

    col_chart, col_text = st.columns([1, 1])

    with col_chart:
        pie_data = pd.DataFrame({
            "\u0417\u043e\u043d\u0430": [
                "\u0411\u043b\u0430\u0433\u043e\u043f\u0440\u0438\u044f\u0442\u043d\u0430\u044f \u0437\u043e\u043d\u0430 \u043f\u043e\u0441\u0430\u0434\u043a\u0438",
                "\u0417\u043e\u043d\u0430 \u0440\u0438\u0441\u043a\u0430",
                "\u0417\u043e\u043d\u0430 \u0440\u0438\u0441\u043a\u0430 (\u0441\u043e\u043b\u043e\u043d\u0447\u0430\u043a\u0438)",
                "\u041f\u0440\u0435\u043f\u044f\u0442\u0441\u0442\u0432\u0438\u044f (\u043e\u0431\u0440\u044b\u0432\u044b)",
            ],
            "\u0414\u043e\u043b\u044f (%)": [65.0, 18.6, 16.0, 0.4],
        })
        fig = px.pie(
            pie_data,
            values="\u0414\u043e\u043b\u044f (%)",
            names="\u0417\u043e\u043d\u0430",
            color="\u0417\u043e\u043d\u0430",
            color_discrete_map={
                "\u0411\u043b\u0430\u0433\u043e\u043f\u0440\u0438\u044f\u0442\u043d\u0430\u044f \u0437\u043e\u043d\u0430 \u043f\u043e\u0441\u0430\u0434\u043a\u0438": "#2ecc40",
                "\u0417\u043e\u043d\u0430 \u0440\u0438\u0441\u043a\u0430": "#f39c12",
                "\u0417\u043e\u043d\u0430 \u0440\u0438\u0441\u043a\u0430 (\u0441\u043e\u043b\u043e\u043d\u0447\u0430\u043a\u0438)": "#e74c3c",
                "\u041f\u0440\u0435\u043f\u044f\u0442\u0441\u0442\u0432\u0438\u044f (\u043e\u0431\u0440\u044b\u0432\u044b)": "#95a5a6",
            },
            hole=0.4,
        )
        fig.update_traces(textinfo="label+percent", textposition="outside")
        fig.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        st.write("")
        st.write("")
        st.markdown("### \u041e \u0440\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0438 \u0437\u043e\u043d")
        st.write(
            "\u041a\u0440\u0443\u0433\u043e\u0432\u0430\u044f \u0434\u0438\u0430\u0433\u0440\u0430\u043c\u043c\u0430 "
            "\u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u043f\u043e\u043b\u043d\u044b\u0439 "
            "\u044d\u043a\u043e\u043b\u043e\u0433\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0441\u0440\u0435\u0437 "
            "\u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u0434\u043d\u0430 "
            "\u0410\u0440\u0430\u043b\u044c\u0441\u043a\u043e\u0433\u043e \u043c\u043e\u0440\u044f."
        )
        st.info(
            "\u0412\u043e \u0438\u0437\u0431\u0435\u0436\u0430\u043d\u0438\u0435 \u043f\u0435\u0440\u0435\u0433\u0440\u0443\u0437\u043a\u0438 "
            "\u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u0430 \u0438 \u0434\u043b\u044f \u0443\u0434\u043e\u0431\u0441\u0442\u0432\u0430 "
            "\u043f\u043b\u0430\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f, "
            "\u043d\u0430 \u0438\u043d\u0442\u0435\u0440\u0430\u043a\u0442\u0438\u0432\u043d\u0443\u044e \u043a\u0430\u0440\u0442\u0443 "
            "\u043d\u0438\u0436\u0435 \u0432\u044b\u0432\u0435\u0434\u0435\u043d\u044b \u0442\u043e\u043b\u044c\u043a\u043e "
            "\u0431\u043b\u0430\u0433\u043e\u043f\u0440\u0438\u044f\u0442\u043d\u044b\u0435 \u0437\u043e\u043d\u044b "
            "\u043f\u043e\u0441\u0430\u0434\u043a\u0438 (\u0437\u0435\u043b\u0435\u043d\u044b\u0439 \u0446\u0432\u0435\u0442). "
            "\u0417\u043e\u043d\u044b \u0440\u0438\u0441\u043a\u0430 \u0438 \u043e\u0431\u0440\u044b\u0432\u044b "
            "\u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u043e \u0441\u043a\u0440\u044b\u0442\u044b, "
            "\u0442\u0430\u043a \u043a\u0430\u043a \u0442\u0443\u0434\u0430 \u043d\u0435 \u043d\u0430\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u0442\u0441\u044f "
            "\u0442\u0435\u0445\u043d\u0438\u043a\u0430."
        )

    st.divider()
    st.markdown("### \u041a\u0430\u0440\u0442\u0430 \u043f\u0440\u0438\u0433\u043e\u0434\u043d\u044b\u0445 \u0443\u0447\u0430\u0441\u0442\u043a\u043e\u0432")

    if MAP_PATH.exists():
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=750, scrolling=True)
    else:
        st.warning(
            f"\u0424\u0430\u0439\u043b \u043a\u0430\u0440\u0442\u044b \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d: {MAP_PATH}. "
            "\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 `python scripts/phase5_v4_export.py`"
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ⚙️ Технические параметры моделей
# ══════════════════════════════════════════════════════════════════════

with tab_dev:
    st.info(
        "\u26a0\ufe0f \u0420\u0430\u0437\u0434\u0435\u043b \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 "
        "\u0441\u044b\u0440\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435, "
        "\u0442\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u043c\u0435\u0442\u0440\u0438\u043a\u0438 "
        "\u0438 \u0433\u0438\u043f\u0435\u0440\u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u043c\u043e\u0434\u0435\u043b\u0438."
    )

    st.subheader("\u0410\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u0430 \u043a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438")

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
            SI \u043e\u0442\u0431\u0440\u0430\u043a\u043e\u0432\u0430\u043d \u2014 Spearman
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
            "Legacy \u043c\u0430\u0441\u043a\u0430 `suitability_full.tif` (V1.0) "
            "\u0431\u043e\u043b\u044c\u0448\u0435 \u043d\u0435 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f."
        )

        st.markdown("**\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 V4:**")
        stats_data = {
            "Class": ["1 Optimal", "2 Risk", "3 Dead", "4 Topo", "0 Water"],
            "px": ["100.6M", "28.8M", "24.7M", "557K", "11.5M"],
            "%": ["65.0%", "18.6%", "16.0%", "0.4%", "\u2014"],
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
        "AOI": ["BBOX", "BBOX", "BBOX", "V1.0 mask", "data valid", "**\u0432\u0435\u043a\u0442\u043e\u0440**"],
        "Optimal": ["6,558 km\u00b2", "\u2014", "75,165 km\u00b2", "14,962 km\u00b2", "~90K km\u00b2*", "**TBD**"],
        "\u0421\u0442\u0430\u0442\u0443\u0441": ["Archive", "Frozen", "Archive", "Archive", "Beta", "**Goal**"],
    }
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, width="stretch")
    st.caption(
        "* 90K km\u00b2 \u0432\u043a\u043b\u044e\u0447\u0430\u0435\u0442 \u043f\u0443\u0441\u0442\u044b\u043d\u0438 "
        "\u0432\u043d\u0435 \u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u0410\u0440\u0430\u043b\u0430. "
        "\u0424\u0438\u043d\u0430\u043b\u044c\u043d\u0430\u044f \u0446\u0438\u0444\u0440\u0430 \u0431\u0443\u0434\u0435\u0442 "
        "\u043f\u043e\u0441\u043b\u0435 \u043a\u043b\u0438\u043f\u043f\u0438\u043d\u0433\u0430 "
        "\u043f\u043e \u0431\u0435\u0440\u0435\u0433\u043e\u0432\u043e\u0439 \u043b\u0438\u043d\u0438\u0438 1960 \u0433\u043e\u0434\u0430."
    )

    # ── Ground Truth ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("V2.0 Ground Truth \u2014 11 \u0442\u043e\u0447\u0435\u043a")

    try:
        df_gt = pd.read_csv(GT_PATH)
        df_gt["pit_code"] = df_gt["pit_code"].str.replace("\u0410", "A")

        cols = [
            "pit_code", "S_Point", "Lon_DD", "Lat_DD", "Salinity_pct",
            "SI", "NDMI", "NDWI", "Cl", "SO4", "Na",
            "Sand_pct", "Silt_pct",
        ]
        available = [c for c in cols if c in df_gt.columns]
        st.dataframe(df_gt[available], hide_index=True, width='stretch')

        st.markdown("### \u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442")
        st.markdown(
            "\u2013 **01/20\u0410\u201307/20\u0410:** ODT DMM "
            "(\u0441\u043a\u0440\u0438\u043d\u0448\u043e\u0442\u044b "
            "cross-reference \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430)\n"
            "\u2013 **08/20\u0410\u201311/20\u0410:** AralField DD "
            "\u0441\u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u044b "
            "\u043d\u0430 mean offset vector "
            "(dLon=1.124\u00b0, dLat=0.895\u00b0)"
        )

        st.markdown("### Spearman \u043a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u0438")
        from scipy.stats import spearmanr
        r_si, p_si = spearmanr(df_gt["Salinity_pct"], df_gt["SI"])
        r_ndmi, p_ndmi = spearmanr(df_gt["Salinity_pct"], df_gt["NDMI"])
        corr_data = {
            "\u041f\u0430\u0440\u0430": ["Salinity vs SI", "Salinity vs NDMI"],
            "Spearman r": [f"{r_si:.4f}", f"{r_ndmi:.4f}"],
            "p-value": [f"{p_si:.4f}", f"{p_ndmi:.4f}"],
            "\u0412\u0435\u0440\u0434\u0438\u043a\u0442": [
                "\u0421\u043b\u0430\u0431\u0430\u044f (p>0.05)",
                "**\u0421\u0438\u043b\u044c\u043d\u0430\u044f (p<0.05)**",
            ],
        }
        st.dataframe(pd.DataFrame(corr_data), hide_index=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.scatter(df_gt["SI"], df_gt["Salinity_pct"], c="#3498DB", s=80, alpha=0.7)
        ax1.set_xlabel("SI (Salinity Index)")
        ax1.set_ylabel("Salinity_pct (%)")
        ax1.set_title(f"SI vs Salinity (r={r_si:.2f}, p={p_si:.3f})")
        ax1.axhline(y=2.0, color="green", linestyle="--", alpha=0.5, label="<2% Optimal")
        ax1.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label=">5% Dead")
        for _, row in df_gt.iterrows():
            ax1.annotate(
                row["S_Point"], (row["SI"], row["Salinity_pct"]),
                fontsize=7, alpha=0.7,
            )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.scatter(df_gt["NDMI"], df_gt["Salinity_pct"], c="#2ECC40", s=80, alpha=0.7)
        ax2.set_xlabel("NDMI (Normalized Difference Moisture Index)")
        ax2.set_ylabel("Salinity_pct (%)")
        ax2.set_title(f"NDMI vs Salinity (r={r_ndmi:.2f}, p={p_ndmi:.3f})")
        ax2.axhline(y=2.0, color="green", linestyle="--", alpha=0.5, label="<2% Optimal")
        ax2.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label=">5% Dead")
        ax2.axvline(x=-0.055, color="blue", linestyle=":", alpha=0.5, label="NDMI Optimal threshold")
        ax2.axvline(x=-0.025, color="red", linestyle=":", alpha=0.5, label="NDMI Dead threshold")
        for _, row in df_gt.iterrows():
            ax2.annotate(
                row["S_Point"], (row["NDMI"], row["Salinity_pct"]),
                fontsize=7, alpha=0.7,
            )
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SI \u043d\u0435 \u0440\u0430\u0437\u0434\u0435\u043b\u044f\u0435\u0442 \u0437\u043e\u043d\u044b. "
            "NDMI: \u0447\u0435\u043c \u0432\u043b\u0430\u0436\u043d\u0435\u0435, "
            "\u0442\u0435\u043c \u0441\u043e\u043b\u043e\u043d\u0435\u0435 "
            "(\u043a\u0430\u043f\u0438\u043b\u043b\u044f\u0440\u043d\u044b\u0439 \u043f\u043e\u0434\u044a\u0451\u043c)."
        )

    except Exception as e:
        st.warning(f"Ground truth \u043d\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d: {e}")
        st.info("\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 `python scripts/build_ground_truth.py`")
