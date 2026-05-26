import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import folium
from folium import plugins
from shapely.geometry import box, shape
from pathlib import Path
import plotly.express as px
try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
GT_PATH = BASE_DIR / "outputs" / "data" / "ground_truth_v2.csv"
AOI_VECTOR_PATH = BASE_DIR / "outputs" / "aoi" / "aral_sea_1960.geojson"

TASKS_PATH = BASE_DIR / "outputs" / "logistics" / "tasks_index_v5_enriched.csv"
ROADS_PATH = BASE_DIR / "outputs" / "logistics" / "aralkum_roads.geojson"
GRID_STEP = 0.1

NDMI_OPTIMAL = -0.055
NDMI_DEAD = -0.025
NDWI_WATER = 0.0
SLOPE_MAX = 5.0

# ── V5.0 paths (strict — no V4 fallback) ──────────────────────────────
V5_MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v5.html"
V5_OPERATIONAL_PATH = BASE_DIR / "outputs" / "data" / "operational_zones_v5.geojson"
V5_THRESHOLDS_PATH = BASE_DIR / "outputs" / "data" / "thresholds_v5.json"
V5_STATS_PATH = BASE_DIR / "outputs" / "data" / "v5_stats.json"

st.set_page_config(page_title="Aral Saxaul: Платформа Фитомелиорации", layout="wide")

# UI/UX: ограничение ширины дашборда для комфортного чтения на широких экранах
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


# ── Cached V5 data loaders (heavy I/O → once per session) ──────────────
@st.cache_data
def load_v5_stats():
    path = BASE_DIR / "outputs" / "data" / "v5_stats.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_v5_class_pixels():
    pixels = {}
    total_px = 0
    path = BASE_DIR / "outputs" / "data" / "suitability_map_v5_filtered.tif"
    if path.exists() and rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read(1)
        total_px = arr.size
        for cls_val in [0, 1, 3, 4, 5, 10]:
            pixels[cls_val] = int((arr == cls_val).sum())
    return pixels, total_px


@st.cache_data
def load_v5_thresholds():
    path = BASE_DIR / "outputs" / "data" / "thresholds_v5.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_map_html_str():
    if V5_MAP_PATH.exists():
        with open(V5_MAP_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return None


@st.cache_data
def make_audit_fig(pixels_json_str, total_px_int):
    pixels = json.loads(pixels_json_str)
    if not pixels or total_px_int == 0:
        return None
    water_pct = pixels.get("0", 0) / total_px_int * 100
    opt_pct = pixels.get("1", 0) / total_px_int * 100
    salt_pct = pixels.get("3", 0) / total_px_int * 100
    brine_pct = pixels.get("4", 0) / total_px_int * 100
    obst_pct = pixels.get("5", 0) / total_px_int * 100
    veg_pct = pixels.get("10", 0) / total_px_int * 100

    audit_data = pd.DataFrame({
        "Зона": [
            "Optimal (Оптимально для посадки)",
            "Vegetation (Естественная вегетация)",
            "Dead / Wet Toxic (Капиллярный подъем рапы)",
            "Risk / Dry Salt (Сухая солевая корка)",
            "Obstacle / Topo (Крутые склоны и обрывы)",
            "Water / NoData (Акватория / Нет данных)",
        ],
        "Доля (%)": [
            round(opt_pct, 1), round(veg_pct, 1),
            round(brine_pct, 1), round(salt_pct, 1),
            round(obst_pct, 1), round(water_pct, 1),
        ],
    })

    audit_colors = {
        "Optimal (Оптимально для посадки)": "#2ecc40",
        "Vegetation (Естественная вегетация)": "#7FCDBB",
        "Dead / Wet Toxic (Капиллярный подъем рапы)": "#D95F02",
        "Risk / Dry Salt (Сухая солевая корка)": "#E6AB02",
        "Obstacle / Topo (Крутые склоны и обрывы)": "#636363",
        "Water / NoData (Акватория / Нет данных)": "#BDBDBD",
    }

    fig = px.pie(
        audit_data,
        values="Доля (%)",
        names="Зона",
        color="Зона",
        color_discrete_map=audit_colors,
        hole=0.4,
    )
    fig.update_traces(textinfo="label+percent", textposition="outside", textfont_size=10)
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=10, b=20))
    return fig


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
    try:
        tasks_df = load_tasks()
        roads_gdf = load_roads()
        v5_stats = load_v5_stats()
    except FileNotFoundError:
        st.error("\u26a0\ufe0f \u041b\u043e\u0433\u0438\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 V5.0 \u043d\u0435 \u043e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u044b.")
        st.info(
            "\u041f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430, \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 \u0431\u0430\u0437\u043e\u0432\u044b\u0439 \u0441\u043a\u0440\u0438\u043f\u0442 \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438 \u0441\u0435\u0442\u043a\u0438 \u043d\u0430\u0440\u044f\u0434\u043e\u0432 \u0438 \u0440\u0430\u0441\u0447\u0435\u0442\u0430 \u0434\u043e\u0440\u043e\u0433:\n"
            "`python scripts/v5_logistics_prep.py`"
        )
        st.stop()

    if tasks_df.empty:
        st.warning(
            f"Файл не найден: {TASKS_PATH}. "
            "Запустите `python scripts/v5_logistics_prep.py`"
        )
    else:
        max_dist = float(tasks_df["distance_to_road_km"].max())
        max_cell_ha = float(tasks_df["area_ha"].max())

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            road_scenarios = {
                "\U0001f680 Прибрежная зона (До 120 км — высокая доступность)": 120.0,
                "\U0001f42b Глубокий Аралкум (До 250 км — автономная экспедиция)": 250.0,
                "\U0001f30d Максимальный охват (Вся доступная территория)": float(tasks_df["distance_to_road_km"].max()),
            }
            selected_road_scen = st.selectbox(
                "\U0001f4cd Транспортная доступность участков:",
                options=list(road_scenarios.keys()),
                index=2,
            )
            dist_thresh = road_scenarios[selected_road_scen]

        with col_f2:
            area_scenarios = {
                "\U0001f331 Локальные питомники (От 10 до 1 000 га)": (10, 1000),
                "\U0001f333 Крупные лесничества (От 1 000 до 5 000 га)": (1000, 5000),
                "\U0001f985 Стратегические хабы (Более 5 000 га)": (5000, int(max_cell_ha)),
                "\U0001f4ca Все размеры кластеров": (0, int(max_cell_ha)),
            }
            selected_area_scen = st.selectbox(
                "\U0001f4d0 Масштаб планируемого кластера:",
                options=list(area_scenarios.keys()),
                index=3,
            )
            min_area, max_area = area_scenarios[selected_area_scen]

        filtered = tasks_df[
            (tasks_df["distance_to_road_km"] <= dist_thresh)
            & (tasks_df["area_ha"] >= min_area)
            & (tasks_df["area_ha"] <= max_area)
        ]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        total_ha = v5_stats.get("area_ha", 0)
        col_m1.metric(
            "Доступно гектаров (V5.0)",
            f"{total_ha:,.0f}",
            delta=f"{filtered['area_ha'].sum():,.0f} отфильтровано" if not filtered.empty else None,
        )
        col_m2.metric("Количество участков", f"{len(filtered):,}")
        col_m3.metric("Из общего числа ячеек", f"{len(tasks_df):,}")
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

        st.html(m.get_root().render())

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
    # ── All heavy I/O goes through @st.cache_data (runs once) ────────────
    try:
        v5_stats = load_v5_stats()
        v5_class_pixels, total_px = load_v5_class_pixels()
        v5_thresholds = load_v5_thresholds()
    except FileNotFoundError:
        st.error("⚠️ Данные V5.0 не обнаружены.")
        st.info(
            "Пожалуйста, запустите сначала локальные скрипты генерации данных:\n"
            "1. `python scripts/run_inference_v5.py` — расчет маски\n"
            "2. `python scripts/v5_extract_stats.py` — извлечение статистики"
        )
        st.stop()

    # ── Top metrics panel (data from v5_stats.json) ──────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Доступно гектаров",
        f"{v5_stats.get('area_ha', 0):,.0f}",
    )
    col2.metric(
        "Количество участков",
        f"{v5_stats.get('clusters', 0):,}",
    )
    col3.metric(
        "Общая площадь посадки",
        f"{v5_stats.get('area_km2', 0):,.0f} км²",
    )
    col4.metric("Минимальный размер участка", "≥10 га")

    # ── Карта на самом видном месте ─────────────────────────────────
    st.markdown("### 🗺️ Карта пригодных участков")
    map_html = load_map_html_str()
    if map_html:
        st.html(map_html)
        st.caption("🟢 V5.0 карта пригодности (разрешение 10 м)")
        if V5_OPERATIONAL_PATH.exists():
            gj_size_mb = V5_OPERATIONAL_PATH.stat().st_size / (1024 * 1024)
            if gj_size_mb < 50:
                gj_bytes = V5_OPERATIONAL_PATH.read_bytes()
                st.download_button(
                    label="📥 Скачать GPS-полигоны участков (GeoJSON, ≥10 га)",
                    data=gj_bytes,
                    file_name=V5_OPERATIONAL_PATH.name,
                    mime="application/geo+json",
                    help="Экспорт контуров оптимальных зон (только кластеры ≥10 га) для загрузки в GPS-навигаторы лесопосадочной техники",
                )
            else:
                st.info(
                    f"📥 Файл: `{V5_OPERATIONAL_PATH.name}` ({gj_size_mb:.0f} MB). "
                    "Копируйте из `outputs/data/`."
                )
    else:
        st.warning(
            f"Файл карты не найден: {V5_MAP_PATH.name}. "
            "Запустите `python scripts/run_inference_v5.py`"
        )

    # ── Two-column layout: Resource Calculator (left) + Eco Audit (right)
    col_calc, col_audit = st.columns([1, 1])

    with col_calc:
        st.markdown("### 🚜 Калькулятор ресурсов экспедиции")

        area_ha = v5_stats.get("area_ha", 0)
        if area_ha > 0:
            density = st.slider(
                "Плотность посадки (саженцев/га)",
                min_value=1000, max_value=3000, value=1500, step=100,
            )
            productivity = st.slider(
                "Производительность 1 трактора (га/смена)",
                min_value=5, max_value=20, value=10, step=1,
            )
            fuel_rate = st.slider(
                "Расход топлива трактора (л/га)",
                min_value=10.0, max_value=30.0, value=15.0, step=0.5,
            )

            total_saplings = int(area_ha * density)
            total_fuel = area_ha * fuel_rate
            total_machine_shifts = area_ha / productivity

            st.markdown("**Расчетные потребности:**")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Всего саженцев", f"{total_saplings:,}")
            col_r2.metric("Объем ГСМ (дизель, л)", f"{total_fuel:,.0f}")
            col_r3.metric("Машино-смен (всего)", f"{total_machine_shifts:,.0f}")

            st.caption(
                f"Расчет на базе {area_ha:,} га пригодных земель (V5.0). "
                "Изменяйте параметры для разных сценариев."
            )
        else:
            st.info("Загрузите данные V5.0 для расчета.")

    with col_audit:
        st.markdown("### 🔬 Спектральный аудит и структура рисков")

        if total_px > 0:
            pixels_json = json.dumps({str(k): v for k, v in v5_class_pixels.items()})
            fig_audit = make_audit_fig(pixels_json, total_px)
            if fig_audit is not None:
                st.plotly_chart(fig_audit, width='stretch')
        else:
            st.info("Растровые данные классификации не найдены.")

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
        st.markdown("**\u041f\u0440\u0430\u0432\u0438\u043b\u0430 \u043a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438 V5.0 (\u0430\u0434\u0430\u043f\u0442\u0438\u0432\u043d\u044b\u0435 \u043f\u043e\u0440\u043e\u0433\u0438):**")
        st.markdown(
            """
            | \u041a\u043b\u0430\u0441\u0441 | \u041f\u0440\u0430\u0432\u0438\u043b\u043e |
            |---|---|
            | **0 Water/NoData** | SCL\u2208[3,8,9,10] \u0438\u043b\u0438 MNDWI>0 \u0438\u043b\u0438 BI<0.15 |
            | **5 Obstacle** | Slope > 5\u00b0 |
            | **10 Vegetation** | NDVI > 0.08 |
            | **4 Dead (brine)** | NDMI > P85 **\u0438** B8/B12 > P85 |
            | **3 Risk (dry salt)** | NDSI_Green_SWIR2 > P85 **\u0438** NDMI < P15 |
            | **1 Optimal** | \u0412\u0441\u0451 \u043e\u0441\u0442\u0430\u043b\u044c\u043d\u043e\u0435 |

            \u041f\u043e\u0440\u043e\u0433\u0438 P15/P85 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u044e\u0442\u0441\u044f \u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043f\u043e \u0432\u044b\u0431\u043e\u0440\u043a\u0435 \u0438\u0437 31 564 km\u00b2.
            """
        )

        st.markdown("**\u041a\u043b\u044e\u0447\u0435\u0432\u043e\u0435 \u043e\u0442\u043a\u0440\u044b\u0442\u0438\u0435:**")
        st.markdown(
            "SI \u043e\u0442\u0431\u0440\u0430\u043a\u043e\u0432\u0430\u043d \u2014 Spearman "
            "r(Salinity vs SI) = +0.41 (p=0.21).  "
            "NDMI: **r = +0.69** (p=0.02)."
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

        st.markdown("**\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 V5.0 (10 \u043c):**")
        v5_pcts = {}
        if total_px and v5_class_pixels:
            for c in [1, 3, 4, 5, 10]:
                v5_pcts[c] = v5_class_pixels.get(c, 0) / total_px * 100
        stats_data = {
            "Class": ["1 Optimal", "3 Dry Salt", "4 Brine", "5 Obstacle", "10 Veg", "0 Water"],
            "\u041f\u0438\u043a\u0441\u0435\u043b\u0438": [
                f"{v5_class_pixels.get(1, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(3, 0)/1e3:.0f}K" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(4, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(5, 0)/1e3:.0f}K" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(10, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
                "\u2014",
            ],
            "%": [
                f"{v5_pcts.get(1, 0):.1f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(3, 0):.2f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(4, 0):.1f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(5, 0):.2f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(10, 0):.1f}%" if v5_pcts else "\u2014",
                "\u2014",
            ],
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

    # ── V5 Dynamic Thresholds ──────────────────────────────────────
    if v5_thresholds:
        st.markdown("---")
        st.subheader("\U0001f50d \u0410\u0434\u0430\u043f\u0442\u0438\u0432\u043d\u044b\u0435 \u0441\u043f\u0435\u043a\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0435 \u043f\u043e\u0440\u043e\u0433\u0438 V5.0")
        st.caption("\u041f\u043e\u0440\u043e\u0433\u0438 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u043d\u044b \u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043f\u043e \u043f\u0440\u043e\u0440\u0435\u0436\u0435\u043d\u043d\u043e\u0439 \u0432\u044b\u0431\u043e\u0440\u043a\u0435 \u0441\u043d\u0438\u043c\u043a\u0430 Sentinel-2 (\u0432\u0435\u0441\u043d\u0430 2024)")

        th_cols = st.columns(3)
        metrics_def = [
            ("\u0412\u043b\u0430\u0436\u043d\u043e\u0441\u0442\u044c (NDMI P15)", "NDMI_P15",
             "\u041d\u0438\u0436\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u0432\u043b\u0430\u0436\u043d\u043e\u0441\u0442\u0438. "
             "NDMI \u043d\u0438\u0436\u0435 \u044d\u0442\u043e\u0433\u043e \u043f\u043e\u0440\u043e\u0433\u0430 \u0443\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u043d\u0430 \u0441\u0443\u0445\u0438\u0435 \u0441\u043e\u043b\u043e\u043d\u0447\u0430\u043a\u0438 (\u043a\u043b\u0430\u0441\u0441 3)."),
            ("\u0412\u043b\u0430\u0436\u043d\u043e\u0441\u0442\u044c (NDMI P85)", "NDMI_P85",
             "\u0412\u0435\u0440\u0445\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u0432\u043b\u0430\u0436\u043d\u043e\u0441\u0442\u0438. "
             "NDMI \u0432\u044b\u0448\u0435 \u044d\u0442\u043e\u0433\u043e \u043f\u043e\u0440\u043e\u0433\u0430 \u0441\u0438\u0433\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u0443\u0435\u0442 \u043e\u0431 \u043e\u043f\u0430\u0441\u043d\u043e\u0441\u0442\u0438 \u043a\u0430\u043f\u0438\u043b\u043b\u044f\u0440\u043d\u043e\u0439 \u0440\u0430\u043f\u044b (\u043a\u043b\u0430\u0441\u0441 4)."),
            ("\u0417\u0430\u0441\u043e\u043b\u0435\u043d\u0438\u0435 (NDSI G/SWIR2 P15)", "NDSI_Green_SWIR2_P15",
             "\u041d\u0438\u0436\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u0441\u043e\u043b\u0435\u0432\u043e\u0433\u043e \u0438\u043d\u0434\u0435\u043a\u0441\u0430. "
             "\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f \u0434\u043b\u044f \u0432\u044b\u0434\u0435\u043b\u0435\u043d\u0438\u044f \u0441\u043e\u043b\u0435\u043d\u043e\u0441\u043d\u0430\u043a\u043e\u043f\u043b\u0435\u043d\u0438\u0439."),
            ("\u0417\u0430\u0441\u043e\u043b\u0435\u043d\u0438\u0435 (NDSI G/SWIR2 P85)", "NDSI_Green_SWIR2_P85",
             "\u0412\u0435\u0440\u0445\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u0441\u043e\u043b\u0435\u0432\u043e\u0433\u043e \u0438\u043d\u0434\u0435\u043a\u0441\u0430. "
             "\u041f\u0440\u0435\u0432\u044b\u0448\u0435\u043d\u0438\u0435 \u0432\u043c\u0435\u0441\u0442\u0435 \u0441 NDMI < P15 \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u044f\u0435\u0442 \u0441\u0443\u0445\u0438\u0435 \u0441\u043e\u043b\u043e\u043d\u0447\u0430\u043a\u0438 (\u043a\u043b\u0430\u0441\u0441 3)."),
            ("\u0420\u0430\u043f\u0430 (B8/B12 P15)", "BR_NIR_SWIR2_P15",
             "\u041d\u0438\u0436\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f NIR/SWIR2. "
             "\u0425\u0430\u0440\u0430\u043a\u0442\u0435\u0440\u0438\u0437\u0443\u0435\u0442 \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0443 \u043f\u043e\u0432\u0435\u0440\u0445\u043d\u043e\u0441\u0442\u0438."),
            ("\u0420\u0430\u043f\u0430 (B8/B12 P85)", "BR_NIR_SWIR2_P85",
             "\u0412\u0435\u0440\u0445\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f NIR/SWIR2. "
             "\u041f\u0440\u0435\u0432\u044b\u0448\u0435\u043d\u0438\u0435 \u0432\u043c\u0435\u0441\u0442\u0435 \u0441 NDMI > P85 \u0443\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u043d\u0430 \u043a\u0430\u043f\u0438\u043b\u043b\u044f\u0440\u043d\u0443\u044e \u0440\u0430\u043f\u0443 (\u043a\u043b\u0430\u0441\u0441 4)."),
        ]
        for i, (label, key, help_text) in enumerate(metrics_def):
            val = v5_thresholds.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            th_cols[i % 3].metric(label, val, help=help_text)

    st.markdown("---")
    st.subheader("\u0421\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u0432\u0435\u0440\u0441\u0438\u0439")

    comp_data = {
        "\u0412\u0435\u0440\u0441\u0438\u044f": ["V1.0", "V2.0", "V3.2", "V4", "**V5.0**"],
        "\u0420\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0435": ["30 m", "30 m", "30 m", "30 m", "**10 m**"],
        "\u041a\u043b\u0430\u0441\u0441\u044b": ["2", "\u2014", "3", "5", "**6**"],
        "\u0418\u043d\u0434\u0435\u043a\u0441\u044b": ["XGBoost", "\u2014", "NDMI", "NDMI+Slope", "**NDMI+NDSI+BR+NDVI+BI**"],
        "\u041f\u043e\u0440\u043e\u0433\u0438": ["ML", "\u2014", "\u0424\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0435", "\u0424\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0435", "**\u0410\u0434\u0430\u043f\u0442\u0438\u0432\u043d\u044b\u0435 P15/P85**"],
        "\u0421\u0442\u0430\u0442\u0443\u0441": ["Archive", "Frozen", "Archive", "Beta", "**Production**"],
    }
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, width="stretch")
    st.caption(
        "V5.0 \u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 \u043d\u0430 \u043d\u0430\u0442\u0438\u0432\u043d\u043e\u043c "
        "\u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0438 Sentinel-2 (10 \u043c) \u0441 "
        "\u0430\u0434\u0430\u043f\u0442\u0438\u0432\u043d\u044b\u043c\u0438 \u043f\u043e\u0440\u043e\u0433\u0430\u043c\u0438 "
        "\u043f\u043e\u0434 \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u0443\u044e \u0441\u0446\u0435\u043d\u0443."
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
