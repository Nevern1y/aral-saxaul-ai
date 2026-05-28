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
import html
try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None

os.environ["MPLBACKEND"] = "Agg"

BASE_DIR = Path(__file__).resolve().parent
AOI_VECTOR_PATH = BASE_DIR / "outputs" / "aoi" / "aral_sea_1960.geojson"

TASKS_PATH = BASE_DIR / "outputs" / "logistics" / "tasks_index_v5_enriched.csv"
ROADS_PATH = BASE_DIR / "outputs" / "logistics" / "aralkum_roads.geojson"
GRID_STEP = 0.1

# ── V5.0 paths (strict — no V4 fallback) ──────────────────────────────
V5_MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v5.html"
V5_OPERATIONAL_PATH = BASE_DIR / "outputs" / "data" / "operational_zones_v5.geojson"
V5_THRESHOLDS_PATH = BASE_DIR / "outputs" / "data" / "thresholds_v5.json"
V5_STATS_PATH = BASE_DIR / "outputs" / "data" / "v5_stats.json"
SCIENCE_DIR = BASE_DIR / "outputs" / "science"
V5_POINT_SAMPLES_PATH = SCIENCE_DIR / "v5_point_samples.csv"
V5_VALIDATION_SUMMARY_PATH = SCIENCE_DIR / "v5_validation_summary.json"
V5_UNCERTAINTY_SUMMARY_PATH = SCIENCE_DIR / "v5_uncertainty_summary.json"
V5_VALIDATION_REPORT_PATH = SCIENCE_DIR / "v5_validation_report.md"
V5_COORDINATE_ADJUDICATION_REPORT_PATH = SCIENCE_DIR / "v5_coordinate_adjudication_report.md"
V5_UNCERTAINTY_REPORT_PATH = SCIENCE_DIR / "v5_uncertainty_report.md"

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


def _render_map(html_str, height=700):
    """Embed a full HTML page (Folium map) via components.html to avoid CSP issues."""
    components.html(html_str, height=height, scrolling=False)


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
    pixel_area_ha = 0.01
    path = BASE_DIR / "outputs" / "data" / "suitability_map_v5_filtered.tif"
    if path.exists() and rasterio is not None:
        with rasterio.open(path) as src:
            pixel_area_ha = abs(src.res[0] * src.res[1]) / 10000.0
            arr = src.read(1)
        total_px = arr.size
        for cls_val in [0, 1, 3, 4, 5, 10]:
            pixels[cls_val] = int((arr == cls_val).sum())
    return pixels, total_px, pixel_area_ha


@st.cache_data
def load_v5_thresholds():
    path = BASE_DIR / "outputs" / "data" / "thresholds_v5.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_v5_point_samples():
    if V5_POINT_SAMPLES_PATH.exists():
        return pd.read_csv(V5_POINT_SAMPLES_PATH)
    return pd.DataFrame()


@st.cache_data
def load_v5_validation_summary():
    if V5_VALIDATION_SUMMARY_PATH.exists():
        with open(V5_VALIDATION_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_v5_uncertainty_summary():
    if V5_UNCERTAINTY_SUMMARY_PATH.exists():
        with open(V5_UNCERTAINTY_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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
            "Кандидатные зоны",
            "Есть растительность",
            "Риск влажной рапы",
            "Риск сухой соли",
            "Сложный рельеф",
            "Вода / тень / нет данных",
        ],
        "Доля (%)": [
            round(opt_pct, 1), round(veg_pct, 1),
            round(brine_pct, 1), round(salt_pct, 1),
            round(obst_pct, 1), round(water_pct, 1),
        ],
    })

    audit_colors = {
        "Кандидатные зоны": "#2ecc40",
        "Есть растительность": "#7FCDBB",
        "Риск влажной рапы": "#D95F02",
        "Риск сухой соли": "#E6AB02",
        "Сложный рельеф": "#636363",
        "Вода / тень / нет данных": "#BDBDBD",
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


st.title("Aral Saxaul V5.1: карта предварительного отбора")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "Карта показывает, где условия по спутниковым данным выглядят менее рискованными. "
    "Это не окончательное решение о посадке, а список мест для проверки в поле."
    "</p>",
    unsafe_allow_html=True,
)

tab_analytics, tab_dev, tab_logistics = st.tabs([
    "Карта и итоги",
    "Проверка и данные",
    "Планирование работ",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: 📍 Карта рабочих участков
# ══════════════════════════════════════════════════════════════════════

with tab_logistics:
    st.subheader("Планирование работ по выбранным зонам")
    st.caption(
        "Здесь карта используется для грубого планирования выездов: можно отфильтровать участки по расстоянию до дорог и размеру. Это не подтверждает пригодность места для посадки."
    )
    try:
        tasks_df = load_tasks()
        roads_gdf = load_roads()
        v5_stats = load_v5_stats()
    except FileNotFoundError:
        st.error("Данные для планирования V5.1 не найдены.")
        st.info(
            "Пожалуйста, запустите базовые скрипты генерации дорог и сетки нарядов:\n"
            "1. `python scripts/v5_roads_prep.py`\n"
            "2. `python scripts/v5_logistics_prep.py`"
        )
        st.stop()

    if tasks_df.empty:
        st.warning(
            f"Файл не найден: {TASKS_PATH}. "
            "Запустите `python scripts/v5_logistics_prep.py`"
        )
    else:
        if "territory_scope" in tasks_df.columns and set(tasks_df["territory_scope"].dropna()) == {"kazakhstan"}:
            st.caption("Логистика ниже посчитана только для candidate-зон внутри территории Казахстана.")

        tasks_df["distance_to_road_km"] = pd.to_numeric(tasks_df["distance_to_road_km"], errors="coerce")
        if "distance_to_kazakhstan_road_km" in tasks_df.columns:
            tasks_df["distance_to_kazakhstan_road_km"] = pd.to_numeric(
                tasks_df["distance_to_kazakhstan_road_km"],
                errors="coerce",
            )
        access_options = {"Любые дороги OSM": "distance_to_road_km"}
        if "distance_to_kazakhstan_road_km" in tasks_df.columns and tasks_df["distance_to_kazakhstan_road_km"].notna().any():
            access_options["Казахстанский подъезд"] = "distance_to_kazakhstan_road_km"

        max_cell_ha = float(tasks_df["area_ha"].max())

        col_f0, col_f1, col_f2 = st.columns(3)
        with col_f0:
            selected_access = st.selectbox(
                "Как считать подъезд:",
                options=list(access_options.keys()),
                index=1 if "Казахстанский подъезд" in access_options else 0,
            )
            distance_col = access_options[selected_access]

        with col_f1:
            max_dist = float(tasks_df[distance_col].max())
            road_scenarios = {
                "Близко к дорогам (до 120 км)": 120.0,
                "Дальние выезды (до 250 км от дорог)": 250.0,
                "Показать весь охват": max_dist,
            }
            selected_road_scen = st.selectbox(
                "Доступность по дорогам:",
                options=list(road_scenarios.keys()),
                index=2,
            )
            dist_thresh = road_scenarios[selected_road_scen]

        with col_f2:
            area_scenarios = {
                "Малые участки (10-1 000 га)": (10, 1000),
                "Крупные участки (1 000-5 000 га)": (1000, 5000),
                "Очень крупные участки (>5 000 га)": (5000, int(max_cell_ha)),
                "Все размеры": (0, int(max_cell_ha)),
            }
            selected_area_scen = st.selectbox(
                "Размер участка:",
                options=list(area_scenarios.keys()),
                index=3,
            )
            min_area, max_area = area_scenarios[selected_area_scen]

        filtered = tasks_df[
            (tasks_df[distance_col] <= dist_thresh)
            & (tasks_df["area_ha"] >= min_area)
            & (tasks_df["area_ha"] <= max_area)
        ]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        total_ha = v5_stats.get("candidate_100m_area_ha", v5_stats.get("area_ha", 0))
        col_m1.metric(
            "Оценка площади по сетке 100 м",
            f"{total_ha:,.0f}",
            delta=f"{filtered['area_ha'].sum():,.0f} га выбрано" if not filtered.empty else None,
        )
        col_m2.metric("Выбрано ячеек", f"{len(filtered):,}")
        col_m3.metric("Всего ячеек", f"{len(tasks_df):,}")
        col_m4.metric(
            "Доля выбранных",
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
                        "dist_km": round(row[distance_col], 2),
                        "dist_kz_km": round(row["distance_to_kazakhstan_road_km"], 2)
                        if "distance_to_kazakhstan_road_km" in row and pd.notna(row["distance_to_kazakhstan_road_km"])
                        else None,
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
                    fields=["filename", "area_ha", "dist_km", "dist_kz_km"],
                    aliases=["Файл:", "Площадь (га):", "До выбранной дороги (км):", "До дороги KZ (км):"],
                    localize=True,
                ),
                highlight_function=lambda f: {"weight": 2.0, "color": "#007bff"},
            ).add_to(m)

            top5 = filtered.nsmallest(5, distance_col)
            for _, row in top5.iterrows():
                folium.Marker(
                    location=[row["centroid_lat"], row["centroid_lon"]],
                    popup=(
                        f"{row['filename']}<br>"
                        f"{row['area_ha']:.0f} ha, {row[distance_col]:.2f} km"
                    ),
                    icon=folium.Icon(color="green", icon="ok-sign", prefix="glyphicon"),
                ).add_to(m)

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MousePosition().add_to(m)

        _render_map(m.get_root().render())

        with st.expander("\U0001f4c2 Список маршрутных файлов (KML)"):
            display_cols = ["filename", "centroid_lat", "centroid_lon", "area_ha", "distance_to_road_km"]
            if "distance_to_kazakhstan_road_km" in filtered.columns:
                display_cols.append("distance_to_kazakhstan_road_km")
            display_df = filtered[display_cols].copy()
            display_df.columns = [
                "Файл KML", "Широта", "Долгота", "Площадь (га)", "До любой дороги (км)",
                *(["До дороги KZ (км)"] if "distance_to_kazakhstan_road_km" in filtered.columns else []),
            ]
            st.dataframe(
                display_df.sort_values("До дороги KZ (км)" if distance_col == "distance_to_kazakhstan_road_km" else "До любой дороги (км)", ascending=True),
                hide_index=True,
                width="stretch",
            )

        with st.expander("Примерный расчет ресурсов для выбранных участков", expanded=False):
            selected_area_ha = float(filtered["area_ha"].sum()) if not filtered.empty else 0.0
            st.caption(
                "Это только предварительный расчет для выбранных ячеек. Он не означает, что все эти места уже можно засаживать."
            )
            if selected_area_ha > 0:
                density = st.slider(
                    "Плотность посадки (саженцев/га)",
                    min_value=1000, max_value=3000, value=1500, step=100,
                )
                productivity = st.slider(
                    "Производительность трактора (га/смена)",
                    min_value=5, max_value=20, value=10, step=1,
                )
                fuel_rate = st.slider(
                    "Расход дизеля (л/га)",
                    min_value=10.0, max_value=30.0, value=15.0, step=0.5,
                )
                total_saplings = int(selected_area_ha * density)
                total_fuel = selected_area_ha * fuel_rate
                total_machine_shifts = selected_area_ha / productivity

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Площадь в выбранном фильтре", f"{selected_area_ha:,.0f} га")
                col_r2.metric("Саженцы", f"{total_saplings:,}")
                col_r3.metric("Машино-смены", f"{total_machine_shifts:,.0f}")
                st.metric("Дизель, примерно", f"{total_fuel:,.0f} л")
            else:
                st.info("По текущим фильтрам участков нет.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2: 📊 Общая статистика
# ══════════════════════════════════════════════════════════════════════

with tab_analytics:
    # ── All heavy I/O goes through @st.cache_data (runs once) ────────────
    try:
        v5_stats = load_v5_stats()
        v5_class_pixels, total_px, pixel_area_ha = load_v5_class_pixels()
        v5_thresholds = load_v5_thresholds()
    except FileNotFoundError:
        st.error("Данные карты V5.1 не найдены.")
        st.info(
            "Пожалуйста, запустите сначала локальные скрипты генерации данных:\n"
            "1. `python scripts/run_inference_v5.py` — расчет маски\n"
            "2. `python scripts/v5_finalize_viz.py` — фильтр и карта\n"
            "3. `python scripts/v5_extract_stats.py` — извлечение статистики"
        )
        st.stop()

    # ── Top metrics panel (data from v5_stats.json) ──────────────────
    candidate_ha_exact = v5_class_pixels.get(1, 0) * pixel_area_ha if v5_class_pixels else 0
    candidate_km2_exact = candidate_ha_exact / 100.0
    candidate_100m_area_ha = v5_stats.get("candidate_100m_area_ha", v5_stats.get("area_ha", 0))
    operational_area_ha = v5_stats.get("operational_area_ha", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Кандидатные зоны, 10 м",
        f"{candidate_ha_exact:,.0f} га",
    )
    col2.metric(
        "Участки >=10 га",
        f"{v5_stats.get('clusters', 0):,}",
    )
    col3.metric(
        "Площадь 10 м",
        f"{candidate_km2_exact:,.0f} км²",
    )
    col4.metric("Оценка по сетке 100 м", f"{candidate_100m_area_ha:,.0f} га")

    # ── Карта на самом видном месте ─────────────────────────────────
    st.markdown("### Карта предварительного отбора")
    if V5_MAP_PATH.exists():
        _render_map(V5_MAP_PATH.read_text(encoding="utf-8"))
        st.caption("Карта 10 м: предварительный отбор мест для проверки. Это не доказательство, что посадка там точно приживется.")
        if V5_OPERATIONAL_PATH.exists():
            gj_size_mb = V5_OPERATIONAL_PATH.stat().st_size / (1024 * 1024)
            if gj_size_mb < 50:
                gj_bytes = V5_OPERATIONAL_PATH.read_bytes()
                st.download_button(
                    label="Скачать полигоны участков (GeoJSON, >=10 га)",
                    data=gj_bytes,
                    file_name=V5_OPERATIONAL_PATH.name,
                    mime="application/geo+json",
                    help="Экспорт контуров candidate-зон (только кластеры ≥10 га) для GPS и дальнейшей полевой проверки",
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

    st.markdown("### Что посчитала карта")
    st.info(
        "Карта делит территорию на 6 классов по спутниковым данным Sentinel-2 с шагом 10 м. "
        "`Кандидатные зоны` — это места, которые не попали в воду/тень, сложный рельеф, существующую растительность, сухую соль или влажную рапу. "
        "Их все равно нужно проверять на месте."
    )

    if total_px and v5_class_pixels:
        class_meta = {
            0: ("Вода / тень / нет данных", "Исключено: вода, тени, нет данных или очень темные пиксели"),
            1: ("Кандидатные зоны", "Главный результат карты: места для первоочередной проверки"),
            3: ("Риск сухой соли", "Возможная сухая солевая корка по спутниковым признакам"),
            4: ("Риск влажной рапы", "Возможная влажная соленая поверхность или подъем рапы"),
            5: ("Сложный рельеф", "Склоны или резкие формы рельефа"),
            10: ("Есть растительность", "Там уже есть растительность; это не класс новой посадки"),
        }
        class_rows = []
        for cls_val in [1, 10, 4, 3, 5, 0]:
            pixel_count = v5_class_pixels.get(cls_val, 0)
            area_ha = pixel_count * pixel_area_ha
            class_rows.append(
                {
                    "Код": cls_val,
                    "Класс": class_meta[cls_val][0],
                    "Площадь, га (карта 10 м)": round(area_ha, 1),
                    "Площадь, км²": round(area_ha / 100.0, 1),
                    "Доля территории, %": round(pixel_count / total_px * 100.0, 2),
                    "Как понимать": class_meta[cls_val][1],
                }
            )
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")

    top10_ha = v5_stats.get("top10_ha", [])
    if top10_ha:
        top10_share = sum(top10_ha) / candidate_100m_area_ha * 100 if candidate_100m_area_ha else 0
        st.caption(
            f"10 самых крупных связанных зон по сетке 100 м: {sum(top10_ha):,.0f} га "
            f"({top10_share:.1f}% от candidate-оценки по сетке 100 м). "
            "То есть результат карты в основном собран в нескольких крупных массивах, а не только в мелких пятнах."
        )
    if operational_area_ha:
        st.caption(
            f"Площадь operational-полигонов >=10 га: {operational_area_ha:,.0f} га. "
            "Она меньше общей оценки по сетке 100 м, потому что мелкие пятна исключены из рабочего GeoJSON."
        )

    # ── Scientific interpretation (left) + spectral audit (right)
    col_interp, col_audit = st.columns([1, 1])

    with col_interp:
        st.markdown("### Главные выводы без преувеличения")
        conclusion_rows = [
            {
                "Вопрос": "Что это за карта",
                "Что есть сейчас": "Карта предварительного отбора по правилам и спутниковым данным",
                "Как понимать": "Она помогает выбрать места для проверки, но не доказывает, что посадка там точно приживется.",
            },
            {
                "Вопрос": "Площадь кандидатных зон",
                "Что есть сейчас": f"{candidate_ha_exact:,.0f} га по карте 10 м",
                "Как понимать": "Это площадь, которую карта не исключила по основным признакам риска.",
            },
            {
                "Вопрос": "Участки для выезда",
                "Что есть сейчас": f"{v5_stats.get('clusters', 0):,} связанных зон >=10 га по упрощенной сетке",
                "Как понимать": "Это слой для навигации и планирования, не замена исходной карты 10 м.",
            },
            {
                "Вопрос": "Главное ограничение",
                "Что есть сейчас": "Для 11 полевых точек есть конфликт координат",
                "Как понимать": "Пока не выбраны правильные координаты, проверка карты остается предварительной.",
            },
        ]
        st.dataframe(pd.DataFrame(conclusion_rows), hide_index=True, width="stretch")

    with col_audit:
        st.markdown("### Доля классов на карте")

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
        "Здесь собраны правила карты, пороги, отчеты проверки и ограничения. Сложные слова ниже поясняются простым языком."
    )

    st.subheader("Как работает карта и что еще нужно проверить")

    roadmap_rows = [
        {
            "Очередь": 1,
            "Что сделать": "Разобраться с координатами",
            "Что нужно": "Правильный GPS/ODT источник для S124-S134 и координаты для профилей 12/20A-21/20A",
            "Зачем": "Чтобы проверять карту по правильным точкам, а не по спорным координатам.",
        },
        {
            "Очередь": 2,
            "Что сделать": "Добавить слой уверенности",
            "Что нужно": "Проверить, насколько зона меняется при других порогах и на разных датах снимков",
            "Зачем": "Отделить устойчивые зоны от пограничных мест, где карта может ошибаться.",
        },
        {
            "Очередь": 3,
            "Что сделать": "Полевые точки по всем классам",
            "Что нужно": "Точки не только в кандидатных зонах, но и в соли, рапе, растительности, воде/тени",
            "Зачем": "Понять, где карта работает хорошо, а где ошибается.",
        },
        {
            "Очередь": 4,
            "Что сделать": "Связать карту с почвой по глубине",
            "Что нужно": "Соленость, EC, pH, мехсостав и гипс по слоям почвы",
            "Зачем": "Спутник видит поверхность, а корни зависят от почвы глубже.",
        },
        {
            "Очередь": 5,
            "Что сделать": "Сделать балл пригодности",
            "Что нужно": "Больше проверенных точек или данные о приживаемости посадок",
            "Зачем": "Тогда можно будет оценивать не только классы риска, но и степень пригодности.",
        },
    ]
    with st.expander("Что улучшать дальше", expanded=True):
        st.dataframe(pd.DataFrame(roadmap_rows), hide_index=True, width="stretch")

    col_rules, col_stats = st.columns([1, 1])

    with col_rules:
        st.markdown("**Правила карты V5.1:**")
        st.markdown(
            """
            | Класс | Простое правило |
            |---|---|
            | **0 Вода / тень / нет данных** | Пиксель похож на воду, тень или плохие данные |
            | **5 Сложный рельеф** | Уклон больше 5 градусов |
            | **10 Есть растительность** | NDVI выше 0.08, то есть уже есть зеленая растительность |
            | **4 Риск влажной рапы** | Признаки высокой влажности и соленой поверхности одновременно |
            | **3 Риск сухой соли** | Признаки сухой солевой корки |
            | **1 Кандидатные зоны** | Все, что не попало в перечисленные выше риски |

            Пороги P15/P85 считаются автоматически по текущему снимку. P15 и P85 - это нижняя и верхняя границы, которые помогают отделить обычные значения от крайних.
            """
        )

        st.markdown("**Что показывает текущая проверка:**")
        st.markdown(
            "По 11 точкам видно, что карта уже полезна для первичного отбора, но координаты этих точек конфликтуют между собой. "
            "Поэтому это пока предварительная проверка, а не окончательная оценка точности."
        )

    with col_stats:
        st.markdown("**Граница расчетной области:**")
        st.markdown(
            "Расчет идет только там, где есть рабочие спутниковые данные и где пиксель не отнесен к воде. "
            "Старая маска V1.0 больше не используется."
        )

        st.markdown("**Распределение классов V5.1 (10 м):**")
        v5_pcts = {}
        if total_px and v5_class_pixels:
            for c in [0, 1, 3, 4, 5, 10]:
                v5_pcts[c] = v5_class_pixels.get(c, 0) / total_px * 100
        stats_data = {
            "Класс": ["1 Кандидат", "3 Сухая соль", "4 Влажная рапа", "5 Рельеф", "10 Растительность", "0 Вода/тень"],
            "\u041f\u0438\u043a\u0441\u0435\u043b\u0438": [
                f"{v5_class_pixels.get(1, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(3, 0)/1e3:.0f}K" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(4, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(5, 0)/1e3:.0f}K" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(10, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
                f"{v5_class_pixels.get(0, 0)/1e6:.1f}M" if v5_class_pixels else "\u2014",
            ],
            "%": [
                f"{v5_pcts.get(1, 0):.1f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(3, 0):.2f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(4, 0):.1f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(5, 0):.2f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(10, 0):.1f}%" if v5_pcts else "\u2014",
                f"{v5_pcts.get(0, 0):.1f}%" if v5_pcts else "\u2014",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)

        st.markdown("**Важное ограничение:**")
        st.markdown(
            "Граница расчетной области должна быть проверена отдельно. Если нужна более строгая обрезка по старому дну Арала, нужно использовать точную береговую линию."
        )

    # ── V5 Dynamic Thresholds ──────────────────────────────────────
    if v5_thresholds:
        st.markdown("---")
        st.subheader("Пороги, по которым карта делит классы")
        st.caption("Пороги посчитаны автоматически по снимку Sentinel-2. Они нужны, чтобы отделить обычные значения от явно сухих, влажных или соленых участков.")

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
    st.subheader("Отчеты проверки")
    st.caption(
        "Эти отчеты показывают, как проверялась карта, где есть спорные координаты и насколько результат зависит от порогов. Это не финальная оценка точности."
    )
    report_cols = st.columns(3)
    report_defs = [
        ("Проверка по точкам", V5_VALIDATION_REPORT_PATH, "python scripts/v5_validation_report.py"),
        (
            "Выбор правильных координат",
            V5_COORDINATE_ADJUDICATION_REPORT_PATH,
            "python scripts/v5_coordinate_adjudication_report.py",
        ),
        ("Чувствительность порогов", V5_UNCERTAINTY_REPORT_PATH, "python scripts/v5_uncertainty_report.py"),
    ]
    for idx, (label, path, command) in enumerate(report_defs):
        with report_cols[idx]:
            if path.exists():
                st.download_button(
                    label=f"Скачать: {label}",
                    data=path.read_text(encoding="utf-8"),
                    file_name=path.name,
                    mime="text/markdown",
                )
                with st.expander(f"Посмотреть: {label}"):
                    st.markdown(path.read_text(encoding="utf-8"))
            else:
                st.info(f"Чтобы создать отчет, запустите `{command}`.")

    st.markdown("---")
    st.subheader("Что изменилось по версиям")

    comp_data = {
        "\u0412\u0435\u0440\u0441\u0438\u044f": ["V1.0", "V2.0", "V3.2", "V4", "**V5.1**"],
        "\u0420\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0435": ["30 m", "30 m", "30 m", "30 m", "**10 m**"],
        "\u041a\u043b\u0430\u0441\u0441\u044b": ["2", "\u2014", "3", "5", "**6**"],
        "\u0418\u043d\u0434\u0435\u043a\u0441\u044b": ["XGBoost", "\u2014", "NDMI", "NDMI+Slope", "**NDMI+NDSI+BR+NDVI+BI**"],
        "\u041f\u043e\u0440\u043e\u0433\u0438": ["ML", "\u2014", "\u0424\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0435", "\u0424\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0435", "**\u0410\u0434\u0430\u043f\u0442\u0438\u0432\u043d\u044b\u0435 P15/P85**"],
        "\u0421\u0442\u0430\u0442\u0443\u0441": ["Архив", "Заморожена", "Архив", "Пробная", "**Текущая рабочая версия**"],
    }
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, width="stretch")
    st.caption(
        "V5.1 \u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 \u043d\u0430 \u043d\u0430\u0442\u0438\u0432\u043d\u043e\u043c "
        "\u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0438 Sentinel-2 (10 \u043c) \u0441 "
        "\u0430\u0434\u0430\u043f\u0442\u0438\u0432\u043d\u044b\u043c\u0438 \u043f\u043e\u0440\u043e\u0433\u0430\u043c\u0438 "
        "\u043f\u043e\u0434 \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u0443\u044e \u0441\u0446\u0435\u043d\u0443."
    )

    # ── Pilot validation ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Проверка по 11 полевым точкам")

    validation_summary = load_v5_validation_summary()
    uncertainty_summary = load_v5_uncertainty_summary()
    point_samples = load_v5_point_samples()

    if point_samples.empty or not validation_summary:
        st.warning("Файлы проверки V5.1 не найдены.")
        st.info(
            "Сначала запустите полный расчет:\n"
            "1. `python scripts/build_v5_science_dataset.py`\n"
            "2. `python scripts/v5_validation_report.py`\n"
            "3. `python scripts/v5_uncertainty_report.py`"
        )
    else:
        conflict_m = validation_summary.get("coordinate_conflict_median")
        col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
        col_v1.metric("Статус координат", validation_summary.get("coordinate_policy", "n/a"))
        col_v2.metric(
            "Расхождение координат",
            f"{conflict_m / 1000:.1f} km" if conflict_m is not None else "n/a",
        )
        col_v3.metric("Профили с координатами", f"{validation_summary.get('n_profiles_with_coordinates', 0):,}")
        col_v4.metric("Только лабораторные", f"{validation_summary.get('n_lab_only_profiles', 0):,}")
        col_v5.metric("Подтвержденные точки", f"{validation_summary.get('n_authoritative_point_samples', 0):,}")

        st.caption(
            "Сейчас хранятся два набора координат: исходные и сдвинутые. Они проверяются отдельно. "
            "Когда будут выбраны правильные координаты, они появятся отдельной группой подтвержденных точек."
        )

        st.markdown("### В какие классы попали точки")
        class_dist = (
            point_samples.groupby(["coordinate_source", "class_filtered_name"])
            .size()
            .reset_index(name="count")
            .sort_values(["coordinate_source", "count"], ascending=[True, False])
        )
        class_dist.columns = ["Источник координат", "Класс карты", "Количество"]
        st.dataframe(class_dist, hide_index=True, width="stretch")

        corr_df = pd.DataFrame(validation_summary.get("correlations", []))
        if not corr_df.empty:
            corr_view = corr_df[
                (corr_df["target"] == "top_salinity_pct")
                & (corr_df["feature"].isin(["ndmi", "ndsi_green_swir2", "br_nir_swir2"]))
            ][["coordinate_source", "feature", "n", "spearman_r", "p_value", "bootstrap_ci95", "status"]].copy()
            corr_view.columns = [
                "Источник координат",
                "Показатель",
                "n",
                "Spearman r",
                "p-value",
                "Интервал",
                "Статус",
            ]
            st.markdown("### Связь карты с верхним слоем солености")
            st.caption("Spearman r показывает, идут ли два показателя в одну сторону. При 11 точках это только подсказка, а не доказательство.")
            st.dataframe(corr_view, hide_index=True, width="stretch")

        plot_df = point_samples.copy()
        for col in ["top_salinity_pct", "ndmi", "br_nir_swir2", "field_ec_0_20"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=["top_salinity_pct"])

        if not plot_df.empty:
            st.markdown("### Графики по точкам")
            fig_ndmi = px.scatter(
                plot_df.dropna(subset=["ndmi"]),
                x="ndmi",
                y="top_salinity_pct",
                color="coordinate_source",
                symbol="class_filtered_name",
                hover_data=["S_Point", "pit_code", "field_ec_0_20"],
                labels={
                    "ndmi": "V5 NDMI",
                    "top_salinity_pct": "Соленость верхнего слоя (%)",
                    "coordinate_source": "Источник координат",
                    "class_filtered_name": "Класс карты",
                },
                title="NDMI и соленость верхнего слоя",
            )
            fig_ndmi.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ndmi, width="stretch")

            fig_br = px.scatter(
                plot_df.dropna(subset=["br_nir_swir2"]),
                x="br_nir_swir2",
                y="top_salinity_pct",
                color="coordinate_source",
                symbol="class_filtered_name",
                hover_data=["S_Point", "pit_code", "field_ec_0_20"],
                labels={
                    "br_nir_swir2": "Отношение B8/B12",
                    "top_salinity_pct": "Соленость верхнего слоя (%)",
                    "coordinate_source": "Источник координат",
                    "class_filtered_name": "Класс карты",
                },
                title="B8/B12 и соленость верхнего слоя",
            )
            fig_br.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_br, width="stretch")

        if uncertainty_summary:
            st.markdown("### Насколько результат зависит от порогов")
            scenario_rows = []
            for scenario, rows in uncertainty_summary.get("class_area_by_scenario", {}).items():
                candidate = next((row for row in rows if row.get("class") == 1), None)
                if candidate is None:
                    continue
                scenario_rows.append(
                    {
                        "Сценарий": scenario,
                        "Кандидатная площадь, га (примерно)": candidate.get("area_ha_approx"),
                        "Кандидатная площадь, км² (примерно)": candidate.get("area_km2_approx"),
                        "Доля сетки, %": candidate.get("pct_of_sample_grid"),
                    }
                )
            if scenario_rows:
                st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, width="stretch")

            stability_rows = []
            stability_df = pd.DataFrame(uncertainty_summary.get("point_stability", []))
            if not stability_df.empty:
                for source, group in stability_df.groupby("coordinate_source"):
                    stable = int(group["stable_across_scenarios"].sum())
                    stability_rows.append(
                        {
                            "Источник координат": source,
                            "Стабильных точек": stable,
                            "Всего точек": int(len(group)),
                            "Доля стабильных, %": round(stable / len(group) * 100, 1) if len(group) else 0,
                        }
                    )
            if stability_rows:
                st.dataframe(pd.DataFrame(stability_rows), hide_index=True, width="stretch")

        st.markdown("### Таблица проверки по точкам")
        display_cols = [
            "coordinate_source", "S_Point", "pit_code", "top_salinity_pct", "field_salinity_0_20",
            "field_ec_0_20", "class_filtered_name", "ndmi", "ndsi_green_swir2", "br_nir_swir2",
            "coordinate_conflict_m",
        ]
        display_df = point_samples[[col for col in display_cols if col in point_samples.columns]].copy()
        display_df.columns = [
            "Источник координат", "Точка", "Разрез", "Соленость сверху, %", "Полев. соленость 0-20",
            "Полев. EC 0-20", "Класс карты", "NDMI", "NDSI G/SWIR2", "B8/B12", "Конфликт координат, м",
        ][:len(display_df.columns)]
        st.dataframe(display_df, hide_index=True, width="stretch")

        st.caption(
            "Эта проверка предварительная. Главное ограничение сейчас - нужно выбрать правильные координаты для 11 точек."
        )
