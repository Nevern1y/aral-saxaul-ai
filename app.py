import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium import plugins
from shapely.geometry import box

try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None

os.environ["MPLBACKEND"] = "Agg"

BASE_DIR = Path(__file__).resolve().parent
AOI_VECTOR_PATH = BASE_DIR / "outputs" / "aoi" / "aral_sea_1960.geojson"

TASKS_PATH = BASE_DIR / "outputs" / "logistics" / "tasks_index_v5_enriched.csv"
ROADS_PATH = BASE_DIR / "outputs" / "logistics" / "aralkum_roads.geojson"
KML_TASKS_DIR = BASE_DIR / "outputs" / "logistics" / "tractor_tasks_v5"
GRID_STEP = 0.1

# ── V5.0 paths (strict — no V4 fallback) ──────────────────────────────
V5_MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v5.html"
V6_MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v6.html"
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

# ── V6 science paths (lab-data layer; JSON/CSV tracked, rasters regenerated) ──
CANON_DIR = BASE_DIR / "data" / "canonical"
V6_SALINITY_MODEL_PATH = BASE_DIR / "outputs" / "models" / "salinity_v6_logit.json"
V6_SUIT_STATS_PATH = BASE_DIR / "outputs" / "data" / "suitability_v6_stats.json"
V6_PIT_VALIDATION_PATH = BASE_DIR / "outputs" / "data" / "suitability_v6_pit_validation_summary.json"
V6_SPATIAL_PATH = BASE_DIR / "outputs" / "data" / "spatial_validation_v6.json"
V6_BENCHMARK_PATH = CANON_DIR / "model_v6_benchmark.json"
V6_PIT_TABLE_PATH = CANON_DIR / "suitability_v6_pit_validation.csv"

st.set_page_config(page_title="Aral Saxaul: Платформа Фитомелиорации", layout="wide")
if not hasattr(st, "iframe"):
    st.error("Нужен Streamlit 1.57+ для отображения интерактивных карт.")
    st.stop()

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
    """Embed a full Folium HTML page in Streamlit's current iframe API."""
    st.iframe(html_str, height=height, width="stretch")


@st.cache_data
def load_tasks():
    if not TASKS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TASKS_PATH)
    return df


@st.cache_data
def load_roads():
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        return None
    if ROADS_PATH.exists():
        return gpd.read_file(ROADS_PATH)
    return None


def zip_kml_files(filenames):
    """Pack selected KML files for GPS/Google Earth handoff."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            path = KML_TASKS_DIR / str(name)
            if path.exists():
                zf.write(path, arcname=path.name)
    buf.seek(0)
    return buf.getvalue()


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


# ── Cached V5 data loaders (heavy I/O → once per session) ──────────────
@st.cache_data
def load_v5_stats():
    path = BASE_DIR / "outputs" / "data" / "v5_stats.json"
    if path.exists():
        with open(path) as f:
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
    else:
        stats_path = BASE_DIR / "outputs" / "data" / "v5_stats.json"
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
            pixels = {
                int(cls_val): int(count)
                for cls_val, count in stats.get("class_pixels_10m", {}).items()
            }
            total_px = int(stats.get("class_total_pixels_10m", 0))
            pixel_area_ha = float(stats.get("pixel_area_ha_10m", pixel_area_ha))
    return pixels, total_px, pixel_area_ha


@st.cache_data
def load_v5_thresholds():
    path = BASE_DIR / "outputs" / "data" / "thresholds_v5.json"
    if path.exists():
        with open(path) as f:
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
        with open(V5_VALIDATION_SUMMARY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_v5_uncertainty_summary():
    if V5_UNCERTAINTY_SUMMARY_PATH.exists():
        with open(V5_UNCERTAINTY_SUMMARY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def _load_json(path_str):
    p = Path(path_str)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_v6_science():
    """Load the V6 lab-data science artifacts (graceful if absent)."""
    return {
        "salinity": _load_json(str(V6_SALINITY_MODEL_PATH)),
        "suit_stats": _load_json(str(V6_SUIT_STATS_PATH)),
        "pit_validation": _load_json(str(V6_PIT_VALIDATION_PATH)),
        "spatial": _load_json(str(V6_SPATIAL_PATH)),
        "benchmark": _load_json(str(V6_BENCHMARK_PATH)),
        "pit_table": (pd.read_csv(V6_PIT_TABLE_PATH) if V6_PIT_TABLE_PATH.exists() else pd.DataFrame()),
    }


def v6_model_metrics(v6):
    """Return current V6 headline metrics, preferring the benchmarked shipped model."""
    rec = v6.get("benchmark", {}).get("recommendation", {})
    sal = v6.get("salinity", {})
    spatial = v6.get("spatial", {}).get("salinity_model", {})
    training = sal.get("training", {})
    return {
        "auc": rec.get("baseline_loo_auc", spatial.get("loo_auc", training.get("loo_auc"))),
        "ci": rec.get("baseline_loo_auc_ci95", spatial.get("loo_auc_ci95")),
        "n": v6.get("benchmark", {}).get("n_total", training.get("n", spatial.get("n"))),
        "n_saline": training.get("n_saline", spatial.get("n_saline")),
    }


def v6_aoi_split(v6):
    split = v6.get("benchmark", {}).get("aoi_split", {})
    return {
        "n_in": split.get("n_in_aoi"),
        "n_out": split.get("n_out_of_aoi"),
    }


def fmt_metric(value, digits=3):
    return f"{float(value):.{digits}f}" if value is not None else "—"


def render_technical_commands(commands):
    with st.expander("Для технического специалиста"):
        for command in commands:
            st.code(command, language="bash")


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


st.title("Aral Saxaul V6: предварительный отбор участков по риску засоления")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "Карта помогает выбрать, куда ехать на полевую проверку: она оценивает риск засоления, "
    "подсказывает следующий шаг по выбранной точке и не заменяет почвенную пробу или решение о посадке. "
    "V6 — основной слой, V5.1 используется как детализация и основа KML-планирования."
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
    st.subheader("Планирование полевого выезда")
    st.info(
        "Рабочий порядок: 1) выберите доступные по дороге участки, 2) скачайте KML, "
        "3) проверьте почву и координаты в поле, 4) только после этого считайте посадку и бюджет."
    )
    tasks_df = load_tasks()
    roads_gdf = load_roads()
    v5_stats = load_v5_stats()

    if tasks_df.empty:
        st.warning("Данные планирования не загружены. Карта V6 доступна, но KML-наряды пока нельзя показать.")
        render_technical_commands([
            "python scripts/v5_roads_prep.py",
            "python scripts/v5_logistics_prep.py",
        ])
    else:
        if "territory_scope" in tasks_df.columns and set(tasks_df["territory_scope"].dropna()) == {"kazakhstan"}:
            st.caption("Планирование ниже построено только по участкам внутри территории Казахстана.")

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
                help="Для первого выезда обычно лучше считать подъезд со стороны Казахстана, если этот слой есть.",
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
                index=0,
                help="Рекомендуемый старт — близко к дорогам. Так проще проверить модель в поле без дорогой экспедиции.",
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
                index=0,
                help="Для первичной проверки удобнее малые и средние участки: меньше риск, проще объехать и отобрать пробы.",
            )
            min_area, max_area = area_scenarios[selected_area_scen]

        filtered = tasks_df[
            (tasks_df[distance_col] <= dist_thresh)
            & (tasks_df["area_ha"] >= min_area)
            & (tasks_df["area_ha"] <= max_area)
        ]

        selected_area_ha = float(filtered["area_ha"].sum()) if not filtered.empty else 0.0
        total_task_area_ha = float(tasks_df["area_ha"].sum()) if not tasks_df.empty else 0.0
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(
            "Площадь выбранных участков",
            f"{selected_area_ha:,.0f} га",
            help="Сумма ячеек, которые прошли текущие фильтры. Это не план посадки до полевой проверки.",
        )
        col_m2.metric("Выбрано ячеек", f"{len(filtered):,}")
        col_m3.metric("Всего в индексе", f"{len(tasks_df):,}")
        col_m4.metric(
            "Доля площади индекса",
            f"{selected_area_ha / total_task_area_ha * 100:.1f}%" if total_task_area_ha else "0%",
        )
        if v5_stats:
            full_aoi_ha = v5_stats.get("candidate_100m_area_ha", v5_stats.get("area_ha", 0))
            st.caption(
                f"Для справки: вся V5.1 candidate-оценка по AOI — {full_aoi_ha:,.0f} га. "
                "Здесь фильтруется только индекс KML-участков для выезда."
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
                name="Участки для проверки",
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

        st.caption(
            "На карте: зелёные квадраты — участки, прошедшие фильтры; маркеры — 5 ближайших к выбранной дороге точек для первого выезда."
        )
        _render_map(m.get_root().render())

        with st.expander("Маршрутные файлы KML", expanded=True):
            st.caption(
                "KML можно открыть в GPS-навигаторе, Google Earth или QGIS. Начните с ближайших участков, "
                "а решение о посадке принимайте только после почвенной проверки."
            )
            st.caption(f"Маршрутные KML лежат в `{rel_path(KML_TASKS_DIR)}`.")
            sorted_filtered = filtered.sort_values(distance_col, ascending=True)
            display_cols = ["filename", "centroid_lat", "centroid_lon", "area_ha", "distance_to_road_km"]
            if "distance_to_kazakhstan_road_km" in filtered.columns:
                display_cols.append("distance_to_kazakhstan_road_km")
            display_df = sorted_filtered[display_cols].copy()
            display_df.columns = [
                "Файл KML", "Широта", "Долгота", "Площадь (га)", "До любой дороги (км)",
                *(["До дороги KZ (км)"] if "distance_to_kazakhstan_road_km" in filtered.columns else []),
            ]
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
            )
            kml_bytes = zip_kml_files(sorted_filtered["filename"].head(25)) if not filtered.empty else b""
            if kml_bytes:
                st.download_button(
                    "Скачать первые 25 KML по текущему фильтру",
                    data=kml_bytes,
                    file_name="aral_saxaul_field_tasks_top25.kml.zip",
                    mime="application/zip",
                    help="Файлы отсортированы текущей таблицей по расстоянию до выбранной дороги.",
                )
            elif not filtered.empty:
                st.info(f"KML-файлы не найдены в `{rel_path(KML_TASKS_DIR)}`.")

        st.warning(
            "Расчёт ресурсов ниже не является планом посадки. Сначала подтвердите выбранные участки в поле."
        )
        with st.expander("Предварительная оценка ресурсов после полевой проверки", expanded=False):
            st.caption(
                "Не используйте этот расчёт как план посадки. Он нужен только для оценки масштаба после того, как участок подтвердили в поле."
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
    v6 = load_v6_science()
    v6_metrics = v6_model_metrics(v6)
    try:
        v5_stats = load_v5_stats()
        v5_class_pixels, total_px, pixel_area_ha = load_v5_class_pixels()
        v5_thresholds = load_v5_thresholds()
    except FileNotFoundError:
        v5_stats = {}
        v5_class_pixels, total_px, pixel_area_ha = {}, 0, 0.01
        v5_thresholds = {}
        st.warning("Данные V5.1 для детализации не загружены. Основная карта V6 остаётся доступной.")
        render_technical_commands([
            "python scripts/run_inference_v5.py",
            "python scripts/v5_finalize_viz.py",
            "python scripts/v5_extract_stats.py",
        ])

    # ── V6-first metrics + V5 logistics context ───────────────────────
    candidate_100m_area_ha = v5_stats.get("candidate_100m_area_ha", v5_stats.get("area_ha", 0))
    operational_area_ha = v5_stats.get("operational_area_ha", 0)
    v6_stats = v6.get("suit_stats", {})
    v6_zone_ha = v6_stats.get("zone_area_ha", {})
    v6_low_salt_ha = float(v6_zone_ha.get("1", 0) or 0)
    v6_coverage = v6_stats.get("valid_fraction_of_aoi")
    v6_ci = v6_metrics.get("ci")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Низкий риск соли V6",
        f"{v6_low_salt_ha:,.0f} га" if v6_low_salt_ha else "—",
        help="Зона V6 с низким риском засоления. Это кандидаты для проверки, не готовая площадь посадки.",
    )
    col2.metric(
        "Оценено V6",
        f"{v6_coverage * 100:.0f}% AOI" if v6_coverage is not None else "—",
        help="Доля расчетной области, где V6 смогла дать оценку риска соли.",
    )
    col3.metric(
        "Проверка модели",
        fmt_metric(v6_metrics.get("auc")),
        delta=f"ДИ {v6_ci[0]:.3f}–{v6_ci[1]:.3f}" if v6_ci else None,
        help="LOO AUC: способность модели отличать солёные почвенные профили от несолёных.",
    )
    col4.metric(
        "KML-контуров V5.1",
        f"{v5_stats.get('clusters', 0):,}",
        help="Вспомогательные контуры V5.1 для организации полевых выездов.",
    )

    # ── Карта на самом видном месте: V6 главная, V5.1 как слой детализации ──
    st.markdown("### Карта риска соли V6 (предварительный отбор)")
    if V6_MAP_PATH.exists():
        _render_map(V6_MAP_PATH.read_text(encoding="utf-8"))
        auc_txt = fmt_metric(v6_metrics["auc"])
        ci = v6_metrics.get("ci")
        ci_txt = f", 95% ДИ {ci[0]:.3f}–{ci[1]:.3f}" if ci else ""
        n_txt = v6_metrics.get("n") or 70
        st.info(
            "Как пользоваться: наведите курсор на цветной участок или нажмите на него. "
            "Панель на карте покажет балл 0–100, риск соли и практический следующий шаг. "
            "Зелёный цвет — не разрешение на посадку, а приоритет для полевой проверки."
        )
        with st.expander("Почему это только предварительный отбор"):
            st.markdown(
                f"V6 обучена на {n_txt} измеренных почвенных профилях и проверяет риск "
                f"засоления (LOO AUC {auc_txt}{ci_txt}). Она не знает приживаемость саксаула, "
                "глубину грунтовых вод и свежие полевые условия, поэтому окончательное решение "
                "принимается только после выезда и почвенной пробы."
            )
    else:
        st.info(
            f"Карта V6 не найдена: {V6_MAP_PATH.name}. "
            "Ниже показана карта V5.1, если она есть."
        )
        render_technical_commands([
            "python scripts/v6/build_suitability_index.py",
            "python scripts/v6/render_v6_map.py",
        ])

    # V5.1 — детализация 10 м + источник логистики (сохранена, не удалена)
    with st.expander("Детальная карта V5.1 (10 м, правила Sentinel-2)", expanded=not V6_MAP_PATH.exists()):
        if V5_MAP_PATH.exists():
            _render_map(V5_MAP_PATH.read_text(encoding="utf-8"))
            st.caption(
                "V5.1 — вспомогательный слой 10 м: помогает смотреть мелкие границы, воду/рельеф "
                "и формировать KML для выездов. Приоритет участка сверяйте с V6 и полевой пробой."
            )
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
                f"Файл карты V5.1 не найден: {V5_MAP_PATH.name}. "
                "Основная карта V6 от этого не зависит."
            )

    st.markdown("### Что делать с результатом")
    action_rows = [
        {
            "Если на карте": "Высокий балл и низкий риск соли",
            "Что это значит": "Участок стоит поставить выше в список полевой проверки.",
            "Следующее действие": "Сверить подъезд, открыть KML и взять почвенную пробу.",
        },
        {
            "Если на карте": "Средний балл",
            "Что это значит": "Участок не исключён, но риск соли заметный.",
            "Следующее действие": "Проверять после лучших участков или как контрольную точку.",
        },
        {
            "Если на карте": "Высокий риск соли или уже есть растительность",
            "Что это значит": "Для новой посадки это слабый кандидат без дополнительных причин.",
            "Следующее действие": "Обычно пропустить или обследовать отдельно.",
        },
    ]
    st.dataframe(pd.DataFrame(action_rows), hide_index=True, width="stretch")

    with st.expander("Что посчитал вспомогательный слой V5.1 (10 м)"):
        st.info(
            "V5.1 делит территорию на 6 классов по Sentinel-2 с шагом 10 м. "
            "Он нужен для детализации и KML-планирования, а не как основной балл V6."
        )
        if total_px and v5_class_pixels:
            class_meta = {
                0: ("Вода / тень / нет данных", "Исключено: вода, тени, нет данных или очень темные пиксели"),
                1: ("Кандидатные зоны", "Места для проверки после сверки с V6"),
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
                        "Класс V5.1": class_meta[cls_val][0],
                        "Площадь, га (10 м)": round(area_ha, 1),
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
            f"10 самых крупных связанных V5.1-зон по сетке 100 м: {sum(top10_ha):,.0f} га "
            f"({top10_share:.1f}% от candidate-оценки по сетке 100 м). "
            "То есть результат карты в основном собран в нескольких крупных массивах, а не только в мелких пятнах."
        )
    if operational_area_ha:
        st.caption(
            f"Площадь V5.1-контуров для выезда >=10 га: {operational_area_ha:,.0f} га. "
            "Она меньше общей оценки по сетке 100 м, потому что мелкие пятна исключены из рабочего GeoJSON."
        )

    # ── Scientific interpretation (left) + spectral audit (right)
    col_interp, col_audit = st.columns([1, 1])

    with col_interp:
        st.markdown("### Главные выводы без преувеличения")
        conclusion_rows = [
            {
                "Вопрос": "Что это за карта",
                "Что есть сейчас": "V6 оценивает риск засоления по лабораторно проверенной связи NDMI и соли",
                "Как понимать": "Она помогает выбрать места для проверки, но не доказывает, что посадка там точно приживется.",
            },
            {
                "Вопрос": "Где начинать",
                "Что есть сейчас": f"{v6_low_salt_ha:,.0f} га зоны V6 с низким риском соли" if v6_low_salt_ha else "V6-статистика недоступна",
                "Как понимать": "Это верхний список для полевой проверки, не площадь посадки.",
            },
            {
                "Вопрос": "Участки для выезда",
                "Что есть сейчас": f"{v5_stats.get('clusters', 0):,} V5.1-контуров >=10 га для KML/логистики",
                "Как понимать": "Их нужно сверять с V6 и проверять почвой в поле.",
            },
            {
                "Вопрос": "Главное ограничение",
                "Что есть сейчас": "V6 проверяет соль, но не приживаемость саксаула и не все местные условия",
                "Как понимать": "Сравнивайте участки внутри района и подтверждайте результат полевой пробой.",
            },
        ]
        st.dataframe(pd.DataFrame(conclusion_rows), hide_index=True, width="stretch")

    with col_audit:
        st.markdown("### Доля классов V5.1 (детализация 10 м)")

        if total_px > 0:
            pixels_json = json.dumps({str(k): v for k, v in v5_class_pixels.items()})
            fig_audit = make_audit_fig(pixels_json, total_px)
            if fig_audit is not None:
                st.plotly_chart(fig_audit, width="stretch")
        else:
            st.info("Растровые данные классификации не найдены.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ⚙️ Технические параметры моделей
# ══════════════════════════════════════════════════════════════════════

with tab_dev:
    st.subheader("Как безопасно пользоваться результатом")
    st.info(
        "Эта вкладка отвечает на вопрос, где границы доверия карты. Основной результат — V6-риск засоления; "
        "V5.1 нужен для детализации 10 м и KML-планирования."
    )
    safety_rows = [
        {
            "Можно делать": "Выбирать места для полевой проверки",
            "Нельзя делать": "Считать зелёные зоны готовым планом посадки",
            "Что проверить в поле": "Соль в верхнем слое почвы, подъезд, фактическую воду/растительность, координаты",
        },
        {
            "Можно делать": "Сравнивать участки внутри одного района",
            "Нельзя делать": "Механически сравнивать удалённые районы без калибровки",
            "Что проверить в поле": "Локальный фон засоления и свежие контрольные пробы",
        },
        {
            "Можно делать": "Использовать KML как маршрут для обследования",
            "Нельзя делать": "Считать KML разрешением на посадку",
            "Что проверить в поле": "Границы участка и пригодность техники/логистики",
        },
    ]
    st.dataframe(pd.DataFrame(safety_rows), hide_index=True, width="stretch")

    st.subheader("Что еще нужно улучшить")

    roadmap_rows = [
        {
            "Очередь": 1,
            "Что сделать": "Больше полевых проб внутри дна моря",
            "Что нужно": "70 профилей обучают модель, но внутри целевого контура дна Арала проверочных точек мало",
            "Зачем": "Главный ограничитель точности — свежие полевые данные внутри целевой территории, а не интерфейс.",
        },
        {
            "Очередь": 2,
            "Что сделать": "Калибровать уровень засоления между районами",
            "Что нужно": "Опорные пробы в разных блоках для общей шкалы (при объединении удалённых районов качество заметно ниже — см. блок «Пространственная проверка»)",
            "Зачем": "Внутри участка модель ранжирует верно, но абсолютный уровень дрейфует между районами.",
        },
        {
            "Очередь": 3,
            "Что сделать": "Согласовать пробы с датой снимков",
            "Что нужно": "Свежий отбор под текущий NDMI (лабораторные данные — 2012–2014)",
            "Зачем": "Связь NDMI→соль проверяется на разнесённых во времени данных; совмещение по дате усилит её.",
        },
        {
            "Очередь": 4,
            "Что сделать": "Независимая проверка по кампании 2020/2021",
            "Что нужно": "Использовать пробы 2020/2021 как отдельный тест (не смешивать с обучением 2012–2014)",
            "Зачем": "Честная внешняя проверка без утечки между наборами данных.",
        },
        {
            "Очередь": 5,
            "Что сделать": "Связать балл с приживаемостью посадок",
            "Что нужно": "Данные о выживаемости реальных посадок саксаула по участкам",
            "Зачем": "Перейти от «риска засоления» к проверенному прогнозу успеха посадки.",
        },
    ]
    with st.expander("План улучшений модели и данных", expanded=False):
        st.dataframe(pd.DataFrame(roadmap_rows), hide_index=True, width="stretch")

    with st.expander("Техническая справка по вспомогательному слою V5.1"):
        col_rules, col_stats = st.columns([1, 1])

        with col_rules:
            st.markdown("**Что делает V5.1:** исключает воду/тень, сложный рельеф, существующую растительность и признаки солёных поверхностей. Оставшееся — candidate-зоны для проверки, не доказательство для посадки.")
            st.markdown(
                """
                | Класс | Простое правило |
                |---|---|
                | **0 Вода / тень / нет данных** | Пиксель похож на воду, тень или плохие данные |
                | **5 Сложный рельеф** | Уклон больше 5 градусов |
                | **10 Есть растительность** | Уже есть зеленая растительность |
                | **4 Риск влажной рапы** | Признаки высокой влажности и соленой поверхности одновременно |
                | **3 Риск сухой соли** | Признаки сухой солевой корки |
                | **1 Кандидатные зоны** | Все, что не попало в перечисленные выше риски |
                """
            )

        with col_stats:
            st.markdown("**Распределение классов V5.1 (10 м):**")
            v5_pcts = {}
            if total_px and v5_class_pixels:
                for c in [0, 1, 3, 4, 5, 10]:
                    v5_pcts[c] = v5_class_pixels.get(c, 0) / total_px * 100
            stats_data = {
                "Класс": ["1 Кандидат", "3 Сухая соль", "4 Влажная рапа", "5 Рельеф", "10 Растительность", "0 Вода/тень"],
                "Пиксели": [
                    f"{v5_class_pixels.get(1, 0)/1e6:.1f}M" if v5_class_pixels else "—",
                    f"{v5_class_pixels.get(3, 0)/1e3:.0f}K" if v5_class_pixels else "—",
                    f"{v5_class_pixels.get(4, 0)/1e6:.1f}M" if v5_class_pixels else "—",
                    f"{v5_class_pixels.get(5, 0)/1e3:.0f}K" if v5_class_pixels else "—",
                    f"{v5_class_pixels.get(10, 0)/1e6:.1f}M" if v5_class_pixels else "—",
                    f"{v5_class_pixels.get(0, 0)/1e6:.1f}M" if v5_class_pixels else "—",
                ],
                "%": [
                    f"{v5_pcts.get(1, 0):.1f}%" if v5_pcts else "—",
                    f"{v5_pcts.get(3, 0):.2f}%" if v5_pcts else "—",
                    f"{v5_pcts.get(4, 0):.1f}%" if v5_pcts else "—",
                    f"{v5_pcts.get(5, 0):.2f}%" if v5_pcts else "—",
                    f"{v5_pcts.get(10, 0):.1f}%" if v5_pcts else "—",
                    f"{v5_pcts.get(0, 0):.1f}%" if v5_pcts else "—",
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True)
            st.caption("V5.1 ограничен рабочими спутниковыми пикселями и не заменяет V6-балл засоления.")

    # ── V5 Dynamic Thresholds ──────────────────────────────────────
    if v5_thresholds:
        st.markdown("---")
        with st.expander("Пороги V5.1 для специалистов"):
            st.caption("Пороги посчитаны автоматически по снимку Sentinel-2. Они нужны для вспомогательной 10 м классификации V5.1.")
            th_cols = st.columns(3)
            metrics_def = [
                ("Влажность (NDMI P15)", "NDMI_P15", "Нижняя граница NDMI для признаков сухих солончаков."),
                ("Влажность (NDMI P85)", "NDMI_P85", "Верхняя граница NDMI для признаков капиллярной рапы."),
                ("Засоление (NDSI P15)", "NDSI_Green_SWIR2_P15", "Нижняя граница солевого индекса."),
                ("Засоление (NDSI P85)", "NDSI_Green_SWIR2_P85", "Верхняя граница солевого индекса."),
                ("Рапа (B8/B12 P15)", "BR_NIR_SWIR2_P15", "Нижняя граница отношения NIR/SWIR2."),
                ("Рапа (B8/B12 P85)", "BR_NIR_SWIR2_P85", "Верхняя граница отношения NIR/SWIR2."),
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
    missing_report_commands = []
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
                st.info(f"Отчет «{label}» пока не создан.")
                missing_report_commands.append(command)
    if missing_report_commands:
        render_technical_commands(missing_report_commands)

    st.markdown("---")
    st.subheader("Какие слои сейчас использовать")
    current_layers = [
        {
            "Слой": "V6",
            "Для чего": "Основной ответ: где ниже риск засоления",
            "Как использовать": "Наводить на карту, смотреть балл 0–100 и выбирать точки для полевой проверки",
        },
        {
            "Слой": "V5.1",
            "Для чего": "Детализация 10 м и KML-контуры для выезда",
            "Как использовать": "Проверять границы, воду/рельеф и строить маршрут, но не заменять V6-балл",
        },
    ]
    st.dataframe(pd.DataFrame(current_layers), hide_index=True, width="stretch")
    with st.expander("Архивные версии V1–V4"):
        st.markdown(
            "V1–V4 оставлены только как история разработки. Они не используются для текущих решений, "
            "не должны возвращаться в интерфейс и не заменяют V6/V5.1."
        )

    # ── V6 lab-data science layer ──────────────────────────────────────
    v6 = load_v6_science()
    v6_metrics = v6_model_metrics(v6)
    if v6["salinity"]:
        st.markdown("---")
        st.subheader("V6 — текущий слой риска засоления (70 почвенных профилей)")
        spatial = v6.get("spatial", {})
        sm = spatial.get("salinity_model", {})
        aoi_split = v6_aoi_split(v6)
        st.caption(
            "Главный слой для предварительного отбора по риску соли. Он обучен на измеренной "
            "солёности почвы из отчёта Пачикина–Козыбаевой (2012–2014, 70 georeferenced разрезов). "
            "V5.1 (10 м, скрининг по правилам) сохранён как слой детализации и логистики."
        )
        v6_summary_rows = [
            {
                "Вопрос": "Что проверяет V6",
                "Ответ": "Риск засоления верхнего слоя почвы, а не приживаемость саксаула",
            },
            {
                "Вопрос": "Как читать балл",
                "Ответ": "Больше балл — ниже ожидаемый риск соли; это приоритет для выезда, не план посадки",
            },
            {
                "Вопрос": "На каких данных основано",
                "Ответ": f"{v6_metrics.get('n') or 70} почвенных профилей с лабораторно измеренной солью",
            },
            {
                "Вопрос": "Где осторожность",
                "Ответ": "Удалённые районы нельзя сравнивать механически без локальной калибровки",
            },
        ]
        st.dataframe(pd.DataFrame(v6_summary_rows), hide_index=True, width="stretch")

        st.markdown(
            "**Что это значит простыми словами:** V6 ищет участки с меньшим риском засоления. "
            "Более высокий балл означает ниже ожидаемый риск соли, но не гарантирует приживаемость посадки."
        )
        with st.expander("Для специалистов: метрики V6"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Профилей (обучение)",
                f"{v6_metrics.get('n') or '—'}",
                help="Почвенные разрезы с измеренной солёностью верхнего слоя.",
            )
            c2.metric("Солёных (>1%)", f"{v6_metrics.get('n_saline') or '—'}")
            loo = v6_metrics.get("auc")
            ci = v6_metrics.get("ci")
            c3.metric(
                "AUC (LOO)",
                fmt_metric(loo),
                help="Площадь под ROC по схеме leave-one-out: способность отличать солёные точки от несолёных.",
            )
            c4.metric(
                "95% интервал",
                f"{ci[0]:.3f}–{ci[1]:.3f}" if ci else "—",
                help="Бутстрап-интервал AUC. Не пересекает 0.5 — связь устойчива.",
            )
            st.markdown(
                "Спутниковый индекс NDMI связан с измеренной солёностью верхнего слоя почвы "
                "(Spearman ρ ≈ +0.66, p < 1e-9, n=70). Балл V6 считается как "
                "`1 − вероятность засоления` и используется только для ранжирования кандидатов."
            )
            af = spatial.get("independent_aralfield")
            if af:
                st.markdown(
                    f"**Независимая проверка (AralField 2018, саксаул):** AUC {af.get('auc')}, "
                    f"n={af.get('n')} ({af.get('n_present')} с саксаулом), "
                    f"интервал {af.get('ci95')[0] if af.get('ci95') else '—'}–"
                    f"{af.get('ci95')[1] if af.get('ci95') else '—'}. "
                    "Точек слишком мало для надёжной оценки — только как направление."
                )

        # spatial validation honesty
        if sm:
            pb = sm.get("spatial_lbo_perblock_auc")
            pooled = sm.get("spatial_lbo_pooled_auc")
            sign = sm.get("within_block_sign_positive", "—")
            with st.expander("Пространственная проверка (честно про ограничения)", expanded=False):
                st.markdown(
                    f"""
                    Чтобы проверить, не завышена ли точность из-за того, что близкие точки похожи,
                    мы делили данные на пространственные блоки (~{sm.get('block_km', 20):.0f} км) и
                    обучали модель без каждого блока по очереди.

                    - **Средняя AUC по блокам: {pb}** — внутри каждого участка модель верно ранжирует
                      солёные и несолёные точки.
                    - Объединённая AUC по всем блокам: {pooled} — ниже, потому что **базовый уровень
                      засоления различается между районами** (в одном блоке солёные почти все точки,
                      в другом — почти ни одной). Это смещение калибровки между районами, **а не
                      потеря сигнала**.
                    - Знак связи NDMI→соль положительный в **{sign}** проверяемых блоках.

                    Вывод: модель лучше использовать для локального ранжирования внутри района.
                    Сравнение удалённых районов по одному абсолютному уровню требует дополнительной
                    калибровки. Это ограничение показано явно, а не скрыто.
                    """
                )

        # suitability zones from the wall-to-wall 30m layer
        stats = v6.get("suit_stats", {})
        zone_ha = stats.get("zone_area_ha", {})
        if zone_ha:
            names = {"1": "1 Кандидат (низкая соль)", "3": "3 Умеренное засоление",
                     "4": "4 Сильное засоление", "10": "10 Растительность", "0": "0 Вода/нет данных"}
            land = sum(float(zone_ha.get(k, 0)) for k in ("1", "3", "4", "10"))
            rows = []
            for k in ("1", "3", "4", "10", "0"):
                ha = float(zone_ha.get(k, 0))
                rows.append({
                    "Зона V6": names[k],
                    "Площадь, га": f"{ha:,.0f}",
                    "% суши": f"{ha / land * 100:.1f}%" if (land and k != "0") else "—",
                })
            st.markdown("**Зоны V6 на сплошном слое 30 м (та же область, что у V5.1):**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            vf = stats.get("valid_fraction_of_aoi")
            st.caption(
                f"Покрытие: {vf*100:.0f}% области исследования получает оценку (сплошной слой 30 м), "
                "против ~46% у композита 10 м. Зоны 3/4 — это градация тяжести засоления по одной "
                "проверенной оси NDMI."
                if vf is not None else
                "Зоны 3/4 — это градация тяжести засоления по одной проверенной оси NDMI."
            )

        # ground-truth + independent validation
        pv = v6.get("pit_validation", {})
        if pv:
            det = pv.get("saline_detector_zone34", {})
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("V6 покрывает точек", f"{pv.get('v6_scored_nonwater', '—')}/70",
                       help="Не-водные точки наземной правды, попавшие в оцениваемые зоны.")
            cc2.metric("V5.1 покрывает точек", f"{pv.get('v5_covered_nonwater', '—')}/70",
                       help="Для сравнения: замороженный продукт 10 м покрывает меньше.")
            cc3.metric("Детектор засоления",
                       f"чувств. {det.get('sensitivity', '—')} / спец. {det.get('specificity', '—')}",
                       help="Зоны 3/4 как детектор солёности >1% на покрытых точках.")
            n_in = aoi_split.get("n_in")
            n_out = aoi_split.get("n_out")
            st.caption(
                f"{n_out or 'Часть'} из {v6_metrics.get('n') or 70} обучающих разрезов лежат вне контура моря 1960 г.; "
                f"внутри целевого дна Арала для проверки около {n_in or pv.get('v6_scored_nonwater', '—')} точек. "
                f"Полная таблица: `{rel_path(V6_PIT_TABLE_PATH)}`."
            )

    # ── Pilot validation ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Проверка слоя V5.1 по 11 полевым точкам")

    validation_summary = load_v5_validation_summary()
    uncertainty_summary = load_v5_uncertainty_summary()
    point_samples = load_v5_point_samples()

    if point_samples.empty or not validation_summary:
        st.warning("Файлы проверки V5.1 не найдены.")
        st.info(
            "Сначала нужно пересобрать вспомогательные отчеты V5.1. "
            "Технические команды скрыты ниже."
        )
        render_technical_commands([
            "python scripts/build_v5_science_dataset.py",
            "python scripts/v5_validation_report.py",
            "python scripts/v5_uncertainty_report.py",
        ])
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
        st.warning(
            "Это проверка вспомогательного слоя V5.1, а не финальная оценка точности V6. "
            "Из-за конфликта координат её нужно читать как предварительную диагностику."
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

        with st.expander("Для специалистов: корреляции, графики и чувствительность V5.1", expanded=False):
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
                st.caption(
                    "Spearman r показывает, идут ли два показателя в одну сторону. "
                    "При 11 точках это только подсказка, а не доказательство."
                )
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
        rename_map = {
            "coordinate_source": "Источник координат",
            "S_Point": "Точка",
            "pit_code": "Разрез",
            "top_salinity_pct": "Соленость сверху, %",
            "field_salinity_0_20": "Полев. соленость 0-20",
            "field_ec_0_20": "Полев. EC 0-20",
            "class_filtered_name": "Класс карты",
            "ndmi": "NDMI",
            "ndsi_green_swir2": "NDSI G/SWIR2",
            "br_nir_swir2": "B8/B12",
            "coordinate_conflict_m": "Конфликт координат, м",
        }
        display_df = display_df.rename(columns=rename_map)
        st.dataframe(display_df, hide_index=True, width="stretch")

        st.caption(
            "Эта проверка предварительная. Главное ограничение сейчас - нужно выбрать правильные координаты для 11 точек."
        )
