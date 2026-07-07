import json
import math
import os
import zipfile
from io import BytesIO
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium import plugins
from shapely.geometry import box

os.environ["MPLBACKEND"] = "Agg"

BASE_DIR = Path(__file__).resolve().parent
AOI_VECTOR_PATH = BASE_DIR / "outputs" / "aoi" / "aral_sea_1960.geojson"

# V6-first: the field-trip task grid comes from the V6 salinity-risk zones
# (candidate + moderate risk, Kazakhstan-clipped), regenerated from the current
# suitability_zones_v6.tif. The V5.1 10 m pipeline stays a frozen backend that only
# supplies the screening summary numbers via load_screening_stats() below.
TASKS_PATH = BASE_DIR / "outputs" / "logistics" / "tasks_index_v6_enriched.csv"
ROADS_PATH = BASE_DIR / "outputs" / "logistics" / "aralkum_roads.geojson"
KML_TASKS_DIR = BASE_DIR / "outputs" / "logistics" / "tractor_tasks_v6"
GRID_STEP = 0.1

# ── Primary V6 map (the 10 m helper pipeline only feeds the KML route files) ──
V6_MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_v6.html"

# ── V6 science paths (lab-data layer; JSON/CSV tracked, rasters regenerated) ──
CANON_DIR = BASE_DIR / "data" / "canonical"
V6_SALINITY_MODEL_PATH = BASE_DIR / "outputs" / "models" / "salinity_v6_logit.json"
V6_SUIT_STATS_PATH = BASE_DIR / "outputs" / "data" / "suitability_v6_stats.json"
V6_PIT_VALIDATION_PATH = BASE_DIR / "outputs" / "data" / "suitability_v6_pit_validation_summary.json"
V6_SPATIAL_PATH = BASE_DIR / "outputs" / "data" / "spatial_validation_v6.json"
V6_BENCHMARK_PATH = CANON_DIR / "model_v6_benchmark.json"
V6_PIT_TABLE_PATH = CANON_DIR / "suitability_v6_pit_validation.csv"

st.set_page_config(page_title="Aral Saxaul: карта риска засоления", layout="wide")
if not hasattr(st, "iframe"):
    st.error("Для интерактивных карт нужен Streamlit 1.57 или новее.")
    st.stop()

# UI/UX: limit the dashboard width for comfortable reading on wide screens
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


@st.cache_data
def load_screening_stats():
    """Summary stats from the 10 m helper pipeline that builds the KML/roads logistics."""
    path = BASE_DIR / "outputs" / "data" / "v5_stats.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
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


def fmt_int(value):
    return f"{float(value):,.0f}".replace(",", " ")


def render_technical_commands(commands):
    with st.expander("Для технических пользователей"):
        for command in commands:
            st.code(command, language="bash")


st.title("Aral Saxaul: карта риска засоления для полевых обследований")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "Карта помогает выбрать участки, которые стоит проверить на месте. Она оценивает вероятный "
    "риск засоления, показывает следующий шаг для выбранной точки и не заменяет анализ почвы. "
    "Решение о посадке принимают только после полевого обследования."
    "</p>",
    unsafe_allow_html=True,
)

tab_analytics, tab_dev, tab_logistics = st.tabs([
    "Карта и сводка",
    "Как проверяли модель",
    "План полевых работ",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: 📍 Map of work sites
# ══════════════════════════════════════════════════════════════════════

with tab_logistics:
    st.subheader("План полевых работ")
    st.info(
        "Порядок простой: выберите участки, до которых реально доехать, скачайте KML-файлы, "
        "затем проверьте координаты и почву на месте. Посадки и бюджет имеет смысл обсуждать "
        "только после этого."
    )
    tasks_df = load_tasks()
    roads_gdf = load_roads()
    screening_stats = load_screening_stats()

    if tasks_df.empty:
        st.warning("Данные для планирования пока не загружены. Карта V6 работает, но список KML-файлов недоступен.")
        render_technical_commands([
            "python scripts/v6/build_v6_vectors.py",
            "python scripts/v6/v6_logistics_prep.py",
        ])
    else:
        if "territory_scope" in tasks_df.columns and set(tasks_df["territory_scope"].dropna()) == {"kazakhstan"}:
            st.caption("Ниже показаны только участки в пределах Казахстана.")

        tasks_df["distance_to_road_km"] = pd.to_numeric(tasks_df["distance_to_road_km"], errors="coerce")
        if "distance_to_kazakhstan_road_km" in tasks_df.columns:
            tasks_df["distance_to_kazakhstan_road_km"] = pd.to_numeric(
                tasks_df["distance_to_kazakhstan_road_km"],
                errors="coerce",
            )
        access_options = {"Все дороги OSM": "distance_to_road_km"}
        if "distance_to_kazakhstan_road_km" in tasks_df.columns and tasks_df["distance_to_kazakhstan_road_km"].notna().any():
            access_options["Дороги Казахстана"] = "distance_to_kazakhstan_road_km"

        max_cell_ha = float(tasks_df["area_ha"].max())

        col_f0, col_f1, col_f2 = st.columns(3)
        with col_f0:
            selected_access = st.selectbox(
                "От какой дорожной сети считать расстояние:",
                options=list(access_options.keys()),
                index=1 if "Дороги Казахстана" in access_options else 0,
                help="Для первой поездки обычно полезнее считать расстояние от дорог Казахстана, если этот слой доступен.",
            )
            distance_col = access_options[selected_access]

        with col_f1:
            # Fixed, human-friendly road-access rungs (km) for planning field routes.
            # A rung is only offered when at least one site sits beyond it: a threshold
            # >= the farthest site would pass every cell and make the control a no-op —
            # exactly the old 120/250 km bug (V6 sites top out at ~73 km from a road).
            # This keeps round numbers while staying immune to data-range drift.
            dist_series = tasks_df[distance_col].dropna()
            max_dist = float(dist_series.max()) if not dist_series.empty else 0.0

            FIXED_ROAD_RUNGS_KM = [10, 25, 50, 100, 150, 250]
            road_scenarios = {
                f"До {rung} км от дороги": float(rung)
                for rung in FIXED_ROAD_RUNGS_KM
                if rung < max_dist
            }
            road_scenarios[f"Показать все участки (до {max_dist:.0f} км)"] = max_dist

            selected_road_scen = st.selectbox(
                "Доступность по дорогам:",
                options=list(road_scenarios.keys()),
                index=0,
                help=(
                    "Показываем только такие пороги расстояния, которые реально сужают текущий "
                    "список участков. Более дальние значения объединены в \"Показать все участки\"."
                ),
            )
            dist_thresh = road_scenarios[selected_road_scen]

        with col_f2:
            # Use math.ceil for the open-ended upper bounds: int() truncates the largest
            # cell's area (e.g. 6591.3 -> 6591) and would silently drop that cell from the
            # "Very large" and "All sizes" buckets.
            area_cap = math.ceil(max_cell_ha)
            area_scenarios = {
                "Небольшие участки (10-1 000 га)": (10, 1000),
                "Крупные участки (1 000-5 000 га)": (1000, 5000),
                "Очень крупные участки (>5 000 га)": (5000, area_cap),
                "Все размеры": (0, area_cap),
            }
            selected_area_scen = st.selectbox(
                "Размер участка:",
                options=list(area_scenarios.keys()),
                index=0,
                help="Для первой проверки проще брать небольшие и средние участки: меньше риска, меньше переездов, проще отбор проб.",
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
            f"{fmt_int(selected_area_ha)} га",
            help="Суммарная площадь ячеек, прошедших текущие фильтры. Пока участки не проверены на месте, это не план посадок.",
        )
        col_m2.metric("Выбрано участков", f"{len(filtered):,}".replace(",", " "))
        col_m3.metric("Всего в индексе", f"{len(tasks_df):,}".replace(",", " "))
        col_m4.metric(
            "Доля площади индекса",
            f"{selected_area_ha / total_task_area_ha * 100:.1f}%" if total_task_area_ha else "0%",
        )
        if screening_stats:
            full_aoi_ha = screening_stats.get("candidate_100m_area_ha", screening_stats.get("area_ha", 0))
            st.caption(
                f"Для масштаба: по всей исследуемой области предварительный отбор выделяет {fmt_int(full_aoi_ha)} га перспективной зоны. "
                "Фильтры выше работают только с KML-индексом для полевых выездов."
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
                        f"{row['area_ha']:.0f} га, {row[distance_col]:.2f} км"
                    ),
                    icon=folium.Icon(color="green", icon="ok-sign", prefix="glyphicon"),
                ).add_to(m)

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MousePosition().add_to(m)

        st.caption(
            "Зелёные квадраты прошли ваши фильтры. Пять маркеров показывают ближайшие к дороге участки — с них удобно начинать."
        )
        _render_map(m.get_root().render())

        with st.expander("KML-файлы маршрутов", expanded=True):
            st.caption(
                "Эти KML-файлы можно открыть в GPS-навигаторе, Google Earth или QGIS. Начинайте с ближайших участков "
                "и не планируйте посадку, пока почва не проверена."
            )
            st.caption(f"KML-файлы маршрутов лежат в `{rel_path(KML_TASKS_DIR)}`.")
            sorted_filtered = filtered.sort_values(distance_col, ascending=True)
            has_low_risk = "low_risk_ha" in filtered.columns
            display_cols = ["filename", "centroid_lat", "centroid_lon", "area_ha"]
            # V6 tasks carry the low-salinity share per cell; show it so users can
            # favour cells whose area is mostly candidate (low-risk) land.
            if has_low_risk:
                display_cols.append("low_risk_ha")
            display_cols.append("distance_to_road_km")
            if "distance_to_kazakhstan_road_km" in filtered.columns:
                display_cols.append("distance_to_kazakhstan_road_km")
            display_df = sorted_filtered[display_cols].copy()
            display_df.columns = [
                "KML-файл", "Широта", "Долгота", "Площадь (га)",
                *(["Площадь с низким риском засоления (га)"] if has_low_risk else []),
                "До любой дороги (км)",
                *(["До дороги Казахстана (км)"] if "distance_to_kazakhstan_road_km" in filtered.columns else []),
            ]
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
            )
            kml_bytes = zip_kml_files(sorted_filtered["filename"].head(25)) if not filtered.empty else b""
            if kml_bytes:
                st.download_button(
                    "Скачать 25 ближайших участков (KML)",
                    data=kml_bytes,
                    file_name="aral_saxaul_field_tasks_top25.kml.zip",
                    mime="application/zip",
                    help="Порядок такой же, как в таблице: по расстоянию до выбранной дороги.",
                )
            elif not filtered.empty:
                st.info(f"KML-файлы не найдены в `{rel_path(KML_TASKS_DIR)}`.")

        st.warning(
            "Расчёт ниже показывает только грубый масштаб. Сначала подтвердите участки в поле, потом используйте цифры для планирования."
        )
        with st.expander("Черновой расчёт ресурсов (для подтверждённых участков)", expanded=False):
            st.caption(
                "Прикидка того, сколько ресурсов потребует посадка на такой площади. "
                "Смысл у неё появляется только после проверки участков на месте."
            )
            if selected_area_ha > 0:
                density = st.slider(
                    "Плотность посадки (саженцев/га)",
                    min_value=1000, max_value=3000, value=1500, step=100,
                )
                productivity = st.slider(
                    "Производительность трактора (га/смену)",
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
                col_r1.metric("Площадь в текущем фильтре", f"{fmt_int(selected_area_ha)} га")
                col_r2.metric("Саженцы", f"{total_saplings:,}".replace(",", " "))
                col_r3.metric("Машино-смены", fmt_int(total_machine_shifts))
                st.metric("Дизель, примерно", f"{fmt_int(total_fuel)} л")
            else:
                st.info("Под текущие фильтры не подходит ни один участок.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2: 📊 Overall statistics
# ══════════════════════════════════════════════════════════════════════

with tab_analytics:
    # ── All heavy I/O goes through @st.cache_data (runs once) ────────────
    v6 = load_v6_science()
    v6_metrics = v6_model_metrics(v6)
    screening_stats = load_screening_stats()

    # ── V6 headline metrics + logistics context ───────────────────────
    operational_area_ha = screening_stats.get("operational_area_ha", 0)
    v6_stats = v6.get("suit_stats", {})
    v6_zone_ha = v6_stats.get("zone_area_ha", {})
    v6_low_salt_ha = float(v6_zone_ha.get("1", 0) or 0)
    v6_coverage = v6_stats.get("valid_fraction_of_aoi")
    v6_ci = v6_metrics.get("ci")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Низкий риск засоления (V6)",
        f"{fmt_int(v6_low_salt_ha)} га" if v6_low_salt_ha else "—",
        help="Территории, где модель видит наименьший риск засоления. Это кандидаты для проверки, а не готовая площадь под посадку.",
    )
    col2.metric(
        "Оценено моделью V6",
        f"{v6_coverage * 100:.0f}% области" if v6_coverage is not None else "—",
        help="Какая часть исследуемой территории получила оценку модели.",
    )
    col3.metric(
        "Проверка модели",
        fmt_metric(v6_metrics.get("auc")),
        delta=f"ДИ {v6_ci[0]:.3f}-{v6_ci[1]:.3f}" if v6_ci else None,
        help="LOO AUC: насколько хорошо модель отличает засолённые почвенные профили от незасолённых.",
    )
    col4.metric(
        "KML-контуры участков",
        f"{screening_stats.get('clusters', 0):,}".replace(",", " "),
        help="Готовые контуры участков для организации полевых выездов. Они находятся во вкладке планирования.",
    )

    # ── Map in the most visible spot ───────────────────────────────────
    st.markdown("### Карта риска засоления V6")
    if V6_MAP_PATH.exists():
        _render_map(V6_MAP_PATH.read_text(encoding="utf-8"))
        auc_txt = fmt_metric(v6_metrics["auc"])
        ci = v6_metrics.get("ci")
        ci_txt = f", 95% ДИ {ci[0]:.3f}-{ci[1]:.3f}" if ci else ""
        n_txt = v6_metrics.get("n") or 70
        st.info(
            "Наведите курсор на окрашенный участок или щёлкните по нему. Панель на карте покажет оценку от 0 до 100, "
            "примерный риск засоления и следующий шаг. Зелёный цвет значит: этот участок стоит проверить раньше других. "
            "Это не разрешение на посадку."
        )
        with st.expander("Почему это предварительный отбор, а не окончательный вывод"):
            st.markdown(
                f"V6 обучена на {n_txt} почвенных профилях с лабораторным анализом солей. "
                f"Она оценивает только риск засоления (LOO AUC {auc_txt}{ci_txt}). "
                "Модель не знает, как здесь приживётся саксаул, как глубоко лежат грунтовые воды "
                "и как участок выглядит сейчас. Перед решением нужен выезд и проба почвы."
            )
    else:
        st.info(f"Карта V6 не найдена ({V6_MAP_PATH.name}).")
        render_technical_commands([
            "python scripts/v6/build_suitability_index.py",
            "python scripts/v6/render_v6_map.py",
        ])

    st.markdown("### Как читать карту")
    action_rows = [
        {
            "Что показывает карта": "Высокая оценка, низкий риск засоления",
            "Что это значит": "Поставьте участок выше в списке полевой проверки.",
            "Следующий шаг": "Проверьте подъезд, откройте KML-файл, отберите пробу почвы.",
        },
        {
            "Что показывает карта": "Средняя оценка",
            "Что это значит": "Участок не стоит исключать, но риск засоления заметный.",
            "Следующий шаг": "Езжайте сюда после лучших участков или оставьте как контрольную точку.",
        },
        {
            "Что показывает карта": "Высокий риск засоления или уже есть растительность",
            "Что это значит": "Для новых посадок участок слабый, если нет другой причины его изучать.",
            "Следующий шаг": "Обычно его пропускают или обследуют отдельно.",
        },
    ]
    st.dataframe(pd.DataFrame(action_rows), hide_index=True, width="stretch")

    if operational_area_ha:
        st.caption(
            f"Готовые контуры для полевых выездов площадью от 10 га суммарно дают {fmt_int(operational_area_ha)} га. "
            "KML-файлы и расстояния до дорог находятся во вкладке планирования."
        )

    st.markdown("### Коротко и честно")
    conclusion_rows = [
        {
            "Вопрос": "Что показывает карта",
            "Ответ": "Риск засоления, рассчитанный по связи между спутниковым индексом влажности и измеренными солями",
            "Как использовать": "Карта подсказывает, куда ехать. Она не обещает, что посадка сработает.",
        },
        {
            "Вопрос": "С чего начать",
            "Ответ": f"{fmt_int(v6_low_salt_ha)} га в зоне V6 с низким риском засоления" if v6_low_salt_ha else "Статистика V6 недоступна",
            "Как использовать": "Это верх списка для проверки в поле, а не площадь под посадку.",
        },
        {
            "Вопрос": "Участки для выезда",
            "Ответ": f"{screening_stats.get('clusters', 0):,}".replace(",", " ") + " готовых контуров от 10 га с KML-файлами",
            "Как использовать": "Сверьте их с картой, затем подтвердите почвой на месте.",
        },
        {
            "Вопрос": "Главное ограничение",
            "Ответ": "Модель видит риск засоления. Она не видит приживаемость саксаула и все местные условия",
            "Как использовать": "Сравнивайте участки внутри одного района, а последнее слово оставляйте за полевой пробой.",
        },
    ]
    st.dataframe(pd.DataFrame(conclusion_rows), hide_index=True, width="stretch")

    st.markdown("### Солончаки и саксаул: что видно по данным")
    st.markdown(
        """
        На части высохшего дна Аральского моря встречаются солончаки. Это почвы, где соли накопились у поверхности, иногда в виде светлой корки. Они появились естественно: вода ушла, а минеральные соли остались в верхних горизонтах. Многие такие места почти голые.

        Соль мешает корням, и саксаул здесь не исключение. Он выносливее многих растений Приаралья, но на сильно засолённой земле тоже приживается хуже. В полевых данных проекта есть бывшее солёное озеро, где саксаула не нашли, и есть неудачная посадка на участке с похожими условиями. Ещё две неудачные посадки были на голой или распаханной земле; по полевым заметкам там, скорее всего, сработали нарушение почвы и отсутствие растительности, а не измеренная соль. Есть и обратный пример: на одном участке живой саксаул рос рядом с другими солеустойчивыми кустарниками и галофитами. Значит, соль снижает шансы, но не ставит абсолютный запрет. Подтверждённых точек с саксаулом в проекте всего шесть, поэтому это подсказка для проверки, а не доказанное правило.

        Проще сказать так: высокая засолённость резко снижает шанс, что саксаул укоренится без подготовки почвы. Но она не означает, что саксаул невозможен везде. Используйте засоление как фильтр для первичного отбора. Карта показывает зоны с более высоким или низким риском, а решение всё равно начинается с выезда и анализа почвы.

        Если засолённый участок всё же нужно использовать, почву сначала готовят: промывают соли, вносят гипс или органику, улучшают дренаж. Это обычные методы, но у проекта пока нет полевых данных о том, насколько хорошо они работают именно здесь. Сажать прямо в неподготовленную сильно засолённую почву рискованно. Карта помогает выбрать, куда смотреть дальше, но не заменяет почвенный анализ и работу специалиста в поле.
        """
    )
    st.caption(
        "Этот вывод опирается на модель V6 по 70 почвенным профилям и небольшой набор полевых записей о наличии или отсутствии саксаула. Полные числа собраны в техническом разделе о саксауле."
    )

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ⚙️ Technical model parameters
# ══════════════════════════════════════════════════════════════════════

with tab_dev:
    st.subheader("Насколько можно доверять карте")
    st.info(
        "Здесь показано, как проверяли карту и где её границы. Главный результат — оценка риска засоления V6. "
        "Сетка участков для выезда и KML-файлы построены по тем же зонам V6: перспективные участки и умеренный риск "
        "в пределах Казахстана. Отдельный замороженный 10-метровый конвейер даёт только сводные числа для отбора."
    )
    safety_rows = [
        {
            "Как использовать": "Выбирать места для полевой проверки",
            "Как не использовать": "Считать зелёные зоны готовым планом посадки",
            "Что проверить в поле": "Соль в верхнем слое почвы, подъезд, воду и растительность на месте, координаты",
        },
        {
            "Как использовать": "Сравнивать участки внутри одного района",
            "Как не использовать": "Сравнивать далёкие районы напрямую без калибровки",
            "Что проверить в поле": "Местный фон засоления и свежие контрольные пробы",
        },
        {
            "Как использовать": "Прокладывать маршрут обследования по KML-файлу",
            "Как не использовать": "Считать KML-файл разрешением на посадку",
            "Что проверить в поле": "Границы участка, технику, доступность и реальную логистику",
        },
    ]
    st.dataframe(pd.DataFrame(safety_rows), hide_index=True, width="stretch")

    st.subheader("Что улучшит модель")

    roadmap_rows = [
        {
            "Приоритет": 1,
            "Что сделать": "Собрать больше почвенных проб внутри высохшего дна",
            "Что нужно": "Модель обучена на 70 профилях, но внутри целевого контура Арала точек проверки мало",
            "Зачем": "Свежие полевые данные внутри целевой зоны сильнее всего повысят точность. Интерфейс здесь не главное.",
        },
        {
            "Приоритет": 2,
            "Что сделать": "Откалибровать уровни засоления между районами",
            "Что нужно": "Эталонные пробы в разных блоках, чтобы собрать общую шкалу. При объединении дальних районов точность заметно падает.",
            "Зачем": "Внутри одного участка модель ранжирует нормально, но абсолютный уровень между районами смещается.",
        },
        {
            "Приоритет": 3,
            "Что сделать": "Согласовать даты проб и спутниковых снимков",
            "Что нужно": "Свежие пробы, взятые близко к дате съёмки. Сейчас лабораторные данные относятся к 2012-2014 годам.",
            "Зачем": "Связь между снимком и солью станет надёжнее, если данные собраны в один период.",
        },
        {
            "Приоритет": 4,
            "Что сделать": "Проверить модель на кампании 2020-2021 годов",
            "Что нужно": "Держать эти пробы отдельным тестовым набором и не смешивать с обучением 2012-2014 годов",
            "Зачем": "Так получится независимая проверка без утечки между наборами данных.",
        },
        {
            "Приоритет": 5,
            "Что сделать": "Связать оценку с реальной приживаемостью саксаула",
            "Что нужно": "Данные по выживаемости посадок на каждом участке",
            "Зачем": "Только так риск засоления можно превратить в проверенный прогноз успешности посадки.",
        },
    ]
    with st.expander("План улучшения модели и данных", expanded=False):
        st.dataframe(pd.DataFrame(roadmap_rows), hide_index=True, width="stretch")

    with st.expander("Откуда берутся KML-контуры участков"):
        st.markdown(
            "Контуры участков и расстояния до дорог во вкладке **План полевых работ** приходят из отдельного "
            "10-метрового анализа Sentinel-2. Он отсекает воду и тень, крутые склоны, существующую растительность "
            "и поверхности, которые со спутника выглядят засолёнными. Оставшиеся территории собираются в контуры "
            "от 10 га, сопоставляются с дорожной сетью OSM и выгружаются в KML. Эти контуры помогают организовать выезд. "
            "Оценка засоления всё равно остаётся за V6 и почвенной пробой."
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
            "Это основной слой предварительного отбора. Он обучен на лабораторно измеренной засолённости: "
            "70 почвенных профилей с координатами из отчёта Пачикина-Козыбаевой за 2012-2014 годы."
        )
        v6_summary_rows = [
            {
                "Вопрос": "Что проверяет V6",
                "Ответ": "Риск засоления верхнего слоя почвы, а не выживаемость саксаула",
            },
            {
                "Вопрос": "Как читать оценку",
                "Ответ": "Чем выше оценка, тем ниже ожидаемый риск соли. Это приоритет для выезда, а не план посадки",
            },
            {
                "Вопрос": "На чём построена модель",
                "Ответ": f"{v6_metrics.get('n') or 70} почвенных профилей с лабораторно измеренным содержанием солей",
            },
            {
                "Вопрос": "Где нужна осторожность",
                "Ответ": "Не сравнивайте далёкие районы напрямую без локальной калибровки",
            },
        ]
        st.dataframe(pd.DataFrame(v6_summary_rows), hide_index=True, width="stretch")

        st.markdown(
            "**Простыми словами:** V6 ищет участки, где соли, вероятно, меньше. "
            "Высокая оценка говорит о низком солевом риске, но не гарантирует, что посадка приживётся."
        )
        with st.expander("Для технических пользователей: показатели V6"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Профили (обучение)",
                f"{v6_metrics.get('n') or '—'}",
                help="Почвенные профили с лабораторно измеренной засолённостью верхнего слоя.",
            )
            c2.metric("Засолённые (>1%)", f"{v6_metrics.get('n_saline') or '—'}")
            loo = v6_metrics.get("auc")
            ci = v6_metrics.get("ci")
            c3.metric(
                "AUC (LOO)",
                fmt_metric(loo),
                help="AUC с исключением по одной точке: насколько хорошо модель разделяет засолённые и незасолённые точки.",
            )
            c4.metric(
                "95% интервал",
                f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "—",
                help="Бутстрэп-интервал для AUC. Он остаётся выше 0.5, значит связь не разваливается.",
            )
            st.markdown(
                "Спутниковый индекс влажности NDMI связан с измеренной засолённостью верхнего слоя "
                "(rho Спирмена ≈ +0.66, p < 1e-9, n=70). Оценка V6 — это просто "
                "`1 - вероятность засоления`; она нужна только для ранжирования участков."
            )
            af = spatial.get("independent_aralfield")
            if af:
                st.markdown(
                    f"**Независимая проверка (AralField 2018, саксаул):** AUC {af.get('auc')}, "
                    f"n={af.get('n')} ({af.get('n_present')} с саксаулом), "
                    f"интервал {af.get('ci95')[0] if af.get('ci95') else '—'}-"
                    f"{af.get('ci95')[1] if af.get('ci95') else '—'}. "
                    "Точек слишком мало для надёжной оценки. Это только ориентир, не вывод."
                )

        with st.expander("Данные о саксауле, солончаках и засолении (технические детали)"):
            st.markdown(
                """
**Модель засоления — основной слой V6:**

Модель — логистическая регрессия с L2-регуляризацией. В ней один предиктор: спутниковый NDMI с разрешением 30 м. Обучение построено на 70 геопривязанных почвенных профилях из отчёта Пачикина-Козыбаевой за 2012-2014 годы. Целевая метка — содержание солей в верхнем слое почвы выше 1%; таких профилей 27 из 70. Качество по leave-one-out AUC: 0.682, 95% ДИ [0.556, 0.802].

Главное ограничение — региональный сдвиг калибровки. Объединённая пространственная проверка leave-block-out даёт AUC 0.385, среднее AUC внутри отдельных блоков — 0.792. Внутри одной местности модель ранжирует риск засоления заметно лучше, чем при прямом сравнении далёких районов. Около 15 из 70 обучающих профилей лежат внутри реального контура дна Арала 1960 года. Остальные помогают обучению по более широкому Приаралью, но не являются независимой проверкой внутри целевой зоны. LOO AUC внутри AOI — 0.614, вне AOI — 0.671.

**Данные именно о саксауле слабее:**

Во всей базе есть только 6 геопривязанных положительных полевых меток саксаула из 70 профилей. Прямой классификатор пригодности саксаула по NDMI+MSAVI получил LOO AUC около 0.48, то есть почти не отличился от случайного угадывания. В проекте он прямо помечен как исследовательский и не используется для решений.

Связи между почвенными свойствами, спутниковыми индексами и саксаулом идут в экологически разумную сторону, но не проходят внутренние пороги устойчивости проекта (MIN_N=12, MIN_N_POS=8, MIN_AUC=0.62). Низкие хлориды, меньше солей, меньше обменного натрия, больше карбонатов и более песчаный состав выглядят лучше для саксаула, но при шести положительных точках это только ориентир.

Корреляции Спирмена с меткой y_suitable тоже читаются как подсказки, а не как доказательство: top_caco3_pct rho=+0.39, p=0.05, n=26; salt_cl_pct rho=-0.27, p=0.07, n=49; rs30_ndwi rho=-0.25, p=0.06, n=56; exch_na/exch_sum rho=-0.21, p=0.13, n=52; sand_pct rho=+0.21, p=0.16, n=47.

**Как ставились отрицательные метки:**

По правилам QA проекта (`SAXAUL_LABELS_QA.md`) сильные отрицательные метки ставились только там, где были документированные неудачные посадки или действительно голая земля. Их не ставили просто из-за высокой засолённости, чтобы не зашить это предположение в обучающие данные. Поэтому часть записей без саксаула не является участками с высокой солью по измерениям проекта.

**Полевые примеры:**

Разрез 08/14 (2014): "Такыр без растительности". В верхнем слое 0.43% солей, ниже порога 1%. Это отрицательная метка по голой земле, а не подтверждённый пример высокой засолённости.

Разрез S134 из AralField: "Former Sorkol lake bed, very saline, solonchak"; haloxylon = 0. Это отдельный внешний набор, не входящий в 70 профилей, и это реальный пример отсутствия саксаула на сильно засолённом месте.

Разрез 13/13 (2013): документированная неудачная посадка, "Сажали саксаул, ничего прижилось". Содержание солей 1.004%, чуть выше порога 1%.

Разрез 24/14 (2014): неудачная посадка на распаханном нарушенном участке. Содержание солей 0.143%, заметно ниже порога. Полевое описание связывает проблему с нарушением почвы, а не с измеренной солью.

Разрез 4А/12 (2012): сильная положительная метка. Описание растительности включает тамарикс, саксаул, селитрянку и другие галофиты. Этот пример показывает: засоление снижает шансы, но не делает саксаул невозможным автоматически.

**Ограничение интерпретации:** модель засоления (LOO AUC 0.682, ДИ [0.556, 0.802], n=70) достаточно обоснована для карты риска соли. Данные о самом саксауле, пороги и полевые примеры — только подсказки для отбора. Их нельзя подавать как прогноз приживаемости или доказательство пригодности конкретного участка.
                """
            )

        # spatial validation honesty
        if sm:
            pb = sm.get("spatial_lbo_perblock_auc")
            pooled = sm.get("spatial_lbo_pooled_auc")
            sign = sm.get("within_block_sign_positive", "—")
            with st.expander("Пространственная проверка: где границы модели", expanded=False):
                st.markdown(
                    f"""
                    Близкие почвенные точки часто похожи друг на друга, и это может завышать оценку качества.
                    Чтобы проверить модель жёстче, данные разделили на пространственные блоки примерно по {sm.get('block_km', 20):.0f} км
                    и каждый раз обучали модель без одного блока.

                    - **Среднее AUC по блокам: {pb}**. Внутри отдельных участков модель правильно ранжирует
                      засолённые и незасолённые точки.
                    - Общее AUC по всем блокам: {pooled}. Оно ниже, потому что **базовый уровень засоления
                      отличается между районами**: один блок почти весь засолён, другой почти нет. Это проблема
                      межрегиональной калибровки, а не исчезновение сигнала.
                    - Связь между NDMI и солью сохраняет положительный знак в **{sign}** проверенных блоках.

                    Вывод: используйте модель для ранжирования участков внутри одного района. Для сравнения далёких
                    районов по одной абсолютной шкале нужна дополнительная калибровка.
                    """
                )

        # suitability zones from the wall-to-wall 30m layer
        stats = v6.get("suit_stats", {})
        zone_ha = stats.get("zone_area_ha", {})
        if zone_ha:
            names = {"1": "1 Перспективная зона (низкое засоление)", "3": "3 Умеренное засоление",
                     "4": "4 Сильное засоление", "10": "10 Растительность", "0": "0 Вода/нет данных"}
            land = sum(float(zone_ha.get(k, 0)) for k in ("1", "3", "4", "10"))
            rows = []
            for k in ("1", "3", "4", "10", "0"):
                ha = float(zone_ha.get(k, 0))
                rows.append({
                    "Зона V6": names[k],
                    "Площадь, га": fmt_int(ha),
                    "% суши": f"{ha / land * 100:.1f}%" if (land and k != "0") else "—",
                })
            st.markdown("**Зоны V6 на сплошном 30-метровом слое:**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            vf = stats.get("valid_fraction_of_aoi")
            st.caption(
                f"30-метровый слой оценивает {vf*100:.0f}% исследуемой территории. "
                "Зоны 3 и 4 — не разные типы земель, а ступени одной проверенной шкалы риска засоления."
                if vf is not None else
                "Зоны 3 и 4 — не разные типы земель, а ступени одной проверенной шкалы риска засоления."
            )

        # ground-truth + independent validation
        pv = v6.get("pit_validation", {})
        if pv:
            det = pv.get("saline_detector_zone34", {})
            cc1, cc2 = st.columns(2)
            cc1.metric("Точки, покрытые V6", f"{pv.get('v6_scored_nonwater', '—')}/70",
                       help="Полевые точки без воды, которые попали в оценённые зоны.")
            sens_ci = det.get("sensitivity_ci95")
            spec_ci = det.get("specificity_ci95")
            spec_n = det.get("specificity_n")
            sens_n = det.get("sensitivity_n")
            cc2.metric("Выявление засоления",
                       f"чувств. {det.get('sensitivity', '—')} / специф. {det.get('specificity', '—')}",
                       help="Насколько хорошо зоны 3/4 находят точки с засолением выше 1%. "
                            "Читайте вместе с размерами выборки и 95% интервалами ниже: специфичность 1.0 "
                            "на нескольких отрицательных точках не означает идеальный фильтр.")
            if spec_ci and sens_ci:
                st.caption(
                    f"Чувствительность {det.get('sensitivity','—')} при n={sens_n} (95% ДИ "
                    f"[{sens_ci[0]}, {sens_ci[1]}]); специфичность {det.get('specificity','—')} только на "
                    f"n={spec_n} отрицательных точках (95% ДИ [{spec_ci[0]}, {spec_ci[1]}] — широкий интервал, малая выборка)."
                )
            n_in = aoi_split.get("n_in")
            n_out = aoi_split.get("n_out")
            st.caption(
                f"Из {v6_metrics.get('n') or 70} обучающих профилей {n_out or 'часть'} лежат за границей моря 1960 года. "
                f"Внутри целевого высохшего дна остаётся около {n_in or pv.get('v6_scored_nonwater', '—')} точек для проверки. "
                f"Полная таблица: `{rel_path(V6_PIT_TABLE_PATH)}`."
            )

