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


st.title("Карта предварительной оценки засоленности почв для обследования участков под посадку саксаула")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "Эта карта помогает определить участки, которые целесообразно обследовать в полевых условиях. "
    "Она оценивает вероятную степень засоленности почвы и предлагает рекомендуемый следующий этап "
    "обследования для выбранного участка. Карта не заменяет лабораторный анализ почвы и не предназначена "
    "для принятия окончательных решений о проведении посадок."
    "</p>",
    unsafe_allow_html=True,
)

tab_analytics, tab_dev, tab_logistics = st.tabs([
    "Карта и краткая сводка",
    "Методика оценки",
    "План полевых работ",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: 📍 Map of work sites
# ══════════════════════════════════════════════════════════════════════

with tab_logistics:
    st.subheader("План полевых работ")
    st.info(
        "Порядок работы. Сначала выберите участки, до которых можно добраться по автомобильным дорогам, "
        "затем загрузите соответствующие KML-файлы и проведите полевое обследование с проверкой координат "
        "и анализом почвы. Планирование посадок и расчёт бюджета выполняются только после завершения этих этапов."
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
            st.caption("Ниже представлены только участки, расположенные на территории Республики Казахстан.")

        tasks_df["distance_to_road_km"] = pd.to_numeric(tasks_df["distance_to_road_km"], errors="coerce")
        if "distance_to_kazakhstan_road_km" in tasks_df.columns:
            tasks_df["distance_to_kazakhstan_road_km"] = pd.to_numeric(
                tasks_df["distance_to_kazakhstan_road_km"],
                errors="coerce",
            )
        access_options = {"Все дороги OpenStreetMap": "distance_to_road_km"}
        if "distance_to_kazakhstan_road_km" in tasks_df.columns and tasks_df["distance_to_kazakhstan_road_km"].notna().any():
            access_options["Автомобильные дороги Казахстана"] = "distance_to_kazakhstan_road_km"

        max_cell_ha = float(tasks_df["area_ha"].max())

        col_f0, col_f1, col_f2 = st.columns(3)
        with col_f0:
            selected_access = st.selectbox(
                "Исходные данные для расчёта расстояния:",
                options=list(access_options.keys()),
                index=1 if "Автомобильные дороги Казахстана" in access_options else 0,
                help="Для первого выезда обычно удобнее считать расстояние от дорожной сети Казахстана, если этот слой доступен.",
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
                f"До {rung} км от автомобильной дороги": float(rung)
                for rung in FIXED_ROAD_RUNGS_KM
                if rung < max_dist
            }
            road_scenarios[f"Показать все участки (до {max_dist:.0f} км)"] = max_dist

            selected_road_scen = st.selectbox(
                "Удалённость от автомобильных дорог:",
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
                "Небольшие участки (10–1 000 га)": (10, 1000),
                "Крупные участки (1 000–5 000 га)": (1000, 5000),
                "Очень крупные участки (>5 000 га)": (5000, area_cap),
                "Все участки": (0, area_cap),
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
                f"Для справки: на всей исследуемой территории модель предварительной оценки выделяет {fmt_int(full_aoi_ha)} га перспективных участков. "
                "Применяемые выше фильтры используются только для отбора участков из KML-базы, предназначенной для планирования полевых обследований."
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
            "Зелёными квадратами показаны участки, соответствующие выбранным критериям. Пять маркеров обозначают участки, "
            "расположенные ближе всего к автомобильным дорогам, что делает их наиболее удобными для первоочередного обследования."
        )
        _render_map(m.get_root().render())

        with st.expander("KML-файлы маршрутов", expanded=True):
            st.caption(
                "Эти KML-файлы можно открыть в GPS-навигаторе, Google Earth или QGIS. Рекомендуется начинать обследование "
                "с ближайших участков. Решение о проведении посадок следует принимать только после полевого обследования "
                "и анализа почвенных образцов."
            )
            st.caption(f"KML-файлы маршрутов находятся в каталоге: `{rel_path(KML_TASKS_DIR)}`.")
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
                *(["Низкая засолённость (га)"] if has_low_risk else []),
                "До ближайшей дороги (км)",
                *(["До ближайшей дороги Казахстана (км)"] if "distance_to_kazakhstan_road_km" in filtered.columns else []),
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
            "Представленная ниже оценка является предварительной. Перед использованием её при планировании работ все участки должны быть проверены в полевых условиях."
        )
        with st.expander("Предварительный расчёт ресурсов", expanded=False):
            st.caption(
                "Расчёт предназначен только для ориентировочной оценки ресурсов. "
                "Перед использованием этих данных при планировании работ участки должны быть проверены в полевых условиях."
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
    st.markdown("### Карта риска засоления (версия V6)")
    if V6_MAP_PATH.exists():
        _render_map(V6_MAP_PATH.read_text(encoding="utf-8"))
        auc_txt = fmt_metric(v6_metrics["auc"])
        ci = v6_metrics.get("ci")
        ci_txt = f", 95% ДИ {ci[0]:.3f}-{ci[1]:.3f}" if ci else ""
        n_txt = v6_metrics.get("n") or 70
        st.info(
            "Наведите курсор или щёлкните по любому окрашенному участку. Информационная панель отобразит оценку "
            "по шкале от 0 до 100, предполагаемый риск засоления и рекомендуемый следующий этап обследования. "
            "Зелёный цвет означает: «этот участок следует проверить в первую очередь». Это не является рекомендацией "
            "или разрешением на проведение посадки."
        )
        with st.expander("Почему эта карта является инструментом предварительной оценки, а не окончательным заключением"):
            st.markdown(
                f"Модель V6 обучена на основе {n_txt} почвенных профилей с лабораторно определённым содержанием "
                f"солей и предназначена исключительно для оценки риска засоления почв (LOO AUC {auc_txt}{ci_txt}).\n\n"
                "Модель не учитывает такие важные факторы, как приживаемость саксаула, глубина залегания грунтовых вод, "
                "текущее состояние поверхности участка и другие полевые условия. Поэтому любое окончательное решение "
                "о проведении посадок должно приниматься только после полевого обследования участка и лабораторного "
                "анализа почвенных образцов."
            )
    else:
        st.info(f"Карта V6 не найдена ({V6_MAP_PATH.name}).")
        render_technical_commands([
            "python scripts/v6/build_suitability_index.py",
            "python scripts/v6/render_v6_map.py",
        ])

    st.markdown("### Пояснения к карте")
    action_rows = [
        {
            "Класс": "Высокий индекс благоприятности, низкий риск засоления",
            "Пояснение": "Этот участок следует включить в число первоочередных для полевого обследования.",
            "Рекомендации": "Оцените доступность участка, откройте KML-файл и выполните отбор почвенных проб.",
        },
        {
            "Класс": "Средний индекс благоприятности",
            "Пояснение": "Участок нельзя исключать, однако риск засоления остаётся значимым.",
            "Рекомендации": "Обследуйте его после наиболее перспективных участков либо используйте в качестве контрольного.",
        },
        {
            "Класс": "Высокий риск засоления или существующая растительность",
            "Пояснение": "Участок малопригоден для новых посадок, если отсутствуют дополнительные благоприятные факторы.",
            "Рекомендации": "Обычно рекомендуется исключить его из первоочередных обследований либо изучать отдельно.",
        },
    ]
    st.dataframe(pd.DataFrame(action_rows), hide_index=True, width="stretch")

    if operational_area_ha:
        st.caption(
            f"Подготовленные для полевого обследования контуры участков площадью 10 га и более охватывают {fmt_int(operational_area_ha)} га. "
            "Их KML-файлы и информация о транспортной доступности доступны во вкладке «План полевых работ»."
        )

    st.markdown("### Основные сведения о карте")
    conclusion_rows = [
        {
            "Вопрос": "Что показывает карта",
            "Ответ": "Карта отображает вероятность засоления почв, рассчитанную по спутниковым данным и модели, обученной на лабораторных анализах почв.",
            "Практическое значение": "Карта помогает определить, какие участки следует обследовать в первую очередь, но не определяет их пригодность для посадки.",
        },
        {
            "Вопрос": "С чего начать обследование",
            "Ответ": f"В зоне V6 с низким риском засоления выделено {fmt_int(v6_low_salt_ha)} га потенциально перспективных территорий." if v6_low_salt_ha else "Статистика V6 недоступна.",
            "Практическое значение": "Начинайте полевые работы именно с этих участков. Они являются приоритетными для обследования, а не автоматически пригодными для посадки.",
        },
        {
            "Вопрос": "Сколько участков подготовлено для обследования",
            "Ответ": f"Подготовлено {screening_stats.get('clusters', 0):,}".replace(",", " ") + " участков площадью 10 га и более, для каждого доступен KML-файл.",
            "Практическое значение": "Используйте KML-файлы для планирования маршрутов, затем подтвердите результаты полевым обследованием и анализом почвы.",
        },
        {
            "Вопрос": "Какие ограничения имеет модель",
            "Ответ": "Модель оценивает только риск засоления почв. Она не учитывает приживаемость саксаула, глубину грунтовых вод и другие местные условия.",
            "Практическое значение": "Окончательное решение о посадке должно приниматься только после полевого обследования и лабораторного анализа почвы.",
        },
    ]
    st.dataframe(pd.DataFrame(conclusion_rows), hide_index=True, width="stretch")

    st.markdown("### Засоление почв и произрастание саксаула")
    st.markdown(
        """
        На отдельных участках высохшего дна Аральского моря распространены солончаки — почвы с высоким содержанием легкорастворимых солей, накапливающихся у поверхности. Такие территории часто можно распознать по характерной светлой соляной корке. Солончаки сформировались естественным образом после высыхания морского дна, когда вода испарилась, а минеральные соли остались в верхних горизонтах почвы. Многие из этих участков практически лишены растительного покрова.

        Высокая засоленность неблагоприятно влияет на развитие корневой системы растений, и саксаул не является исключением. Несмотря на то что саксаул относится к числу наиболее устойчивых к засолению древесно-кустарниковых пород, пригодных для условий Приаралья, его способность приживаться существенно снижается на сильно засолённых почвах.

        Полевые материалы проекта подтверждают эту тенденцию. На одном из обследованных участков бывшего солёного озера, характеризующемся очень высокой засолённостью, саксаул обнаружен не был. Попытка создания посадок на другом участке с аналогичными почвенными условиями также оказалась неуспешной. Ещё две неудачные посадки были зафиксированы на открытых или ранее распаханных территориях. Однако в этих случаях полевые наблюдения указывают, что основной причиной неудачи, вероятнее всего, стали нарушение почвенного покрова и отсутствие естественной растительности, а не подтверждённая засолённость.

        Вместе с тем имеются и противоположные примеры. По крайней мере на одном участке с хорошо сохранившимися насаждениями саксаула одновременно произрастали другие солеустойчивые кустарники и растения-галофиты. Это свидетельствует о том, что наличие засоления не исключает возможность произрастания саксаула.

        Следует учитывать, что в рамках проекта подтверждённое присутствие саксаула зарегистрировано лишь в шести почвенных разрезах, поэтому выявленные закономерности следует рассматривать как предварительные выводы, требующие дальнейшей проверки в ходе полевых исследований.

        **Основной вывод**

        Высокая засолённость почвы существенно снижает вероятность успешного укоренения саксаула без проведения предварительных мероприятий по улучшению почвенных условий, однако сама по себе не означает полной непригодности участка для посадки. Поэтому уровень засолённости следует использовать как критерий предварительного отбора участков, а не как окончательное основание для принятия решения.

        Представленная на веб-карте модель оценивает вероятность засоления на основе спутниковых данных и результатов анализа почв. Она помогает определить, какие территории следует обследовать в первую очередь, однако перед началом посадочных работ необходимо провести полевое обследование и лабораторный анализ почвы.

        **Если посадка планируется на засолённом участке**

        Если использование участка с повышенной засолённостью неизбежно, рекомендуется предварительно выполнить мероприятия по мелиорации почвы. К числу общепринятых методов относятся: промывка почвы для удаления избыточных солей, внесение гипса и органических удобрений, улучшение дренажа.

        Эти методы широко применяются для снижения засолённости почв, однако в рамках данного проекта пока отсутствуют данные полевых наблюдений, позволяющие оценить их эффективность именно на обследованных территориях. Посадка саксаула непосредственно на сильно засолённых почвах без предварительной подготовки участка имеет низкую вероятность успеха.

        Во всех случаях карта должна рассматриваться как инструмент предварительной оценки, помогающий определить приоритетные участки для обследования. Она не заменяет полевые исследования, лабораторный анализ почвы и экспертную оценку специалистов.
        """
    )
    st.caption(
        "Этот вывод опирается на модель V6 по 70 почвенным профилям и небольшой набор полевых записей о наличии или отсутствии саксаула. Полные числа собраны в техническом разделе о саксауле."
    )

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ⚙️ Technical model parameters
# ══════════════════════════════════════════════════════════════════════

with tab_dev:
    st.subheader("Возможности и ограничения карты")
    st.info(
        "В этом разделе описано, как проводилась проверка модели и какие ограничения следует учитывать при интерпретации результатов. "
        "Основным результатом модели является карта риска засоления V6. Задания для полевых обследований и KML-файлы "
        "сформированы на основе тех же зон V6 (перспективные участки и зоны умеренного риска в пределах территории Казахстана). "
        "Отдельный фиксированный конвейер обработки данных с пространственным разрешением 10 м используется только для расчёта "
        "сводных показателей предварительной оценки."
    )
    safety_rows = [
        {
            "Рекомендуемое использование": "Выбор участков для полевого обследования",
            "Не рекомендуется": "Рассматривать зелёные зоны как окончательно пригодные для посадки",
            "Что необходимо проверить в полевых условиях": "Засолённость верхнего слоя почвы, транспортную доступность, наличие воды и растительности, координаты участка",
        },
        {
            "Рекомендуемое использование": "Сравнение участков в пределах одного района",
            "Не рекомендуется": "Сравнивать удалённые районы без дополнительной калибровки",
            "Что необходимо проверить в полевых условиях": "Региональный фон засоления и результаты контрольных почвенных проб",
        },
        {
            "Рекомендуемое использование": "Планирование маршрутов по KML-файлам",
            "Не рекомендуется": "Рассматривать KML-файлы как разрешение на проведение посадок",
            "Что необходимо проверить в полевых условиях": "Границы участков и соответствие условий имеющемуся оборудованию и логистическим возможностям",
        },
    ]
    st.dataframe(pd.DataFrame(safety_rows), hide_index=True, width="stretch")

    st.subheader("Направления развития модели")

    roadmap_rows = [
        {
            "Приоритет": 1,
            "Развитие модели": "Собрать больше полевых образцов в пределах высохшего дна Аральского моря",
            "Необходимые данные": "Сейчас модель обучена на 70 почвенных профилях, однако внутри целевой территории имеется недостаточное количество точек наблюдений.",
            "Ожидаемый результат": "Дополнительные полевые данные непосредственно в пределах исследуемой территории являются главным условием повышения точности модели.",
        },
        {
            "Приоритет": 2,
            "Развитие модели": "Выполнить калибровку уровней засоления между районами",
            "Необходимые данные": "Необходимы эталонные почвенные образцы из различных районов для формирования единой шкалы оценки.",
            "Ожидаемый результат": "В пределах одного района модель ранжирует участки достаточно корректно, однако абсолютные значения могут различаться между районами.",
        },
        {
            "Приоритет": 3,
            "Развитие модели": "Согласовать сроки отбора почвенных проб с датами спутниковой съёмки",
            "Необходимые данные": "Требуются почвенные образцы, собранные максимально близко к дате получения спутниковых данных (используемые лабораторные данные относятся к 2012–2014 гг.).",
            "Ожидаемый результат": "Это позволит повысить достоверность связи между спутниковыми показателями и фактической засолённостью почв.",
        },
        {
            "Приоритет": 4,
            "Развитие модели": "Провести независимую проверку модели по материалам полевой кампании 2020–2021 гг.",
            "Необходимые данные": "Использовать данные 2020–2021 гг. исключительно как независимый тестовый набор, не включая их в обучающую выборку.",
            "Ожидаемый результат": "Это позволит выполнить объективную независимую оценку качества модели без смешения обучающих и тестовых данных.",
        },
        {
            "Приоритет": 5,
            "Развитие модели": "Связать результаты модели с фактической приживаемостью саксаула",
            "Необходимые данные": "Необходимы данные о приживаемости посадок саксаула для каждого участка.",
            "Ожидаемый результат": "Это позволит перейти от оценки риска засоления к прогнозированию успешности посадок.",
        },
    ]
    with st.expander("План улучшения модели и данных", expanded=False):
        st.dataframe(pd.DataFrame(roadmap_rows), hide_index=True, width="stretch")

    with st.expander("Как формируются KML-контуры участков"):
        st.markdown(
            "Контуры участков и информация о расстоянии до автомобильных дорог, представленные во вкладке **План полевых работ**, "
            "формируются с использованием отдельного алгоритма предварительного анализа спутниковых данных Sentinel-2 с пространственным "
            "разрешением 10 м.\n\n"
            "В процессе обработки последовательно исключаются водные объекты и затенённые участки, территории с крутыми склонами, "
            "участки, покрытые существующей растительностью, а также поверхности с признаками повышенной засолённости, выявленными "
            "по спутниковым данным.\n\n"
            "Оставшиеся территории объединяются в контуры площадью не менее 10 га, сопоставляются с дорожной сетью OpenStreetMap (OSM), "
            "после чего экспортируются в формате KML. Контуры KML предназначены для планирования маршрутов полевых работ и не используются "
            "для оценки засолённости. Решение о степени засоления участка основывается на результатах модели V6 и должно подтверждаться "
            "анализом почвенных образцов."
        )

    # ── V6 lab-data science layer ──────────────────────────────────────
    v6 = load_v6_science()
    v6_metrics = v6_model_metrics(v6)
    if v6["salinity"]:
        st.markdown("---")
        st.subheader("Модель V6 для оценки риска засоления почв (обучена на данных 70 почвенных профилей)")
        spatial = v6.get("spatial", {})
        sm = spatial.get("salinity_model", {})
        aoi_split = v6_aoi_split(v6)
        st.caption(
            "Основной слой предварительной оценки. Модель обучена на данных о засолённости почв, определённой по результатам "
            "лабораторных анализов. Для обучения использованы 70 почвенных профилей с известными координатами из отчёта "
            "Пачикина–Козыбаевой (2012–2014 гг.)."
        )
        v6_summary_rows = [
            {
                "Вопрос": "Что оценивает модель V6",
                "Ответ": "Риск засоления верхнего слоя почвы, а не вероятность успешного произрастания саксаула.",
            },
            {
                "Вопрос": "Как интерпретировать оценку",
                "Ответ": "Чем выше значение, тем ниже предполагаемый риск засоления. Оценка используется для определения приоритетов полевых обследований и не является рекомендацией для посадки.",
            },
            {
                "Вопрос": "На каких данных основана модель",
                "Ответ": f"На {v6_metrics.get('n') or 70} почвенных профилях с лабораторно определённым содержанием солей.",
            },
            {
                "Вопрос": "Что следует учитывать",
                "Ответ": "Не рекомендуется напрямую сравнивать удалённые районы без предварительной локальной калибровки модели.",
            },
        ]
        st.dataframe(pd.DataFrame(v6_summary_rows), hide_index=True, width="stretch")

        st.markdown(
            "**Простыми словами.** Модель V6 помогает находить участки с наиболее низкой вероятностью засоления. "
            "Высокое значение оценки означает лишь то, что участок выглядит более перспективным с точки зрения засолённости. "
            "Это не гарантирует успешную приживаемость саксаула."
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

        with st.expander("Данные о саксауле, солончаках и засолении почв (техническая информация)"):
            st.markdown(
                """
**Модель оценки засоления почв (основной слой V6)**

Это основная модель, используемая для оценки риска засоления почв. Модель: логистическая регрессия с L2-регуляризацией. Единственный предиктор — спутниковый NDMI с пространственным разрешением 30 м.

Обучающая выборка: 70 геопривязанных почвенных профилей из отчёта Пачикина–Козыбаевой (2012–2014 гг.). Целевая переменная — содержание легкорастворимых солей в верхнем слое почвы более 1%; из 70 профилей 27 относятся к категории засолённых. Качество модели: LOO AUC = 0,682; 95% доверительный интервал: 0,556–0,802.

Ограничения модели: при пространственной проверке выявлена зависимость качества модели от региона исследования. При объединённой пространственной проверке (leave-block-out) получено AUC = 0,385. Среднее значение AUC внутри отдельных пространственных блоков составляет 0,792.

Из 70 обучающих почвенных профилей только около 15 расположены непосредственно в пределах исследуемой территории высохшего дна Аральского моря (AOI). Остальные используются для расширения обучающей выборки и не являются независимой проверкой модели внутри исследуемой области. Качество модели: внутри AOI: LOO AUC = 0,614; вне AOI: LOO AUC = 0,671.

**Данные о пригодности саксаула**

Во всей базе данных имеется только 6 геопривязанных положительных наблюдений саксаула (из 70 почвенных профилей). Остальные записи представлены отрицательными, условно отрицательными или исключёнными из анализа наблюдениями.

Попытка построить классификатор пригодности местообитаний саксаула на основе спутниковых индексов NDMI и MSAVI показала качество LOO AUC ≈ 0,48, что статистически не отличается от случайного угадывания. В документации проекта эта модель обозначена как «только для исследовательских целей, не для принятия решений».

Выявленные зависимости между почвенными характеристиками, данными дистанционного зондирования и наличием саксаула соответствуют известным экологическим особенностям саксаула, однако не удовлетворяют критериям статистической устойчивости, принятым в проекте (MIN_N = 12; MIN_N_POS = 8; MIN_AUC = 0,62). Поэтому все они рассматриваются только как предварительные индикаторы.

Установлены следующие тенденции: низкое содержание хлоридов (Cl ≤ 0,059%): ориентированный AUC = 0,733; низкое общее содержание солей (≤ 0,311%): ориентированный AUC = 0,642; низкое содержание обменного натрия (≤ 0,262): ориентированный AUC = 0,692; повышенное содержание карбонатов (CaCO₃ ≥ 5,3%): ориентированный AUC = 0,855; более лёгкий песчаный механический состав (содержание песка ≥ 62,56%): ориентированный AUC = 0,695; более низкие значения NDMI и NDWI, характеризующие более сухую поверхность: AUC = 0,647 (NDMI) и 0,737 (NDWI).

Полученные направления связей согласуются с известными экологическими особенностями саксаула: он чаще встречается на карбонатных, песчаных почвах с низким содержанием хлоридов и обменного натрия. Однако статистическая надёжность этих результатов ограничена малым числом положительных наблюдений.

**Особенности формирования обучающих данных**

Согласно правилам контроля качества проекта (`SAXAUL_LABELS_QA.md`), отрицательные метки присваивались только документально подтверждённым случаям неудачных посадок саксаула и участкам с полностью отсутствующей растительностью. Они не присваивались автоматически на основании высокой засолённости почвы, чтобы избежать включения этого предположения в обучающие данные модели.

Поэтому часть наблюдений, где саксаул отсутствует, не относится к участкам с высокой засолённостью, несмотря на внешнее сходство с такими территориями.

**Полевые примеры**

Разрез 08/14 (2014). Описание участка: «Такыр без растительности». Содержание солей в верхнем слое почвы составило 0,43%, что ниже принятого в проекте порога засолённости (1%). Этот участок классифицирован как безрастительная территория, а не как подтверждённый пример высокой засолённости.

Разрез S134 (AralField). Независимый набор данных, не входящий в обучающую выборку из 70 почвенных профилей. Описание: «бывшее озеро Сорколь, очень высокая засолённость, солончак». Саксаул отсутствует. Это подтверждённый пример, в котором отсутствие саксаула связано с высокой засолённостью.

Разрез 13/13 (2013). Зафиксирована неудачная посадка: «Сажали саксаул, ничего не прижилось». Содержание солей в верхнем слое почвы составило 1,004%, что превышает установленный порог 1%.

Разрез 24/14 (2014). Документально подтверждённая неудачная посадка на распаханном и нарушенном участке. Описание: «Посадка неудачно прижилась, меньше, чем на естественном участке, только землю испортили». Содержание солей составило 0,143%, что значительно ниже порога 1%. Полевые наблюдения связывают причину неудачи с нарушением почвенного покрова, а не с засолённостью.

Разрез 4А/12 (2012). Подтверждённое присутствие саксаула. Описание растительности: «кустарниково-солянковая растительность (тамарикс, саксаул, селитрянка, солянка супротивнолистная, лебеда солончаковая и др.)». Этот пример показывает, что саксаул может произрастать совместно с галофитной растительностью на засолённых почвах, то есть засолённость является ограничивающим фактором, но не исключает полностью возможность его произрастания.

**Ограничение интерпретации результатов**

Модель оценки засолённости (LOO AUC = 0,682; 95% ДИ: 0,556–0,802; n = 70) имеет достаточное статистическое обоснование для оценки пространственного распределения риска засоления.

В отличие от неё, зависимости, связанные с произрастанием саксаула, пороговые значения и приведённые полевые примеры основаны только на шести подтверждённых наблюдениях присутствия саксаула и небольшом числе дополнительных отрицательных наблюдений. Поэтому эти результаты следует рассматривать исключительно как предварительные ориентиры для отбора участков. Они не должны интерпретироваться как модель прогнозирования приживаемости саксаула или как доказательство пригодности территории для проведения посадок.
                """
            )

        # spatial validation honesty
        if sm:
            pb = sm.get("spatial_lbo_perblock_auc")
            pooled = sm.get("spatial_lbo_pooled_auc")
            sign = sm.get("within_block_sign_positive", "—")
            with st.expander("Пространственная проверка модели", expanded=False):
                st.markdown(
                    f"""
                    Почвенные точки, расположенные близко друг к другу, как правило, обладают сходными характеристиками, что может приводить к завышенной оценке качества модели. Чтобы исключить этот эффект, обучающая выборка была разделена на пространственные блоки размером около {sm.get('block_km', 20):.0f} км, после чего модель многократно переобучалась, последовательно исключая один блок из обучения и используя его для проверки.

                    Среднее значение AUC по отдельным блокам: **{pb}**. Это показывает, что в пределах каждого исследуемого района модель успешно различает участки с высокой и низкой засолённостью.

                    Общее значение AUC для всех блоков: **{pooled}**. Более низкий показатель обусловлен различиями в исходном уровне засолённости между районами: в одних районах большинство участков засолены, в других — наоборот. Это отражает необходимость дополнительной межрегиональной калибровки модели, а не отсутствие связи между спутниковыми данными и засолённостью почв.

                    Связь между индексом влажности (NDMI) и засолённостью сохраняет одинаковое направление в **{sign}** протестированных пространственных блоках.

                    Основной вывод: модель следует использовать для ранжирования участков в пределах одного района. Для сопоставления абсолютных уровней засолённости между удалёнными районами требуется дополнительная локальная калибровка модели.
                    """
                )

        # suitability zones from the wall-to-wall 30m layer
        stats = v6.get("suit_stats", {})
        zone_ha = stats.get("zone_area_ha", {})
        if zone_ha:
            names = {"1": "1. Перспективная зона (низкая засолённость)", "3": "3. Умеренная засолённость",
                     "4": "4. Высокая засолённость", "10": "10. Существующая растительность", "0": "0. Водоёмы / отсутствуют данные"}
            land = sum(float(zone_ha.get(k, 0)) for k in ("1", "3", "4", "10"))
            rows = []
            for k in ("1", "3", "4", "10", "0"):
                ha = float(zone_ha.get(k, 0))
                rows.append({
                    "Зона V6": names[k],
                    "Площадь, га": fmt_int(ha),
                    "Доля территории": f"{ha / land * 100:.1f}%" if (land and k != "0") else "—",
                })
            st.markdown("**Распределение классов модели V6 (30 м):**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            vf = stats.get("valid_fraction_of_aoi")
            st.caption(
                f"30-метровый слой охватывает {vf*100:.0f}% исследуемой территории. "
                "Зоны 3 и 4 не являются самостоятельными категориями, а представляют собой разные уровни единой шкалы оценки риска засоления, прошедшей валидацию."
                if vf is not None else
                "Зоны 3 и 4 не являются самостоятельными категориями, а представляют собой разные уровни единой шкалы оценки риска засоления, прошедшей валидацию."
            )

        # ground-truth + independent validation
        pv = v6.get("pit_validation", {})
        if pv:
            det = pv.get("saline_detector_zone34", {})
            cc1, cc2 = st.columns(2)
            cc1.metric("Количество точек, охваченных моделью", f"{pv.get('v6_scored_nonwater', '—')}/70",
                       help="Полевые точки без воды, которые попали в оценённые зоны.")
            sens_ci = det.get("sensitivity_ci95")
            spec_ci = det.get("specificity_ci95")
            spec_n = det.get("specificity_n")
            sens_n = det.get("sensitivity_n")
            cc2.metric("Показатели модели выявления засоления",
                       f"чувствительность {det.get('sensitivity', '—')} / специфичность {det.get('specificity', '—')}",
                       help="Насколько хорошо зоны 3/4 находят точки с засолением выше 1%. "
                            "Читайте вместе с размерами выборки и 95% интервалами ниже: специфичность 1.0 "
                            "на нескольких отрицательных точках не означает идеальный фильтр.")
            if spec_ci and sens_ci:
                st.caption(
                    f"Чувствительность {det.get('sensitivity','—')} рассчитана по {sens_n} положительным наблюдениям (95% доверительный интервал: "
                    f"{sens_ci[0]}–{sens_ci[1]}). Специфичность {det.get('specificity','—')} рассчитана только по {spec_n} "
                    f"отрицательным наблюдениям (95% доверительный интервал: {spec_ci[0]}–{spec_ci[1]}), поэтому её оценку следует интерпретировать с осторожностью из-за небольшого объёма выборки."
                )
            n_in = aoi_split.get("n_in")
            n_out = aoi_split.get("n_out")
            st.caption(
                f"Из {v6_metrics.get('n') or 70} обучающих профилей {n_out or 'часть'} лежат за границей моря 1960 года. "
                f"Внутри целевого высохшего дна остаётся около {n_in or pv.get('v6_scored_nonwater', '—')} точек для проверки. "
                f"Полная таблица: `{rel_path(V6_PIT_TABLE_PATH)}`."
            )

