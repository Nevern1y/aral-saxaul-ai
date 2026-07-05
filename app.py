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

st.set_page_config(page_title="Aral Saxaul: Phytomelioration Platform", layout="wide")
if not hasattr(st, "iframe"):
    st.error("This dashboard needs Streamlit 1.57 or newer to show interactive maps.")
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
    with st.expander("For technical users"):
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
        "Zone": [
            "Candidate zone",
            "Existing vegetation",
            "Wet brine risk",
            "Dry salt risk",
            "Steep terrain",
            "Water / shadow / no data",
        ],
        "Share (%)": [
            round(opt_pct, 1), round(veg_pct, 1),
            round(brine_pct, 1), round(salt_pct, 1),
            round(obst_pct, 1), round(water_pct, 1),
        ],
    })

    audit_colors = {
        "Candidate zone": "#2ecc40",
        "Existing vegetation": "#7FCDBB",
        "Wet brine risk": "#D95F02",
        "Dry salt risk": "#E6AB02",
        "Steep terrain": "#636363",
        "Water / shadow / no data": "#BDBDBD",
    }

    fig = px.pie(
        audit_data,
        values="Share (%)",
        names="Zone",
        color="Zone",
        color_discrete_map=audit_colors,
        hole=0.4,
    )
    fig.update_traces(textinfo="label+percent", textposition="outside", textfont_size=10)
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=10, b=20))
    return fig


st.title("Aral Saxaul V6: preliminary screening of sites by salinity risk")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "This map helps you choose where to go for a field check. It estimates salinity risk, "
    "suggests the next step for the site you select, and does not replace a soil sample or a planting decision. "
    "V6 is the main layer; V5.1 gives extra detail and is the basis for KML route planning."
    "</p>",
    unsafe_allow_html=True,
)

tab_analytics, tab_dev, tab_logistics = st.tabs([
    "Map and summary",
    "Checks and data",
    "Work planning",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: 📍 Map of work sites
# ══════════════════════════════════════════════════════════════════════

with tab_logistics:
    st.subheader("Field visit planning")
    st.info(
        "Steps: 1) choose sites with road access, 2) download the KML files, "
        "3) check the soil and coordinates in the field, 4) only then plan planting and budget."
    )
    tasks_df = load_tasks()
    roads_gdf = load_roads()
    v5_stats = load_v5_stats()

    if tasks_df.empty:
        st.warning("Planning data is not loaded. The V6 map is available, but the KML task files cannot be shown yet.")
        render_technical_commands([
            "python scripts/v5_roads_prep.py",
            "python scripts/v5_logistics_prep.py",
        ])
    else:
        if "territory_scope" in tasks_df.columns and set(tasks_df["territory_scope"].dropna()) == {"kazakhstan"}:
            st.caption("The planning below only covers sites inside Kazakhstan's territory.")

        tasks_df["distance_to_road_km"] = pd.to_numeric(tasks_df["distance_to_road_km"], errors="coerce")
        if "distance_to_kazakhstan_road_km" in tasks_df.columns:
            tasks_df["distance_to_kazakhstan_road_km"] = pd.to_numeric(
                tasks_df["distance_to_kazakhstan_road_km"],
                errors="coerce",
            )
        access_options = {"Any OSM road": "distance_to_road_km"}
        if "distance_to_kazakhstan_road_km" in tasks_df.columns and tasks_df["distance_to_kazakhstan_road_km"].notna().any():
            access_options["Kazakhstan road access"] = "distance_to_kazakhstan_road_km"

        max_cell_ha = float(tasks_df["area_ha"].max())

        col_f0, col_f1, col_f2 = st.columns(3)
        with col_f0:
            selected_access = st.selectbox(
                "How to measure road access:",
                options=list(access_options.keys()),
                index=1 if "Kazakhstan road access" in access_options else 0,
                help="For the first field visit, it is usually better to measure access from the Kazakhstan side, if this layer is available.",
            )
            distance_col = access_options[selected_access]

        with col_f1:
            max_dist = float(tasks_df[distance_col].max())
            road_scenarios = {
                "Close to roads (up to 120 km)": 120.0,
                "Far visits (up to 250 km from roads)": 250.0,
                "Show full coverage": max_dist,
            }
            selected_road_scen = st.selectbox(
                "Road access:",
                options=list(road_scenarios.keys()),
                index=0,
                help="We recommend starting close to roads. This makes it easier to check the model in the field without a costly expedition.",
            )
            dist_thresh = road_scenarios[selected_road_scen]

        with col_f2:
            area_scenarios = {
                "Small sites (10-1,000 ha)": (10, 1000),
                "Large sites (1,000-5,000 ha)": (1000, 5000),
                "Very large sites (>5,000 ha)": (5000, int(max_cell_ha)),
                "All sizes": (0, int(max_cell_ha)),
            }
            selected_area_scen = st.selectbox(
                "Site size:",
                options=list(area_scenarios.keys()),
                index=0,
                help="Small and medium sites are easier for an initial check: lower risk, easier to visit, and easier to sample.",
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
            "Area of selected sites",
            f"{selected_area_ha:,.0f} ha",
            help="Total area of the cells that pass the current filters. This is not a planting plan until sites are checked in the field.",
        )
        col_m2.metric("Cells selected", f"{len(filtered):,}")
        col_m3.metric("Total in index", f"{len(tasks_df):,}")
        col_m4.metric(
            "Share of index area",
            f"{selected_area_ha / total_task_area_ha * 100:.1f}%" if total_task_area_ha else "0%",
        )
        if v5_stats:
            full_aoi_ha = v5_stats.get("candidate_100m_area_ha", v5_stats.get("area_ha", 0))
            st.caption(
                f"For reference: the full V5.1 candidate-zone estimate for the AOI is {full_aoi_ha:,.0f} ha. "
                "This filter only covers the KML site index used for field visits."
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
                name="Roads (OSM)",
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
                name="Sites to check",
                style_function=lambda f: {
                    "fillColor": "#2ecc40", "color": "#27ae60",
                    "weight": 1.0, "fillOpacity": 0.3,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["filename", "area_ha", "dist_km", "dist_kz_km"],
                    aliases=["File:", "Area (ha):", "To selected road (km):", "To KZ road (km):"],
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
            "On the map: green squares are sites that pass the filters; markers show the 5 sites closest to the selected road, for the first field visit."
        )
        _render_map(m.get_root().render())

        with st.expander("KML route files", expanded=True):
            st.caption(
                "You can open these KML files in a GPS device, Google Earth, or QGIS. Start with the closest sites, "
                "and make any planting decision only after a soil check."
            )
            st.caption(f"KML route files are stored in `{rel_path(KML_TASKS_DIR)}`.")
            sorted_filtered = filtered.sort_values(distance_col, ascending=True)
            display_cols = ["filename", "centroid_lat", "centroid_lon", "area_ha", "distance_to_road_km"]
            if "distance_to_kazakhstan_road_km" in filtered.columns:
                display_cols.append("distance_to_kazakhstan_road_km")
            display_df = sorted_filtered[display_cols].copy()
            display_df.columns = [
                "KML file", "Latitude", "Longitude", "Area (ha)", "Distance to any road (km)",
                *(["Distance to KZ road (km)"] if "distance_to_kazakhstan_road_km" in filtered.columns else []),
            ]
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
            )
            kml_bytes = zip_kml_files(sorted_filtered["filename"].head(25)) if not filtered.empty else b""
            if kml_bytes:
                st.download_button(
                    "Download the first 25 KML files for the current filter",
                    data=kml_bytes,
                    file_name="aral_saxaul_field_tasks_top25.kml.zip",
                    mime="application/zip",
                    help="Files are sorted by the current table, by distance to the selected road.",
                )
            elif not filtered.empty:
                st.info(f"No KML files found in `{rel_path(KML_TASKS_DIR)}`.")

        st.warning(
            "The resource estimate below is not a planting plan. First confirm the selected sites in the field."
        )
        with st.expander("Preliminary resource estimate after a field check", expanded=False):
            st.caption(
                "Do not use this estimate as a planting plan. It is only meant to show scale, after a site has been confirmed in the field."
            )
            if selected_area_ha > 0:
                density = st.slider(
                    "Planting density (seedlings/ha)",
                    min_value=1000, max_value=3000, value=1500, step=100,
                )
                productivity = st.slider(
                    "Tractor output (ha/shift)",
                    min_value=5, max_value=20, value=10, step=1,
                )
                fuel_rate = st.slider(
                    "Diesel use (L/ha)",
                    min_value=10.0, max_value=30.0, value=15.0, step=0.5,
                )
                total_saplings = int(selected_area_ha * density)
                total_fuel = selected_area_ha * fuel_rate
                total_machine_shifts = selected_area_ha / productivity

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Area in the current filter", f"{selected_area_ha:,.0f} ha")
                col_r2.metric("Seedlings", f"{total_saplings:,}")
                col_r3.metric("Machine-shifts", f"{total_machine_shifts:,.0f}")
                st.metric("Diesel, approximate", f"{total_fuel:,.0f} L")
            else:
                st.info("No sites match the current filters.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2: 📊 Overall statistics
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
        st.warning("V5.1 detail data is not loaded. The main V6 map is still available.")
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
        "Low salinity risk (V6)",
        f"{v6_low_salt_ha:,.0f} ha" if v6_low_salt_ha else "—",
        help="V6 zone with low salinity risk. These are candidates to check, not a ready planting area.",
    )
    col2.metric(
        "Scored by V6",
        f"{v6_coverage * 100:.0f}% AOI" if v6_coverage is not None else "—",
        help="Share of the study area where V6 could produce a salinity-risk score.",
    )
    col3.metric(
        "Model validation",
        fmt_metric(v6_metrics.get("auc")),
        delta=f"CI {v6_ci[0]:.3f}-{v6_ci[1]:.3f}" if v6_ci else None,
        help="LOO AUC: how well the model tells saline soil profiles apart from non-saline ones.",
    )
    col4.metric(
        "V5.1 KML contours",
        f"{v5_stats.get('clusters', 0):,}",
        help="Helper-layer V5.1 contours used to organize field visits.",
    )

    # ── Map in the most visible spot: V6 is the headline, V5.1 gives extra detail ──
    st.markdown("### V6 salinity-risk map (preliminary screening)")
    if V6_MAP_PATH.exists():
        _render_map(V6_MAP_PATH.read_text(encoding="utf-8"))
        auc_txt = fmt_metric(v6_metrics["auc"])
        ci = v6_metrics.get("ci")
        ci_txt = f", 95% CI {ci[0]:.3f}-{ci[1]:.3f}" if ci else ""
        n_txt = v6_metrics.get("n") or 70
        st.info(
            "How to use this: hover over or click a colored site on the map. "
            "The panel on the map shows a 0-100 score, the salinity risk, and a practical next step. "
            "Green does not mean planting is approved. It means the site is a priority for a field check."
        )
        with st.expander("Why this is only preliminary screening"):
            st.markdown(
                f"V6 was trained on {n_txt} measured soil profiles and screens for salinity "
                f"risk (LOO AUC {auc_txt}{ci_txt}). It does not know saxaul survival rates, "
                "groundwater depth, or current field conditions. The final decision should only be "
                "made after a field visit and a soil sample."
            )
    else:
        st.info(
            f"V6 map not found: {V6_MAP_PATH.name}. "
            "The V5.1 map is shown below, if it exists."
        )
        render_technical_commands([
            "python scripts/v6/build_suitability_index.py",
            "python scripts/v6/render_v6_map.py",
        ])

    # V5.1 — 10 m detail + source of logistics data (kept, not removed)
    with st.expander("Detailed V5.1 map (10 m, Sentinel-2 rules)", expanded=not V6_MAP_PATH.exists()):
        if V5_MAP_PATH.exists():
            _render_map(V5_MAP_PATH.read_text(encoding="utf-8"))
            st.caption(
                "V5.1 is a 10 m helper layer: it helps you see small boundaries, water/terrain, "
                "and build KML files for field visits. Check site priority against V6 and a field soil sample."
            )
            if V5_OPERATIONAL_PATH.exists():
                gj_size_mb = V5_OPERATIONAL_PATH.stat().st_size / (1024 * 1024)
                if gj_size_mb < 50:
                    gj_bytes = V5_OPERATIONAL_PATH.read_bytes()
                    st.download_button(
                        label="Download site polygons (GeoJSON, >=10 ha)",
                        data=gj_bytes,
                        file_name=V5_OPERATIONAL_PATH.name,
                        mime="application/geo+json",
                        help="Exports candidate-zone contours (clusters >=10 ha only) for GPS use and further field checks.",
                    )
                else:
                    st.info(
                        f"📥 File: `{V5_OPERATIONAL_PATH.name}` ({gj_size_mb:.0f} MB). "
                        "Copy it from `outputs/data/`."
                    )
        else:
            st.warning(
                f"V5.1 map file not found: {V5_MAP_PATH.name}. "
                "The main V6 map does not depend on this file."
            )

    st.markdown("### What to do with the result")
    action_rows = [
        {
            "If the map shows": "High score and low salinity risk",
            "What it means": "Put this site higher on the field-check list.",
            "Next step": "Check road access, open the KML file, and take a soil sample.",
        },
        {
            "If the map shows": "Medium score",
            "What it means": "The site is not ruled out, but salinity risk is notable.",
            "Next step": "Check it after the best sites, or use it as a control point.",
        },
        {
            "If the map shows": "High salinity risk, or existing vegetation",
            "What it means": "This is a weak candidate for new planting unless there is another reason to check it.",
            "Next step": "Usually skip it, or survey it separately.",
        },
    ]
    st.dataframe(pd.DataFrame(action_rows), hide_index=True, width="stretch")

    with st.expander("What the V5.1 helper layer (10 m) found"):
        st.info(
            "V5.1 splits the area into 6 classes using Sentinel-2 imagery at 10 m resolution. "
            "It is meant for extra detail and KML planning, not as the main V6 score."
        )
        if total_px and v5_class_pixels:
            class_meta = {
                0: ("Water / shadow / no data", "Excluded: water, shadow, no data, or very dark pixels"),
                1: ("Candidate zone", "Places to check after comparing with V6"),
                3: ("Dry salt risk", "Possible dry salt crust, based on satellite signals"),
                4: ("Wet brine risk", "Possible wet, salty surface or rising brine"),
                5: ("Steep terrain", "Slopes or sharp terrain features"),
                10: ("Existing vegetation", "Vegetation is already present; not a class for new planting"),
            }
            class_rows = []
            for cls_val in [1, 10, 4, 3, 5, 0]:
                pixel_count = v5_class_pixels.get(cls_val, 0)
                area_ha = pixel_count * pixel_area_ha
                class_rows.append(
                    {
                        "Code": cls_val,
                        "V5.1 class": class_meta[cls_val][0],
                        "Area, ha (10 m)": round(area_ha, 1),
                        "Area, km2": round(area_ha / 100.0, 1),
                        "Share of area, %": round(pixel_count / total_px * 100.0, 2),
                        "How to read it": class_meta[cls_val][1],
                    }
                )
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")

    top10_ha = v5_stats.get("top10_ha", [])
    if top10_ha:
        top10_share = sum(top10_ha) / candidate_100m_area_ha * 100 if candidate_100m_area_ha else 0
        st.caption(
            f"The 10 largest connected V5.1 zones on the 100 m grid cover {sum(top10_ha):,.0f} ha "
            f"({top10_share:.1f}% of the candidate-zone estimate on the 100 m grid). "
            "In other words, the map result is mostly concentrated in a few large areas, not just scattered small spots."
        )
    if operational_area_ha:
        st.caption(
            f"Area of V5.1 field-visit contours >=10 ha: {operational_area_ha:,.0f} ha. "
            "This is smaller than the overall 100 m grid estimate because small spots are excluded from the working GeoJSON file."
        )

    # ── Scientific interpretation (left) + spectral audit (right)
    col_interp, col_audit = st.columns([1, 1])

    with col_interp:
        st.markdown("### Main conclusions, without exaggeration")
        conclusion_rows = [
            {
                "Question": "What this map shows",
                "Current answer": "V6 screens for salinity risk using a lab-verified link between NDMI and salt",
                "How to read it": "It helps you choose places to check. It does not prove that planting there will succeed.",
            },
            {
                "Question": "Where to start",
                "Current answer": f"{v6_low_salt_ha:,.0f} ha in the V6 low-salinity-risk zone" if v6_low_salt_ha else "V6 statistics are not available",
                "How to read it": "This is the top of the field-check list, not a planting area.",
            },
            {
                "Question": "Sites for field visits",
                "Current answer": f"{v5_stats.get('clusters', 0):,} V5.1 contours >=10 ha for KML and logistics",
                "How to read it": "Check these against V6 and confirm with a soil check in the field.",
            },
            {
                "Question": "Main limitation",
                "Current answer": "V6 screens for salt, but not saxaul survival rate or every local condition",
                "How to read it": "Compare sites within the same district, and confirm results with a field soil sample.",
            },
        ]
        st.dataframe(pd.DataFrame(conclusion_rows), hide_index=True, width="stretch")

    with col_audit:
        st.markdown("### Share of V5.1 classes (10 m detail)")

        if total_px > 0:
            pixels_json = json.dumps({str(k): v for k, v in v5_class_pixels.items()})
            fig_audit = make_audit_fig(pixels_json, total_px)
            if fig_audit is not None:
                st.plotly_chart(fig_audit, width="stretch")
        else:
            st.info("No raster classification data found.")

    st.markdown("### Salt flats and saxaul: what the evidence says")
    st.markdown(
        """
        Some patches of the dried Aral seabed are what soil scientists call solonchaks. This is ground where salt has built up at or near the surface, often visible as a whitish crust. These flats formed naturally as the old sea or lake bed dried out and left mineral salts behind. Some of them are almost bare of any plants at all.

        Salt is hard on plant roots, and saxaul is no exception. It is one of the toughest, most salt-tolerant shrubs available for this landscape, but even it struggles on the most heavily salted ground. The project's field records include a former salt lake bed, logged as very saline, where no saxaul was found, and one planting attempt on similarly salty ground that failed to establish. Two other failed plantings in the records sit on bare or ploughed ground; the field notes there point to disturbance and lack of vegetation, not measured salt, as the likely reason. At least one site with healthy saxaul also had other salt-tolerant shrubs and salt-marsh plants growing right alongside it, so saxaul and salty ground are not mutually exclusive. Across the whole project, only six pits have a confirmed saxaul presence record, so this pattern is a hint worth checking in the field, not a proven rule.

        Put simply: high salt content makes it much less likely that saxaul will take root without treatment. It does not rule saxaul out everywhere. Treat salinity as a screening flag, not a hard line. The dashboard's map flags zones with higher or lower salinity risk, using satellite and soil data. Use it to decide which sites to visit first, and check the salt on the ground before committing to plant.

        If a salty site must be used, treat the soil first. Standard steps include flushing out excess salt with irrigation, adding gypsum or organic matter, and improving drainage. These are recognized methods, though this project has no field outcome data yet on how well they work at these specific sites. Planting directly into untreated salty ground, without any of this, is a poor bet. In every case, the map is a starting point for deciding where to look next. It is not a substitute for a soil test and a trained eye in the field.
        """
    )
    st.caption(
        "This finding is supported by the project's 70-profile V6 salinity model plus a small set of field "
        "records on saxaul presence/absence. See 'Saxaul-specific evidence' in the Technical tab for the full numbers."
    )

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ⚙️ Technical model parameters
# ══════════════════════════════════════════════════════════════════════

with tab_dev:
    st.subheader("How to use the result safely")
    st.info(
        "This tab explains how much to trust the map, and where its limits are. The main result is the V6 salinity risk score; "
        "V5.1 provides 10 m detail and KML planning."
    )
    safety_rows = [
        {
            "OK to do": "Choose places for a field check",
            "Not OK to do": "Treat green zones as a finished planting plan",
            "What to verify in the field": "Salt in the topsoil, road access, actual water/vegetation, and coordinates",
        },
        {
            "OK to do": "Compare sites within the same district",
            "Not OK to do": "Mechanically compare distant districts without calibration",
            "What to verify in the field": "Local salinity background and fresh control samples",
        },
        {
            "OK to do": "Use the KML file as a route for a survey",
            "Not OK to do": "Treat the KML file as approval to plant",
            "What to verify in the field": "Site boundaries and whether equipment/logistics fit",
        },
    ]
    st.dataframe(pd.DataFrame(safety_rows), hide_index=True, width="stretch")

    st.subheader("What still needs improvement")

    roadmap_rows = [
        {
            "Priority": 1,
            "What to do": "Collect more field samples inside the seabed area",
            "What is needed": "70 profiles train the model, but there are few check points inside the target Aral seabed contour",
            "Why": "The main limit on accuracy is fresh field data inside the target area, not the interface.",
        },
        {
            "Priority": 2,
            "What to do": "Calibrate salinity levels across districts",
            "What is needed": "Reference samples in different blocks to build a shared scale (accuracy drops noticeably when combining distant districts; see the \"Spatial validation\" section)",
            "Why": "Within one site, the model ranks correctly, but the absolute level drifts between districts.",
        },
        {
            "Priority": 3,
            "What to do": "Match soil samples to the imagery date",
            "What is needed": "Fresh sampling that matches current NDMI imagery (lab data is from 2012-2014)",
            "Why": "The NDMI-to-salt link is checked against data collected at a different time; matching the dates would strengthen it.",
        },
        {
            "Priority": 4,
            "What to do": "Independent check using the 2020/2021 field campaign",
            "What is needed": "Use 2020/2021 samples as a separate test set (do not mix them with the 2012-2014 training data)",
            "Why": "This gives a fair, independent check with no leakage between data sets.",
        },
        {
            "Priority": 5,
            "What to do": "Link the score to planting survival",
            "What is needed": "Data on the survival of real saxaul plantings, by site",
            "Why": "This would move the tool from \"salinity risk\" to a verified forecast of planting success.",
        },
    ]
    with st.expander("Model and data improvement roadmap", expanded=False):
        st.dataframe(pd.DataFrame(roadmap_rows), hide_index=True, width="stretch")

    with st.expander("Technical notes on the V5.1 helper layer"):
        col_rules, col_stats = st.columns([1, 1])

        with col_rules:
            st.markdown("**What V5.1 does:** it excludes water/shadow, steep terrain, existing vegetation, and signs of salty surfaces. What is left forms candidate zones to check; this is not proof that planting will work.")
            st.markdown(
                """
                | Class | Simple rule |
                |---|---|
                | **0 Water / shadow / no data** | Pixel looks like water, shadow, or bad data |
                | **5 Steep terrain** | Slope greater than 5 degrees |
                | **10 Existing vegetation** | Green vegetation is already present |
                | **4 Wet brine risk** | Signs of high moisture and a salty surface at the same time |
                | **3 Dry salt risk** | Signs of a dry salt crust |
                | **1 Candidate zone** | Everything not caught by the risks listed above |
                """
            )

        with col_stats:
            st.markdown("**V5.1 class distribution (10 m):**")
            v5_pcts = {}
            if total_px and v5_class_pixels:
                for c in [0, 1, 3, 4, 5, 10]:
                    v5_pcts[c] = v5_class_pixels.get(c, 0) / total_px * 100
            stats_data = {
                "Class": ["1 Candidate", "3 Dry salt", "4 Wet brine", "5 Terrain", "10 Vegetation", "0 Water/shadow"],
                "Pixels": [
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
            st.caption("V5.1 is limited to usable satellite pixels, and does not replace the V6 salinity score.")

    # ── V5 Dynamic Thresholds ──────────────────────────────────────
    if v5_thresholds:
        st.markdown("---")
        with st.expander("V5.1 thresholds, for technical users"):
            st.caption("These thresholds are calculated automatically from the Sentinel-2 image. They are used for the V5.1 helper 10 m classification.")
            th_cols = st.columns(3)
            metrics_def = [
                ("Moisture (NDMI P15)", "NDMI_P15", "Lower NDMI bound for signs of a dry solonchak (a salt-affected soil type)."),
                ("Moisture (NDMI P85)", "NDMI_P85", "Upper NDMI bound for signs of capillary brine (salty groundwater drawn up to the surface)."),
                ("Salinity (NDSI P15)", "NDSI_Green_SWIR2_P15", "Lower bound of the salinity index."),
                ("Salinity (NDSI P85)", "NDSI_Green_SWIR2_P85", "Upper bound of the salinity index."),
                ("Brine (B8/B12 P15)", "BR_NIR_SWIR2_P15", "Lower bound of the NIR/SWIR2 ratio."),
                ("Brine (B8/B12 P85)", "BR_NIR_SWIR2_P85", "Upper bound of the NIR/SWIR2 ratio."),
            ]
            for i, (label, key, help_text) in enumerate(metrics_def):
                val = v5_thresholds.get(key, "N/A")
                if isinstance(val, float):
                    val = f"{val:.4f}"
                th_cols[i % 3].metric(label, val, help=help_text)

    st.markdown("---")
    st.subheader("Validation reports")
    st.caption(
        "These reports show how the map was checked, where there are disputed coordinates, and how much the result depends on the thresholds used. This is not a final accuracy assessment."
    )
    report_cols = st.columns(3)
    report_defs = [
        ("Point-by-point check", V5_VALIDATION_REPORT_PATH, "python scripts/v5_validation_report.py"),
        (
            "Choosing correct coordinates",
            V5_COORDINATE_ADJUDICATION_REPORT_PATH,
            "python scripts/v5_coordinate_adjudication_report.py",
        ),
        ("Threshold sensitivity", V5_UNCERTAINTY_REPORT_PATH, "python scripts/v5_uncertainty_report.py"),
    ]
    missing_report_commands = []
    for idx, (label, path, command) in enumerate(report_defs):
        with report_cols[idx]:
            if path.exists():
                st.download_button(
                    label=f"Download: {label}",
                    data=path.read_text(encoding="utf-8"),
                    file_name=path.name,
                    mime="text/markdown",
                )
                with st.expander(f"View: {label}"):
                    st.markdown(path.read_text(encoding="utf-8"))
            else:
                st.info(f"The \"{label}\" report has not been created yet.")
                missing_report_commands.append(command)
    if missing_report_commands:
        render_technical_commands(missing_report_commands)

    st.markdown("---")
    st.subheader("Which layers to use now")
    current_layers = [
        {
            "Layer": "V6",
            "What it is for": "Main answer: where salinity risk is lower",
            "How to use it": "Hover over the map, check the 0-100 score, and choose sites for a field check",
        },
        {
            "Layer": "V5.1",
            "What it is for": "10 m detail and KML contours for field visits",
            "How to use it": "Check boundaries, water/terrain, and build a route, but do not use it in place of the V6 score",
        },
    ]
    st.dataframe(pd.DataFrame(current_layers), hide_index=True, width="stretch")
    with st.expander("Archived versions V1-V4"):
        st.markdown(
            "V1-V4 are kept only as development history. They are not used for current decisions, "
            "should not return to the interface, and do not replace V6/V5.1."
        )

    # ── V6 lab-data science layer ──────────────────────────────────────
    v6 = load_v6_science()
    v6_metrics = v6_model_metrics(v6)
    if v6["salinity"]:
        st.markdown("---")
        st.subheader("V6 — current salinity-risk layer (70 soil profiles)")
        spatial = v6.get("spatial", {})
        sm = spatial.get("salinity_model", {})
        aoi_split = v6_aoi_split(v6)
        st.caption(
            "This is the main layer for preliminary screening by salinity risk. It is trained on measured "
            "soil salinity from the Pachikin-Kozybaeva report (2012-2014, 70 georeferenced soil profiles). "
            "V5.1 (10 m, rule-based screening) is kept as a detail and logistics layer."
        )
        v6_summary_rows = [
            {
                "Question": "What V6 checks",
                "Answer": "Salinity risk in the topsoil, not saxaul survival rate",
            },
            {
                "Question": "How to read the score",
                "Answer": "A higher score means lower expected salinity risk; it is a priority for a field visit, not a planting plan",
            },
            {
                "Question": "What data this is based on",
                "Answer": f"{v6_metrics.get('n') or 70} soil profiles with lab-measured salt content",
            },
            {
                "Question": "Where to be careful",
                "Answer": "Distant districts should not be compared directly without local calibration",
            },
        ]
        st.dataframe(pd.DataFrame(v6_summary_rows), hide_index=True, width="stretch")

        st.markdown(
            "**In plain terms:** V6 looks for sites with lower salinity risk. "
            "A higher score means lower expected salt risk, but it does not guarantee that a planting will survive."
        )
        with st.expander("For technical users: V6 metrics"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Profiles (training)",
                f"{v6_metrics.get('n') or '—'}",
                help="Soil profiles with measured topsoil salinity.",
            )
            c2.metric("Saline (>1%)", f"{v6_metrics.get('n_saline') or '—'}")
            loo = v6_metrics.get("auc")
            ci = v6_metrics.get("ci")
            c3.metric(
                "AUC (LOO)",
                fmt_metric(loo),
                help="Area under the ROC curve using leave-one-out validation: how well the model tells saline points apart from non-saline ones.",
            )
            c4.metric(
                "95% interval",
                f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "—",
                help="Bootstrap confidence interval for the AUC. It does not cross 0.5, so the relationship is stable.",
            )
            st.markdown(
                "The satellite NDMI index is linked to measured topsoil salinity "
                "(Spearman rho ≈ +0.66, p < 1e-9, n=70). The V6 score is calculated as "
                "`1 - probability of salinity` and is used only to rank candidates."
            )
            af = spatial.get("independent_aralfield")
            if af:
                st.markdown(
                    f"**Independent check (AralField 2018, saxaul):** AUC {af.get('auc')}, "
                    f"n={af.get('n')} ({af.get('n_present')} with saxaul), "
                    f"interval {af.get('ci95')[0] if af.get('ci95') else '—'}-"
                    f"{af.get('ci95')[1] if af.get('ci95') else '—'}. "
                    "There are too few points for a reliable estimate; treat this only as a directional signal."
                )

        with st.expander("Saxaul-specific evidence: solonchaks and salinity (technical detail)"):
            st.markdown(
                """
**Salinity model (well-supported, this is the main V6 layer):**
- Model: L2-regularized logistic regression, single predictor = satellite NDMI at 30 m resolution.
- Training data: 70 georeferenced soil profiles (Pachikin/Kozybaeva soil report, 2012-2014); target = topsoil total salts > 1%; 27/70 profiles positive.
- Performance: leave-one-out AUC 0.682, 95% CI [0.556, 0.802].
- Known caveat (regional calibration drift): pooled spatial leave-block-out AUC is only 0.385, while mean per-block spatial AUC is 0.792. The model ranks salinity risk reliably *within* a local area but should not be used to compare absolute salinity levels *across* distant regions without local calibration.
- Only about 15 of the 70 training profiles fall inside the actual 1960 seabed AOI; the rest are wider-region training support, not independent in-AOI validation. In-AOI LOO AUC is 0.614; out-of-AOI is 0.671.

**Saxaul-specific evidence (weak, exploratory only, NOT decision-grade):**
- Only 6 georeferenced positive saxaul field labels exist in the entire dataset (out of 70 profiles), against a much larger pool of negatives, weak negatives, and excluded rows.
- A direct NDMI+MSAVI to saxaul-suitability classifier scores LOO AUC approximately 0.48, statistically indistinguishable from chance. It is explicitly documented in the project as "exploratory only, not for decisions."
- Directional soil/remote-sensing correlations with saxaul suitability (all n_pos = 6 unless noted) match known saxaul ecology but fall below the project's own stability gates (MIN_N=12, MIN_N_POS=8, MIN_AUC=0.62) and are flagged `indicative_only=true`:
  - Low chloride (Cl <= 0.059%): oriented AUC 0.733, n=49, n_pos=6.
  - Low total salts (<= 0.311%): oriented AUC 0.642, n=56, n_pos=6.
  - Low exchangeable sodium (<= 0.262): oriented AUC 0.692, n=52, n_pos=6.
  - Higher carbonate/CaCO3 (>= 5.3%): oriented AUC 0.855, n=26, n_pos=3 (smallest, most fragile sample).
  - Sandier texture (sand >= 62.56%): oriented AUC 0.695, n=47, n_pos=5.
  - Lower NDMI/NDWI (drier surface signal): oriented AUC 0.647 (NDMI) / 0.737 (NDWI), n_pos=6.
- Spearman correlations against the y_suitable label (ML_DATASET_QA.md): top_caco3_pct rho=+0.39, p=0.05, n=26 (the only nominally significant one); salt_cl_pct rho=-0.27, p=0.07, n=49; rs30_ndwi rho=-0.25, p=0.06, n=56; exch_na/exch_sum rho=-0.21, p=0.13, n=52; sand_pct rho=+0.21, p=0.16, n=47. All directions are ecologically coherent (saxaul favors carbonate, sandy, low-chloride, low-sodium substrates) but magnitudes and significance are weak given n approximately 6 positives.

**Labeling methodology note (important for interpreting the examples below):** per the project's own QA rules (`SAXAUL_LABELS_QA.md`), hard negative labels are restricted to documented plantation failures and genuinely barren ground. They are deliberately **not** assigned on the basis of measured salinity, to avoid baking a salinity assumption into the very labels the model is supposed to learn from. This means some of the "no saxaul" field records below are not, in the project's own numbers, high-salinity sites, even though they look salt-related at first glance.

**Concrete field examples underlying the headline claim:**
- Pit 08/14 (2014): "Такыр без растительности" (a barren takyr, a bare clay flat, with no vegetation at all). Measured topsoil salt content here is 0.43%, below the project's own 1% saline threshold, so this is a "barren ground" negative, not a confirmed high-salinity negative. Labeled negative/strong, absent_recorded.
- AralField pit S134 (separate external validation dataset, not part of the 70-profile lab set): "Former Sorkol lake bed, very saline, solonchak"; haloxylon = 0 (absent). This is a genuinely salt-linked negative example.
- Pit 13/13 (2013): documented planting failure, "Сажали саксаул, ничего прижилось" (saxaul was planted, nothing survived). Measured topsoil salt content is 1.004%, just over the project's 1% saline threshold, so this is a genuinely salt-linked negative example. Labeled negative/strong, gold record.
- Pit 24/14 (2014): documented planting failure on ploughed/disturbed ground, "Посадка неудачно прижилась, меньше, чем на естественном участке, только землю испортили" (planting failed to establish well, worse than the natural stand, and degraded the soil). Measured topsoil salt content is only 0.143%, well below the 1% threshold; the field note attributes the failure to ploughing and soil disturbance, not to measured salt. Labeled negative/strong, gold record.
- Pit 4А/12 (2012): positive/strong presence label; vegetation description "кустарниково-солянковой растительностью (кустарники - тамарикс, саксаул, селитрянка; растительность - солянка супротивнолистная, лебеда солончаковая и др.)", i.e. a shrub-saltwort community including live saxaul growing alongside tamarisk, Nitraria, and other halophytes on salt-affected ground. This is the field counter-example showing salinity is a limiting factor that reduces the odds, not a modeled or dataset-confirmed absolute exclusion rule.

**Explicit caveat:** the salinity model (LOO AUC 0.682, CI [0.556, 0.802], n=70) is reasonably supported evidence about salinity distribution. The saxaul-specific correlations, thresholds, and field examples above are indicative screening hints only, built on just 6 positive field records (with a small number of separately sourced negative field examples, not all of which are actually high-salinity per the project's own numbers, as noted above). They must not be presented with the same confidence as the salinity model, nor described as predicting saxaul survival or proving planting suitability anywhere on the map.
                """
            )

        # spatial validation honesty
        if sm:
            pb = sm.get("spatial_lbo_perblock_auc")
            pooled = sm.get("spatial_lbo_pooled_auc")
            sign = sm.get("within_block_sign_positive", "—")
            with st.expander("Spatial validation (an honest look at the limits)", expanded=False):
                st.markdown(
                    f"""
                    To check whether accuracy looks better than it is just because nearby points are similar,
                    we split the data into spatial blocks (~{sm.get('block_km', 20):.0f} km) and
                    trained the model leaving out each block in turn.

                    - **Average AUC per block: {pb}** — within each site, the model correctly ranks
                      saline and non-saline points.
                    - Pooled AUC across all blocks: {pooled} — lower, because **the baseline salinity
                      level differs between districts** (in one block almost every point is saline,
                      in another almost none is). This is a cross-region calibration gap, **not
                      a loss of signal**.
                    - The NDMI-to-salt relationship has a positive sign in **{sign}** of the tested blocks.

                    Conclusion: the model is best used for local ranking within a district.
                    Comparing distant districts on one absolute scale needs extra
                    calibration. This limitation is stated openly here, not hidden.
                    """
                )

        # suitability zones from the wall-to-wall 30m layer
        stats = v6.get("suit_stats", {})
        zone_ha = stats.get("zone_area_ha", {})
        if zone_ha:
            names = {"1": "1 Candidate (low salinity)", "3": "3 Moderate salinity",
                     "4": "4 Strong salinity", "10": "10 Vegetation", "0": "0 Water/no data"}
            land = sum(float(zone_ha.get(k, 0)) for k in ("1", "3", "4", "10"))
            rows = []
            for k in ("1", "3", "4", "10", "0"):
                ha = float(zone_ha.get(k, 0))
                rows.append({
                    "V6 zone": names[k],
                    "Area, ha": f"{ha:,.0f}",
                    "% of land": f"{ha / land * 100:.1f}%" if (land and k != "0") else "—",
                })
            st.markdown("**V6 zones on the wall-to-wall 30 m layer (full coverage, no gaps; same area as V5.1):**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            vf = stats.get("valid_fraction_of_aoi")
            st.caption(
                f"Coverage: {vf*100:.0f}% of the study area gets a score (wall-to-wall 30 m layer), "
                "compared to about 46% for the 10 m composite. Zones 3 and 4 are a severity gradient of salinity along a single, "
                "validated NDMI axis."
                if vf is not None else
                "Zones 3 and 4 are a severity gradient of salinity along a single, validated NDMI axis."
            )

        # ground-truth + independent validation
        pv = v6.get("pit_validation", {})
        if pv:
            det = pv.get("saline_detector_zone34", {})
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Points covered by V6", f"{pv.get('v6_scored_nonwater', '—')}/70",
                       help="Non-water ground-truth points that fall inside scored zones.")
            cc2.metric("Points covered by V5.1", f"{pv.get('v5_covered_nonwater', '—')}/70",
                       help="For comparison: the frozen 10 m product covers fewer points.")
            cc3.metric("Salinity detector",
                       f"sensitivity {det.get('sensitivity', '—')} / specificity {det.get('specificity', '—')}",
                       help="Zones 3/4 used as a detector for salinity >1% on the covered points.")
            n_in = aoi_split.get("n_in")
            n_out = aoi_split.get("n_out")
            st.caption(
                f"{n_out or 'Some'} of the {v6_metrics.get('n') or 70} training soil profiles lie outside the 1960 sea boundary; "
                f"about {n_in or pv.get('v6_scored_nonwater', '—')} points inside the target Aral seabed are available for checking. "
                f"Full table: `{rel_path(V6_PIT_TABLE_PATH)}`."
            )

    # ── Pilot validation ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("V5.1 layer check against 11 field points")

    validation_summary = load_v5_validation_summary()
    uncertainty_summary = load_v5_uncertainty_summary()
    point_samples = load_v5_point_samples()

    if point_samples.empty or not validation_summary:
        st.warning("V5.1 validation files were not found.")
        st.info(
            "The V5.1 helper reports need to be rebuilt first. "
            "The technical commands are hidden below."
        )
        render_technical_commands([
            "python scripts/build_v5_science_dataset.py",
            "python scripts/v5_validation_report.py",
            "python scripts/v5_uncertainty_report.py",
        ])
    else:
        conflict_m = validation_summary.get("coordinate_conflict_median")
        col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
        col_v1.metric("Coordinate status", validation_summary.get("coordinate_policy", "n/a"))
        col_v2.metric(
            "Coordinate discrepancy",
            f"{conflict_m / 1000:.1f} km" if conflict_m is not None else "n/a",
        )
        col_v3.metric("Profiles with coordinates", f"{validation_summary.get('n_profiles_with_coordinates', 0):,}")
        col_v4.metric("Lab data only", f"{validation_summary.get('n_lab_only_profiles', 0):,}")
        col_v5.metric("Confirmed points", f"{validation_summary.get('n_authoritative_point_samples', 0):,}")

        st.caption(
            "Two sets of coordinates are currently kept: the original set and a shifted set. They are checked separately. "
            "Once the correct coordinates are chosen, they will appear as a separate group of confirmed points."
        )
        st.warning(
            "This checks the V5.1 helper layer. It is not a final accuracy assessment of V6. "
            "Because of the coordinate conflict, treat it as a preliminary diagnostic only."
        )

        st.markdown("### Which map classes the points fell into")
        class_dist = (
            point_samples.groupby(["coordinate_source", "class_filtered_name"])
            .size()
            .reset_index(name="count")
            .sort_values(["coordinate_source", "count"], ascending=[True, False])
        )
        class_dist.columns = ["Coordinate source", "Map class", "Count"]
        st.dataframe(class_dist, hide_index=True, width="stretch")

        with st.expander("For technical users: V5.1 correlations, charts, and sensitivity", expanded=False):
            corr_df = pd.DataFrame(validation_summary.get("correlations", []))
            if not corr_df.empty:
                corr_view = corr_df[
                    (corr_df["target"] == "top_salinity_pct")
                    & (corr_df["feature"].isin(["ndmi", "ndsi_green_swir2", "br_nir_swir2"]))
                ][["coordinate_source", "feature", "n", "spearman_r", "p_value", "bootstrap_ci95", "status"]].copy()
                corr_view.columns = [
                    "Coordinate source",
                    "Metric",
                    "n",
                    "Spearman r",
                    "p-value",
                    "Interval",
                    "Status",
                ]
                st.markdown("### Link between the map and topsoil salinity")
                st.caption(
                    "Spearman r shows whether two measures move in the same direction. "
                    "With only 11 points, this is a hint, not proof."
                )
                st.dataframe(corr_view, hide_index=True, width="stretch")

            plot_df = point_samples.copy()
            for col in ["top_salinity_pct", "ndmi", "br_nir_swir2", "field_ec_0_20"]:
                if col in plot_df.columns:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
            plot_df = plot_df.dropna(subset=["top_salinity_pct"])

            if not plot_df.empty:
                st.markdown("### Charts by point")
                fig_ndmi = px.scatter(
                    plot_df.dropna(subset=["ndmi"]),
                    x="ndmi",
                    y="top_salinity_pct",
                    color="coordinate_source",
                    symbol="class_filtered_name",
                    hover_data=["S_Point", "pit_code", "field_ec_0_20"],
                    labels={
                        "ndmi": "V5 NDMI",
                        "top_salinity_pct": "Topsoil salinity (%)",
                        "coordinate_source": "Coordinate source",
                        "class_filtered_name": "Map class",
                    },
                    title="NDMI vs. topsoil salinity",
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
                        "br_nir_swir2": "B8/B12 ratio",
                        "top_salinity_pct": "Topsoil salinity (%)",
                        "coordinate_source": "Coordinate source",
                        "class_filtered_name": "Map class",
                    },
                    title="B8/B12 vs. topsoil salinity",
                )
                fig_br.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_br, width="stretch")

            if uncertainty_summary:
                st.markdown("### How much the result depends on the thresholds")
                scenario_rows = []
                for scenario, rows in uncertainty_summary.get("class_area_by_scenario", {}).items():
                    candidate = next((row for row in rows if row.get("class") == 1), None)
                    if candidate is None:
                        continue
                    scenario_rows.append(
                        {
                            "Scenario": scenario,
                            "Candidate area, ha (approx.)": candidate.get("area_ha_approx"),
                            "Candidate area, km2 (approx.)": candidate.get("area_km2_approx"),
                            "Share of grid, %": candidate.get("pct_of_sample_grid"),
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
                                "Coordinate source": source,
                                "Stable points": stable,
                                "Total points": int(len(group)),
                                "Share stable, %": round(stable / len(group) * 100, 1) if len(group) else 0,
                            }
                        )
                if stability_rows:
                    st.dataframe(pd.DataFrame(stability_rows), hide_index=True, width="stretch")

        st.markdown("### Point-by-point validation table")
        display_cols = [
            "coordinate_source", "S_Point", "pit_code", "top_salinity_pct", "field_salinity_0_20",
            "field_ec_0_20", "class_filtered_name", "ndmi", "ndsi_green_swir2", "br_nir_swir2",
            "coordinate_conflict_m",
        ]
        display_df = point_samples[[col for col in display_cols if col in point_samples.columns]].copy()
        rename_map = {
            "coordinate_source": "Coordinate source",
            "S_Point": "Sample point",
            "pit_code": "Pit ID",
            "top_salinity_pct": "Topsoil salinity, %",
            "field_salinity_0_20": "Field salinity 0-20",
            "field_ec_0_20": "Field EC 0-20",
            "class_filtered_name": "Map class",
            "ndmi": "NDMI",
            "ndsi_green_swir2": "NDSI G/SWIR2",
            "br_nir_swir2": "B8/B12",
            "coordinate_conflict_m": "Coordinate conflict, m",
        }
        display_df = display_df.rename(columns=rename_map)
        st.dataframe(display_df, hide_index=True, width="stretch")

        st.caption(
            "This check is preliminary. The main limitation right now is choosing the correct coordinates for the 11 points."
        )
