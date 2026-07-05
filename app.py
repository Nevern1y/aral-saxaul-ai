import json
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


def render_technical_commands(commands):
    with st.expander("For technical users"):
        for command in commands:
            st.code(command, language="bash")


st.title("Aral Saxaul: a salinity screening map for planting surveys")
st.markdown(
    '<p style="font-size:0.9rem; color:#6c757d;">'
    "This map points you to the ground worth checking in person. It estimates how salty the soil "
    "is likely to be and suggests a next step for any site you pick — it does not replace a soil "
    "sample, and it does not make planting decisions."
    "</p>",
    unsafe_allow_html=True,
)

tab_analytics, tab_dev, tab_logistics = st.tabs([
    "Map and summary",
    "How it was checked",
    "Plan a field trip",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: 📍 Map of work sites
# ══════════════════════════════════════════════════════════════════════

with tab_logistics:
    st.subheader("Plan a field trip")
    st.info(
        "The workflow: pick sites you can actually reach by road, download their KML files, "
        "then go check the soil and coordinates on the ground. Planting and budget planning "
        "come only after that."
    )
    tasks_df = load_tasks()
    roads_gdf = load_roads()
    screening_stats = load_screening_stats()

    if tasks_df.empty:
        st.warning("Planning data isn't loaded yet. The V6 map still works, but the KML task files can't be listed.")
        render_technical_commands([
            "python scripts/v6/build_v6_vectors.py",
            "python scripts/v6/v6_logistics_prep.py",
        ])
    else:
        if "territory_scope" in tasks_df.columns and set(tasks_df["territory_scope"].dropna()) == {"kazakhstan"}:
            st.caption("Everything below covers sites inside Kazakhstan only.")

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
                "Measure road access from:",
                options=list(access_options.keys()),
                index=1 if "Kazakhstan road access" in access_options else 0,
                help="For a first trip, measuring from the Kazakhstan road network usually makes more sense, when that layer is available.",
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
                help="Start close to roads — you can check the model against real ground without mounting a full expedition.",
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
                help="Small and medium sites make the easiest first checks — less risk, less driving, simpler sampling.",
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
            help="Total area of the cells passing your current filters. Until someone checks these sites on the ground, this number is not a planting plan.",
        )
        col_m2.metric("Cells selected", f"{len(filtered):,}")
        col_m3.metric("Total in index", f"{len(tasks_df):,}")
        col_m4.metric(
            "Share of index area",
            f"{selected_area_ha / total_task_area_ha * 100:.1f}%" if total_task_area_ha else "0%",
        )
        if screening_stats:
            full_aoi_ha = screening_stats.get("candidate_100m_area_ha", screening_stats.get("area_ha", 0))
            st.caption(
                f"For scale: across the whole study area, the screening estimates {full_aoi_ha:,.0f} ha of candidate zone. "
                "The filter above only touches the KML site index used for field trips."
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
            "Green squares pass your filters. The five pins mark the sites closest to a road — natural first stops."
        )
        _render_map(m.get_root().render())

        with st.expander("KML route files", expanded=True):
            st.caption(
                "These KML files open in a GPS unit, Google Earth, or QGIS. Start with the nearest sites — "
                "and hold off on any planting decision until the soil has been checked."
            )
            st.caption(f"KML route files are stored in `{rel_path(KML_TASKS_DIR)}`.")
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
                "KML file", "Latitude", "Longitude", "Area (ha)",
                *(["Low-salinity area (ha)"] if has_low_risk else []),
                "Distance to any road (km)",
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
                    "Download the 25 nearest sites (KML)",
                    data=kml_bytes,
                    file_name="aral_saxaul_field_tasks_top25.kml.zip",
                    mime="application/zip",
                    help="Sorted the same way as the table: by distance to the selected road.",
                )
            elif not filtered.empty:
                st.info(f"No KML files found in `{rel_path(KML_TASKS_DIR)}`.")

        st.warning(
            "The estimate below shows rough scale, nothing more. Confirm the sites in the field before treating it as a plan."
        )
        with st.expander("Rough resource estimate (for confirmed sites)", expanded=False):
            st.caption(
                "A back-of-the-envelope calculation of what planting this much area would take. "
                "It becomes meaningful only once the sites are confirmed on the ground."
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
        "Low salinity risk (V6)",
        f"{v6_low_salt_ha:,.0f} ha" if v6_low_salt_ha else "—",
        help="Ground the model flags as least salty. Candidates for a field check — not a ready planting area.",
    )
    col2.metric(
        "Scored by V6",
        f"{v6_coverage * 100:.0f}% AOI" if v6_coverage is not None else "—",
        help="How much of the study area the model could actually score.",
    )
    col3.metric(
        "Model validation",
        fmt_metric(v6_metrics.get("auc")),
        delta=f"CI {v6_ci[0]:.3f}-{v6_ci[1]:.3f}" if v6_ci else None,
        help="LOO AUC: how well the model separates saline soil profiles from non-saline ones.",
    )
    col4.metric(
        "KML site contours",
        f"{screening_stats.get('clusters', 0):,}",
        help="Ready-made site contours used to organize field trips (see the Plan a field trip tab).",
    )

    # ── Map in the most visible spot ───────────────────────────────────
    st.markdown("### V6 salinity risk map")
    if V6_MAP_PATH.exists():
        _render_map(V6_MAP_PATH.read_text(encoding="utf-8"))
        auc_txt = fmt_metric(v6_metrics["auc"])
        ci = v6_metrics.get("ci")
        ci_txt = f", 95% CI {ci[0]:.3f}-{ci[1]:.3f}" if ci else ""
        n_txt = v6_metrics.get("n") or 70
        st.info(
            "Hover over or click any colored site. The panel on the map gives a 0-100 score, "
            "the estimated salinity risk, and a practical next step. Green means \"check this one first\" — "
            "it is not an approval to plant."
        )
        with st.expander("Why this map is a screening tool, not a verdict"):
            st.markdown(
                f"V6 learned from {n_txt} soil profiles with lab-measured salt content, and salt is "
                f"the only thing it screens for (LOO AUC {auc_txt}{ci_txt}). It knows nothing about "
                "saxaul survival rates, groundwater depth, or what the ground looks like today. "
                "A field visit and a soil sample come before any final decision."
            )
    else:
        st.info(f"V6 map not found ({V6_MAP_PATH.name}).")
        render_technical_commands([
            "python scripts/v6/build_suitability_index.py",
            "python scripts/v6/render_v6_map.py",
        ])

    st.markdown("### Reading the map")
    action_rows = [
        {
            "If the map shows": "High score, low salinity risk",
            "What it means": "Move this site up the field-check list.",
            "Next step": "Check road access, open the KML file, take a soil sample.",
        },
        {
            "If the map shows": "Medium score",
            "What it means": "Not ruled out, but the salt risk is real.",
            "Next step": "Visit after the best sites, or keep it as a control point.",
        },
        {
            "If the map shows": "High salinity risk, or vegetation already growing",
            "What it means": "A weak candidate for new planting, unless something else makes it interesting.",
            "Next step": "Usually skip it, or survey it separately.",
        },
    ]
    st.dataframe(pd.DataFrame(action_rows), hide_index=True, width="stretch")

    if operational_area_ha:
        st.caption(
            f"The ready field-visit contours (10 ha and larger) add up to {operational_area_ha:,.0f} ha. "
            "Their KML files and road distances live in the Plan a field trip tab."
        )

    st.markdown("### The honest summary")
    conclusion_rows = [
        {
            "Question": "What this map shows",
            "Current answer": "Salinity risk, from a lab-verified link between a satellite moisture index and measured salt",
            "How to read it": "It tells you where to look. It doesn't promise that planting there will work.",
        },
        {
            "Question": "Where to start",
            "Current answer": f"{v6_low_salt_ha:,.0f} ha in the V6 low-salinity-risk zone" if v6_low_salt_ha else "V6 statistics are not available",
            "How to read it": "The top of the field-check list — not a planting area.",
        },
        {
            "Question": "Sites for field visits",
            "Current answer": f"{screening_stats.get('clusters', 0):,} ready site contours (10 ha and up) with KML files",
            "How to read it": "Cross-check against the map, then confirm with a soil check on the ground.",
        },
        {
            "Question": "Main limitation",
            "Current answer": "The model sees salt. It doesn't see saxaul survival or every local condition",
            "How to read it": "Compare sites within one district, and let the field sample have the last word.",
        },
    ]
    st.dataframe(pd.DataFrame(conclusion_rows), hide_index=True, width="stretch")

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
    st.subheader("How far to trust this map")
    st.info(
        "This tab shows how the map was checked and where its limits are. The headline result is the V6 "
        "salinity risk score; the field-trip task grid and its KML files are derived from the same V6 zones "
        "(candidate + moderate risk, Kazakhstan-clipped). A separate frozen 10 m pipeline only supplies the "
        "screening summary numbers."
    )
    safety_rows = [
        {
            "Good use": "Picking places for a field check",
            "Bad use": "Treating green zones as a finished planting plan",
            "Verify in the field": "Topsoil salt, road access, actual water and vegetation, coordinates",
        },
        {
            "Good use": "Comparing sites within one district",
            "Bad use": "Comparing distant districts head-to-head without calibration",
            "Verify in the field": "The local salinity background, plus fresh control samples",
        },
        {
            "Good use": "Driving a survey route from the KML file",
            "Bad use": "Reading the KML file as approval to plant",
            "Verify in the field": "Site boundaries, and whether the equipment and logistics actually fit",
        },
    ]
    st.dataframe(pd.DataFrame(safety_rows), hide_index=True, width="stretch")

    st.subheader("What would make the model better")

    roadmap_rows = [
        {
            "Priority": 1,
            "What to do": "Collect more field samples inside the seabed area",
            "What is needed": "The model trains on 70 profiles, but few check points fall inside the target Aral seabed contour",
            "Why": "Fresh field data inside the target area is the main limit on accuracy — not the interface.",
        },
        {
            "Priority": 2,
            "What to do": "Calibrate salinity levels across districts",
            "What is needed": "Reference samples in different blocks to build a shared scale (accuracy drops noticeably when distant districts are pooled; see \"Spatial validation\")",
            "Why": "Within one site the model ranks correctly, but the absolute level drifts between districts.",
        },
        {
            "Priority": 3,
            "What to do": "Match soil samples to the imagery date",
            "What is needed": "Fresh sampling taken close to the date of the satellite imagery (the lab data is from 2012-2014)",
            "Why": "Right now the satellite-to-salt link leans on data collected years apart; matching the dates would firm it up.",
        },
        {
            "Priority": 4,
            "What to do": "Independent check against the 2020/2021 field campaign",
            "What is needed": "Keep the 2020/2021 samples as a separate test set, never mixed into the 2012-2014 training data",
            "Why": "A clean, independent check with no leakage between data sets.",
        },
        {
            "Priority": 5,
            "What to do": "Link the score to planting survival",
            "What is needed": "Survival data from real saxaul plantings, site by site",
            "Why": "This is what would turn \"salinity risk\" into a verified forecast of planting success.",
        },
    ]
    with st.expander("Model and data improvement roadmap", expanded=False):
        st.dataframe(pd.DataFrame(roadmap_rows), hide_index=True, width="stretch")

    with st.expander("Where the KML site contours come from"):
        st.markdown(
            "The site contours and road distances in the **Plan a field trip** tab come from a separate "
            "10 m Sentinel-2 screening pipeline. It rules out water and shadow, steep terrain, existing "
            "vegetation, and ground that looks salty from orbit; whatever survives those filters is grouped "
            "into contours of 10 ha and up, matched to the OSM road network, and exported as KML files. "
            "Those contours organize the driving — the salinity call always stays with the V6 score and the soil sample."
        )

    # ── V6 lab-data science layer ──────────────────────────────────────
    v6 = load_v6_science()
    v6_metrics = v6_model_metrics(v6)
    if v6["salinity"]:
        st.markdown("---")
        st.subheader("V6 — the current salinity layer (70 soil profiles)")
        spatial = v6.get("spatial", {})
        sm = spatial.get("salinity_model", {})
        aoi_split = v6_aoi_split(v6)
        st.caption(
            "The main screening layer. It is trained on soil salinity actually measured in the lab — "
            "70 soil profiles with coordinates from the Pachikin-Kozybaeva report (2012-2014)."
        )
        v6_summary_rows = [
            {
                "Question": "What V6 checks",
                "Answer": "Salinity risk in the topsoil — not saxaul survival",
            },
            {
                "Question": "How to read the score",
                "Answer": "Higher score, lower expected salt risk. It sets field-visit priority, not a planting plan",
            },
            {
                "Question": "What it's built on",
                "Answer": f"{v6_metrics.get('n') or 70} soil profiles with lab-measured salt content",
            },
            {
                "Question": "Where to be careful",
                "Answer": "Don't compare distant districts head-to-head without local calibration",
            },
        ]
        st.dataframe(pd.DataFrame(v6_summary_rows), hide_index=True, width="stretch")

        st.markdown(
            "**In plain terms:** V6 hunts for ground with less salt. "
            "A high score means the salt risk looks low — it doesn't guarantee a planting will survive there."
        )
        with st.expander("For technical users: V6 metrics"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Profiles (training)",
                f"{v6_metrics.get('n') or '—'}",
                help="Soil profiles with lab-measured topsoil salinity.",
            )
            c2.metric("Saline (>1%)", f"{v6_metrics.get('n_saline') or '—'}")
            loo = v6_metrics.get("auc")
            ci = v6_metrics.get("ci")
            c3.metric(
                "AUC (LOO)",
                fmt_metric(loo),
                help="Leave-one-out AUC: how well the model separates saline points from non-saline ones.",
            )
            c4.metric(
                "95% interval",
                f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "—",
                help="Bootstrap confidence interval for the AUC. It stays above 0.5, so the relationship holds up.",
            )
            st.markdown(
                "The satellite moisture index (NDMI) tracks measured topsoil salinity "
                "(Spearman rho ≈ +0.66, p < 1e-9, n=70). The V6 score is simply "
                "`1 - probability of salinity`, used only to rank candidates."
            )
            af = spatial.get("independent_aralfield")
            if af:
                st.markdown(
                    f"**Independent check (AralField 2018, saxaul):** AUC {af.get('auc')}, "
                    f"n={af.get('n')} ({af.get('n_present')} with saxaul), "
                    f"interval {af.get('ci95')[0] if af.get('ci95') else '—'}-"
                    f"{af.get('ci95')[1] if af.get('ci95') else '—'}. "
                    "Too few points for a reliable estimate — read it as a directional signal, nothing more."
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
                    Nearby soil points tend to resemble each other, which can flatter a model's accuracy.
                    To rule that out, we cut the data into spatial blocks (~{sm.get('block_km', 20):.0f} km)
                    and retrained the model with each block held out in turn.

                    - **Average AUC per block: {pb}** — within each site, the model ranks
                      saline and non-saline points correctly.
                    - Pooled AUC across all blocks: {pooled} — lower, because **the baseline salinity
                      level differs between districts** (one block is almost entirely saline,
                      another almost entirely not). That's a cross-region calibration gap, **not
                      a loss of signal**.
                    - The moisture-to-salt relationship keeps its positive sign in **{sign}** of the tested blocks.

                    The takeaway: use the model to rank sites within a district. Putting distant
                    districts on one absolute scale needs extra calibration — a limit we'd rather
                    state here than hide.
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
            st.markdown("**V6 zones on the full-coverage 30 m layer (no gaps):**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            vf = stats.get("valid_fraction_of_aoi")
            st.caption(
                f"The 30 m layer scores {vf*100:.0f}% of the study area. "
                "Zones 3 and 4 aren't separate categories — they're steps on one validated salinity scale."
                if vf is not None else
                "Zones 3 and 4 aren't separate categories — they're steps on one validated salinity scale."
            )

        # ground-truth + independent validation
        pv = v6.get("pit_validation", {})
        if pv:
            det = pv.get("saline_detector_zone34", {})
            cc1, cc2 = st.columns(2)
            cc1.metric("Points covered by V6", f"{pv.get('v6_scored_nonwater', '—')}/70",
                       help="Ground-truth points (excluding water) that land inside scored zones.")
            sens_ci = det.get("sensitivity_ci95")
            spec_ci = det.get("specificity_ci95")
            spec_n = det.get("specificity_n")
            sens_n = det.get("sensitivity_n")
            cc2.metric("Salinity detector",
                       f"sensitivity {det.get('sensitivity', '—')} / specificity {det.get('specificity', '—')}",
                       help="How well zones 3/4 flag points with salinity above 1%. "
                            "Read with the sample sizes and 95% intervals below — a specificity of 1.0 "
                            "on a handful of negatives is not a validated perfect screen.")
            if spec_ci and sens_ci:
                st.caption(
                    f"Sensitivity {det.get('sensitivity','—')} on n={sens_n} (95% CI "
                    f"[{sens_ci[0]}, {sens_ci[1]}]); specificity {det.get('specificity','—')} on only "
                    f"n={spec_n} negatives (95% CI [{spec_ci[0]}, {spec_ci[1]}] — wide, small sample)."
                )
            n_in = aoi_split.get("n_in")
            n_out = aoi_split.get("n_out")
            st.caption(
                f"Of the {v6_metrics.get('n') or 70} training profiles, {n_out or 'some'} lie outside the 1960 sea boundary, "
                f"leaving about {n_in or pv.get('v6_scored_nonwater', '—')} points inside the target seabed for checking. "
                f"Full table: `{rel_path(V6_PIT_TABLE_PATH)}`."
            )

