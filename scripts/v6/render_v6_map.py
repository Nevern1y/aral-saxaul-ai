"""Phase 8b — render the V6 suitability layer as a self-contained Folium map.

Produces outputs/reports/suitability_map_v6.html with two togglable overlays
(base64-embedded PNGs, so the HTML works on Streamlit Cloud where the source
.tif files are gitignored):

  1. "V6 salinity risk zones" — suitability_zones_v6.tif in the SAME palette/legend
     codes as V5.1 (candidate / moderate-salt / strong-salt / vegetation), so the
     map reads like the existing product but is built on the lab-calibrated layer.
  2. "V6 score: higher = lower salinity risk (0..1)" — suitability_index_v6.tif as a green→red
     gradient (high score = less saline = greener), off by default.

Both rasters are EPSG:4326. They are downsampled to a light RGBA overlay and
Folium projects the arrays to Web Mercator for zoom-stable display; mode-resampling
preserves zone classes, average resampling is used for the continuous score.

This does NOT touch V5: the V5 map, thresholds, and logistics stay frozen. The
dashboard shows V6 as the primary map and keeps V5.1 as a 10 m detail layer.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

try:
    import folium
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from shapely.geometry import Polygon, box, mapping, shape
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"render_v6_map needs rasterio + folium + Pillow + shapely: {e}")

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
ODATA = BASE / "outputs" / "data"
AOI_DIR = BASE / "outputs" / "aoi"
REPORTS = BASE / "outputs" / "reports"

ZONES = ODATA / "suitability_zones_v6.tif"
INDEX = ODATA / "suitability_index_v6.tif"
AOI_BOUNDARY = AOI_DIR / "aral_sea_1960.geojson"
KZ_BOUNDARY = BASE / "outputs" / "logistics" / "kazakhstan_boundary.geojson"
HTML_PATH = REPORTS / "suitability_map_v6.html"
PNG_ZONES = REPORTS / "suitability_map_v6_zones.png"
PNG_INDEX = REPORTS / "suitability_map_v6_score.png"
PNG_LOOKUP = REPORTS / "suitability_map_v6_lookup.png"
BENCHMARK = CANON / "model_v6_benchmark.json"
SPATIAL = ODATA / "spatial_validation_v6.json"

NODATA_F = -9999.0
MAX_DIM = 2200

# Same palette as V5.1 (UX parity). V6 has no topo(5) class; 3/4 are salinity severity.
PALETTE = {
    0:  (0, 0, 0, 0),          # water / nodata -> transparent
    1:  (6, 95, 70, 220),      # candidate (low salinity)  #065F46
    3:  (253, 230, 138, 220),  # moderate salinity         #FDE68A
    4:  (153, 27, 27, 220),    # strong salinity           #991B1B
    10: (167, 243, 208, 220),  # vegetation                #A7F3D0
}
DISPLAY_NAMES_EN = {
    1: "Candidate zone (low salinity)",
    3: "Moderate salinity",
    4: "Strong salinity",
    10: "Existing vegetation",
    0: "Water / no data",
}


def downsampled(src, band, resampling, masked=False):
    """Read a band downsampled so width <= MAX_DIM, returning (array, bounds)."""
    scale = min(1.0, MAX_DIM / src.width)
    out_w = max(1, int(src.width * scale))
    out_h = max(1, int(src.height * scale))
    arr = src.read(band, out_shape=(out_h, out_w), resampling=resampling, masked=masked)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(src.nodata if src.nodata is not None else NODATA_F)
    b = src.bounds
    bounds = [[b.bottom, b.left], [b.top, b.right]]
    return arr, bounds, out_w, out_h


def majority_smooth(arr, size=3):
    """Categorical majority (mode) filter for DISPLAY only.

    The zoned raster flips class pixel-by-pixel wherever NDMI sits near a threshold,
    so the map fill looks like salt-and-pepper 'chopped' noise along the shore. A
    per-class neighbourhood vote reassigns each pixel to the locally dominant class,
    which reads as clean patches at map scale. This runs ONLY on the downsampled
    display array — the analytical suitability_zones_v6.tif stays pixel-exact, so
    the vector/logistics/pit-validation products are unaffected. NoData (0) also
    votes, so it correctly reclaims stray specks at the coastline edge.
    """
    from scipy import ndimage

    classes = list(PALETTE.keys())  # 0,1,3,4,10
    best = np.zeros_like(arr)
    best_count = np.full(arr.shape, -1.0)
    for c in classes:
        count = ndimage.uniform_filter((arr == c).astype(np.float32), size=size)
        upd = count > best_count
        best[upd] = c
        best_count[upd] = count[upd]
    return best


def zones_rgba(arr):
    img = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    for cls, rgba in PALETTE.items():
        img[arr == cls] = rgba
    return img


def score_rgba(arr, valid_mask=None):
    """Continuous score -> green(high)→yellow→red(low); NoData transparent."""
    img = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    valid = arr != NODATA_F
    if valid_mask is not None:
        valid &= valid_mask
    v = np.clip(arr, 0.0, 1.0)
    # simple 3-stop ramp: 0=red(153,27,27) -> 0.5=amber(253,230,138) -> 1=green(6,95,70)
    r = np.where(v < 0.5, 153 + (253 - 153) * (v / 0.5), 253 + (6 - 253) * ((v - 0.5) / 0.5))
    g = np.where(v < 0.5, 27 + (230 - 27) * (v / 0.5), 230 + (95 - 230) * ((v - 0.5) / 0.5))
    bl = np.where(v < 0.5, 27 + (138 - 27) * (v / 0.5), 138 + (70 - 138) * ((v - 0.5) / 0.5))
    img[..., 0] = np.where(valid, r, 0).astype(np.uint8)
    img[..., 1] = np.where(valid, g, 0).astype(np.uint8)
    img[..., 2] = np.where(valid, bl, 0).astype(np.uint8)
    img[..., 3] = np.where(valid, 210, 0).astype(np.uint8)
    return img


def interaction_lookup_rgba(score_arr, zone_arr):
    """Encode zone + score into a hidden PNG for browser-side hover lookup."""
    valid = (zone_arr != 0) & (score_arr != NODATA_F)
    score_byte = np.clip(np.rint(np.clip(score_arr, 0.0, 1.0) * 255), 0, 255).astype(np.uint8)
    img = np.zeros((zone_arr.shape[0], zone_arr.shape[1], 4), dtype=np.uint8)
    img[..., 0] = np.where(valid, zone_arr, 0).astype(np.uint8)
    img[..., 1] = np.where(valid, score_byte, 0)
    img[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return img


def image_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def project_rgba_for_leaflet(img: np.ndarray, bounds: list[list[float]]) -> np.ndarray:
    """Project a north-up geodetic RGBA raster to Web Mercator with crisp alpha edges."""
    south = max(bounds[0][0], -85.05112877980659)
    north = min(bounds[1][0], 85.05112877980659)
    height = img.shape[0]

    def mercator(lat):
        return np.arcsinh(np.tan(np.deg2rad(lat))) * 180.0 / np.pi

    input_lats = south + (np.arange(height) + 0.5) / height * (north - south)
    output_merc = mercator(south) + (np.arange(height) + 0.5) / height * (
        mercator(north) - mercator(south)
    )
    input_merc = mercator(input_lats)
    right = np.searchsorted(input_merc, output_merc, side="left")
    right = np.clip(right, 0, height - 1)
    left = np.clip(right - 1, 0, height - 1)
    choose_right = np.abs(input_merc[right] - output_merc) < np.abs(output_merc - input_merc[left])
    row_idx = np.where(choose_right, right, left)
    return img[::-1][row_idx][::-1]


def load_aoi_boundary_geojson(hole_area_min_deg2: float = 0.002, simplify_tol_deg: float = 0.004) -> dict | None:
    """Load the 1960 AOI polygon and lighten it for map display.

    The raster-derived contour is extremely ragged (~110k vertices, thousands of
    tiny rings), which draws as a shimmering "chopped" coastline. We drop small
    interior holes (< hole_area_min_deg2, i.e. vectorization/NDMI-dropout noise) and
    simplify the remainder hard (simplify_tol_deg ~0.004 deg ≈ 400 m). Combined with
    the upstream mask cleanup in build_aoi_mask.py this yields a thin, smooth outline
    (~2,000 vertices, ~10 real holes) instead of tens of thousands of jagged points.
    Tune down only if the coastline shape starts to look boxy at max zoom.
    """
    if not AOI_BOUNDARY.exists():
        return None
    data = json.loads(AOI_BOUNDARY.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    if not feats:
        return None
    geom = shape(feats[0]["geometry"])
    kept_holes = [h for h in geom.interiors if Polygon(h).area > hole_area_min_deg2]
    light = Polygon(geom.exterior, kept_holes).simplify(simplify_tol_deg, preserve_topology=True)
    return {
        "type": "Feature",
        "properties": {"name": "AOI boundary (1960 shoreline)"},
        "geometry": mapping(light),
    }


def load_kazakhstan_boundary_geojson(simplify_tol_deg: float = 0.01,
                                     clip_bounds: tuple | None = None) -> dict | None:
    """Load the Kazakhstan ADM0 boundary (geoBoundaries) for an optional overlay.

    The saxaul-restoration programme is a Kazakhstan effort, so field users can
    toggle a "Kazakhstan border" line to see which part of the mapped seabed lies
    on the Kazakh side (the southern Large-Aral seabed is on the Uzbek side). This
    is a reference line only — it never alters the salinity layer. The polygon is
    simplified and optionally cropped to the map's bounds so the embedded HTML stays
    light (the raw KAZ border spans to 87 E).
    """
    if not KZ_BOUNDARY.exists():
        return None
    data = json.loads(KZ_BOUNDARY.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    if not feats:
        return None
    geom = shape(feats[0]["geometry"])
    if clip_bounds is not None:
        west, south, east, north = clip_bounds
        geom = geom.intersection(box(west, south, east, north))
    if geom.is_empty:
        return None
    geom = geom.simplify(simplify_tol_deg, preserve_topology=True)
    return {
        "type": "Feature",
        "properties": {"name": "Kazakhstan border"},
        "geometry": mapping(geom),
    }


def load_map_metrics() -> dict:
    """Read the current shipped-model metrics for map text; never hand-type them."""
    metrics = {"auc": None, "ci": None, "n": None}
    if BENCHMARK.exists():
        bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        rec = bench.get("recommendation", {})
        metrics.update({
            "auc": rec.get("baseline_loo_auc"),
            "ci": rec.get("baseline_loo_auc_ci95"),
            "n": bench.get("n_total"),
        })
    if SPATIAL.exists() and (metrics["auc"] is None or metrics["ci"] is None):
        sm = json.loads(SPATIAL.read_text(encoding="utf-8")).get("salinity_model", {})
        metrics["auc"] = metrics["auc"] if metrics["auc"] is not None else sm.get("loo_auc")
        metrics["ci"] = metrics["ci"] if metrics["ci"] is not None else sm.get("loo_auc_ci95")
        metrics["n"] = metrics["n"] if metrics["n"] is not None else sm.get("n")
    metrics["auc"] = float(metrics["auc"]) if metrics["auc"] is not None else None
    return metrics


def metric_note(metrics: dict) -> str:
    auc = metrics.get("auc")
    ci = metrics.get("ci")
    n = metrics.get("n")
    auc_txt = f"LOO AUC {auc:.3f}" if auc is not None else "LOO AUC n/a"
    if ci:
        auc_txt += f", CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    n_txt = f"{n} lab soil profiles" if n else "lab soil profiles"
    return f"30 m · salinity model from {n_txt} ({auc_txt})"


def decision_help_html(metrics: dict) -> str:
    return f"""
<style>
#decision-help, #v6-map-legend {{
    background: rgba(255, 255, 255, 0.94);
    backdrop-filter: blur(2px);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.18);
    font-family: 'Segoe UI', Arial, sans-serif;
}}
@media (max-width: 520px) {{
  #decision-help {{
    top: 8px !important;
    left: 8px !important;
    right: 8px !important;
    width: auto !important;
    max-height: 30vh;
    overflow-y: auto;
    padding: 8px 10px !important;
    font-size: 11px !important;
  }}
  #v6-map-legend {{
    left: 8px !important;
    bottom: 8px !important;
    max-width: calc(100vw - 16px) !important;
    max-height: 28vh;
    overflow-y: auto;
    padding: 7px 9px !important;
    font-size: 10.5px !important;
  }}
  .leaflet-control-layers {{
    max-width: calc(100vw - 16px);
    font-size: 11px;
  }}
  /* The help panel spans the top on phones — drop the zoom/layers controls below it. */
  .leaflet-top.leaflet-left {{
    top: 31vh;
  }}
}}
</style>
<div id="decision-help" style="position:fixed; top:12px; right:12px; z-index:9999;
            padding:10px 12px; font-size:12px; width:250px; line-height:1.35;">
    <div style="font-weight:700; font-size:12.5px; margin-bottom:3px; color:#0F172A;">What the selected spot shows</div>
    <div id="decision-main" style="color:#334155;">
        Hover over a colored area on the map, or click it.
    </div>
    <div id="decision-detail" style="margin-top:5px; color:#64748B; font-size:11px;">
        The map will translate the color into a plain result: salinity risk, a 0-100 score, and the next step for a field visit.
    </div>
    <div style="margin-top:7px; color:#94A3B8; font-size:10px; border-top:1px solid #E2E8F0; padding-top:5px;">
        {metric_note(metrics)}. Preliminary screening, not permission to plant.
    </div>
</div>
"""


def interaction_script(map_name: str, bounds: list[list[float]], lookup_uri: str) -> str:
    south, west = bounds[0]
    north, east = bounds[1]
    return f"""
<script>
(function() {{
  const panelMain = document.getElementById('decision-main');
  const panelDetail = document.getElementById('decision-detail');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
  const img = new Image();
  const bounds = {{south:{south:.12f}, west:{west:.12f}, north:{north:.12f}, east:{east:.12f}}};
  const zoneInfo = {{
    1: {{
      title: 'Worth considering for a field visit',
      label: 'Candidate zone: low salinity',
      advice: 'First check road access and take a field soil sample. If the sample confirms low salt, this site can be ranked higher in the work plan.',
      color: '#065F46'
    }},
    3: {{
      title: 'Needs caution',
      label: 'Moderate salinity risk',
      advice: 'Do not plan planting without a field soil sample. Suitable as a backup option or for a spot check.',
      color: '#B45309'
    }},
    4: {{
      title: 'High salinity risk',
      label: 'Strong salinity',
      advice: 'A poor candidate for saxaul planting without soil remediation or a special reason. Best skipped in a normal field-visit plan.',
      color: '#991B1B'
    }},
    10: {{
      title: 'Already has vegetation',
      label: 'Not a new-planting zone',
      advice: 'Vegetation is already visible here. Use this as a control or to survey current conditions, not as a primary new-planting site.',
      color: '#047857'
    }}
  }};

  function describeScore(score) {{
    if (score >= 0.66) return 'high score of low salinity risk';
    if (score >= 0.33) return 'medium score of low salinity risk';
    return 'low score of low salinity risk';
  }}

  function setEmpty() {{
    panelMain.innerHTML = 'No score at this point';
    panelDetail.innerHTML = 'Hover over a colored area inside the V6 layer. Transparent areas are not used for decisions.';
  }}

  function update(latlng) {{
    if (!img.complete || !img.naturalWidth) return;
    if (latlng.lat < bounds.south || latlng.lat > bounds.north ||
        latlng.lng < bounds.west || latlng.lng > bounds.east) {{
      setEmpty();
      return;
    }}
    const x = Math.floor((latlng.lng - bounds.west) / (bounds.east - bounds.west) * img.naturalWidth);
    const y = Math.floor((bounds.north - latlng.lat) / (bounds.north - bounds.south) * img.naturalHeight);
    if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) {{
      setEmpty();
      return;
    }}
    const p = ctx.getImageData(x, y, 1, 1).data;
    const zone = p[0];
    const score = p[1] / 255;
    const info = zoneInfo[zone];
    if (!info || p[3] === 0) {{
      setEmpty();
      return;
    }}
    const pct = Math.round(score * 100);
    const saline = Math.round((1 - score) * 100);
    const scoreText = describeScore(score);
    panelMain.innerHTML = '<div style="font-weight:700;color:' + info.color + ';">' + info.title + '</div>' +
      '<div style="margin-top:4px;">' + info.label + '</div>' +
      '<div style="margin-top:6px;"><b>Low-salinity-risk score:</b> ' + pct + ' out of 100 (' + scoreText + ')</div>' +
      '<div><b>Estimated salinity risk (&gt;1%):</b> ' + saline + ' out of 100, per the V6 model</div>' +
      '<div><b>Zone class:</b> ' + info.label + '</div>';
    panelDetail.innerHTML = '<b>What to do:</b> ' + info.advice +
      '<div style="margin-top:5px;color:#64748B;">Coordinates: ' + latlng.lat.toFixed(5) + ', ' + latlng.lng.toFixed(5) + '</div>';
  }}

  img.onload = function() {{
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
  }};
  img.src = '{lookup_uri}';

  function attach(attempt) {{
    const map = window['{map_name}'];
    if (!map || !window.L) {{
      if (attempt < 50) window.setTimeout(function() {{ attach(attempt + 1); }}, 100);
      return;
    }}
    map.on('mousemove', function(e) {{ update(e.latlng); }});
    map.on('click', function(e) {{ update(e.latlng); }});
    map.on('mouseout', setEmpty);
  }}
  attach(0);
}})();
</script>
"""


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not ZONES.exists():
        raise SystemExit(f"missing {ZONES} — run build_suitability_index.py first")
    metrics = load_map_metrics()

    print("rendering V6 zones overlay ...", flush=True)
    with rasterio.open(ZONES) as src:
        zarr, bounds, w, h = downsampled(src, 1, Resampling.mode, masked=True)
    # Display-only de-speckle: reassign each display pixel to its local majority class
    # so the shoreline reads as clean patches, not chopped salt-and-pepper. Applied once
    # here so the zone fill, the hover lookup, and the score valid-mask stay consistent
    # (what you see == what you hover). The analytical raster on disk is untouched.
    zarr = majority_smooth(zarr, size=3)
    zrgba = project_rgba_for_leaflet(zones_rgba(zarr), bounds)
    Image.fromarray(zrgba, "RGBA").save(PNG_ZONES)

    score_ok = INDEX.exists()
    if score_ok:
        print("rendering V6 score overlay ...", flush=True)
        with rasterio.open(INDEX) as src:
            iarr, _, _, _ = downsampled(src, 1, Resampling.average, masked=True)
        lookup_rgba = interaction_lookup_rgba(iarr, zarr)
        Image.fromarray(lookup_rgba, "RGBA").save(PNG_LOOKUP)
        srgba = project_rgba_for_leaflet(score_rgba(iarr, valid_mask=(zarr != 0)), bounds)
        Image.fromarray(srgba, "RGBA").save(PNG_INDEX)

    south, west = bounds[0]
    north, east = bounds[1]

    m = folium.Map(location=[45.0, 60.5], zoom_start=7, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite imagery",  # human label in the layers control (not the raw URL)
        control=False,
    ).add_to(m)
    map_name = m.get_name()

    folium.raster_layers.ImageOverlay(
        image=zrgba, bounds=bounds, opacity=0.80,
        name="V6 salinity risk zones", show=True, pixelated=True,
    ).add_to(m)

    if score_ok:
        folium.raster_layers.ImageOverlay(
            image=srgba, bounds=bounds, opacity=0.80,
            name="V6 score: higher = lower salinity risk (0..1)", show=False, pixelated=True,
        ).add_to(m)

    aoi_feature = load_aoi_boundary_geojson()
    if aoi_feature is not None:
        folium.GeoJson(
            aoi_feature,
            name="AOI boundary (1960 shoreline)",
            style_function=lambda f: {
                "color": "#38BDF8",
                "weight": 1.5,
                "fill": False,
                "opacity": 0.85,
                "dashArray": "4 3",
            },
            show=True,
        ).add_to(m)

    # Optional Kazakhstan-border reference line. The restoration programme is a
    # Kazakhstan effort; toggling this shows which mapped seabed is on the Kazakh
    # side (the southern Large-Aral seabed is Uzbek). Reference only — off by default,
    # never alters the salinity layer. Clipped to the map extent to keep the HTML light.
    kz_feature = load_kazakhstan_boundary_geojson(clip_bounds=(west, south, east, north))
    if kz_feature is not None:
        folium.GeoJson(
            kz_feature,
            name="Kazakhstan border (restoration programme area)",
            style_function=lambda f: {
                "color": "#F59E0B",
                "weight": 2.0,
                "fill": False,
                "opacity": 0.9,
                "dashArray": "6 4",
            },
            show=False,
        ).add_to(m)

    # Collapsed and top-left (under the zoom buttons): the expanded control used to
    # sit top-right, underneath the fixed decision panel, so the two overlapped.
    folium.LayerControl(collapsed=True, position="topleft").add_to(m)

    legend = """
<div id="v6-map-legend" style="position:fixed; bottom:24px; left:12px; z-index:9999;
            padding:8px 11px; font-size:11px; max-width:200px; line-height:1.45; color:#334155;">
    <b style="font-size:11.5px; color:#0F172A;">Legend</b><br>
"""
    for cls in (1, 3, 4, 10, 0):
        r, g, b, a = PALETTE[cls]
        sw = "background:#ffffff;border:1px solid #bbb;" if cls == 0 else f"background:rgb({r},{g},{b});"
        legend += (f'<span style="display:inline-block;width:10px;height:10px;'
                   f'{sw}border-radius:2px;margin-right:5px;"></span>'
                   f'{DISPLAY_NAMES_EN[cls]}<br>')
    legend += """
    <div style="height:8px; margin:7px 0 3px 0; border-radius:999px;
                background:linear-gradient(90deg,#991B1B 0%,#FDE68A 50%,#065F46 100%);
                border:1px solid #ddd;"></div>
    <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B;">
        <span>0 high salt risk</span><span>1 low risk</span>
    </div>
    <div style="margin-top:7px; padding-top:5px; border-top:1px solid #E2E8F0;
                font-size:9.5px; color:#94A3B8; line-height:1.35;">
        The zoned map is a conservative (high-sensitivity) salt screen: it flags
        moderate/strong salinity earlier than the continuous score, so a spot can be
        zoned &ldquo;risk&rdquo; while the 0&ndash;1 score is mid-range. Candidate = the
        strictest low-salt class.
    </div>
</div>
"""
    m.get_root().html.add_child(folium.Element(legend))
    m.get_root().html.add_child(folium.Element(decision_help_html(metrics)))
    if score_ok:
        m.get_root().html.add_child(
            folium.Element(interaction_script(map_name, bounds, image_data_uri(PNG_LOOKUP)))
        )
    m.save(str(HTML_PATH))
    html = HTML_PATH.read_text(encoding="utf-8")
    HTML_PATH.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")

    print(f"  PNG zones: {PNG_ZONES.name} ({w}x{h})")
    if score_ok:
        print(f"  PNG score: {PNG_INDEX.name}")
        print(f"  PNG lookup: {PNG_LOOKUP.name}")
    print(f"  HTML: {HTML_PATH.name} ({HTML_PATH.stat().st_size / 1e3:.0f} KB)")
    print(f"  bounds south={south:.4f} west={west:.4f} north={north:.4f} east={east:.4f}")
    print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
