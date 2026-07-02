"""Phase 8b — render the V6 suitability layer as a self-contained Folium map.

Produces outputs/reports/suitability_map_v6.html with two togglable overlays
(base64-embedded PNGs, so the HTML works on Streamlit Cloud where the source
.tif files are gitignored):

  1. "Зоны риска соли V6" — suitability_zones_v6.tif in the SAME palette/legend
     codes as V5.1 (candidate / moderate-salt / strong-salt / vegetation), so the
     map reads like the existing product but is built on the lab-calibrated layer.
  2. "Балл V6: ниже риск соли (0..1)" — suitability_index_v6.tif as a green→red
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
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"render_v6_map needs rasterio + folium + Pillow: {e}")

BASE = Path(__file__).resolve().parent.parent.parent
CANON = BASE / "data" / "canonical"
ODATA = BASE / "outputs" / "data"
REPORTS = BASE / "outputs" / "reports"

ZONES = ODATA / "suitability_zones_v6.tif"
INDEX = ODATA / "suitability_index_v6.tif"
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
DISPLAY_NAMES_RU = {
    1: "Кандидат (низкое засоление)",
    3: "Умеренное засоление",
    4: "Сильное засоление",
    10: "Есть растительность",
    0: "Вода / нет данных",
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
    n_txt = f"{n} лаб. профилям" if n else "лабораторным профилям"
    return f"30 м · модель солёности по {n_txt} ({auc_txt})"


def decision_help_html(metrics: dict) -> str:
    return f"""
<style>
@media (max-width: 520px) {{
  #decision-help {{
    top: 8px !important;
    left: 8px !important;
    right: 8px !important;
    width: auto !important;
    max-height: 34vh;
    overflow-y: auto;
    padding: 9px 10px !important;
    font-size: 12px !important;
  }}
  #v6-map-legend {{
    left: 8px !important;
    bottom: 8px !important;
    max-width: calc(100vw - 16px) !important;
    max-height: 30vh;
    overflow-y: auto;
    padding: 8px 10px !important;
    font-size: 11px !important;
  }}
  .leaflet-control-layers {{
    max-width: calc(100vw - 16px);
    font-size: 11px;
  }}
}}
</style>
<div id="decision-help" style="position:fixed; top:14px; right:14px; z-index:9999;
            background:white; padding:12px 14px; border-radius:10px;
            box-shadow:0 0 14px rgba(0,0,0,0.20); font-size:13px;
            font-family:'Segoe UI',Arial,sans-serif; width:310px; line-height:1.35;">
    <div style="font-weight:700; font-size:14px; margin-bottom:4px;">Что показывает выбранное место</div>
    <div id="decision-main" style="color:#334155;">
        Наведите курсор на цветной участок карты или нажмите на него.
    </div>
    <div id="decision-detail" style="margin-top:6px; color:#64748B; font-size:12px;">
        Карта переведёт цвет в понятный вывод: риск соли, балл 0–100 и следующий шаг для выезда.
    </div>
    <div style="margin-top:8px; color:#64748B; font-size:11px; border-top:1px solid #E2E8F0; padding-top:6px;">
        {metric_note(metrics)}. Это предварительный отбор, не разрешение на посадку.
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
      title: 'Можно рассматривать для выезда',
      label: 'Кандидатная зона: низкая солёность',
      advice: 'Сначала проверьте доступ к дороге и сделайте полевую пробу почвы. Если проба подтвердит низкую соль, участок можно ставить выше в план работ.',
      color: '#065F46'
    }},
    3: {{
      title: 'Нужна осторожность',
      label: 'Средний риск соли',
      advice: 'Не планируйте посадку без полевой пробы. Подходит как запасной вариант или для точечной проверки.',
      color: '#B45309'
    }},
    4: {{
      title: 'Высокий риск засоления',
      label: 'Сильное засоление',
      advice: 'Для посадки саксаула это плохой кандидат без мелиорации или специальных причин. В обычном плане выезда лучше пропустить.',
      color: '#991B1B'
    }},
    10: {{
      title: 'Уже есть растительность',
      label: 'Не зона новой посадки',
      advice: 'Здесь уже видна растительность. Используйте как контроль или для обследования текущего состояния, а не как основной участок новой посадки.',
      color: '#047857'
    }}
  }};

  function describeScore(score) {{
    if (score >= 0.66) return 'высокий балл низкого риска соли';
    if (score >= 0.33) return 'средний балл низкого риска соли';
    return 'низкий балл низкого риска соли';
  }}

  function setEmpty() {{
    panelMain.innerHTML = 'В этой точке нет оценки';
    panelDetail.innerHTML = 'Наведите на цветной участок внутри слоя V6. Прозрачные места не используются для решения.';
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
      '<div style="margin-top:6px;"><b>Балл низкого риска соли:</b> ' + pct + ' из 100 (' + scoreText + ')</div>' +
      '<div><b>Примерный риск соли (&gt;1%):</b> ' + saline + ' из 100 по модели V6</div>' +
      '<div><b>Класс зоны:</b> ' + info.label + '</div>';
    panelDetail.innerHTML = '<b>Что делать:</b> ' + info.advice +
      '<div style="margin-top:5px;color:#64748B;">Координаты: ' + latlng.lat.toFixed(5) + ', ' + latlng.lng.toFixed(5) + '</div>';
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

    m = folium.Map(location=[45.0, 60.5], zoom_start=7,
                   tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                   attr="Google Satellite", control_scale=True)
    map_name = m.get_name()

    folium.raster_layers.ImageOverlay(
        image=zrgba, bounds=bounds, opacity=0.80,
        name="Зоны риска соли V6", show=True, pixelated=True,
    ).add_to(m)

    if score_ok:
        folium.raster_layers.ImageOverlay(
            image=srgba, bounds=bounds, opacity=0.80,
            name="Балл V6: ниже риск соли (0..1)", show=False, pixelated=True,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend = """
<div id="v6-map-legend" style="position:fixed; bottom:20px; left:20px; z-index:9999;
            background:white; padding:10px 14px; border-radius:8px;
            box-shadow:0 0 10px rgba(0,0,0,0.15); font-size:13px;
            font-family:'Segoe UI',Arial,sans-serif; max-width:270px;">
    <b style="font-size:14px;">Как читать карту</b><br>
"""
    for cls in (1, 3, 4, 10, 0):
        r, g, b, a = PALETTE[cls]
        sw = "background:#ffffff;border:1px solid #bbb;" if cls == 0 else f"background:rgb({r},{g},{b});"
        legend += (f'<span style="display:inline-block;width:12px;height:12px;'
                   f'{sw}border-radius:2px;margin-right:6px;"></span>'
                   f'{DISPLAY_NAMES_RU[cls]}<br>')
    legend += """
    <hr style="margin:6px 0;">
    <b style="font-size:12px;">Наведите на карту</b>
    <div style="margin-top:4px; color:#555; font-size:12px; line-height:1.3;">
        Справа появится вывод по выбранной точке: балл 0–100, риск соли и следующий шаг.
    </div>
    <div style="height:10px; margin:5px 0 3px 0; border-radius:999px;
                background:linear-gradient(90deg,#991B1B 0%,#FDE68A 50%,#065F46 100%);
                border:1px solid #ddd;"></div>
    <div style="display:flex; justify-content:space-between; font-size:11px; color:#444;">
        <span>0.0 высокий риск соли</span><span>1.0 низкий риск</span>
    </div>
    <hr style="margin:6px 0;">
    <span style="color:#666;font-size:11px;">__METRIC_NOTE__<br>
    Зоны 3/4 — сила засоления по NDMI. Решение всегда подтверждается полевой пробой.</span>
</div>
"""
    legend = legend.replace("__METRIC_NOTE__", metric_note(metrics))
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
