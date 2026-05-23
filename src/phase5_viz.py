"""
Phase 5: Decision Boundaries & Interactive Visualisation.

Pipeline
--------
1. Threshold probability raster -> binary mask (prob > 0.85).
2. Morphological cleaning (opening/closing) -> remove salt-and-pepper noise.
3. Vectorise -> GeoDataFrame of suitable-zone polygons.
4. Compute statistics (area, cluster count, size distribution).
5. Build interactive Folium map with satellite basemaps + overlay + stats.

Classes
-------
MorphologicalFilter     - Binary-image cleaning (opening/closing/small-object removal).
RasterVectorizer        - Raster -> GeoDataFrame conversion via rasterio.features.shapes.
DecisionMapper          - Full Phase 5 orchestrator.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    generate_binary_structure,
    label,
)
from shapely.geometry import Polygon, shape

import folium
from folium.plugins import Fullscreen, LayerControl, MeasureControl, MousePosition

from src.config import config
from src.utils import logger


# ============================================================================
# MorphologicalFilter
# ============================================================================


class MorphologicalFilter:
    """Clean a binary suitability mask with morphological operations.

    Motivation
    ----------
    A pixel-level classifier may predict "suitable" for isolated 10x10 m
    patches inside a salt flat.  A forestry tractor cannot access a lone
    pixel - the minimum operational unit is ~1 ha (100x100 m).

    This filter:
        1. Opens (erodes then dilates) - removes isolated false positives.
        2. Closes (dilates then erodes) - fills small holes inside clusters.
        3. Removes connected components smaller than *min_cluster_px*.
    """

    def __init__(
        self,
        structure_radius: int = config.MORPH_RADIUS_PX,
        min_cluster_ha: float = config.MIN_CLUSTER_HA,
        pixel_area_m2: float = config.TARGET_SCALE_M**2,
    ) -> None:
        """
        Args:
            structure_radius: Radius (pixels) of the morphological kernel.
            min_cluster_ha: Minimum cluster area in hectares (1 ha = 10 000 m²).
            pixel_area_m2: Area of one pixel in m².
        """
        self.structure_radius = structure_radius
        self.min_cluster_px = max(1, int(min_cluster_ha * 10_000 / pixel_area_m2))

        logger.info(
            "MorphologicalFilter: kernel r=%d px, min_cluster=%d px (%.2f ha).",
            structure_radius,
            self.min_cluster_px,
            min_cluster_ha,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def apply(self, binary: np.ndarray) -> np.ndarray:
        """Apply the morphological cleaning pipeline.

        Args:
            binary: 2D boolean or uint8 array (1 = suitable, 0 = unsuitable).

        Returns:
            Cleaned boolean array of the same shape.
        """
        if binary.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {binary.shape}")

        binary_bool = binary.astype(bool)
        t0 = time.time()

        # Structuring element: cross (4-connectivity) + iterations = radius
        struct = generate_binary_structure(2, 1)

        # 1. Opening: erosion then dilation -> remove salt noise
        opened = binary_opening(
            binary_bool,
            structure=struct,
            iterations=self.structure_radius,
        )

        # 2. Closing: dilation then erosion -> fill pepper noise (holes)
        closed = binary_closing(
            opened,
            structure=struct,
            iterations=self.structure_radius,
        )

        # 3. Remove small connected components
        cleaned = self._remove_small_objects(closed)

        n_original = int(binary_bool.sum())
        n_cleaned = int(cleaned.sum())

        elapsed = time.time() - t0
        logger.info(
            "Morphology: %d -> %d suitable pixels (%.1f%% retained, %.2f s).",
            n_original,
            n_cleaned,
            100 * n_cleaned / max(1, n_original),
            elapsed,
        )

        return cleaned

    def _remove_small_objects(self, binary: np.ndarray) -> np.ndarray:
        """Remove connected components with area < min_cluster_px."""
        labeled, n_features = label(binary)

        if n_features == 0:
            return binary

        sizes = np.bincount(labeled.ravel())
        # sizes[0] = background count, skip

        mask_sizes = sizes >= self.min_cluster_px
        mask_sizes[0] = False  # background

        cleaned = mask_sizes[labeled]

        logger.debug(
            "Small-object removal: %d components -> %d kept (>=%d px).",
            n_features,
            mask_sizes.sum(),
            self.min_cluster_px,
        )

        return cleaned


# ============================================================================
# RasterVectorizer
# ============================================================================


class RasterVectorizer:
    """Convert a binary raster mask to a GeoDataFrame of polygons.

    Uses ``rasterio.features.shapes()`` - a generator-based method that
    does **not** load the full raster into memory at once.
    """

    def __init__(
        self,
        simplify_tolerance_m: float = config.VECTOR_SIMPLIFY_TOLERANCE_M,
        connectivity: int = 8,
    ) -> None:
        """
        Args:
            simplify_tolerance_m: Douglas–Peucker tolerance in metres for
                polygon simplification.  Lower = more vertices, larger file.
            connectivity: 4 or 8 - pixel connectivity for region grouping.
        """
        self.simplify_tolerance_m = simplify_tolerance_m
        self.connectivity = connectivity

        # Convert simplify tolerance from metres to degrees (approximate)
        # 1 deg ~ 111 320 m at the equator. At ~45N: 1 deg lon ~ 78 800 m, 1 deg lat ~ 111 000 m.
        self._simplify_tol_deg = simplify_tolerance_m / 100_000.0

        logger.info(
            "RasterVectorizer: connectivity=%d, simplify=%.1fm (~%.6f deg).",
            connectivity,
            simplify_tolerance_m,
            self._simplify_tol_deg,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def vectorize(
        self,
        binary: np.ndarray,
        transform: rasterio.Affine,
        crs: rasterio.crs.CRS,
    ) -> gpd.GeoDataFrame:
        """Convert a binary raster to a polygon GeoDataFrame.

        Args:
            binary: 2D uint8 array (1 = suitable, 0 = background).
            transform: Affine transform from the source raster.
            crs: CRS of the source raster.

        Returns:
            ``GeoDataFrame`` with geometry column and ``area_ha`` attribute.
        """
        if binary.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {binary.shape}")

        logger.info("Vectorising raster (%dx%d) ...", binary.shape[1], binary.shape[0])
        t0 = time.time()

        # rasterio.features.shapes yields (polygon_geojson, pixel_value) tuples
        shape_gen = features.shapes(
            binary.astype("uint8"),
            mask=binary.astype(bool),
            transform=transform,
            connectivity=self.connectivity,
        )

        geometries: List[Polygon] = []
        for geojson_geom, value in shape_gen:
            if value != 1:
                continue
            geom = shape(geojson_geom)

            # Skip invalid or empty geometries
            if geom.is_empty or not geom.is_valid:
                continue

            # Simplify to reduce output file size
            if self.simplify_tolerance_m > 0:
                geom = geom.simplify(
                    self._simplify_tol_deg, preserve_topology=True
                )

            if not geom.is_empty:
                geometries.append(geom)

        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)

        elapsed = time.time() - t0
        logger.info(
            "Vectorisation complete: %d polygons (%.1f s).",
            len(gdf),
            elapsed,
        )

        return gdf

    def save_geojson(self, gdf: gpd.GeoDataFrame, path: str, precision: int = 6) -> str:
        """Save GeoDataFrame as GeoJSON with controlled coordinate precision.

        Args:
            gdf: GeoDataFrame to save.
            path: Destination path.
            precision: Decimal places for coordinates.

        Returns:
            Path to the saved file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        gdf.to_file(
            path,
            driver="GeoJSON",
            coordinate_precision=precision,
        )

        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("GeoJSON saved: %s (%.2f MB).", path, size_mb)
        return path


# ============================================================================
# DecisionMapper (Orchestrator)
# ============================================================================


class DecisionMapper:
    """Turn a probability raster into an actionable suitability map.

    Outputs
    -------
    =============================  =========================================
    File                           Description
    =============================  =========================================
    ``outputs/suitable_zones.geojson``   Polygon vector of suitable zones.
    ``outputs/aral_saxaul_map.html``     Interactive Folium map.
    ``outputs/statistics.json``          Aggregated area statistics.
    =============================  =========================================
    """

    def __init__(
        self,
        prob_raster_path: str,
        output_dir: Optional[Path] = None,
        prob_threshold: float = config.PROB_THRESHOLD,
    ) -> None:
        """
        Args:
            prob_raster_path: Path to ``probability_map.tif`` (Phase 4 output).
            output_dir: Output directory (default: config.output_dir).
            prob_threshold: Probability above which a pixel is deemed suitable.
        """
        self.prob_raster_path = prob_raster_path
        self.output_dir = output_dir or config.output_dir
        self.prob_threshold = prob_threshold

        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DecisionMapper: raster='%s', threshold=%.2f.",
            prob_raster_path,
            prob_threshold,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def run(self) -> Tuple[str, str, Dict[str, Any]]:
        """Execute the full Phase 5 pipeline.

        Returns:
            ``(html_map_path, geojson_path, statistics_dict)``.
        """
        t0 = time.time()

        # 1. Load & threshold
        binary, profile = self._load_and_threshold()

        # 2. Morphological cleaning
        morph = MorphologicalFilter()
        cleaned = morph.apply(binary)

        # 3. Vectorise
        vec = RasterVectorizer()
        gdf = vec.vectorize(
            cleaned,
            transform=profile["transform"],
            crs=profile["crs"],
        )

        # 4. Compute areas
        gdf = self._compute_areas(gdf)

        # 5. Save GeoJSON
        geojson_path = str(self.output_dir / "suitable_zones.geojson")
        vec.save_geojson(gdf, geojson_path)

        # 6. Statistics
        stats = self._compute_statistics(binary, cleaned, gdf)
        stats_path = str(self.output_dir / "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Statistics saved: %s", stats_path)

        # 7. Interactive map
        map_path = self._create_folium_map(gdf)

        elapsed = time.time() - t0
        logger.info("Phase 5 complete (%.1f s).", elapsed)

        return map_path, geojson_path, stats

    # ── Internal helpers ───────────────────────────────────────────────

    def _load_and_threshold(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load the probability raster and apply the decision threshold."""
        logger.info("Loading probability raster …")

        with rasterio.open(self.prob_raster_path) as src:
            profile = src.profile.copy()
            data = src.read(1)

        # Replace nodata with 0 (not suitable)
        nodata = profile.get("nodata")
        if nodata is not None:
            data = np.where(np.isnan(data) | (data == nodata), 0.0, data)

        binary = (data > self.prob_threshold).astype("uint8")
        suitable_pct = 100 * binary.sum() / binary.size

        logger.info(
            "Threshold %.2f -> %.2f%% of pixels suitable.",
            self.prob_threshold,
            suitable_pct,
        )

        return binary, profile

    def _compute_areas(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add an ``area_ha`` column computed in a metric CRS (UTM zone 41N for Aral)."""
        utm_crs = "EPSG:32641"

        if len(gdf) == 0:
            gdf["area_ha"] = []
            return gdf

        # Project to UTM for accurate area calculation
        gdf_utm = gdf.to_crs(utm_crs)
        gdf["area_ha"] = gdf_utm.geometry.area / 10_000.0  # m^2 -> hectares

        logger.info(
            "Area stats: total=%.0f ha, mean=%.1f ha, median=%.1f ha.",
            gdf["area_ha"].sum(),
            gdf["area_ha"].mean(),
            gdf["area_ha"].median(),
        )

        return gdf

    def _compute_statistics(
        self,
        binary_raw: np.ndarray,
        binary_cleaned: np.ndarray,
        gdf: gpd.GeoDataFrame,
    ) -> Dict[str, Any]:
        """Compute comprehensive statistics for the report."""
        pms = config.TARGET_SCALE_M**2  # pixel area in m²

        total_pixels = binary_raw.size
        raw_suitable_px = int(binary_raw.sum())
        cleaned_suitable_px = int(binary_cleaned.sum())

        raw_ha = raw_suitable_px * pms / 10_000
        cleaned_ha = cleaned_suitable_px * pms / 10_000

        if len(gdf) > 0 and "area_ha" in gdf.columns:
            areas = gdf["area_ha"].values
            total_gdf_ha = float(areas.sum())

            # Size bins
            bins = {
                "<1 ha": int((areas < 1).sum()),
                "1–10 ha": int(((areas >= 1) & (areas < 10)).sum()),
                "10–100 ha": int(((areas >= 10) & (areas < 100)).sum()),
                "100–1000 ha": int(((areas >= 100) & (areas < 1000)).sum()),
                ">1000 ha": int((areas >= 1000).sum()),
            }
        else:
            areas = np.array([])
            total_gdf_ha = 0.0
            bins = {}

        stats = {
            "threshold": self.prob_threshold,
            "pixels_total": total_pixels,
            "raw_suitable_pixels": raw_suitable_px,
            "raw_suitable_ha": round(raw_ha, 2),
            "cleaned_suitable_pixels": cleaned_suitable_px,
            "cleaned_suitable_ha": round(cleaned_ha, 2),
            "retained_pct": round(100 * cleaned_suitable_px / max(1, raw_suitable_px), 2),
            "n_clusters": len(gdf),
            "total_suitable_ha": round(total_gdf_ha, 2),
            "mean_cluster_ha": round(float(areas.mean()), 2) if len(areas) > 0 else 0.0,
            "median_cluster_ha": round(float(np.median(areas)), 2) if len(areas) > 0 else 0.0,
            "largest_cluster_ha": round(float(areas.max()), 2) if len(areas) > 0 else 0.0,
            "clusters_by_size_bins": bins,
            "suitable_pct_of_aoi": round(100 * cleaned_suitable_px / max(1, total_pixels), 2),
        }

        logger.info("=== Suitability Statistics ===")
        logger.info("  Total suitable:  %.0f ha", stats["total_suitable_ha"])
        logger.info("  Number of zones: %d", stats["n_clusters"])
        logger.info("  Largest zone:    %.0f ha", stats["largest_cluster_ha"])

        return stats

    # ── Folium map ─────────────────────────────────────────────────────

    def _create_folium_map(self, gdf: gpd.GeoDataFrame) -> str:
        """Build an interactive Folium map with satellite basemaps and overlay.

        Args:
            gdf: GeoDataFrame of suitable-zone polygons (WGS84).

        Returns:
            Path to the saved HTML map.
        """
        logger.info("Building Folium map …")

        center_lat, center_lon = config.MAP_CENTER

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=config.MAP_DEFAULT_ZOOM,
            control_scale=True,
            tiles=None,  # custom tile layers below
            prefer_canvas=True,
            max_zoom=18,
        )

        # ── Satellite basemaps ────────────────────────────────────
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Google Satellite",
            overlay=False,
            max_zoom=22,
        ).add_to(m)

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics",
            name="ESRI World Imagery",
            overlay=False,
            max_zoom=19,
        ).add_to(m)

        # ── Suitable zones overlay ────────────────────────────────
        if len(gdf) > 0:
            folium.GeoJson(
                gdf.__geo_interface__,
                name=f"Suitable Zones (prob > {self.prob_threshold:.0%})",
                style_function=lambda x: {
                    "fillColor": "#2ecc40",
                    "color": "#1a7a28",
                    "weight": 1.2,
                    "fillOpacity": 0.45,
                },
                highlight_function=lambda x: {
                    "fillColor": "#00ff00",
                    "color": "#00aa00",
                    "weight": 2.5,
                    "fillOpacity": 0.7,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["area_ha"],
                    aliases=["Area (ha):"],
                    labels=True,
                    localize=True,
                ),
                popup=folium.GeoJsonPopup(
                    fields=["area_ha"],
                    aliases=["Area (hectares):"],
                    labels=True,
                ),
            ).add_to(m)

        # ── Statistics widget ─────────────────────────────────────
        stats_html = self._build_stats_widget(gdf)
        m.get_root().html.add_child(folium.Element(stats_html))

        # ── Plugins ───────────────────────────────────────────────
        Fullscreen(
            title="Full screen",
            title_cancel="Exit full screen",
        ).add_to(m)

        MeasureControl(
            position="topleft",
            primary_length_unit="kilometers",
            secondary_length_unit="miles",
            primary_area_unit="hectares",
            secondary_area_unit="acres",
        ).add_to(m)

        formatter = "function(num) {return L.Util.formatNum(num, 4) + ' &deg;';};"
        MousePosition(
            position="bottomright",
            separator=" | ",
            empty_string="NaN",
            lng_first=False,
            num_digits=20,
            prefix="Координаты:",
            lat_formatter=formatter,
            lng_formatter=formatter,
        ).add_to(m)

        LayerControl(position="topright", collapsed=False).add_to(m)

        # ── Save ──────────────────────────────────────────────────
        html_path = str(self.output_dir / "aral_saxaul_map.html")
        m.save(html_path)

        size_mb = os.path.getsize(html_path) / (1024 * 1024)
        logger.info("Folium map saved: %s (%.2f MB).", html_path, size_mb)

        return html_path

    def _build_stats_widget(self, gdf: gpd.GeoDataFrame) -> str:
        """Build an HTML statistics panel for the Folium map."""
        total_ha = gdf["area_ha"].sum() if len(gdf) > 0 and "area_ha" in gdf.columns else 0
        n_zones = len(gdf)

        html = textwrap.dedent(f"""\
        <div style="
            position: fixed;
            top: 10px;
            right: 10px;
            width: 300px;
            max-height: 80vh;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.85);
            color: #ecf0f1;
            border-radius: 8px;
            padding: 16px;
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            font-size: 13px;
            z-index: 9999;
            box-shadow: 0 2px 12px rgba(0,0,0,0.5);
            line-height: 1.5;
        ">
            <h3 style="margin:0 0 10px;font-size:16px;color:#2ecc40;">
                Aral Saxaul &#8226; Suitability Map
            </h3>
            <table style="width:100%;border-collapse:collapse;">
                <tr>
                    <td style="color:#95a5a6;">Total suitable</td>
                    <td style="text-align:right;font-weight:bold;color:#2ecc40;">
                        {total_ha:,.0f} ha
                    </td>
                </tr>
                <tr>
                    <td style="color:#95a5a6;">Suitable zones</td>
                    <td style="text-align:right;font-weight:bold;color:#2ecc40;">
                        {n_zones:,}
                    </td>
                </tr>
                <tr>
                    <td style="color:#95a5a6;">Probability threshold</td>
                    <td style="text-align:right;font-weight:bold;color:#2ecc40;">
                        &gt;{self.prob_threshold:.0%}
                    </td>
                </tr>
            </table>
            <hr style="border-color:#444;margin:10px 0;">
            <p style="font-size:11px;color:#7f8c8d;margin:0;">
                Model: XGBoost Phase 3 |
                Features: NDMI, MSAVI, SI, Slope, TWI, VH, NDWI<br>
                Data: Sentinel-2 + Sentinel-1 + Copernicus DEM<br>
                Season: Aug–Sep 2025 (dry season)
            </p>
        </div>
        """)

        return html


# ============================================================================
# Phase 5 entrypoint
# ============================================================================


def run_phase5(
    prob_raster_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    prob_threshold: float = config.PROB_THRESHOLD,
) -> Tuple[str, str, Dict[str, Any]]:
    """Convenience function: full Phase 5 pipeline.

    Args:
        prob_raster_path: Path to probability_map.tif.
        output_dir: Output directory.
        prob_threshold: Decision threshold.

    Returns:
        ``(html_map_path, geojson_path, stats_dict)``.
    """
    prob_raster_path = prob_raster_path or str(config.output_dir / "probability_map.tif")

    if not os.path.exists(prob_raster_path):
        raise FileNotFoundError(
            f"Probability raster not found: {prob_raster_path}. Run Phase 4 first."
        )

    mapper = DecisionMapper(
        prob_raster_path=prob_raster_path,
        output_dir=output_dir,
        prob_threshold=prob_threshold,
    )

    return mapper.run()


# ============================================================================
# CLI entrypoint
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5: Decision Boundaries & Visualisation"
    )
    parser.add_argument(
        "--raster", type=str, default=None,
        help="Path to probability_map.tif.",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.PROB_THRESHOLD,
        help="Probability threshold (default: 0.85).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    map_path, geojson_path, stats = run_phase5(
        prob_raster_path=args.raster,
        output_dir=output_dir,
        prob_threshold=args.threshold,
    )

    logger.info("Phase 5 complete.")
    logger.info("  Map:     %s", map_path)
    logger.info("  GeoJSON: %s", geojson_path)
    print(json.dumps(stats, indent=2))
