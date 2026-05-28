"""Prepare V5 road layers, including Kazakhstan access roads from OSM.

The existing logistics workflow uses ``aralkum_roads.geojson`` for distance
calculations. This script refreshes a Kazakhstan-only access layer from OSM and
merges it into the common road layer while preserving a separate file for
Kazakhstan-specific distance metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString


sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
LOGISTICS = BASE / "outputs" / "logistics"
ROADS_OUT = LOGISTICS / "aralkum_roads.geojson"
KZ_ROADS_OUT = LOGISTICS / "kazakhstan_access_roads.geojson"
METADATA_OUT = LOGISTICS / "v5_roads_metadata.json"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Covers North/East Aralkum access through Kazakhstan, including Aralsk,
# Kazaly/Aiteke Bi, and the corridor toward Kyzylorda.
KZ_ACCESS_BBOX = (56.0, 43.0, 66.5, 47.8)  # min_lon, min_lat, max_lon, max_lat
TILE_STEP_DEG = 1.0
MAX_SPLIT_DEPTH = 2
HIGHWAY_FILTER = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|road|track|service|"
    "residential|living_street|primary_link|secondary_link|tertiary_link"
)


def overpass_query(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    return f"""
    [out:json][timeout:180];
    area["ISO3166-1"="KZ"][admin_level=2]->.kz;
    (
      way["highway"~"{HIGHWAY_FILTER}"]({min_lat},{min_lon},{max_lat},{max_lon})(area.kz);
    );
    out geom;
    """


def fetch_tile(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[dict[str, Any]]:
    query = overpass_query(min_lon, min_lat, max_lon, max_lat)
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=300,
        headers={"User-Agent": "AralSaxaulAI/1.1 (V5 roads prep)"},
    )
    response.raise_for_status()
    return response.json().get("elements", [])


def fetch_tile_with_split(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    depth: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    tile_label = f"[{min_lon:.3f},{min_lat:.3f},{max_lon:.3f},{max_lat:.3f}]"
    try:
        return fetch_tile(min_lon, min_lat, max_lon, max_lat), []
    except Exception as exc:
        if depth >= MAX_SPLIT_DEPTH:
            return [], [f"{tile_label}: {exc}"]

        mid_lon = (min_lon + max_lon) / 2
        mid_lat = (min_lat + max_lat) / 2
        elements: list[dict[str, Any]] = []
        failures: list[str] = []
        for sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat in [
            (min_lon, min_lat, mid_lon, mid_lat),
            (mid_lon, min_lat, max_lon, mid_lat),
            (min_lon, mid_lat, mid_lon, max_lat),
            (mid_lon, mid_lat, max_lon, max_lat),
        ]:
            sub_elements, sub_failures = fetch_tile_with_split(
                sub_min_lon,
                sub_min_lat,
                sub_max_lon,
                sub_max_lat,
                depth + 1,
            )
            elements.extend(sub_elements)
            failures.extend(sub_failures)
        return elements, failures


def element_to_feature(element: dict[str, Any]) -> dict[str, Any] | None:
    if element.get("type") != "way":
        return None
    points = element.get("geometry") or []
    if len(points) < 2:
        return None
    line = LineString((point["lon"], point["lat"]) for point in points)
    if line.is_empty:
        return None
    tags = element.get("tags", {})
    return {
        "osm_id": element.get("id"),
        "fclass": tags.get("highway", "unknown"),
        "name": tags.get("name", ""),
        "ref": tags.get("ref", ""),
        "surface": tags.get("surface", ""),
        "tracktype": tags.get("tracktype", ""),
        "access_scope": "kazakhstan_osm",
        "geometry": line,
    }


def download_kazakhstan_roads() -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = KZ_ACCESS_BBOX
    x_steps = int((max_lon - min_lon) / TILE_STEP_DEG) + 1
    y_steps = int((max_lat - min_lat) / TILE_STEP_DEG) + 1
    total_tiles = x_steps * y_steps

    features: list[dict[str, Any]] = []
    failed_tiles: list[str] = []

    print("Downloading Kazakhstan OSM access roads ...", flush=True)
    for i in range(x_steps):
        for j in range(y_steps):
            tile_min_lon = min_lon + i * TILE_STEP_DEG
            tile_max_lon = min(tile_min_lon + TILE_STEP_DEG, max_lon)
            tile_min_lat = min_lat + j * TILE_STEP_DEG
            tile_max_lat = min(tile_min_lat + TILE_STEP_DEG, max_lat)
            tile_id = f"{i + 1}.{j + 1}"
            print(
                f"  Tile {tile_id}/{x_steps}.{y_steps}: "
                f"[{tile_min_lon:.1f},{tile_min_lat:.1f}] -> [{tile_max_lon:.1f},{tile_max_lat:.1f}] ...",
                end=" ",
                flush=True,
            )
            elements, failures = fetch_tile_with_split(tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat)
            failed_tiles.extend(f"{tile_id} {failure}" for failure in failures)
            if failures:
                print(f"partial: {len(elements):,} segments, {len(failures)} failed subtiles", flush=True)
            elif elements:
                print(f"{len(elements):,} elements", end="; ", flush=True)

            tile_features = [feature for element in elements if (feature := element_to_feature(element)) is not None]
            features.extend(tile_features)
            if not failures:
                print(f"{len(tile_features):,} segments", flush=True)

    if not features:
        raise RuntimeError("No Kazakhstan OSM roads were downloaded.")

    roads = gpd.GeoDataFrame(features, crs="EPSG:4326")
    roads = roads.drop_duplicates(subset=["osm_id"]).copy()
    roads["geometry_wkb"] = roads.geometry.to_wkb().map(bytes.hex)
    roads = roads.drop_duplicates(subset=["geometry_wkb"]).drop(columns=["geometry_wkb"])
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads.attrs["failed_tiles"] = failed_tiles
    return roads


def load_existing_roads() -> gpd.GeoDataFrame:
    if not ROADS_OUT.exists():
        return gpd.GeoDataFrame(columns=["fclass", "geometry"], geometry="geometry", crs="EPSG:4326")
    roads = gpd.read_file(ROADS_OUT).to_crs("EPSG:4326")
    if "access_scope" not in roads.columns:
        roads["access_scope"] = "existing_osm"
    if "osm_id" not in roads.columns:
        roads["osm_id"] = None
    for col in ["name", "ref", "surface", "tracktype"]:
        if col not in roads.columns:
            roads[col] = ""
    return roads


def merge_roads(existing: gpd.GeoDataFrame, kazakhstan: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = ["osm_id", "fclass", "name", "ref", "surface", "tracktype", "access_scope", "geometry"]
    for df in [existing, kazakhstan]:
        for col in cols:
            if col not in df.columns:
                df[col] = None if col == "osm_id" else ""
    merged = pd.concat([existing[cols], kazakhstan[cols]], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged["geometry_wkb"] = merged.geometry.to_wkb().map(bytes.hex)
    merged = merged.drop_duplicates(subset=["geometry_wkb"]).drop(columns=["geometry_wkb"])
    return merged[merged.geometry.notna() & ~merged.geometry.is_empty].copy()


def main() -> None:
    t0 = time.time()
    LOGISTICS.mkdir(parents=True, exist_ok=True)

    kz_roads = download_kazakhstan_roads()
    existing_roads = load_existing_roads()
    merged_roads = merge_roads(existing_roads, kz_roads)

    kz_roads.to_file(KZ_ROADS_OUT, driver="GeoJSON")
    merged_roads.to_file(ROADS_OUT, driver="GeoJSON")

    metadata = {
        "version": "V5.1-roads",
        "source": "OpenStreetMap via Overpass API",
        "kazakhstan_access_bbox": KZ_ACCESS_BBOX,
        "highway_filter": HIGHWAY_FILTER,
        "kazakhstan_road_segments": int(len(kz_roads)),
        "merged_road_segments": int(len(merged_roads)),
        "failed_tiles": kz_roads.attrs.get("failed_tiles", []),
        "outputs": {
            "kazakhstan_access_roads": str(KZ_ROADS_OUT),
            "merged_roads": str(ROADS_OUT),
        },
    }
    METADATA_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nV5 roads prepared", flush=True)
    print(f"  Kazakhstan access roads: {len(kz_roads):,} -> {KZ_ROADS_OUT.name}", flush=True)
    print(f"  Merged road layer:       {len(merged_roads):,} -> {ROADS_OUT.name}", flush=True)
    print(f"  Metadata:                {METADATA_OUT.name}", flush=True)
    print(f"  Elapsed:                 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
