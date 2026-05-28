"""Download Kazakhstan ADM0 boundary for V5 logistics clipping."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests


sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
LOGISTICS = BASE / "outputs" / "logistics"
BOUNDARY_OUT = LOGISTICS / "kazakhstan_boundary.geojson"
METADATA_OUT = LOGISTICS / "kazakhstan_boundary_metadata.json"

GEOB_API = "https://www.geoboundaries.org/api/current/gbOpen/KAZ/ADM0/"


def main() -> None:
    t0 = time.time()
    LOGISTICS.mkdir(parents=True, exist_ok=True)

    print("Downloading Kazakhstan ADM0 boundary ...", flush=True)
    metadata = requests.get(GEOB_API, timeout=60).json()
    geojson_url = metadata["gjDownloadURL"]
    response = requests.get(geojson_url, timeout=180)
    response.raise_for_status()

    raw_path = LOGISTICS / "kazakhstan_boundary_raw.geojson"
    raw_path.write_bytes(response.content)

    boundary = gpd.read_file(raw_path).to_crs("EPSG:4326")
    boundary = boundary[["geometry"]].dissolve().reset_index(drop=True)
    boundary["name"] = "Kazakhstan"
    boundary["source"] = "geoBoundaries gbOpen KAZ ADM0"
    boundary.to_file(BOUNDARY_OUT, driver="GeoJSON")

    out_metadata = {
        "version": "V5.1-kazakhstan-boundary",
        "source": "geoBoundaries gbOpen KAZ ADM0",
        "source_api": GEOB_API,
        "source_geojson": geojson_url,
        "boundary_id": metadata.get("boundaryID"),
        "boundary_year": metadata.get("boundaryYearRepresented"),
        "boundary_license": metadata.get("boundaryLicense"),
        "features": int(len(boundary)),
        "bounds": [float(value) for value in boundary.total_bounds],
        "outputs": {
            "boundary": str(BOUNDARY_OUT),
            "raw_boundary": str(raw_path),
        },
    }
    METADATA_OUT.write_text(json.dumps(out_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Kazakhstan boundary prepared", flush=True)
    print(f"  Boundary: {BOUNDARY_OUT}", flush=True)
    print(f"  Bounds:   {out_metadata['bounds']}", flush=True)
    print(f"  Elapsed:  {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
