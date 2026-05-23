"""
Wait for GEE batch export to complete and download the result.
Usage: python -m scripts.wait_and_download
"""

import time, os, requests
import ee
from src.utils import initialize_gee, logger
from src.config import config

initialize_gee(project="tribal-dispatch-494405-u4")

TASK_DESC = "aral_saxaul_fs_30m_mosaic"
OUT_PATH = str(config.output_dir / "data" / "feature_stack_30m.tif")

while True:
    tasks = ee.data.listOperations()
    for t in tasks:
        meta = t.get("metadata", {})
        if meta.get("description") == TASK_DESC:
            state = meta.get("state")
            if state == "SUCCEEDED":
                url = meta.get("outputUrl", "")
                if url:
                    logger.info("Downloading from %s ...", url)
                    r = requests.get(url, stream=True, timeout=1200)
                    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
                    with open(OUT_PATH, "wb") as f:
                        f.write(r.content)
                    logger.info("Downloaded: %.0f MB", os.path.getsize(OUT_PATH) / 1e6)
                else:
                    logger.info("SUCCEEDED but no outputUrl. Check Google Drive: ee_exports/")
                exit(0)
            elif state == "FAILED":
                err = meta.get("error_message", "unknown")
                logger.error("FAILED: %s", err)
                exit(1)
            else:
                logger.info("State: %s - waiting 30s...", state)
                time.sleep(30)
                break
    time.sleep(5)
