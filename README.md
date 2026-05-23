# Aral Saxaul AI

**Production-ready machine learning pipeline for mapping saxaul afforestation suitability across 60 000 km^2 of the dried Aral Sea bed.**

---

## Business Context

The Aral Sea — once the world's fourth-largest lake — has shrunk by 90% since the 1960s, leaving behind a toxic salt desert.  Large-scale saxaul (*Haloxylon*) planting can stabilise soils, reduce dust storms, and begin ecosystem restoration.

**The problem:** No ground-truth dataset exists.  Forestry expeditions cannot survey 60 000 km² of remote terrain on foot.  The solution is a synthetic-data-driven XGBoost model that learns suitability patterns directly from satellite observations — without a single human-labelled training point.

---

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  PHASE 1     │    │  PHASE 2         │    │  PHASE 3        │    │  PHASE 4         │    │  PHASE 5       │
│  Data        │───▶│  Synthetic       │───▶│  XGBoost        │───▶│  Distributed     │───▶│  Decision      │
│  Ingestion   │    │  Labels          │    │  + Optuna + SHAP│    │  Inference       │    │  Map           │
│  (GEE)       │    │  (10K points)    │    │  (GPU)          │    │  (Tile-based)    │    │  (Folium)      │
└──────┬───────┘    └────────┬─────────┘    └────────┬────────┘    └────────┬─────────┘    └───────┬────────┘
       │ 7-band              │ 10K rows              │ model.pkl           │ prob_map.tif          │ .html + .geojson
       │ Feature Stack       │ pd.DataFrame          │ scaler.pkl          │ BIGTIFF               │ statistics.json
       │ (60 000 km²)        │                       │ SHAP plots          │                       │
       ▼                     ▼                       ▼                     ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS: Sentinel-2 L2A (10m) · Sentinel-1 GRD (VH) · Copernicus DEM GLO-30 · JRC Global Surface Water  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Features (7 bands)

| Band   | Description                          | Source                |
|--------|--------------------------------------|-----------------------|
| NDMI   | Normalised Difference Moisture Index | Sentinel-2 B8, B11    |
| MSAVI  | Modified Soil-Adjusted Vegetation    | Sentinel-2 B4, B8     |
| SI     | Salinity Index                       | Sentinel-2 B2, B4     |
| Slope  | Terrain slope (degrees)              | Copernicus DEM 30m    |
| TWI    | Topographic Wetness Index            | Copernicus DEM 30m    |
| VH     | Radar backscatter VH (dB)            | Sentinel-1 GRD        |
| NDWI   | Water index                          | Sentinel-2 B3, B8     |

### Key Design Decisions

- **Synthetic labels (cold-start solution):** Class 1 = pixels with stable vegetation (MSAVI > 0.15, NDMI > -0.1) across 2023–2025 dry seasons.  Class 0 = saline crusts (SI > P85) OR steep cliffs (Slope > 25°) OR bare sand.
- **SCL cloud masking** (not QA60): Salt flats are spectrally similar to clouds.  ESA's Scene Classification Layer is more robust in arid environments.
- **Survivor-bias acceptance:** If suitable land exists but no seeds ever arrived, the model marks it Class 0.  We optimise for **Precision** over Recall — false positives (planting in salt) are costlier than false negatives (missing a good spot).
- **Tile-based inference (20×20 km):** A full-probability raster at 10 m over 60 000 km² would require 16.8 GB.  We process ~150 tiles independently and merge via GDAL VRT (O(1) memory).

---

## System Requirements

| Component   | Minimum            | Recommended          |
|-------------|--------------------|----------------------|
| GPU         | NVIDIA >= 8 GB VRAM | RTX 3080 / A4000     |
| RAM         | 16 GB              | 32 GB                |
| Disk        | 20 GB free         | 50 GB free (tiles)   |
| OS          | Windows 10/11, Linux | N/A              |
| Python      | 3.12               | 3.12                 |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> && cd aral-saxaul-ai

# 2. Create the conda environment (all dependencies, CUDA-ready XGBoost)
conda env create -f environment.yml

# 3. Activate
conda activate aral-saxaul

# 4. Verify GPU is visible to XGBoost
python -c "import xgboost as xgb; print('GPU:', xgb.build_info().get('USE_CUDA'))"
# Expected: GPU: True

# 5. Authenticate with Google Earth Engine (one-time)
python -c "import ee; ee.Authenticate()"
# Follow the browser-based OAuth flow.
```

**Troubleshooting:**
- If GDAL DLL errors appear: `conda install -c conda-forge gdal libgdal --force-reinstall`
- If XGBoost cannot see the GPU: install NVIDIA Game-Ready or Studio drivers ≥ 545.x
- Never mix `pip install gdal` with conda-forge geo-stack — stick to conda-forge exclusively for `gdal`, `rasterio`, `fiona`

---

## Usage

### Full pipeline (end-to-end)

```bash
python main.py --all
```

### Individual phases

```bash
python main.py --phase 1              # Build AOI + 7-band Feature Stack
python main.py --phase 2              # Generate 10 000 synthetic labels
python main.py --phase 3              # Train XGBoost (Optuna + SHAP)
python main.py --phase 4              # Distributed inference (tile-based)
python main.py --phase 5              # Decision boundaries + interactive map
```

### Phase ranges

```bash
python main.py --phase 1-3            # Data → Labels → Model
python main.py --phase 3-5            # Training through visualisation
```

### Advanced flags

```bash
# Phase 3: control Optuna trials, skip tuning
python main.py --phase 3 --trials 200 --csv outputs/data/synthetic_labels.csv
python main.py --phase 3 --skip-optuna --csv data.csv

# Phase 4: custom tile directory, 8 inference workers
python main.py --phase 4 --tile-dir /mnt/tiles --workers 8

# Phase 5: adjust confidence threshold
python main.py --phase 5 --raster outputs/probability_map.tif --threshold 0.90

# Skip Phase 4 (model already trained, inference done elsewhere)
python main.py --all --skip-phase4
```

### Standalone module execution

Each phase can also be run directly:

```bash
python src/phase1_ingestion.py --aoi-only           # Build and inspect AOI only
python src/phase2_synthetic.py                       # Requires GEE session
python src/phase3_training.py --input-csv data.csv --trials 50
python src/phase4_inference.py --tile-dir outputs/tiles --workers 4
python src/phase5_viz.py --raster outputs/probability_map.tif --threshold 0.85
```

---

## Output Artifacts

All results are written to the `outputs/` directory:

```
outputs/
├── aoi/
│   └── (AOI geometry)
├── models/
│   ├── xgb_classifier.pkl           ← Trained XGBoost model
│   ├── scaler.pkl                   ← StandardScaler for Phase 4
│   └── feature_names.json           ← Ordered feature list
├── data/
│   ├── synthetic_labels.csv         ← 10K labelled dataset
│   └── feature_importance.csv       ← Gain-based importance
├── reports/
│   ├── shap_summary.png             ← SHAP bee-swarm plot
│   ├── shap_dependence_si.png       ← SHAP dependence (Salinity Index)
│   ├── shap_dependence_msavi.png    ← SHAP dependence (Vegetation)
│   ├── shap_dependence_ndmi.png     ← SHAP dependence (Moisture)
│   ├── optuna_history.png           ← Hyperparameter optimisation trace
│   ├── confusion_matrix.png         ← Test-set confusion matrix
│   └── classification_report.txt    ← Precision/Recall/F1 report
├── tiles/                           ← Intermediate tile files (Phase 4)
│   └── predictions/
├── probability_map.tif              ← Suitability probability raster (BIGTIFF)
├── suitable_zones.geojson           ← Vectorised suitable zones
├── aral_saxaul_map.html             ← Interactive Folium map
├── statistics.json                  ← Aggregated area statistics
└── pipeline_summary.json            ← Full-run metadata
```

Open `outputs/aral_saxaul_map.html` in any browser to explore the final suitability map with Google Satellite basemap.

---

## Interpreting SHAP Results

The SHAP plots in `outputs/reports/` are the primary validation tool:

| Feature | Expected SHAP Direction | Physical Interpretation                          |
|---------|------------------------|---------------------------------------------------|
| SI      | **Negative**           | High salt → low suitability (salt crusts kill saxaul) |
| MSAVI   | **Positive**           | High vegetation → high suitability                |
| NDMI    | **Positive**           | Moist soil → better growing conditions            |
| VH      | **Experimental**       | High backscatter = rough surface (sand); low = smooth (salt). Monitor. |

If SHAP shows the opposite direction (e.g., SI is positive), the model learned noise rather than physics.  Revisit the Phase 2 label-generation rules.

---

## Project Structure

```
aral-saxaul-ai/
├── main.py                         ← Pipeline orchestrator
├── environment.yml                 ← Conda environment specification
├── README.md                       ← This file
├── src/
│   ├── __init__.py
│   ├── config.py                   ← Central configuration (all 5 phases)
│   ├── utils.py                    ← GEE init, SCL masks, spectral indices
│   ├── phase1_ingestion.py         ← AOIBuilder + DataIngestion
│   ├── phase2_synthetic.py         ← TemporalStabilityChecker + AdaptiveThresholdCalibrator + SyntheticLabelGenerator
│   ├── phase3_training.py          ← DatasetBuilder + OptunaOptimizer + ModelTrainer (SHAP)
│   ├── phase4_inference.py         ← TileGridGenerator + GEEExporter + TileProcessor + TileMerger + DistributedInference
│   └── phase5_viz.py               ← MorphologicalFilter + RasterVectorizer + DecisionMapper
└── outputs/                        ← All generated artifacts
    ├── aoi/
    ├── models/
    ├── data/
    ├── reports/
    └── tiles/predictions/
```

---

## License & Citation

This pipeline was designed for the Aral Sea ecological restoration initiative.  If you use it in academic or operational work, please cite appropriately.

---

*Built with: Google Earth Engine · Sentinel-2 · Sentinel-1 · Copernicus DEM · XGBoost · Optuna · SHAP · GDAL · Folium*
