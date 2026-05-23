# Aral Saxaul AI — Полное описание проекта

## Цель
Автономный пайплайн для определения пригодности высохшего дна Аральского моря для посадки саксаула (Haloxylon). Использует спутниковые данные Sentinel-2, Sentinel-1 и Copernicus DEM + Google Earth Engine.

## Результат (итоговый)
- **6,558 км²** (655,773 га) дна Арала пригодны для посадки (13.3% от суши, 4.4% от полного bounding box)
- Модель: XGBoost GPU (RTX 3080), Optuna, SHAP
- Интерактивная карта: `outputs/reports/suitability_map_full.html` (597 полигонов)

---

## Структура проекта

```
aral-saxaul-ai/
├── main.py                          # Оркестратор пайплайна
├── src/
│   ├── config.py                    # Все конфиги, пути, гиперпараметры
│   ├── utils.py                     # Логирование, GEE init, функции индексов
│   ├── phase1_ingestion.py          # AOI (JRC + NDWI) + Feature Stack (GEE)
│   ├── phase2_local.py              # Экспорт raw данных из GEE → локальные индексы → CSV
│   ├── phase3_training.py           # XGBoost + Optuna (100 trial) + SHAP
│   ├── phase4_local.py              # Инференс по тайлам (1024×1024 px)
│   └── phase5_viz.py                # (устарел — используй scripts/phase5_full.py)
├── scripts/
│   ├── test_full_pipeline.py        # Интеграционный тест на тестовой области
│   ├── download_from_drive.py       # Скачивание COG с Google Drive через GEE OAuth2
│   ├── wait_and_download.py         # Ожидание задачи GEE + скачивание
│   ├── phase5_full.py               # Финальная визуализация (Folium + даунсэмплинг 30→300m)
│   └── check_labels.py, check_tile.py, local_training_data.py (вспомогательные)
└── outputs/
    ├── data/                        # Все растры, CSV, feature importance
    ├── models/                      # xgb_classifier.pkl, scaler.pkl, feature_names.json
    ├── reports/                     # HTML-карты, SHAP plots, confusion matrix
    └── aoi/                         # Векторные слои AOI
```

---

## 5 фаз пайплайна

### Phase 1: AOI + Feature Stack (GEE-side)
- AOI = JRC Global Surface Water `max_extent` (историческая вода 1984-2021) AND NOT текущая вода (NDWI > 0 + SCL == 6)
- Bounding box: (58.0, 43.5, 62.0, 46.5) — ~105k км²
- Feature Stack: 7 каналов [NDMI, MSAVI, SI, Slope, TWI, VH, NDWI]
- Для production используется **batch export** в Google Drive с `mosaic()` (не `median()` — слишком долго для GEE)
- Mosaic — последний пиксел без облаков за 15-31 Aug 2025; для Арала эквивалентен median (облаков нет в сухой сезон)

### Phase 2: Синтетические лейблы (локально)
- Экспорт raw S2 (6 bands) + DEM + S1 VH из GEE через `geemap.ee_export_image()` (getPixels) на 100m
- Локально в numpy считаются все 7 индексов, включая Slope (Sobel), TWI (1/tan(slope))
- 5000 случайных точек
- **Правило лейблинга**: SI >= P85 (засоленная почва) AND Slope <= 5° (ровная поверхность)
  - NDWI и MSAVI исключены из правил: NDWI в диапазоне (-0.09, -0.05) не достигает порога, SI-MSAVI корреляция -0.69 избыточна
  - ~15% точек положительные (пригодные)
  - Feature importance модели подтверждает: SI 60% > NDWI 26% > NDMI 7% > MSAVI 5%

### Phase 3: XGBoost + Optuna + SHAP
- Scaler: StandardScaler (сохраняется для инференса)
- Optuna: 100 trial, 5-fold CV, оптимизация **Precision (Class 1)**
- Гиперпараметры: max_depth 3-7, learning_rate 0.01-0.2, n_estimators 50-300 и т.д.
- GPU: `tree_method=hist`, `device=cuda`
- SHAP: TreeExplainer, 2000 сэмплов, summary + 4 dependence plots (SI, MSAVI, NDMI, VH)
- Всего 8 артефактов: optuna_history, optuna_param_importance, confusion_matrix, feature_importance.csv, classification_report.txt, shap_summary, shap_dependence_{si,msavi,ndmi,vh}

### Phase 4: Инференс (локально, по тайлам)
- Читает 7-канальный GeoTIFF, загружает model + scaler
- Разбивает на тайлы 1024×1024 (если >5M пикселей — всегда для прода)
- Внутри тайла — батчи по 100k валидных пикселей через `predict_proba`
- Записывает float32 GeoTIFF (вероятность, nodata=-1)
- Для COG тайлов: **читать напрямую, не через VRT** (GDAL bug с TIFFReadEncodedTile через VRT)

### Phase 5: Визуализация (scripts/phase5_full.py)
- Загружает suitability_full.tif
- Считает статистики (всего/valid/high pixels, площадь)
- Даунсэмплинг 30m → 300m (mean factor=10) для векторизации
- `rasterio.features.shapes()` → GeoJSON с min_px=2
- Folium карта на Esri Satellite, зелёные полигоны с tooltip (ha, km²)
- Сохраняется в `outputs/reports/suitability_map_full.html`

---

## Ключевые архитектурные решения

1. **Offline pipeline**: raw S2/DEM/S1 экспортируются из GEE (лёгкие getPixels), все индексы считаются в numpy — полный контроль, нет таймаутов GEE
2. **mosaic() вместо median()**: O(n) вместо O(n log n) — секунды вместо бесконечности для 30m. На Арале в сухой сезон облаков нет, разницы нет
3. **COG проблемы**: VRT не работает для записи (нужен явный GTiff profile); COG с LZW может иметь повреждённые блоки → перекачка с Drive
4. **OAuth2 GEE → Drive**: credentials содержат Drive scope → `google.oauth2.credentials.Credentials(refresh_token=..., client_id=ee.oauth.CLIENT_ID, client_secret=ee.oauth.CLIENT_SECRET)`

## Окружение
- Windows 11, Python 3.12, conda env `aral-saxaul`
- `& "C:\Users\cynok\miniconda3\shell\condabin\conda-hook.ps1"; conda activate aral-saxaul`
- RTX 3080 (CUDA 12.x)
- GEE project: `tribal-dispatch-494405-u4`
- Все зависимости через conda-forge

## Команды для запуска
```powershell
# Полный пайплайн (Phase 1-5)
python main.py --all --project tribal-dispatch-494405-u4

# Инференс на полной AOI
python -m src.phase4_local --input outputs/data/feature_stack_30m_tile1.tif --output outputs/data/suitability_left.tif
python -m src.phase4_local --input outputs/data/feature_stack_30m_tile0_redo.tif --output outputs/data/suitability_right.tif

# Мёрж
python -c "import rasterio; from rasterio.merge import merge; [d:=rasterio.open(f) for f in ['outputs/data/suitability_left.tif','outputs/data/suitability_right.tif']]... ; m,t=merge(sources); ..."

# Карта
python scripts/phase5_full.py
```

## Известные проблемы
1. **Tile0 COG повреждён**: блоки Y=9984..11197 (1213 rows × 2300 cols = 2.8M px) нечитаемы в первой версии. Решение: перекачка с Drive
2. **VRT → write**: `Writing through VRTSourcedRasterBand is not supported` → нужен явный GTiff profile
3. **PowerShell quoting**: сложные многострочные python -c требуют экранирования → лучше писать скрипты в файлы
4. **Phase 4 пиксели**: 166M пикселей на 30m, обработка ~50 секунд на тайл через тайлы 1024×1024

## Текущее состояние файлов
- `feature_stack_30m_tile1.tif`: 7×11197×12544, 58.0°E–61.38°E, 3.7 GB (ЛЕВЫЙ, хороший)
- `feature_stack_30m_tile0_redo.tif`: 7×11197×2300, 61.38°E–62.0°E, 0.68 GB (ПРАВЫЙ, перекачанный, хороший)
- `suitability_left.tif`: 1×11197×12544 (вероятности левого тайла)
- `suitability_right.tif`: 1×11197×2300 (вероятности правого тайла)
- `suitability_full.tif`: 1×11197×14844 (мёрж левого+правого, 31 MB LZW)
- `outputs/reports/suitability_map_full.html`: 920 KB, интерактивная карта

## Что можно улучшить
1. **Оптимизация даунсэмплинга**: factor=10 сейчас использует `mean()`, можно `max()` или `mode()` для бинарной маски
2. **Тренировка на 5000 точках**: мало для 100k км². Добавить больше сэмплов или стратифицированную выборку
3. **Optuna**: запустить 200+ trial для production модели
4. **Валидация**: полевые данные для Ground Truth отсутствуют — нужна экспедиция
5. **TWI**: упрощённая формула ln(1/tan(slope)) без flow accumulation — для равнины Арала ок, но можно улучшить
6. **NDWI в фичах**: все значения <0 (сухой бассейн) — модель выучила SI порог, NDWI как вторичный сигнал. Можно убрать NDWI для упрощения
