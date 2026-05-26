# Aral Saxaul AI — Полный контекст проекта для планировщика

> Скопируй весь этот файл в новый чат планировщика, чтобы он понимал текущее состояние проекта без необходимости гадать.

---

## 1. ЦЕЛЬ ПРОЕКТА

Автономный пайплайн определения пригодности высохшего дна Аральского моря (~60 000 км², Аралкум) для посадки саксаула (Haloxylon). Использует спутниковые данные Sentinel-2, Sentinel-1 и SRTM3.

**Ключевое научное открытие:** NDMI — главный маркер засоления (Spearman r=+0.69, p=0.02). Чем влажнее, тем солонее (капиллярный подъём). SI отбракован (r=+0.41, p=0.21).

---

## 2. ТЕКУЩАЯ АРХИТЕКТУРА (V4.1 — Elevation Hack)

Весь пайплайн полностью переписан по сравнению с оригинальным `main.py`/`src/`. Теперь он состоит из серии независимых скриптов:

### 2.1 Пайплайн V4.1 (текущий production)

```
# Шаг 1: SRTM данные (30м)
scripts/prepare_slope_data.py
  → outputs/data/dem_slope_30m.tif      (589 MB)
  → outputs/data/dem_elevation_30m.tif   (509 MB)

# Шаг 2: Маска AOI из высоты
scripts/build_aoi_mask.py
  → outputs/data/aoi_mask_v5.tif         (3 MB)

# Шаг 3: Классификация (правила NDMI + уклон)
scripts/run_inference_v4.py
  → outputs/data/suitability_map_v4.tif   (7.8 MB)

# Шаг 4: Векторизация + карта
scripts/phase5_v4_export.py
  → outputs/data/optimal_zones_v4.geojson (30 MB, 20 261 кластер)
  → outputs/reports/suitability_map_v4.html (22 MB, Folium карта)

# QЧ: Red Team аудит
scripts/qa_sanity_check.py
  → 4 теста: геометрия, площадь UTM 41N, целостность растра, чувствительность высоты
```

### 2.2 AOI-маска (Elevation Hack) — ключевое решение

Маска пригодности строится не из JRC/вектора, а **из высоты SRTM3**:
- Уровень Аральского моря в 1960 году: ~53.4 м над уровнем моря
- Маска = `(elevation <= 54.0) AND (elevation > -50.0) AND NDMI валиден`
- Площадь: ~138 000 км² до клиппинга
- Серая зона 54-55 м: всего 3 126 км² (3.9%) — порог стабилен

### 2.3 Правила классификации V4

| Class | Правило |
|-------|---------|
| **1 Optimal** | NDMI < -0.055 AND Slope ≤ 5° |
| **2 Risk** | -0.055 ≤ NDMI ≤ -0.025 AND Slope ≤ 5° |
| **3 Dead** | NDMI > -0.025 (капиллярный подъём) |
| **4 Topo Obstacle** | Slope > 5° |
| **0 Water/NoData** | NDWI > 0 или нет данных |

### 2.4 Результаты V4 (текущие)

| Класс | % |
|-------|----|
| 1 Optimal | 65.0% |
| 2 Risk | 18.6% |
| 3 Dead | 16.0% |
| 4 Topo Obstacle | 0.4% |
| 0 Water | — |

Финальная оптимальная площадь: **31 564 км²** (20 261 кластер), подтверждена в UTM 41N (0.12% vs EPSG:4326).

---

## 3. ФАЙЛОВАЯ СТРУКТУРА

```
aral-saxaul-ai/
├── app.py                              ← Streamlit дашборд
├── main.py                             ← Старый оркестратор (оригинальный V1-V3)
├── V3_ARCHITECTURE.md                  ← Документация V3 (история версий)
├── SCRUM.md                            ← Полное описание проекта
├── PROJECT_CONTEXT.md                  ← ЭТОТ ФАЙЛ — контекст для планировщика
├── README.md                           ← README (слегка устарел)
├── requirements.txt                    ← Только для Streamlit Cloud
├── environment.yml                     ← Conda среда (только для dev)
├── packages.txt                        ← apt пакеты для Streamlit Cloud
│
├── scripts/                            ← СКРИПТЫ V4 (текущий production)
│   ├── prepare_slope_data.py           ← SRTM загрузка + уклон + высота
│   ├── build_aoi_mask.py               ← AOI маска по высоте
│   ├── run_inference_v4.py             ← NDMI + уклон классификация
│   ├── phase5_v4_export.py             ← Векторизация + Folium карта
│   ├── qa_sanity_check.py              ← Red Team аудит
│   ├── build_ground_truth.py           ← Сборка ground truth (11 точек)
│   ├── coordinate_offset_analysis.py   ← Анализ смещения координат
│   ├── temp_threshold_analysis.py      ← Анализ порогов
│   ├── audit_vectorization.py          ← QA аудит V3
│   ├── phase5_v3_export.py             ← V3 экспорт (устарел)
│   ├── run_inference_v3.py/.1/.2       ← V3 инференс (устарел)
│   └── ... (другие вспомогательные)
│
├── src/                                ← ОРИГИНАЛЬНЫЙ ПАЙПЛАЙН V1-V3 (ЗАМОРОЖЕН)
│   ├── config.py                       ← Конфиг с гиперпараметрами
│   ├── utils.py                        ← Утилиты GEE
│   ├── phase1_ingestion.py             ← V1 AOI + Feature Stack
│   ├── phase2_synthetic.py             ← V2 генерация лейблов
│   ├── phase2_local.py                 ← V2 локальная
│   ├── phase3_training.py              ← V3 XGBoost + Optuna
│   ├── phase4_inference.py             ← V3 инференс
│   └── phase5_viz.py                   ← V3 визуализация
│
├── outputs/
│   ├── aoi/
│   │   ├── aral_sea_1960.geojson       ← 5.5 MB (из растра elevation mask)
│   │   └── coastline_raster_mask.geojson
│   ├── data/
│   │   ├── aoi_mask_v5.tif             ← 3 MB (Актуальная маска)
│   │   ├── dem_slope_30m.tif           ← 589 MB
│   │   ├── dem_elevation_30m.tif       ← 509 MB (НЕ в git — >500 MB)
│   │   ├── suitability_map_v4.tif      ← 7.8 MB
│   │   ├── optimal_zones_v4.geojson    ← 30 MB (20 261 кластер)
│   │   ├── feature_stack_30m.vrt       ← VRT на тайлы
│   │   ├── feature_stack_30m_tile1.tif ← 3.5 GB
│   │   ├── feature_stack_30m_tile0_redo.tif ← 660 MB
│   │   ├── ground_truth_v2.csv         ← 11 точек
│   │   ├── synthetic_labels.csv        ← 5K точек
│   │   └── training_test.csv
│   ├── models/
│   │   ├── xgb_classifier.pkl          ← V3 модель (для истории)
│   │   └── scaler.pkl
│   ├── reports/
│   │   ├── suitability_map_v4.html     ← 22 MB (Folium карта)
│   │   ├── suitability_map_v3.html     ← 12.8 MB
│   │   └── ... (SHAP, optuna plots)
│   └── maps/                           ← PNG карты
└── .streamlit/
    └── config.toml
```

---

## 4. GIT ИСТОРИЯ (6 коммитов)

```
4ab2927 Remove environment.yml (Streamlit Cloud uses requirements.txt only)
4ff21b0 Fix Streamlit deprecation: use_container_width -> width='stretch'
9df6a8e V4.1 Elevation Hack — полностью автономный пайплайн
1eb47d6 Remove environment.yml (Streamlit Cloud uses requirements.txt)
12000ee Prepare for Streamlit Cloud: slim deps, packages.txt, track output artifacts
58b79e7 Initial commit
```

---

## 5. STREAMLIT CLOUD ДЕПЛОЙ

**Ветка:** `main`
**URL:** https://aral-saxaul-ai-7skntrgzemfiwqmjd7feoo.streamlit.app/
**Entry point:** `app.py`
**Зависимости:** `requirements.txt` (7 пакетов, без GDAL/rasterio/geopandas)
**Аpt пакеты:** `packages.txt` (пустой, но можно добавить, напр. `libgdal-dev`)

**Проблема:** `environment.yml` не должен быть в git — Streamlit Cloud пытается использовать conda и падает. Коммит `4ab2927` удалил его из репозитория.

**Важно:** GIS-библиотеки (GDAL, rasterio, geopandas, shapely, fiona, pyproj) **не установлены** и **не могут быть установлены** на Streamlit Cloud через pip (нужны системные libgdal-dev). Все скрипты пайплайна на V4 (`scripts/*.py`) работают только локально.

---

## 6. ЧТО РАБОТАЕТ СЕЙЧАС

### На Streamlit Cloud (app.py):
- Загрузка `suitability_map_v4.html` (Folium карта)
- Загрузка `optimal_zones_v4.geojson` (статистика)
- Ground truth таблица + Spearman корреляции
- Scatter plots (SI vs Salinity, NDMI vs Salinity)
- **НЕ РАБОТАЕТ:** `aral_sea_1960.geojson` на карте (включится, когда файл попадёт в git)

### Локально (Windows, conda aral-saxaul):
- **Полный пайплайн V4.1** — все 4 шага отработаны
- **QA аудит** — пройден (4 теста, 3 PASS + 1 WARN на серую зону)
- **dem_elevation_30m.tif (509 MB)** — НЕ в git (превышает лимит GitHub)

---

## 7. ИЗВЕСТНЫЕ ПРОБЛЕМЫ И УЛУЧШЕНИЯ

### Блокеры:
- `dem_elevation_30m.tif` (509 MB) не может быть в git → для Streamlit Cloud AOI маска не может быть перестроена

### Что можно улучшить:
1. **Streamlit Cloud — загружать карту без GIS-зависимостей** — сейчас только показывает HTML, но мог бы показывать больше аналитики
2. **Обновить main.py** — он всё ещё на V1-3 архитектуру, не включает V4 скрипты
3. **README устарел** — описывает conda+V1-3, не упоминает V4.1 Elevation Hack
4. **SCRUM.md устарел** — последние V4 изменения не отражены
5. **feature_stack_30m.vrt** — указывает на тайлы полными путями `F:\OPENCODE...` (непортабельно)
6. **Оптимизация:** V4.1 сейчас выдаёт 65% территории как пригодную — возможно, слишком либеральный порог NDMI -0.055

### Для полевой бригады:
- Файлы оптимальных зон: `outputs/data/optimal_zones_v4.geojson` (30 MB, 20 261 кластер)
- Folium карта: `outputs/reports/suitability_map_v4.html` (22 MB)
- GeoJSON можно конвертировать в KML/Garmin/Google Maps для навигации

---

## 8. КЛЮЧЕВЫЕ РЕШЕНИЯ

1. **Elevation Hack** — замена JRC water mask на высотную маску
2. **AOI на карте = BBOX прямоугольник** (не изрезанная береговая линия)
3. **NDMI-only** — SI и другие индексы отбракованы по Spearman
4. **Минимальная единица картографирования (MMU)** — 1 га (~11 px при 30м)
5. **30м разрешение** (не 10м) — оптимальное для масштаба 60 000 км²
6. **Топографический фильтр Slope ≤ 5°** — добавляет класс 4 (Topo Obstacle)

---

## 9. requirements.txt (Streamlit Cloud)

```
streamlit>=1.35
xgboost>=2.1
scikit-learn>=1.5
numpy>=2.0
pandas>=2.2
matplotlib>=3.9
joblib>=1.4
scipy>=1.14
```

---

## 10. КОМАНДЫ ДЛЯ ЗАПУСКА (локально, conda aral-saxaul)

```powershell
# Весь пайплайн V4.1:
python scripts/prepare_slope_data.py
python scripts/build_aoi_mask.py
python scripts/run_inference_v4.py
python scripts/phase5_v4_export.py

# QA аудит:
python scripts/qa_sanity_check.py

# Streamlit локально:
streamlit run app.py
```
