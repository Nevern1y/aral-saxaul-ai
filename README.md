# Aral Saxaul AI — V4.1

**Оптимизация посадки саксаула на высохшем дне Аральского моря.**
31 564 км² пригодных зон, 782 тракторных наряда с транспортной доступностью, веб-дашборд на Streamlit Cloud.

---

## Ключевые результаты

| Метрика | Значение |
|---|---|
| **AOI** | 57.5°–62.0°E / 43.3°–46.7°N (историческое дно Арала) |
| **Оптимальная зона (V4)** | **31 564 км²** (3 156 366 га) |
| **Рабочих нарядов (KML)** | **782** (сетка 0.1°×0.1°, фильтр ≥5 га) |
| **Дорог в AOI (OSM)** | **2 349 сегментов** (track / service / unclassified) |
| **Золотых нарядов (<2 км от дороги)** | **106** (481 402 га) |
| **Самая удалённая точка** | **81.6 км** до ближайшей дороги |
| **Разрешение** | 30 м (10 м мультиспектральные каналы) |
| **Ground truth** | 11 точек, Spearman r(NDMI vs Salinity) = **+0.69** (p=0.02) |

---

## Архитектура пайплайна

Пайплайн состоит из 7 последовательных фаз — от загрузки спутниковых данных до полевых KML-нарядов:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  PHASE 1     │   │  PHASE 2     │   │  PHASE 3     │   │  PHASE 4     │
│  Data        │──▶│  Synthetic   │──▶│  XGBoost     │──▶│  Distributed │
│  Ingestion   │   │  Labels      │   │  + Optuna    │   │  Inference   │
│  (GEE)       │   │  (10K pts)   │   │  + SHAP      │   │  (tile-based)│
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │ 7-band stack     │ 10K rows         │ model.pkl        │ prob_map.tif
       ▼                  ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INPUTS: Sentinel-2 L2A · Sentinel-1 GRD · Copernicus DEM GLO-30    │
│          SRTM3 (elevation hack) · JRC Global Surface Water           │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│  PHASE 5     │   │  PHASE 6     │   │  PHASE 7     │   │  APP.PY          │
│  Export +    │──▶│  Logistics   │──▶│  Infra       │   │  Streamlit       │
│  Map (V4)    │   │  Export KML  │   │  (OSM roads) │   │  Dashboard       │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────────────────┘
       │ GeoJSON + HTML   │ 782 KML файлов   │ road dist (km)
       ▼                  ▼                  ▼
```

---

## Ключевые научные инсайты

### NDMI — маркер засоления (SI отвергнут)

Spearman correlation на 11 грунтовых образцах с полевых экспедиций:

| Пара | r | p | Вердикт |
|---|---|---|---|
| Salinity vs **SI** | +0.41 | 0.21 | Слабая (p > 0.05) |
| Salinity vs **NDMI** | **+0.69** | **0.02** | Сильная (p < 0.05) |

Засоление в Аралкуме контролируется **капиллярным подъёмом грунтовых вод**, а не поверхностными солевыми корками. NDMI (Normalised Difference Moisture Index) — прямой маркер засоления: чем влажнее грунт, тем выше засоление. SI (яркость соли) — лишь оптическая иллюзия.

### Правила классификации V4

| Класс | Условие | Интерпретация |
|---|---|---|
| **1 Optimal** | NDMI < **-0.055** и Slope ≤ 5° | Пригодно для посадки |
| **2 Risk** | -0.055 ≤ NDMI ≤ -0.025 и Slope ≤ 5° | Пограничная зона (риск) |
| **3 Dead** | NDMI > -0.025 | Капиллярный подъём — засоление |
| **4 Obstacle** | Slope > 5° | Крутые склоны/обрывы |
| **0 Water** | NDWI > 0 | Вода / нет данных |

### SRTM3 Elevation Hack

Платные исторические батиметрические данные заменены бесплатным радаром NASA SRTM3 (30 м). Береговая линия Аральского моря 1960 года восстановлена по изогипсе **54 метра** — физическому следу уровня воды до катастрофы.

### R-tree Spatial Index

Для пересечения 20 261 полигона зон с сеткой 0.1°×0.1° (1 240 ячеек) используется R-tree (`gdf.sindex`). Время выполнения — **20.9 секунд**. Наивный `gpd.overlay` гарантированно вызвал бы OOM на 32 GB RAM.

---

## Установка

Проект развёрнут на Streamlit Cloud, но может быть запущен локально:

```bash
# 1. Clone
git clone <repo-url> && cd aral-saxaul-ai

# 2. Установка зависимостей (Python 3.12)
pip install -r requirements.txt

# 3. Аутентификация GEE (однократно)
python -c "import ee; ee.Authenticate()"
```

**Основные зависимости:** folium · geopandas · matplotlib · numpy · pandas · plotly · scikit-learn · scipy · shapely · streamlit · xgboost

---

## Запуск

### Веб-дашборд (основной интерфейс)

```bash
streamlit run app.py
```

Три вкладки:
1. **📍 Карта рабочих участков** — фильтр по расстоянию до дорог и минимальной площади, интерактивная карта с Google Satellite
2. **📊 Общая статистика** — метрики V4, круговая диаграмма зон, карта пригодности
3. **⚙️ Технические параметры** — правила классификации, распределение V4, ground truth, Spearman корреляции

### Полный пайплайн (7 фаз)

```bash
# По порядку
python scripts/phase1_ingestion.py      # Phase 1 — GEE: AOI + feature stack
python scripts/phase2_synthetic.py      # Phase 2 — синтетические метки
python scripts/phase3_training.py       # Phase 3 — XGBoost + Optuna + SHAP
python scripts/phase4_inference.py      # Phase 4 — инференс (tile-based)
python scripts/phase5_v4_export.py      # Phase 5 — векторизация + Folium карта
python scripts/phase6_logistics_export.py   # Phase 6 — 782 KML наряда
python scripts/phase7_infrastructure.py     # Phase 7 — OSM дороги + расстояния
```

Альтернативно — `main.py` (легаси-оркестратор Phases 1–5):

```bash
python main.py --phase 1-3    # Исследование: данные → метки → модель
python main.py --phase 3-5    # Обучение → инференс → карта
```

---

## Структура проекта

```
aral-saxaul-ai/
├── app.py                          ← Streamlit dashboard (3 вкладки)
├── main.py                         ← Pipeline orchestrator (Phases 1-5)
├── requirements.txt                ← Зависимости для pip
├── README.md
│
├── scripts/                        ← Исполняемые скрипты (Phases 1-7)
│   ├── phase1_ingestion.py         ← GEE: AOI + 7-band feature stack
│   ├── phase2_synthetic.py         ← Генерация синтетических меток
│   ├── phase3_training.py          ← XGBoost + Optuna + SHAP
│   ├── phase4_inference.py         ← Тайловый инференс
│   ├── phase5_v4_export.py         ← Векторизация + карта V4
│   ├── phase6_logistics_export.py  ← Разбивка на 782 KML-наряда
│   ├── phase7_infrastructure.py    ← OSM дороги + distance matrix
│   ├── build_ground_truth.py       ← Ground truth pipeline
│   └── ...                         ← Утилиты и вспомогательные скрипты
│
├── src/                            ← Исходный код библиотеки
│   ├── config.py                   ← Центральная конфигурация
│   ├── utils.py                    ← GEE utils, спектральные индексы
│   ├── phase1_ingestion.py         ← Модуль Phase 1
│   ├── phase2_synthetic.py         ← Модуль Phase 2 (GEE sampling)
│   ├── phase2_local.py             ← Phase 2 (локальный рендеринг)
│   ├── phase3_training.py          ← Модуль Phase 3
│   ├── phase4_inference.py         ← Модуль Phase 4 (GEE export)
│   ├── phase4_local.py             ← Phase 4 (локальный инференс)
│   └── phase5_viz.py               ← Модуль Phase 5
│
└── outputs/
    ├── aoi/                        ← AOI geometry + маски
    ├── data/                       ← Feature stacks, labels, GeoJSON
    ├── models/                     ← xgb_classifier.pkl, scaler.pkl
    ├── reports/                    ← SHAP plots, карты (HTML)
    ├── tiles/                      ← Inference tiles
    ├── logistics/
    │   ├── tractor_tasks/          ← 782 KML для GPS-навигаторов
    │   ├── tasks_index_enriched.csv ← С расстоянием до дорог
    │   └── aralkum_roads.geojson   ← OSM road network
    └── tmp/
```

---

## Ground Truth

11 точек полевых почвенных разрезов (2020 г., экспедиция ОДТ и AralField).

**Spearman корреляции спектральных индексов с засолением:**

| Пара | r | p | Интерпретация |
|---|---|---|---|
| Salinity vs SI | +0.41 | 0.21 | SI не разделяет зоны |
| Salinity vs NDMI | **+0.69** | **0.02** | NDMI — сильный предиктор |

Пороги для правил классификации (NDMI = -0.055, -0.025) калиброваны по этим данным.

---

## Выходные артефакты

### Интерактивная карта
`outputs/reports/suitability_map_v4.html` — Folium с Google Satellite basemap.

### Полевые наряды (Phase 6)
- `outputs/logistics/tractor_tasks/task_grid_{lat}_{lon}.kml` — 782 KML-файла
- Каждый = одна операционная ячейка 0.1°×0.1° с dissolved геометрией зон
- Формат: KML v2.2, совместим с любыми GPS-навигаторами

### Транспортная доступность (Phase 7)
- `tasks_index_enriched.csv` — 782 записи с `distance_to_road_km`
- `aralkum_roads.geojson` — 2 349 сегментов дорог

### ML-модель (Phase 3)
- `models/xgb_classifier.pkl` — обученный XGBoost
- `reports/shap_summary.png` — SHAP bee-swarm
- `reports/optuna_history.png` — история оптимизации гиперпараметров

---

## Распределение доступности

```
< 1 km:    64 tasks  ( 8.2%)    — пешая доступность
< 2 km:   106 tasks  (13.6%)    — быстрый подъезд    ← ЗОЛОТЫЕ НАРЯДЫ
< 5 km:   218 tasks  (27.9%)    — трактор без подготовки
< 10 km:  322 tasks  (41.2%)    — короткое плечо
< 20 km:  466 tasks  (59.6%)    — среднее плечо
< 50 km:  709 tasks  (90.7%)    — с топливозаправщиком
< 100 km: 782 tasks (100%)      — вся территория
```

**Рекомендация:** начать с 106 золотых нарядов (481 тыс. га, <2 км от дороги). Максимальная отдача на литр топлива, минимальный риск для техники.

---

## История версий

| Версия | Описание |
|---|---|
| V1.0 | Baseline: BBOX AOI, SI-based salinity |
| V2.0 | Синтетические метки, XGBoost |
| V3.0–V3.2 | NDMI pivot, slope filter |
| V4.0 | Coastline mask из SRTM3 (54m contour) |
| V4.1 (current) | OSM roads + 782 KML наряда + Streamlit dashboard + Ground truth |

---

## License

Разработано для инициативы по восстановлению экосистемы Аральского моря.

Данные: OpenStreetMap (ODbL), Copernicus (free), NASA SRTM (public domain), JRC Global Surface Water.

*Built with: Google Earth Engine · Sentinel-2 · Sentinel-1 · Copernicus DEM · NASA SRTM · XGBoost · Optuna · SHAP · GeoPandas · OSMnx · Folium · Streamlit · scipy*
