# Aral Saxaul AI — V5.1 Scientific Screening

**Научно воспроизводимый скрининг кандидатных зон посадки саксаула на высохшем дне Аральского моря.**

V5.1 сохраняет V5 rule-based pipeline на нативном разрешении Sentinel-2 (10 м), но добавляет научный слой: provenance, нормализованные почвенные таблицы, dual-coordinate validation и осторожную интерпретацию результата как screening product.

[→ Открыть дашборд](https://aral-saxaul-ai-7skntrgzemfiwqmjd7feoo.streamlit.app/)

---

## Ключевые Метрики V5.1

| Метрика | Значение |
|---|---|
| **AOI** | 58.6°–62.0°E / 43.5°–46.5°N |
| **Candidate screening zone** | **1 598 800 га** по сетке 100 м (15 988 км²); **1 594 115 га** по карте 10 м |
| **Operational area >=10 ha** | **1 564 250 га** по всему AOI |
| **Казахстанская operational area** | **1 092 886 га**, **1 296** clipped-полигонов |
| **Кластеров candidate-зоны >=10 га** | **2 132** |
| **Операционных ячеек (KML)** | **308** по территории Казахстана (сетка 0.1°×0.1°) |
| **Дорог OSM для логистики** | **16 244 сегмента**, включая **14 915** сегментов казахстанского подъезда |
| **Разрешение** | **10 м** (Sentinel-2 L2A) |
| **Классов классификации** | **6**: Candidate suitable, Dry salt risk, Wet brine risk, Obstacle, Vegetation, Water/NoData |
| **Ground truth status** | 11 mapped pilot points with unresolved coordinate authority; profiles 12-21 are lab-only until coordinates are provided |

---

## Архитектура V5.1

Пайплайн V5.1 — последовательность от загрузки сырых спутниковых данных до веб-дашборда и научного validation layer:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  V5 Config   │   │  Inference   │   │  Stats       │   │  Logistics   │
│  + GEE Fetch │──▶│  V5 (rule-   │──▶│  Extraction  │──▶│  Prep        │
│  (Sentinel-2)│   │  based, 10m) │   │  + Viz       │   │  + KML       │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                    │
                                                                    ▼
                                                             ┌──────────────┐
                                                             │  APP.PY      │
                                                             │  Streamlit   │
                                                             │  Dashboard   │
                                                             └──────────────┘
```

### Скрипты V5.1

| Скрипт | Назначение |
|---|---|
| `scripts/fetch_gee_raw_v5.py` | Загрузка каналов Sentinel-2 L2A (B3, B4, B8 — 10 м; B11, B12 — 20 м; SCL) через GEE |
| `scripts/prepare_slope_data.py` | Загрузка SRTM, расчет slope/elevation 30 м (`dem_slope_30m.tif`) — обязательный вход для классификации |
| `scripts/run_inference_v5.py` | Применение правил классификации (NDMI, NDSI, BR, Slope, NDVI) → растровая маска 10 м |
| `scripts/v5_finalize_viz.py` | Категориальный фильтр 3x3, filtered raster и Folium-карта (`suitability_map_v5.html`) |
| `scripts/v5_extract_stats.py` | Статистика по filtered raster и рабочий GeoJSON (`v5_stats.json`, `operational_zones_v5.geojson`) |
| `scripts/v5_kazakhstan_boundary_prep.py` | Загрузка границы Казахстана для обрезки логистики по территории РК |
| `scripts/v5_roads_prep.py` | Обновление OSM-дорог, включая отдельный слой казахстанского подъезда |
| `scripts/v5_logistics_prep.py` | Обрезка candidate-зон по Казахстану, сетка 0.1°×0.1°, KML-наряды и расстояния до дорог |
| `scripts/build_v5_science_dataset.py` | Нормализация field/soil/provenance таблиц в `outputs/science/` |
| `scripts/v5_validation_report.py` | V5 point sampling validation по raw и shifted coordinate-source |
| `scripts/v5_coordinate_adjudication_report.py` | Coordinate authority appendix и review template |
| `scripts/v5_uncertainty_report.py` | Threshold sensitivity и uncertainty report |

---

## Дашборд (Streamlit)

Три вкладки:

### 1. 📍 Карта рабочих участков
- Выбор сценария транспортной доступности (прибрежная зона / глубокий Аралкум / весь охват)
- Выбор масштаба кластеров (локальные питомники / лесничества / стратегические хабы)
- Интерактивная карта Folium с Google Satellite, дорогами OSM и зелёными ячейками нарядов
- Экспорт KML для GPS-навигаторов

### 2. 📊 Общая статистика
- 4 ключевых метрики: доступные гектары, количество участков, площадь, мин. размер
- **Интерактивная screening-карта candidate-зон** (V5.1, 10 м)
- **Калькулятор ресурсов экспедиции** (саженцы, ГСМ, машино-смены)
- **Спектральный аудит** — донут-диаграмма распределения классов

### 3. ⚙️ Технические параметры
- Правила классификации V5.1 с адаптивными порогами P15/P85
- Динамические спектральные пороги (6 метрик)
- Таблица сравнения версий (V1.0 — V5.1)
- Ground Truth: 11 mapped pilot points, coordinate uncertainty, validation/uncertainty reports

---

## Правила Классификации V5.1

| Класс | Правило |
|---|---|
| **0 Water/NoData/Shadow** | SCL ∈ [3,6,7,8,9,10] или NDWI(Green,NIR) > 0 или BI < 0.15 |
| **5 Obstacle** | Slope > 5° |
| **10 Vegetation** | NDVI > 0.08 |
| **4 Wet brine risk** | NDMI > P85 **и** B8/B12 > P85 |
| **3 Risk (dry salt)** | NDSI_Green_SWIR2 > P85 **и** NDMI < P15 |
| **1 Candidate suitable** | Residual screening class: not water/shadow, obstacle, vegetation, dry-salt proxy, or wet-brine proxy |

Пороги P15/P85 рассчитываются автоматически по прореженной выборке сцены Sentinel-2. Класс `Candidate suitable` не является доказательством успешной посадки; это приоритетная зона для дальнейшей полевой проверки.

---

## Установка

```bash
# 1. Clone
git clone <repo-url> && cd aral-saxaul-ai

# 2. Установка зависимостей (Python 3.12)
pip install -r requirements.txt

# 3. Запуск дашборда
streamlit run app.py
```

### Полный пайплайн V5.1 (локально)

```bash
# 1. Загрузка спектральных каналов S2 (требуется GEE)
python scripts/fetch_gee_raw_v5.py

# 1b. Загрузка рельефа SRTM → dem_slope_30m.tif (обязательный вход)
python scripts/prepare_slope_data.py

# 2. Инференс (правила классификации)
python scripts/run_inference_v5.py

# 3. Финальная карта и filtered raster
python scripts/v5_finalize_viz.py

# 4. Извлечение статистик + рабочий GeoJSON
python scripts/v5_extract_stats.py

# 5. Граница Казахстана и дороги OSM, включая Казахстанский подъезд
python scripts/v5_kazakhstan_boundary_prep.py
python scripts/v5_roads_prep.py

# 6. Логистика (KML + расстояния до дорог)
python scripts/v5_logistics_prep.py

# 7. Научный слой: provenance + validation + uncertainty
python scripts/build_v5_science_dataset.py
python scripts/v5_validation_report.py
python scripts/v5_coordinate_adjudication_report.py
python scripts/v5_uncertainty_report.py

# Или после инференса одной командой для products + science + QA:
python scripts/run_v5_science_suite.py --products --refresh-roads --qa
```

---

## Ground Truth And Provenance

V5.1 хранит данные в `outputs/science/`:

- `field_sites_v5.csv`: 11 точек AralField `S124-S134` с GPS, landcover, vegetation, полевыми salinity/EC.
- `soil_layers_v5.csv`: лабораторные слои salinity/ions, humus/CO2/pH.
- `soil_profile_summary_v5.csv`: top-layer summary для 21 разреза.
- `site_profile_mapping_v5.csv`: связь `S_Point ↔ pit_code ↔ Разрез N` и конфликт raw vs shifted coordinates.
- `v5_point_samples.csv`: V5 classes/bands/indices sampled at both coordinate sources.
- `v5_validation_report.md`: pilot validation report.
- `v5_coordinate_adjudication_report.md`: appendix для выбора authoritative coordinate source.
- `v5_coordinate_authority_template.csv`: шаблон для фиксации GPS/ODT evidence по `S124-S134`.

`v5_coordinate_authority_template.csv` безопасен для ручного заполнения: поля reviewer decision сохраняются при повторном запуске `python scripts/run_v5_science_suite.py --qa`, а report показывает `resolved`, `unresolved`, `incomplete` или `invalid` статус по строкам.

Если строки template получают `resolved` статус, `v5_validation_report.py` автоматически добавляет в `v5_point_samples.csv` отдельный `coordinate_source=authoritative_dd`. Raw и shifted координаты остаются в outputs как audit trail, но не должны смешиваться с authoritative validation subset.

Текущий вывод: данные достаточны для pilot validation и screening, но недостаточны для claims уровня “модель доказала пригодность посадки”. Главный blocker — выбор authoritative coordinates для 11 mapped points и добавление координат для разрезов 12-21.

---

## V6 — слой солёности по лабораторным данным (научное дополнение)

<!-- V6_SCOPE_AUTO -->
> **V6 scope (auto-generated — see `data/canonical/SCOPE_AND_LIMITATIONS_V6.md`):** Shipped salinity anchor = **M0 (univariate NDMI logit)**, LOO AUC 0.682 (CI [0.556, 0.802]). **Regional calibration drift (W1):** pooled spatial AUC 0.385 vs per-block 0.792 — the model ranks salinity *locally* but absolute level must be calibrated per region.
<!-- /V6_SCOPE_AUTO -->

V6 не заменяет карту V5.1, а **дополняет** её количественной, проверяемой оценкой
засоления почвы. Он обучен на отчёте Пачикина–Козыбаевой (2012–2014): **70
georeferenced почвенных разрезов** с измеренной солёностью — это ~6× больше точек
наземной правды, чем было у V5.1 (11). UX дашборда не меняется: новый слой живёт во
вкладке «Проверка и данные», а главная карта остаётся V5.1.

**Что валидировано (честно, с интервалами):**

- **Модель солёности (количественное ядро):** логистическая регрессия измеренной
  солёности (>1 %) по 30 м NDMI на всех 70 точках (27 солёных). LOO AUC **0.682**,
  бутстрап-интервал [0.552, 0.800]; калибровочная AUC 0.77. Связь NDMI↔соль
  лабораторно подтверждена (Spearman ρ ≈ +0.66, p < 1e-9). Балл пригодности =
  1 − P(засоление).
- **Пространственная проверка:** leave-block-out (20 км блоки). Средняя per-block
  AUC **0.792** (знак NDMI→соль положителен в 5/5 проверяемых блоках). Объединённая
  AUC 0.385 ниже — это **дрейф калибровки между районами** (разная базовая доля
  засоления), а не потеря сигнала. Записано как ограничение.
- **Сплошной слой 30 м:** `suitability_index_v6.tif` (непрерывный 0..1) +
  `suitability_zones_v6.tif` (те же коды/цвета ZoneClass, что у V5.1). Покрытие
  **96.9 % области** против ~46 % у композита 10 м. NDMI обрезается по training
  support перед применением (без экстраполяции).
- **Слой неопределённости:** `suitability_uncertainty_v6.tif` — SE вероятности
  засоления (дельта-метод по ковариации коэффициентов).
- **Сравнение с замороженным продуктом:** на 70 точках V6 покрывает 15 как
  не-водные, V5.1 — 13. Детектор засоления (зоны 3/4): чувств. 0.70, спец. 0.80.

**Честный отрицательный результат:** прямой классификатор присутствия саксаула
(NDMI+MSAVI → саксаул) даёт LOO AUC ≈ 0.48 (на уровне случайности при 6
положительных метках) и **демотирован до exploratory, не для решений**. Наука
заякорена на сигнале солёности (n=70), а не на 6 точках саксаула.

### Скрипты V6

| Скрипт | Назначение |
|---|---|
| `scripts/v6/extract_docx_tables.py` | Извлечение всех таблиц + нарратива из DOCX-отчёта (22 МБ) в `data/interim/` |
| `scripts/v6/build_canonical_db.py` | Каноническая БД: `profiles_v6.csv` (70 georef), `soil_layers_v6.csv` (369 слоёв) |
| `scripts/v6/extract_saxaul_labels.py` | Метки саксаула (presence/absence/suppressed) с at-pit gating |
| `scripts/v6/build_ml_dataset.py` | ML-таблица: почва + RS (10 м и 30 м) + V5 zone → `ml_dataset_v6.csv` |
| `scripts/v6/calibrate_thresholds.py` | Пороги по измеренной солёности → `thresholds_v6_calibrated.json` |
| `scripts/v6/train_suitability_model.py` | Модель солёности (валидированная) + exploratory логит саксаула |
| `scripts/v6/build_suitability_index.py` | Сплошные растры пригодности/зон + наземная проверка по 70 точкам |
| `scripts/v6/spatial_validation.py` | Пространственная CV, бутстрап-интервалы, слой неопределённости |
| `scripts/v6/qa_science_audit_v6.py` | QA red-team научного слоя (circularity, AUC sanity, reproducibility) |

### Запуск пайплайна V6

```bash
# Полная пересборка из DOCX + растров + QA red-team (нужны локальные большие входы):
python scripts/run_v6_pipeline.py --all --qa

# Только шаги из tracked-входов (без растров/docx) — работает на чистом checkout:
python scripts/run_v6_pipeline.py

# Флаги: --docx (шаг извлечения), --rasters (растровые шаги 4,7,8), --qa (red-team)
```

Растры (`.tif`) gitignored (регенерируются); таблицы/JSON/MD трекаются. QA red-team
проверяет отсутствие circularity, корректность интервалов AUC, воспроизводимость
площадей зон и наличие честных оговорок в отчётах.

---

## Scientific Limitations

- V5.1 is a rule-based remote-sensing screening product, not a trained habitat model.
- V6 adds a soil-salinity-anchored layer validated on 70 lab profiles (LOO AUC 0.68);
  it is a screening aid with quantified uncertainty, **not** a planting guarantee. Its
  saxaul-specific skill rides on small n (6 positives) and is reported with wide CIs.
- V6 zones 3/4 are a *severity split* on a single validated NDMI axis (the 30 m stack
  lacks V5's separate wet/dry-brine spectral bands and the SCL water mask); soil labels
  are 2012–2014 while the NDMI composite is recent (quasi-stationary assumption).
- `Candidate suitable` is a residual class after excluding known spectral/topographic risk classes.
- Remote-sensing indices are surface proxies and cannot replace root-zone salinity, groundwater depth, survival observations, or field agronomy.
- Current spatial validation has only 11 mapped points and unresolved coordinate authority.
- Soil profiles 12-21 are laboratory-only until coordinates are provided.
- Report correlations with `n`, `p`, and bootstrap sensitivity; do not report them as final accuracy.

---

## Выходные артефакты

| Артефакт | Описание |
|---|---|
| `v5_stats.json` | Сводные статистики: 1 598 800 га по сетке 100 м, 2 132 кластера >=10 га, топ-10, гистограмма |
| `thresholds_v5.json` | Адаптивные пороги P15/P85 для NDMI, NDSI, B8/B12 |
| `suitability_map_v5.html` | Folium screening-карта candidate-зон (Google Satellite) |
| `operational_zones_v5_kazakhstan.geojson` | Candidate-полигоны, обрезанные по границе Казахстана |
| `tasks_index_v5_enriched.csv` | 308 казахстанских операционных ячеек с расстояниями до дорог |
| `tractor_tasks_v5/*.kml` | 308 KML-нарядов для GPS-навигаторов |
| `aralkum_roads.geojson` | 16 244 сегмента дорог OSM для общей логистики |
| `kazakhstan_access_roads.geojson` | 14 915 сегментов OSM-дорог Казахстана для расчета казахстанского подъезда |
| `outputs/science/v5_validation_report.md` | Pilot validation по двум coordinate-source |
| `outputs/science/v5_coordinate_adjudication_report.md` | Coordinate authority appendix и review template |
| `outputs/science/v5_uncertainty_report.md` | Threshold sensitivity и stability diagnostics |

---

## История версий

| Версия | Описание |
|---|---|
| V1.0 | Baseline: BBOX AOI, SI-based salinity |
| V2.0 | Синтетические метки, XGBoost |
| V3.0–V3.2 | NDMI pivot, slope filter |
| V4.0–V4.1 | Coastline mask, OSM roads, 782 KML |
| **V5.1** | **10 м screening baseline, adaptive thresholds, provenance, validation, uncertainty diagnostics** |
| **V6** | **Слой солёности по 70 лаб. профилям: валидированная модель (LOO AUC 0.68), сплошной слой 30 м, пространственная CV, неопределённость** |

---

## License

Разработано для инициативы по восстановлению экосистемы Аральского моря.

Данные: OpenStreetMap (ODbL), Copernicus (free), NASA SRTM (public domain), JRC Global Surface Water.

*Built with: Google Earth Engine · Sentinel-2 · GeoPandas · Folium · Streamlit · scipy · Plotly · matplotlib*
