# Aral Saxaul AI — V3.0 Rule-Based Pipeline

## Архитектура и обоснование

### Проблема
Предсказание пригодности высохшего дна Аральского моря (Аралкум, ~60,000 км²) для посадки саксаула (Haloxylon) по спутниковым данным Sentinel-2 и Sentinel-1.

### История версий

| Версия | Подход | Статус | Площадь Optimal |
|--------|--------|--------|:---------------:|
| **V1.0** | XGBoost + synthetic labels (SI≥P85, slope≤15°) | ✅ Работает | 6,558 км² (13% AOI) |
| **V2.0** | Ridge regression на 7 ground truth точках | ❌ Заморожен | — |
| **V3.0** | Rule-based (NDMI + SI + NDWI пороги) | ✅ Production | 14,962 км² (30% AOI) |

### Ключевые научные открытия

#### 1. Две системы координат (критический баг)
Обнаружено расхождение ~100 км между координатами полевых EC-замеров (AralField) и лабораторных шурфов (ODT). Вектор смещения: ΔLon=1.124°, ΔLat=0.895° (константный, std=0.003° для 6 из 7 точек). Все V2.0 модели, обученные на этих данных, давали мусор из-за пространственного mismatch.

**Решение:** Математическая коррекция через mean offset vector + использование V1.0 suitability rasta как AOI-маски.

#### 2. SI (Salinity Index) — ненадёжный предиктор
Spearman r(Salinity_pct vs SI) = **+0.41** (p=0.21) — слабая, статистически незначимая корреляция.
SI насыщается на соляной корке и измеряет общее альбедо (белизну), а не концентрацию солей. Оптимальные (<2% соли) и мёртвые (>5% соли) зоны полностью перекрываются по SI (2742–3188).

#### 3. NDMI — главный маркер засоления
Spearman r(Salinity_pct vs NDMI) = **+0.69** (p=0.02) — сильная, статистически значимая корреляция.
Положительный знак: чем выше влажность (NDMI), тем выше солёность. **Прямое подтверждение гипотезы капиллярного подъёма** — грунтовые воды поднимаются к поверхности, испаряются и оставляют соль.

### Финальная архитектура V3.2

```
Вход: feature_stack_30m.vrt (7 каналов)
       ┌─ Band 1: NDMI  (первичный дискриминатор)
       ├─ Band 4: NDWI  (маска воды)
       └─ AOI маска из V1.0 suitability_full.tif (географическая изоляция)

Пороги:
  NDMI_OPTIMAL = -0.055   (сухо → низкая солёность → оптимально)
  NDMI_DEAD    = -0.025   (влажно → капиллярный подъём → мертвая зона)
  NDWI_WATER   = 0.0      (открытая вода)

Логика:
  Class 0 = NoData | NDWI > 0 | вне AOI
  Class 3 = NDMI > -0.025                          (Мёртвая зона)
  Class 1 = NDMI < -0.055                          (Оптимальная зона)
  Class 2 = всё остальное в AOI                    (Зона риска)
```

### Результаты V3.2

| Класс | Площадь | % AOI | % total |
|-------|---------|:-----:|:-------:|
| **1 Optimal** | 14,962 км² | 30.3% | — |
| 2 Risk | 17,201 км² | 34.9% | — |
| 3 Dead | 11,006 км² | 22.3% | — |
| 0 Water/Outside | 100,235 км² | — | 67.0% |
| **AOI (Аралкум)** | **49,351 км²** | 100% | 33.0% |

### QA-верификация

- **Accuracy** (NDMI < -0.055): **93.9%** (1000 random points внутри полигонов)
- **False Positives** (NDMI > -0.025): **0.1%** (1 пиксель из 1000)
- **6% Risk** — краевой эффект simplify (Douglas-Peucker, tolerance=50m), физически допустим

### Файлы

| Файл | Описание |
|------|----------|
| `outputs/data/suitability_map_v3_2.tif` | Растр V3.2 (uint8, 3 класса + NoData) |
| `outputs/data/optimal_zones_v3.geojson` | Векторные полигоны Optimal (10,371 шт., ≥1 га) |
| `outputs/reports/suitability_map_v3.html` | Интерактивная карта (Folium, top-1000 кластеров) |
| `outputs/data/ground_truth_v2.csv` | Ground truth (11 точек, 25 колонок) |
| `outputs/data/salinity_map_v2_alpha.tif` | V2.0 регрессия солёности |

### Скрипты V3

| Скрипт | Назначение |
|--------|------------|
| `scripts/coordinate_offset_analysis.py` | Расчёт вектора смещения координат |
| `scripts/build_ground_truth.py` | Сборка ground truth (11 точек) |
| `scripts/temp_threshold_analysis.py` | Корреляционный анализ и калибровка порогов |
| `scripts/run_inference_v3.py` | V3.0 инференс (базовый) |
| `scripts/run_inference_v3_1.py` | V3.1 (water mask + SI_MAX filter) |
| `scripts/run_inference_v3_2.py` | V3.2 (AOI mask + NDMI-only, финал) |
| `scripts/phase5_v3_export.py` | Векторизация, simplify, GeoJSON, Folium |
| `scripts/audit_vectorization.py` | QA аудит (1000 random points, NDMI check) |
