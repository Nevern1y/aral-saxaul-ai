import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Aral Saxaul — XAI Dashboard", layout="wide")

import numpy as np
import pandas as pd
import joblib
import json
import base64
import os
from pathlib import Path
import xgboost as xgb

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "xgb_classifier.pkl"
SCALER_PATH = BASE_DIR / "outputs" / "models" / "scaler.pkl"
LABELS_PATH = BASE_DIR / "outputs" / "data" / "synthetic_labels.csv"
MAP_PATH = BASE_DIR / "outputs" / "reports" / "suitability_map_full.html"
FI_PATH = BASE_DIR / "outputs" / "data" / "feature_importance.csv"
SHAP_SUMMARY = BASE_DIR / "outputs" / "reports" / "shap_summary.png"
SHAP_DEP_SI = BASE_DIR / "outputs" / "reports" / "shap_dependence_si.png"
SHAP_DEP_MSAVI = BASE_DIR / "outputs" / "reports" / "shap_dependence_msavi.png"

FEATURE_COLUMNS = ["NDMI", "MSAVI", "SI", "Slope", "TWI", "VH", "NDWI"]
FEATURE_LABELS = {
    "NDMI": "NDMI (Влажность почвы)",
    "MSAVI": "MSAVI (Растительность)",
    "SI": "SI (Засоленность)",
    "Slope": "Slope (Уклон, \u00b0)",
    "TWI": "TWI (Влажность рельефа)",
    "VH": "VH (Радар VH, dB)",
    "NDWI": "NDWI (Водный индекс)",
}
FEATURE_RANGES = {
    "NDMI": (-1.0, 1.0, 0.01),
    "MSAVI": (-1.0, 1.0, 0.01),
    "SI": (-1.0, 1.0, 0.01),
    "Slope": (0.0, 45.0, 0.1),
    "TWI": (0.0, 20.0, 0.1),
    "VH": (-60.0, 0.0, 0.5),
    "NDWI": (-1.0, 1.0, 0.01),
}


def _require(path: Path, label: str) -> None:
    if not path.exists():
        st.error(f"{label} не найден: {path}")
        st.stop()


@st.cache_resource
def load_model():
    _require(MODEL_PATH, "Модель")
    model = joblib.load(MODEL_PATH)
    try:
        model.set_params(device="cpu", predictor="cpu_predictor")
    except Exception:
        pass
    return model


@st.cache_resource
def load_scaler():
    _require(SCALER_PATH, "Scaler")
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_test_data():
    _require(LABELS_PATH, "Тестовый датасет")
    return pd.read_csv(LABELS_PATH)


@st.cache_data
def load_feature_importance():
    _require(FI_PATH, "Feature importance")
    return pd.read_csv(FI_PATH)


@st.cache_data
def get_slider_ranges():
    df = load_test_data()
    ranges = {}
    for col in FEATURE_COLUMNS:
        mn = float(df[col].min())
        mx = float(df[col].max())
        step = float(FEATURE_RANGES[col][2])
        if abs(mx - mn) < step:
            mx = mn + step * 2
        ranges[col] = {
            "min": mn if mn < mx else mx - step * 4,
            "max": mx,
            "mean": float(df[col].mean()),
            "step": step,
        }
    return ranges


@st.cache_data
def compute_xgb_shap_values():
    df = load_test_data()
    scaler = load_scaler()
    model = load_model()
    booster = model.get_booster()
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    X_scaled = scaler.transform(X)
    n_sample = min(2000, len(X))
    idx = np.random.default_rng(42).choice(len(X), n_sample, replace=False)
    X_sub = X_scaled[idx]
    dmat = xgb.DMatrix(X_sub, feature_names=FEATURE_COLUMNS)
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]
    base_val = contribs[0, -1]
    return shap_vals, X_sub, base_val


def plot_waterfall(shap_row, base_val, data_row, feature_names):
    n = len(shap_row)
    idx_sorted = np.argsort(np.abs(shap_row))[::-1]
    sorted_vals = shap_row[idx_sorted]
    sorted_names = [feature_names[i] for i in idx_sorted]
    sorted_data = [data_row[i] for i in idx_sorted]

    cumsum = base_val + np.cumsum(sorted_vals)
    f_x = base_val + sorted_vals.sum()

    y_pos = np.arange(n)[::-1]
    colors = ["#E74C3C" if v < 0 else "#3498DB" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(y_pos, sorted_vals, color=colors, height=0.6)

    for i, (v, name, dv) in enumerate(zip(sorted_vals, sorted_names, sorted_data)):
        label = f"{name} = {dv:.4f}"
        ax.text(
            v + (0.02 if v >= 0 else -0.02),
            y_pos[i],
            label,
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )

    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * n)
    ax.set_xlabel("SHAP value (log-odds contribution)")
    ax.set_title(f"Waterfall Plot  |  f(x) = {f_x:.3f}  |  base = {base_val:.3f}")
    ax.axvline(base_val, color="gray", linestyle="--", linewidth=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#3498DB", label="Повышает пригодность"),
        plt.Rectangle((0, 0), 1, 1, color="#E74C3C", label="Снижает пригодность"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


st.title("Aral Saxaul — XAI Dashboard")
st.markdown(
    "Интерактивный дашборд объяснимого ИИ для карты пригодности посадки саксаула "
    "на высохшем дне Аральского моря"
)

tab1, tab2, tab3 = st.tabs([
    "Карта пригодности",
    "Глобальная аналитика",
    "Симулятор What-If",
])

# ── Tab 1: Map ──

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Пригодная площадь", "6 558 км\u00B2")
    col2.metric("Доля от суши", "13.3%")
    col3.metric("Полигонов", "597")
    col4.metric("Площадь суши AOI", "49 296 км\u00B2")

    if MAP_PATH.exists():
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_html = f.read()
        b64 = base64.b64encode(map_html.encode()).decode()
        st.markdown(
            f'<iframe src="data:text/html;base64,{b64}" '
            f'width="100%" height="700" frameborder="0"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            f"Файл карты не найден: {MAP_PATH}. "
            "Запустите `python scripts/phase5_full.py` для генерации."
        )

# ── Tab 2: Global Analysis ──

with tab2:
    st.subheader("Глобальная важность признаков")
    st.markdown(
        "Feature importance на основе Gain (прирост точности при расщеплении) – "
        "показывает, на какие признаки модель опирается больше всего."
    )

    fi_df = load_feature_importance()
    fi_df_sorted = fi_df.sort_values("importance_gain", ascending=True)

    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.markdown("**Важность признаков (Gain)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(fi_df_sorted["feature"], fi_df_sorted["importance_gain"], color="#2E86AB")
        ax.set_xlabel("Gain")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, fi_df_sorted["importance_gain"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                    va="center", fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("**Интерпретация**")
        si_gain = fi_df_sorted.loc[fi_df_sorted["feature"] == "SI", "importance_gain"].values[0]
        ndwi_gain = fi_df_sorted.loc[fi_df_sorted["feature"] == "NDWI", "importance_gain"].values[0]
        st.markdown(
            f"- **SI (Засоленность)** — **{si_gain:.0%}** — главный фактор\n"
            f"- **NDWI (Водный индекс)** — **{ndwi_gain:.0%}** — вторичный сигнал влажности\n"
            f"- **NDMI / MSAVI** — ~11% суммарно — вегетация и влажность почвы\n"
            f"- **Slope / TWI / VH** — <2% — рельеф и радар почти не влияют\n\n"
            "Модель подтверждает, что засоление — ключевой лимитирующий фактор "
            "для приживаемости саксаула на высохшем дне Арала."
        )

    st.markdown("---")
    st.subheader("SHAP-анализ (предвычисленные графики)")

    col_dep1, col_dep2 = st.columns(2)

    with col_dep1:
        st.markdown("**SHAP Summary — влияние всех признаков**")
        if SHAP_SUMMARY.exists():
            st.image(str(SHAP_SUMMARY), width='stretch')
        else:
            st.warning("Файл shap_summary.png не найден в outputs/reports/")
        st.caption(
            "Каждая точка — одно наблюдение. Цвет — значение признака "
            "(красный = высокое, синий = низкое). "
            "Положительный SHAP = повышает вероятность пригодности."
        )

    with col_dep2:
        st.markdown("**SI (Засоленность) — зависимость SHAP**")
        if SHAP_DEP_SI.exists():
            st.image(str(SHAP_DEP_SI), width='stretch')
        else:
            st.warning("Файл shap_dependence_si.png не найден")
        st.caption(
            "По оси X — значение SI, по оси Y — SHAP value. "
            "Показывает, при каких значениях SI засоление становится "
            "критическим фактором."
        )

    col3_1, col3_2 = st.columns(2)

    with col3_1:
        st.markdown("**MSAVI — зависимость**")
        if SHAP_DEP_MSAVI.exists():
            st.image(str(SHAP_DEP_MSAVI), width='stretch')
        else:
            st.warning("Файл shap_dependence_msavi.png не найден")

    with col3_2:
        st.info(
            "**Физика модели:** SI — главный признак (62% важности). "
            "Графики SHAP показывают направление влияния. "
            "NDWI и NDMI работают как вторичные сигналы влажности. "
            "Slope, TWI и VH имеют минимальный вклад."
        )

# ── Tab 3: What-If Simulator ──

with tab3:
    st.subheader("Интерактивный симулятор пригодности")
    st.markdown(
        "Меняйте ползунки и наблюдайте, как меняется предсказание "
        "и какой вклад вносит каждый признак."
    )

    try:
        ranges = get_slider_ranges()
        model = load_model()
        scaler = load_scaler()
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.stop()

    col_sliders, col_result = st.columns([1, 1.5])

    with col_sliders:
        sliders = {}
        sliders["NDMI"] = st.slider(
            FEATURE_LABELS["NDMI"],
            min_value=ranges["NDMI"]["min"],
            max_value=ranges["NDMI"]["max"],
            value=ranges["NDMI"]["mean"],
            step=ranges["NDMI"]["step"],
        )
        sliders["MSAVI"] = st.slider(
            FEATURE_LABELS["MSAVI"],
            min_value=ranges["MSAVI"]["min"],
            max_value=ranges["MSAVI"]["max"],
            value=ranges["MSAVI"]["mean"],
            step=ranges["MSAVI"]["step"],
        )
        sliders["SI"] = st.slider(
            FEATURE_LABELS["SI"],
            min_value=ranges["SI"]["min"],
            max_value=ranges["SI"]["max"],
            value=ranges["SI"]["mean"],
            step=ranges["SI"]["step"],
        )
        sliders["Slope"] = st.slider(
            FEATURE_LABELS["Slope"],
            min_value=ranges["Slope"]["min"],
            max_value=ranges["Slope"]["max"],
            value=ranges["Slope"]["mean"],
            step=ranges["Slope"]["step"],
        )
        sliders["TWI"] = st.slider(
            FEATURE_LABELS["TWI"],
            min_value=ranges["TWI"]["min"],
            max_value=ranges["TWI"]["max"],
            value=ranges["TWI"]["mean"],
            step=ranges["TWI"]["step"],
        )
        sliders["VH"] = st.slider(
            FEATURE_LABELS["VH"],
            min_value=ranges["VH"]["min"],
            max_value=ranges["VH"]["max"],
            value=ranges["VH"]["mean"],
            step=ranges["VH"]["step"],
        )
        sliders["NDWI"] = st.slider(
            FEATURE_LABELS["NDWI"],
            min_value=ranges["NDWI"]["min"],
            max_value=ranges["NDWI"]["max"],
            value=ranges["NDWI"]["mean"],
            step=ranges["NDWI"]["step"],
        )

    with col_result:
        sample = np.array([[sliders[c] for c in FEATURE_COLUMNS]], dtype=np.float64)
        sample_scaled = scaler.transform(sample)

        prob = float(model.predict_proba(sample_scaled)[0, 1])
        pred_class = "Пригодно" if prob >= 0.5 else "Непригодно"

        st.metric("Вероятность пригодности", f"{prob:.1%}")
        st.markdown(f"**Решение модели:** {pred_class}")

        proba_display = pd.DataFrame(
            {"Класс": ["Непригодно", "Пригодно"], "Вероятность": [f"{1-prob:.1%}", f"{prob:.1%}"]}
        )
        st.dataframe(proba_display, hide_index=True)

        st.markdown("---")
        st.markdown("**Вклад признаков (SHAP Waterfall)**")

        booster = model.get_booster()
        dmat = xgb.DMatrix(sample_scaled, feature_names=FEATURE_COLUMNS)
        contribs = booster.predict(dmat, pred_contribs=True)
        shap_row = contribs[0, :-1]
        base_val = float(contribs[0, -1])

        nan_mask = np.isnan(shap_row)
        if nan_mask.any():
            shap_row = shap_row.copy()
            shap_row[nan_mask] = 0.0
            st.caption(
                "Примечание: VH имеет нулевую дисперсию в обучающих данных. "
                "Его вклад в предсказание равен нулю."
            )

        fig = plot_waterfall(shap_row, base_val, sample[0], FEATURE_COLUMNS)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Как читать:**")
        st.markdown(
            "- Синие полосы = признак **повышает** вероятность пригодности\n"
            "- Красные полосы = признак **снижает** вероятность пригодности\n"
            "- Длина полосы = сила влияния (в log-odds)\n"
            "- **f(x)** = итоговый log-odds модели → конвертируется в вероятность"
        )
