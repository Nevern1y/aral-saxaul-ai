"""
Phase 3: Model Training & Explainable AI (XAI).

Classes
-------
DatasetBuilder
    Clean raw synthetic data, train/test split, StandardScaler.

OptunaOptimizer
    Bayesian hyperparameter search maximising **Precision (Class 1)**
    with hard complexity caps to prevent overfitting on synthetic labels.

ModelTrainer
    Orchestrator: final training, evaluation, SHAP analysis,
    and artifact persistence.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_have_np_trapz = hasattr(np, "trapz")
if not _have_np_trapz:
    np.trapz = np.trapezoid
import optuna
import pandas as pd
import shap
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.utils import logger

# Suppress verbose optuna logging unless debugging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Non-interactive matplotlib backend for server/headless environments
matplotlib.use("Agg")


# ============================================================================
# Constants
# ============================================================================

FEATURE_COLS: List[str] = list(config.FEATURE_COLUMNS)
META_COLS: List[str] = list(config.META_COLUMNS)
TARGET_COL: str = "label"

RANDOM_STATE: int = config.XGB_RANDOM_STATE


# ============================================================================
# DatasetBuilder
# ============================================================================


class DatasetBuilder:
    """Clean, split, and scale the synthetic training dataset.

    Actions (in order)
    -------------------
    1. Detect and drop rows with NaN / ±Inf.
    2. Log class balance - target is 1:1; warn if skewed.
    3. Train / test split (stratified, 80/20).
    4. Fit ``StandardScaler`` on the training set; transform both splits.
    5. Export scaler for use in Phase 4 (distributed inference).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = TARGET_COL,
        test_size: float = config.TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Args:
            df: Raw synthetic DataFrame from Phase 2.
            feature_cols: Ordered feature column names (default: config.FEATURE_COLUMNS).
            target_col: Label column name.
            test_size: Fraction of data held out for testing.
            random_state: RNG seed for split.
        """
        self.raw_df = df
        self.feature_cols = feature_cols or FEATURE_COLS
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        # Results populated by _prepare()
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.cleaned_df: Optional[pd.DataFrame] = None

        logger.info("DatasetBuilder initialised - %d raw rows.", len(df))

    # ── Public API ─────────────────────────────────────────────────────

    def prepare(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the full preprocessing pipeline.

        Returns:
            ``(X_train, X_test, y_train, y_test)`` as numpy float64 arrays.
        """
        t0 = time.time()

        self._clean()
        self._check_balance()
        self._split_and_scale()

        elapsed = time.time() - t0
        logger.info(
            "Preprocessing complete: train=(%d,%d) test=(%d,%d) (%.1f s).",
            self.X_train.shape[0],
            self.X_train.shape[1],
            self.X_test.shape[0],
            self.X_test.shape[1],
            elapsed,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def export_scaler(self, path: Optional[str] = None) -> str:
        """Persist the fitted StandardScaler.

        Args:
            path: Destination path. Defaults to ``outputs/models/scaler.pkl``.

        Returns:
            Absolute path to the saved file.
        """
        if self.scaler is None:
            raise RuntimeError("Call .prepare() before exporting the scaler.")

        if path is None:
            path = str(config.output_dir / "models" / "scaler.pkl")

        joblib.dump(self.scaler, path)
        logger.info("Scaler exported: %s", path)
        return path

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names (post-cleaning)."""
        return list(self.feature_cols)

    # ── Internal ───────────────────────────────────────────────────────

    def _clean(self) -> None:
        """Remove NaN, ±Inf, and extreme outliers from the dataset."""
        df = self.raw_df.copy()

        # Replace Inf with NaN then drop
        df = df.replace([np.inf, -np.inf], np.nan)

        n_before = len(df)
        df = df.dropna(subset=self.feature_cols + [self.target_col])
        n_dropped = n_before - len(df)

        if n_dropped > 0:
            logger.warning(
                "_clean: dropped %d / %d rows (%.1f%%) with NaN/Inf.",
                n_dropped,
                n_before,
                100 * n_dropped / n_before,
            )

        # Clip extreme outliers (>6*std from mean) to reduce influence
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            vals = df[col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                lo, hi = mean - 6 * std, mean + 6 * std
                n_clipped = ((vals < lo) | (vals > hi)).sum()
                if n_clipped > 0:
                    df[col] = vals.clip(lo, hi)
                    logger.debug(
                        "_clean: clipped %d outliers in '%s'.", n_clipped, col
                    )

        self.cleaned_df = df
        logger.info("Cleaned dataset: %d rows, %d features.", len(df), len(self.feature_cols))

    def _check_balance(self) -> Dict[str, Any]:
        """Log class distribution; warn on heavy imbalance."""
        counts = self.cleaned_df[self.target_col].value_counts().to_dict()

        n_pos = counts.get(1, 0)
        n_neg = counts.get(0, 0)
        total = n_pos + n_neg
        ratio = n_pos / total if total > 0 else 0.0

        logger.info("Class balance: %d positive (%.1f%%), %d negative (%.1f%%).",
                     n_pos, 100 * ratio, n_neg, 100 * (1 - ratio))

        if ratio < 0.35 or ratio > 0.65:
            logger.warning(
                "Class imbalance detected (P=%.1f%%). "
                "Consider adjusting sampling or using scale_pos_weight.",
                100 * ratio,
            )

        return {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_ratio": ratio,
        }

    def _split_and_scale(self) -> None:
        """Stratified train/test split + StandardScaler fit/transform."""
        X = self.cleaned_df[self.feature_cols].values.astype(np.float64)
        y = self.cleaned_df[self.target_col].values.astype(np.int64)

        if len(X) < 10:
            raise ValueError(
                f"Dataset too small for split: {len(X)} rows. Check Phase 2 output."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
            shuffle=True,
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logger.info(
            "Split: train %d / test %d. Scaler fitted (mean +/- std).",
            len(X_train),
            len(X_test),
        )


# ============================================================================
# OptunaOptimizer
# ============================================================================


class OptunaOptimizer:
    """Bayesian hyperparameter optimisation for XGBoost via Optuna.

    Search space (tightly constrained for synthetic-data regime)
    ------------------------------------------------------------
    ====================  ========  =========  ===================
    Parameter             Min       Max        Distribution
    ====================  ========  =========  ===================
    max_depth             3         7          int, step=1
    learning_rate         0.01      0.2        float, log-uniform
    n_estimators          50        300        int, step=25
    subsample             0.5       1.0        float, uniform
    colsample_bytree      0.5       1.0        float, uniform
    reg_alpha             1e-8      5.0        float, log-uniform
    reg_lambda            1e-8      5.0        float, log-uniform
    min_child_weight      1         20         int, step=1
    ====================  ========  =========  ===================

    Objective
    --------
    Maximise **mean precision (Class 1)** across 5-fold stratified CV.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = config.OPTUNA_N_TRIALS,
        timeout_sec: int = config.OPTUNA_TIMEOUT_SEC,
        cv_folds: int = config.CV_FOLDS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Args:
            X_train: Scaled training features.
            y_train: Training labels (0/1).
            n_trials: Max Optuna trials.
            timeout_sec: Max wall-clock time for optimisation.
            cv_folds: Number of CV folds.
            random_state: Seed.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.timeout_sec = timeout_sec
        self.cv_folds = cv_folds
        self.random_state = random_state

        self._study: Optional[optuna.Study] = None
        self._best_params: Optional[Dict[str, Any]] = None

        # Detect GPU availability
        self._gpu_available = self._check_gpu()

        logger.info(
            "OptunaOptimizer initialised: %d trials, %d-fold CV, GPU=%s.",
            n_trials,
            cv_folds,
            self._gpu_available,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def optimize(self, show_progress: bool = True) -> optuna.Study:
        """Run the hyperparameter search.

        Args:
            show_progress: If True, show tqdm progress bar.

        Returns:
            Completed ``optuna.Study`` object.
        """
        logger.info("Starting Optuna hyperparameter search …")
        t0 = time.time()

        # Direction: maximise precision (class 1)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            study_name="aral_saxaul_xgb",
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout_sec,
            show_progress_bar=show_progress,
            n_jobs=1,  # single-thread optuna; XGBoost uses GPU internally
        )

        self._study = study
        self._best_params = study.best_params.copy()
        # Inject fixed params that are not tuned
        self._best_params.update(self._fixed_params())

        elapsed = time.time() - t0
        logger.info(
            "Optuna complete - best precision: %.4f (trial #%d, %.1f s).",
            study.best_value,
            study.best_trial.number,
            elapsed,
        )
        logger.info("Best params: %s", self._best_params)

        return study

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameter dict (including fixed GPU/seed params)."""
        if self._best_params is None:
            raise RuntimeError("Call optimize() first.")
        return self._best_params.copy()

    def get_study(self) -> optuna.Study:
        """Return the Optuna study object."""
        if self._study is None:
            raise RuntimeError("Call optimize() first.")
        return self._study

    def plot_optimization_history(self, save_path: Optional[str] = None) -> str:
        """Save Optuna's optimization-history plot.

        Args:
            save_path: Output path. Defaults to ``outputs/reports/optuna_history.png``.

        Returns:
            Path to the saved plot.
        """
        if self._study is None:
            raise RuntimeError("Call optimize() first.")

        if save_path is None:
            save_path = str(config.output_dir / "reports" / "optuna_history.png")

        obj = optuna.visualization.matplotlib.plot_optimization_history(self._study)
        if isinstance(obj, plt.Axes):
            fig = obj.figure
        else:
            fig = obj
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Optuna history plot saved: %s", save_path)
        return save_path

    def plot_param_importances(self, save_path: Optional[str] = None) -> str:
        """Save Optuna's parameter-importance plot."""
        if self._study is None:
            raise RuntimeError("Call optimize() first.")

        if save_path is None:
            save_path = str(config.output_dir / "reports" / "optuna_param_importance.png")

        obj = optuna.visualization.matplotlib.plot_param_importances(self._study)
        if isinstance(obj, plt.Axes):
            fig = obj.figure
        else:
            fig = obj
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Param importance plot saved: %s", save_path)
        return save_path

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _check_gpu() -> bool:
        """Probe whether XGBoost can see a CUDA device."""
        try:
            test_model = xgb.XGBClassifier(
                n_estimators=1,
                max_depth=2,
                tree_method="hist",
                device="cuda",
            )
            X_tiny = np.random.randn(10, 2)
            y_tiny = np.random.randint(0, 2, 10)
            test_model.fit(X_tiny, y_tiny)
            del test_model
            gc.collect()
            logger.info("GPU check: XGBoost CUDA backend is AVAILABLE.")
            return True
        except Exception:
            logger.warning(
            "GPU check: CUDA backend UNAVAILABLE - falling back to CPU (hist). "
                "Training will be slower."
            )
            return False

    def _fixed_params(self) -> Dict[str, Any]:
        """Return parameters that are NOT tuned but always set."""
        return {
            "tree_method": "hist",
            "device": "cuda" if self._gpu_available else "cpu",
            "eval_metric": "logloss",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
            "scale_pos_weight": 1.0,
            "early_stopping_rounds": None,  # early stopping handled in CV
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: 5-fold CV mean precision (Class 1).

        Returns:
            Mean precision score across folds.
        """
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7, step=1),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 5.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 5.0, log=True
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20, step=1),
        }
        params.update(self._fixed_params())

        model = xgb.XGBClassifier(**params)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring="precision",
            n_jobs=1,  # XGBoost GPU doesn't parallelise well across CV folds
            error_score="raise",
        )

        return float(np.mean(scores))


# ============================================================================
# ModelTrainer
# ============================================================================


class ModelTrainer:
    """Train final XGBoost model, evaluate, and produce SHAP explanations.

    Outputs (written to ``outputs/``)
    ---------------------------------
    ==================================  ====================================
    Artifact                            Path
    ==================================  ====================================
    Trained XGBoost model               ``models/xgb_classifier.pkl``
    Feature importance (CSV)            ``data/feature_importance.csv``
    Classification report (TXT)         ``reports/classification_report.txt``
    Confusion matrix (PNG)              ``reports/confusion_matrix.png``
    SHAP summary plot (PNG)             ``reports/shap_summary.png``
    SHAP dependence plots (PNG)         ``reports/shap_dependence_{feat}.png``
    ==================================  ====================================
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        best_params: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            X_train: Scaled training features.
            X_test: Scaled test features.
            y_train: Training labels.
            y_test: Test labels.
            feature_names: Ordered list of feature names.
            best_params: XGBoost hyperparameters from Optuna.
            output_dir: Root output directory (default: config.output_dir).
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.best_params = best_params

        self.output_dir = output_dir or config.output_dir
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"

        for d in [self.models_dir, self.reports_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._model: Optional[xgb.XGBClassifier] = None
        self._metrics: Optional[Dict[str, Any]] = None

        logger.info("ModelTrainer initialised - %d features.", len(feature_names))

    # ── Public API ─────────────────────────────────────────────────────

    def run(self) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Execute the full training -> evaluation -> SHAP pipeline.

        Returns:
            ``(trained_model, metrics_dict)``.
        """
        t0 = time.time()

        # 1. Train final model
        self._train_final_model()

        # 2. Evaluate
        self._metrics = self._evaluate()

        # 3. Feature importance
        self._save_feature_importance()

        # 4. SHAP
        self._compute_shap()

        # 5. Classification report
        self._save_classification_report()

        elapsed = time.time() - t0
        logger.info("Phase 3 complete (%.1f s).", elapsed)

        return self._model, self._metrics

    def get_model(self) -> xgb.XGBClassifier:
        """Return the trained model (call ``run()`` first)."""
        if self._model is None:
            raise RuntimeError("Call run() first.")
        return self._model

    def get_metrics(self) -> Dict[str, Any]:
        """Return the evaluation metrics dict."""
        if self._metrics is None:
            raise RuntimeError("Call run() first.")
        return self._metrics

    # ── Training ───────────────────────────────────────────────────────

    def _train_final_model(self) -> None:
        """Train XGBoost on the full training set with best parameters."""
        logger.info("Training final XGBoost model …")

        self._model = xgb.XGBClassifier(**self.best_params)
        self._model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False,
        )

        logger.info("Final model trained.")

    # ── Evaluation ─────────────────────────────────────────────────────

    def _evaluate(self) -> Dict[str, Any]:
        """Compute comprehensive metrics on the test set."""
        y_pred = self._model.predict(self.X_test)
        y_proba = self._model.predict_proba(self.X_test)[:, 1]

        # Precision, Recall, F1
        prec = float(precision_score(self.y_test, y_pred, zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, zero_division=0))
        f1 = float(f1_score(self.y_test, y_pred, zero_division=0))

        # PR-AUC
        prec_curve, rec_curve, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = float(np.trapz(prec_curve[::-1], rec_curve[::-1]))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "pr_auc": pr_auc,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "test_size": len(self.y_test),
        }

        logger.info(
            "Evaluation - P: %.4f  R: %.4f  F1: %.4f  PR-AUC: %.4f",
            prec, rec, f1, pr_auc,
        )
        logger.info(
            "Confusion: TN=%d  FP=%d  FN=%d  TP=%d", tn, fp, fn, tp
        )

        # Confusion matrix plot
        self._plot_confusion_matrix(cm)

        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray) -> str:
        """Save a styled confusion matrix plot."""
        path = str(self.reports_dir / "confusion_matrix.png")

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Unsuitable (0)", "Suitable (1)"], fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Unsuitable (0)", "Suitable (1)"], fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix - Aral Saxaul XGBoost", fontsize=13)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Confusion matrix plot: %s", path)
        return path

    def _save_classification_report(self) -> str:
        """Write sklearn classification_report to a text file."""
        path = str(self.reports_dir / "classification_report.txt")

        y_pred = self._model.predict(self.X_test)
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=["Unsuitable (0)", "Suitable (1)"],
            digits=4,
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write("Classification Report - Aral Saxaul XGBoost\n")
            f.write("=" * 60 + "\n")
            f.write(f"Features: {', '.join(self.feature_names)}\n")
            f.write(f"Test set: {len(self.y_test)} samples\n\n")
            f.write(report)
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"PR-AUC: {self._metrics.get('pr_auc', 'N/A'):.4f}\n")

        logger.info("Classification report: %s", path)
        return path

    # ── Feature importance ─────────────────────────────────────────────

    def _save_feature_importance(self) -> str:
        """Export feature importance (gain-based) to CSV."""
        path = str(self.data_dir / "feature_importance.csv")

        importances = self._model.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance_gain": importances}
        ).sort_values("importance_gain", ascending=False)

        df.to_csv(path, index=False)

        logger.info("Feature importance saved: %s", path)

        # Quick log
        for _, row in df.iterrows():
            logger.info("  %-8s  %.6f", row["feature"], row["importance_gain"])

        return path

    # ── SHAP (Explainable AI) ──────────────────────────────────────────

    def _compute_shap(self) -> None:
        """Compute and save SHAP explanations.

        Uses TreeExplainer (optimised for XGBoost).  Processes a random
        subset of the test set to keep memory/compute tractable.
        """
        logger.info("Computing SHAP values …")
        t0 = time.time()

        # Subsample test set for SHAP (full set may be too large)
        n_samples = min(config.SHAP_SAMPLE_SIZE, len(self.X_test))
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(len(self.X_test), size=n_samples, replace=False)

        X_sample = self.X_test[indices]

        # TreeExplainer is fast and exact for tree-based models
        explainer = shap.TreeExplainer(
            self._model,
            feature_perturbation="interventional",
        )

        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        # Handle binary classification - shap_values might be (n, d) or (n, d, 2)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class-1 SHAP values

        # ── Summary plot ───────────────────────────────────────────
        self._plot_shap_summary(shap_values, X_sample)

        # ── Dependence plots for key features ──────────────────────
        key_features = ["SI", "MSAVI", "NDMI", "VH"]
        for feat in key_features:
            if feat in self.feature_names:
                self._plot_shap_dependence(shap_values, X_sample, feat)

        elapsed = time.time() - t0
        logger.info("SHAP analysis complete (%.1f s).", elapsed)

    def _plot_shap_summary(
        self, shap_values: np.ndarray, X_sample: np.ndarray
    ) -> str:
        """Generate and save the SHAP summary (bee-swarm) plot.

        Args:
            shap_values: (n_samples, n_features) array.
            X_sample: (n_samples, n_features) array.

        Returns:
            Path to the saved plot.
        """
        path = str(self.reports_dir / "shap_summary.png")

        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            show=False,
            plot_type="dot",
            max_display=min(20, len(self.feature_names)),
        )
        fig = plt.gcf()
        fig.axes[-1].set_title("SHAP Summary - Aral Saxaul XGBoost", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        logger.info("SHAP summary plot: %s", path)
        return path

    def _plot_shap_dependence(
        self,
        shap_values: np.ndarray,
        X_sample: np.ndarray,
        feature_name: str,
    ) -> str:
        """Generate a SHAP dependence plot for a single feature.

        Uses the feature with the highest interaction as the colour axis.
        """
        path = str(self.reports_dir / f"shap_dependence_{feature_name.lower()}.png")

        feat_idx = self.feature_names.index(feature_name)

        # Pick the feature with highest mean absolute SHAP interaction as colour axis
        interaction_idx = None
        best_interaction = 0
        for j in range(len(self.feature_names)):
            if j == feat_idx:
                continue
            mean_interaction = np.abs(
                shap_values[:, feat_idx] * X_sample[:, j]
            ).mean()
            if mean_interaction > best_interaction:
                best_interaction = mean_interaction
                interaction_idx = j

        fig, ax = plt.subplots(figsize=(8, 5))

        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            ax=ax,
            show=False,
        )

        ax.set_title(f"SHAP Dependence - {feature_name}", fontsize=13)

        annotations = {
            "SI": "EXPECT: negative SHAP for high SI (salt ↓ suitability)",
            "MSAVI": "EXPECT: positive SHAP for high MSAVI (veg ↑ suitability)",
            "NDMI": "EXPECT: positive SHAP for high NDMI (moisture ↑ suitability)",
            "VH": "MONITOR: direction unclear (roughness ↔ substrate type)",
        }
        if feature_name in annotations:
            ax.text(
                0.5, 1.02,
                annotations[feature_name],
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=8, style="italic", color="grey",
            )

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("SHAP dependence plot for '%s': %s", feature_name, path)
        return path


# ============================================================================
# Phase 3 entrypoint
# ============================================================================


def run_phase3(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    n_optuna_trials: int = config.OPTUNA_N_TRIALS,
) -> Tuple[xgb.XGBClassifier, StandardScaler, Dict[str, Any]]:
    """Convenience function: full Phase 3 pipeline from DataFrame to trained model.

    Args:
        df: Labelled DataFrame from Phase 2.
        output_dir: Output directory.
        n_optuna_trials: Override number of Optuna trials.

    Returns:
        ``(model, scaler, metrics_dict)``.
    """
    out = output_dir or config.output_dir

    # 1. Preprocess
    builder = DatasetBuilder(df)
    X_train, X_test, y_train, y_test = builder.prepare()
    builder.export_scaler()

    # 2. Hyperparameter tuning
    opt = OptunaOptimizer(
        X_train, y_train,
        n_trials=n_optuna_trials,
    )
    study = opt.optimize(show_progress=True)
    best_params = opt.get_best_params()

    opt.plot_optimization_history()
    opt.plot_param_importances()

    # 3. Train & evaluate
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=builder.get_feature_names(),
        best_params=best_params,
        output_dir=out,
    )
    model, metrics = trainer.run()

    # 4. Save model
    model_path = str(out / "models" / "xgb_classifier.pkl")
    joblib.dump(model, model_path)
    logger.info("Model exported: %s", model_path)

    # 5. Save feature names for Phase 4
    feature_path = str(out / "models" / "feature_names.json")
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(builder.get_feature_names(), f, indent=2)
    logger.info("Feature names saved: %s", feature_path)

    return model, builder.scaler, metrics


# ============================================================================
# CLI entrypoint
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: XGBoost Training + Optuna + SHAP"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to synthetic_labels.csv (skips Phase 2).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=config.OPTUNA_N_TRIALS,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip hyperparameter search; use config defaults.",
    )
    args = parser.parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        logger.info("Loaded dataset from %s: %d rows.", args.input_csv, len(df))
    else:
        logger.error(
            "No --input-csv provided. Run Phase 1 + 2 first, "
            "or specify a path to synthetic_labels.csv."
        )
        sys.exit(1)

    run_phase3(df, n_optuna_trials=0 if args.skip_optuna else args.trials)
    logger.info("Phase 3 complete.")
