"""Matplotlib visualizations used in EDA and reporting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def plot_missing_values(df: pd.DataFrame, top_n: int = 20) -> None:
    missing = df.isna().mean().sort_values(ascending=False).head(top_n) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top-{top_n} признаков по доле пропусков")
    ax.set_xlabel("Пропуски, %")
    ax.set_ylabel("Признак")
    plt.tight_layout()


def plot_target_balance(y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    y.value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Баланс целевого класса RainTomorrow")
    ax.set_xlabel("0 = No, 1 = Yes")
    ax.set_ylabel("Доля")
    plt.tight_layout()


def plot_numeric_correlation(df: pd.DataFrame, target: str = "RainTomorrow") -> None:
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        return
    corr = numeric.corr(numeric_only=True)[target].drop(target).sort_values(key=lambda x: x.abs())
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.tail(15).plot(kind="barh", ax=ax)
    ax.set_title("Числовые признаки с наибольшей корреляцией с RainTomorrow")
    ax.set_xlabel("Корреляция")
    plt.tight_layout()


def plot_roc_pr_curves(model, X: pd.DataFrame, y: pd.Series, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.set_title(f"ROC-кривая {title_suffix}".strip())
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax)
    ax.set_title(f"Precision-Recall кривая {title_suffix}".strip())
    plt.tight_layout()


def plot_confusion(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> None:
    from src.evaluate import predict_proba_positive

    y_score = predict_proba_positive(model, X)
    y_pred = (y_score >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax)
    ax.set_title(f"Confusion matrix, threshold={threshold:.2f}")
    plt.tight_layout()
