"""Evaluation helpers for binary rain prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    """Return probability or decision scores for the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def find_best_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing F1 on validation data."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        scores.append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def evaluate_binary_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Compute metrics used in the experiments table."""
    y_score = predict_proba_positive(model, X)
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y, y_score),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(y, y_score)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def metrics_to_row(model_name: str, split: str, metrics: dict, threshold: float) -> dict:
    """Convert metric dictionary into one table row."""
    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
    }
