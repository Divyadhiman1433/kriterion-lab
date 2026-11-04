from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(errors**2)) / denominator
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2}


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | int]:
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    fpr, tpr, _ = roc_curve_manual(y_true, probabilities)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": trapezoidal_auc(fpr, tpr),
        **counts,
    }


def roc_curve_manual(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_y = y_true[order]
    sorted_probabilities = probabilities[order]
    threshold_indices = np.flatnonzero(
        np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True]
    )
    cumulative_tp = np.cumsum(sorted_y)
    cumulative_fp = np.cumsum(1 - sorted_y)
    tp = cumulative_tp[threshold_indices]
    fp = cumulative_fp[threshold_indices]
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    thresholds = np.r_[np.inf, sorted_probabilities[threshold_indices], -np.inf]
    return fpr, tpr, thresholds


def trapezoidal_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    order = np.argsort(fpr, kind="mergesort")
    x = fpr[order]
    y = tpr[order]
    widths = np.diff(x)
    heights = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(widths * heights))
