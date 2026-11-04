import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kriterion.metrics import (
    classification_metrics,
    confusion_counts,
    regression_metrics,
    roc_curve_manual,
    trapezoidal_auc,
)


def test_regression_metrics_match_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 4.0, 3.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mse"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["r2"] == pytest.approx(0.6)


def test_confusion_counts_use_positive_class_one():
    counts = confusion_counts(
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([0, 1, 0, 1, 0, 1]),
    )
    assert counts == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


def test_classification_metrics_match_manual_calculation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.8, 0.2, 0.9, 0.3, 0.7])
    y_pred = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(y_true, y_pred, probabilities)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["auc"] == pytest.approx(roc_auc_score(y_true, probabilities))


def test_manual_roc_starts_and_ends_at_unit_square_corners():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])
    fpr, tpr, thresholds = roc_curve_manual(y_true, probabilities)
    assert (fpr[0], tpr[0], thresholds[0]) == (0.0, 0.0, np.inf)
    assert (fpr[-1], tpr[-1], thresholds[-1]) == (1.0, 1.0, -np.inf)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)


def test_trapezoidal_auc_for_unit_triangle():
    auc = trapezoidal_auc(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    )
    assert auc == pytest.approx(0.75)

