from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .metrics import classification_metrics, regression_metrics, roc_curve_manual
from .models import LinearRegressionGD, LogisticRegressionGD, ols_univariate

SEED = 42
SELECTED_FEATURE_COUNT = 10


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class ExperimentBundle:
    metadata: dict[str, Any]
    feature_selection: dict[str, Any]
    classification: dict[str, Any]
    regression: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _feature_subsets(
    x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    minmax = MinMaxScaler().fit_transform(x_train)
    filter_selector = SelectKBest(score_func=chi2, k=SELECTED_FEATURE_COUNT)
    filter_selector.fit(minmax, y_train)
    filter_indices = np.flatnonzero(filter_selector.get_support()).tolist()

    standard = StandardScaler().fit_transform(x_train)
    rfe_estimator = LogisticRegression(max_iter=5000, random_state=SEED)
    wrapper_selector = RFE(
        estimator=rfe_estimator,
        n_features_to_select=SELECTED_FEATURE_COUNT,
        step=1,
    )
    wrapper_selector.fit(standard, y_train)
    wrapper_indices = np.flatnonzero(wrapper_selector.get_support()).tolist()

    forest = RandomForestClassifier(
        n_estimators=400,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    forest.fit(x_train, y_train)
    embedded_indices = np.argsort(forest.feature_importances_)[-SELECTED_FEATURE_COUNT:]
    embedded_indices = embedded_indices[np.argsort(-forest.feature_importances_[embedded_indices])]
    embedded_indices = embedded_indices.tolist()

    subsets = {
        "All features": list(range(x_train.shape[1])),
        "Chi-square filter": filter_indices,
        "RFE wrapper": wrapper_indices,
        "Random forest embedded": embedded_indices,
    }
    score_rows = []
    for index, name in enumerate(feature_names):
        score_rows.append(
            {
                "feature": name,
                "chi_square": float(filter_selector.scores_[index]),
                "rfe_rank": int(wrapper_selector.ranking_[index]),
                "random_forest_importance": float(forest.feature_importances_[index]),
            }
        )
    details = {
        "selected_features": {
            method: [feature_names[index] for index in indices]
            for method, indices in subsets.items()
        },
        "selector_scores": score_rows,
        "selected_count": SELECTED_FEATURE_COUNT,
    }
    return subsets, details


def _classification_experiment() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=float)
    y = (np.asarray(dataset.target, dtype=int) == 0).astype(int)
    feature_names = [str(name) for name in dataset.feature_names]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    subsets, selection = _feature_subsets(x_train, y_train, feature_names)
    classification: dict[str, Any] = {}
    comparison_rows = []
    for method, indices in subsets.items():
        scaler = StandardScaler()
        train_selected = scaler.fit_transform(x_train[:, indices])
        test_selected = scaler.transform(x_test[:, indices])

        scratch = LogisticRegressionGD(learning_rate=0.06, epochs=2400)
        scratch.fit(train_selected, y_train)
        scratch_probabilities = scratch.predict_proba(test_selected)
        scratch_predictions = scratch.predict(test_selected)
        scratch_metrics = classification_metrics(
            y_test, scratch_predictions, scratch_probabilities
        )
        fpr, tpr, thresholds = roc_curve_manual(y_test, scratch_probabilities)

        reference = LogisticRegression(max_iter=5000, random_state=SEED)
        reference.fit(train_selected, y_train)
        reference_probabilities = reference.predict_proba(test_selected)[:, 1]
        reference_predictions = reference.predict(test_selected)
        reference_metrics = classification_metrics(
            y_test, reference_predictions, reference_probabilities
        )

        confusion = [
            [scratch_metrics["tn"], scratch_metrics["fp"]],
            [scratch_metrics["fn"], scratch_metrics["tp"]],
        ]
        classification[method] = {
            "n_features": len(indices),
            "features": [feature_names[index] for index in indices],
            "scratch_metrics": scratch_metrics,
            "sklearn_metrics": reference_metrics,
            "confusion_matrix": confusion,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "probabilities": scratch_probabilities,
            "predictions": scratch_predictions,
            "loss_history": scratch.loss_history,
        }
        comparison_rows.append(
            {
                "method": method,
                "n_features": len(indices),
                "scratch_accuracy": scratch_metrics["accuracy"],
                "scratch_auc": scratch_metrics["auc"],
                "sklearn_accuracy": reference_metrics["accuracy"],
                "sklearn_auc": reference_metrics["auc"],
            }
        )
    selection["results"] = comparison_rows
    metadata = {
        "classification_dataset": "Breast Cancer Wisconsin Diagnostic",
        "classification_samples": int(x.shape[0]),
        "classification_features": int(x.shape[1]),
        "classification_train_samples": int(x_train.shape[0]),
        "classification_test_samples": int(x_test.shape[0]),
        "malignant_samples": int(np.sum(y == 1)),
        "benign_samples": int(np.sum(y == 0)),
        "positive_class": "malignant",
        "classification_target_test": y_test,
    }
    return metadata, selection, classification


def _regression_experiment() -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = load_diabetes()
    bmi_index = list(dataset.feature_names).index("bmi")
    x = np.asarray(dataset.data[:, bmi_index], dtype=float)
    y = np.asarray(dataset.target, dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    ols_slope, ols_intercept = ols_univariate(x_train, y_train)
    ols_predictions = ols_slope * x_test + ols_intercept

    x_mean = float(np.mean(x_train))
    x_std = float(np.std(x_train))
    train_scaled = (x_train - x_mean) / x_std
    test_scaled = (x_test - x_mean) / x_std
    gd = LinearRegressionGD(learning_rate=0.08, epochs=1600)
    gd.fit(train_scaled, y_train)
    gd_predictions = gd.predict(test_scaled)
    gd_slope = float(gd.weight / x_std)
    gd_intercept = float(gd.bias - gd.weight * x_mean / x_std)

    reference = LinearRegression()
    reference.fit(x_train.reshape(-1, 1), y_train)
    reference_predictions = reference.predict(x_test.reshape(-1, 1))

    regression = {
        "feature": "bmi",
        "target": "quantitative diabetes progression one year after baseline",
        "ols": {
            "slope": ols_slope,
            "intercept": ols_intercept,
            "metrics": regression_metrics(y_test, ols_predictions),
            "predictions": ols_predictions,
        },
        "gd": {
            "slope": gd_slope,
            "intercept": gd_intercept,
            "metrics": regression_metrics(y_test, gd_predictions),
            "predictions": gd_predictions,
            "loss_history": gd.loss_history,
        },
        "sklearn": {
            "slope": float(reference.coef_[0]),
            "intercept": float(reference.intercept_),
            "metrics": regression_metrics(y_test, reference_predictions),
            "predictions": reference_predictions,
        },
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }
    metadata = {
        "regression_dataset": "Diabetes progression",
        "regression_samples": int(x.shape[0]),
        "regression_features_used": 1,
        "regression_train_samples": int(x_train.shape[0]),
        "regression_test_samples": int(x_test.shape[0]),
    }
    return metadata, regression


def run_all_experiments(output_dir: str | Path | None = None) -> ExperimentBundle:
    classification_metadata, feature_selection, classification = (
        _classification_experiment()
    )
    regression_metadata, regression = _regression_experiment()
    metadata = {
        "project": "Kriterion",
        "purpose": "Auditable clinical prediction with scratch and library models",
        "random_seed": SEED,
        "test_fraction": 0.2,
        **classification_metadata,
        **regression_metadata,
    }
    bundle = ExperimentBundle(
        metadata=metadata,
        feature_selection=feature_selection,
        classification=classification,
        regression=regression,
    )
    if output_dir is not None:
        export_experiment_bundle(bundle, output_dir)
    return bundle


def export_experiment_bundle(
    bundle: ExperimentBundle, output_dir: str | Path
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = bundle.to_dict()
    (output_path / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(bundle.feature_selection["results"]).to_csv(
        output_path / "feature_selection_results.csv", index=False
    )
    classification_rows = []
    for method, result in bundle.classification.items():
        classification_rows.append(
            {
                "method": method,
                "features": result["n_features"],
                **{
                    key: value
                    for key, value in result["scratch_metrics"].items()
                    if key not in {"tn", "fp", "fn", "tp"}
                },
                "tn": result["scratch_metrics"]["tn"],
                "fp": result["scratch_metrics"]["fp"],
                "fn": result["scratch_metrics"]["fn"],
                "tp": result["scratch_metrics"]["tp"],
            }
        )
    pd.DataFrame(classification_rows).to_csv(
        output_path / "classification_metrics.csv", index=False
    )
    regression_rows = [
        {"method": method.upper(), **bundle.regression[method]["metrics"]}
        for method in ("ols", "gd", "sklearn")
    ]
    pd.DataFrame(regression_rows).to_csv(
        output_path / "regression_metrics.csv", index=False
    )
