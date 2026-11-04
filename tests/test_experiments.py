import json

import pandas as pd
import pytest

from kriterion.experiments import export_experiment_bundle, run_all_experiments


@pytest.fixture(scope="module")
def bundle():
    return run_all_experiments()


def test_dataset_metadata_is_complete(bundle):
    assert bundle.metadata["random_seed"] == 42
    assert bundle.metadata["classification_samples"] == 569
    assert bundle.metadata["classification_features"] == 30
    assert bundle.metadata["positive_class"] == "malignant"
    assert bundle.metadata["regression_samples"] == 442
    assert bundle.metadata["regression_features_used"] == 1


def test_feature_selection_methods_have_expected_cardinality(bundle):
    selected = bundle.feature_selection["selected_features"]
    assert set(selected) == {
        "All features",
        "Chi-square filter",
        "RFE wrapper",
        "Random forest embedded",
    }
    assert len(selected["All features"]) == 30
    assert all(len(selected[method]) == 10 for method in selected if method != "All features")
    assert all(len(set(features)) == len(features) for features in selected.values())


def test_classification_outputs_are_valid(bundle):
    for result in bundle.classification.values():
        metrics = result["scratch_metrics"]
        bounded_metrics = ("accuracy", "precision", "recall", "f1", "auc")
        assert all(0.0 <= metrics[name] <= 1.0 for name in bounded_metrics)
        assert sum(metrics[name] for name in ("tn", "fp", "fn", "tp")) == bundle.metadata[
            "classification_test_samples"
        ]
        assert len(result["fpr"]) == len(result["tpr"])
        assert result["loss_history"][-1] < result["loss_history"][0]


def test_scratch_regression_optimizers_agree(bundle):
    regression = bundle.regression
    assert regression["ols"]["slope"] == pytest.approx(regression["gd"]["slope"], rel=1e-4)
    assert regression["ols"]["intercept"] == pytest.approx(
        regression["gd"]["intercept"], rel=1e-4
    )
    assert regression["gd"]["loss_history"][-1] < regression["gd"]["loss_history"][0]
    assert regression["ols"]["metrics"]["mse"] == pytest.approx(
        regression["sklearn"]["metrics"]["mse"], rel=1e-10
    )


def test_export_writes_machine_readable_single_source(bundle, tmp_path):
    export_experiment_bundle(bundle, tmp_path)
    expected = {
        "results.json",
        "feature_selection_results.csv",
        "classification_metrics.csv",
        "regression_metrics.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["project"] == "Kriterion"
    classification = pd.read_csv(tmp_path / "classification_metrics.csv")
    regression = pd.read_csv(tmp_path / "regression_metrics.csv")
    assert len(classification) == 4
    assert set(regression["method"]) == {"OLS", "GD", "SKLEARN"}

