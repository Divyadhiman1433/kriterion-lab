import numpy as np
import pytest

from kriterion.models import (
    LinearRegressionGD,
    LogisticRegressionGD,
    binary_cross_entropy,
    ols_univariate,
    stable_sigmoid,
)


def test_ols_recovers_exact_line():
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    y = 4.25 * x - 1.75
    slope, intercept = ols_univariate(x, y)
    assert slope == pytest.approx(4.25)
    assert intercept == pytest.approx(-1.75)


def test_ols_rejects_constant_predictor():
    with pytest.raises(ValueError, match="non-zero variance"):
        ols_univariate(np.ones(5), np.arange(5))


def test_linear_gradient_descent_converges_to_ols():
    x = np.linspace(-1.0, 1.0, 101)
    y = 2.4 * x + 0.7
    slope, intercept = ols_univariate(x, y)
    model = LinearRegressionGD(learning_rate=0.1, epochs=600).fit(x, y)
    assert model.weight == pytest.approx(slope, abs=1e-5)
    assert model.bias == pytest.approx(intercept, abs=1e-5)
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.loss_history[-1] < 1e-10


def test_stable_sigmoid_is_finite_and_monotonic():
    output = stable_sigmoid(np.array([-1e6, -2.0, 0.0, 2.0, 1e6]))
    assert np.all(np.isfinite(output))
    assert np.all(np.diff(output) > 0)
    assert output[2] == pytest.approx(0.5)


def test_binary_cross_entropy_rewards_better_probabilities():
    y = np.array([0, 0, 1, 1])
    good = binary_cross_entropy(y, np.array([0.05, 0.2, 0.8, 0.95]))
    poor = binary_cross_entropy(y, np.array([0.45, 0.6, 0.4, 0.55]))
    assert good < poor


def test_logistic_gradient_descent_learns_separable_data():
    x = np.array([[-2.0], [-1.4], [-0.8], [0.8], [1.4], [2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegressionGD(learning_rate=0.2, epochs=1200).fit(x, y)
    probabilities = model.predict_proba(x)
    assert np.array_equal(model.predict(x), y)
    assert np.all(np.diff(probabilities) > 0)
    assert model.loss_history[-1] < model.loss_history[0]


def test_logistic_predict_requires_fit():
    with pytest.raises(RuntimeError, match="fitted"):
        LogisticRegressionGD().predict_proba(np.zeros((2, 1)))

