from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.08
    epochs: int = 1800
    weight: float = 0.0
    bias: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegressionGD:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.weight = 0.0
        self.bias = float(np.mean(y))
        self.loss_history = []
        n = x.size
        for _ in range(self.epochs):
            predictions = self.weight * x + self.bias
            errors = predictions - y
            self.loss_history.append(float(np.mean(errors**2)))
            self.weight -= self.learning_rate * float((2.0 / n) * np.dot(errors, x))
            self.bias -= self.learning_rate * float(2.0 * np.mean(errors))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.weight * np.asarray(x, dtype=float).reshape(-1) + self.bias


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.08
    epochs: int = 2200
    threshold: float = 0.5
    weights: np.ndarray | None = None
    bias: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticRegressionGD:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = x.shape
        self.weights = np.zeros(p, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        epsilon = 1e-12
        for _ in range(self.epochs):
            probabilities = stable_sigmoid(x @ self.weights + self.bias)
            loss = -np.mean(
                y * np.log(probabilities + epsilon)
                + (1.0 - y) * np.log(1.0 - probabilities + epsilon)
            )
            self.loss_history.append(float(loss))
            errors = probabilities - y
            self.weights -= self.learning_rate * (x.T @ errors) / n
            self.bias -= self.learning_rate * float(np.mean(errors))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model must be fitted before prediction")
        return stable_sigmoid(np.asarray(x, dtype=float) @ self.weights + self.bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold).astype(int)


def ols_univariate(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    variance = float(np.mean(x_centered**2))
    if variance == 0.0:
        raise ValueError("OLS requires a predictor with non-zero variance")
    slope = float(np.mean(x_centered * y_centered) / variance)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(
            y_true * np.log(probabilities)
            + (1.0 - y_true) * np.log(1.0 - probabilities)
        )
    )
