from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter

from .experiments import ExperimentBundle

ZINC_950 = "#09090b"
ZINC_900 = "#18181b"
ZINC_700 = "#3f3f46"
ZINC_500 = "#71717a"
ZINC_300 = "#d4d4d8"
ZINC_200 = "#e4e4e7"
ZINC_100 = "#f4f4f5"
CYAN = "#06b6d4"
AMBER = "#f59e0b"
EMERALD = "#10b981"
ROSE = "#f43f5e"
VIOLET = "#8b5cf6"
SERIES = [CYAN, AMBER, EMERALD, VIOLET]


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": ZINC_300,
            "axes.labelcolor": ZINC_700,
            "axes.titlecolor": ZINC_950,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "xtick.color": ZINC_500,
            "ytick.color": ZINC_500,
            "grid.color": ZINC_200,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_selection(bundle: ExperimentBundle, output_dir: Path) -> None:
    rows = bundle.feature_selection["results"]
    labels = [row["method"].replace(" ", "\n", 1) for row in rows]
    accuracy = [row["scratch_accuracy"] for row in rows]
    auc = [row["scratch_auc"] for row in rows]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars_a = ax.bar(x - width / 2, accuracy, width, color=CYAN, label="Accuracy")
    bars_b = ax.bar(x + width / 2, auc, width, color=AMBER, label="ROC-AUC")
    ax.set_title("Feature reduction retains diagnostic signal")
    ax.set_ylabel("Test score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.75, 1.015)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, loc="lower right")
    ax.bar_label(bars_a, fmt="%.3f", padding=3, fontsize=8, color=ZINC_700)
    ax.bar_label(bars_b, fmt="%.3f", padding=3, fontsize=8, color=ZINC_700)
    fig.tight_layout()
    _finish(fig, output_dir / "feature_selection_comparison.png")


def plot_selection_matrix(bundle: ExperimentBundle, output_dir: Path) -> None:
    selected = bundle.feature_selection["selected_features"]
    methods = [method for method in selected if method != "All features"]
    feature_order = []
    for method in methods:
        for feature in selected[method]:
            if feature not in feature_order:
                feature_order.append(feature)
    matrix = np.array(
        [[int(feature in selected[method]) for feature in feature_order] for method in methods]
    )
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap([ZINC_100, CYAN]), vmin=0, vmax=1)
    ax.set_title("Different selectors converge on a compact morphology signature")
    ax.set_yticks(range(len(methods)), [method.replace(" ", "\n", 1) for method in methods])
    ax.set_xticks(range(len(feature_order)), feature_order, rotation=55, ha="right", fontsize=8)
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col]:
                ax.text(col, row, "●", ha="center", va="center", color=ZINC_950, fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    _finish(fig, output_dir / "feature_selection_matrix.png")


def plot_regression_fit(bundle: ExperimentBundle, output_dir: Path) -> None:
    regression = bundle.regression
    x_test = np.asarray(regression["x_test"])
    y_test = np.asarray(regression["y_test"])
    grid = np.linspace(float(np.min(x_test)), float(np.max(x_test)), 200)
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.scatter(x_test, y_test, s=32, alpha=0.7, color=ZINC_500, edgecolor="white", linewidth=0.5, label="Held-out patients")
    ax.plot(
        grid,
        regression["ols"]["slope"] * grid + regression["ols"]["intercept"],
        color=CYAN,
        linewidth=2.5,
        label="Closed-form OLS",
    )
    ax.plot(
        grid,
        regression["gd"]["slope"] * grid + regression["gd"]["intercept"],
        color=AMBER,
        linewidth=2.0,
        linestyle="--",
        label="Batch gradient descent",
    )
    ax.set_title("Two optimizers recover the same BMI progression trend")
    ax.set_xlabel("Standardized baseline BMI")
    ax.set_ylabel("Disease progression score after one year")
    ax.grid(alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    _finish(fig, output_dir / "regression_fit.png")


def plot_linear_convergence(bundle: ExperimentBundle, output_dir: Path) -> None:
    loss = np.asarray(bundle.regression["gd"]["loss_history"])
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(np.arange(1, loss.size + 1), loss, color=AMBER, linewidth=2.2)
    ax.scatter([1, loss.size], [loss[0], loss[-1]], color=[ROSE, EMERALD], s=42, zorder=3)
    ax.set_title("Linear gradient descent reaches a stable minimum")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.grid(alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(f"{loss[0]:.1f}", (1, loss[0]), xytext=(10, 8), textcoords="offset points", color=ZINC_700)
    ax.annotate(f"{loss[-1]:.1f}", (loss.size, loss[-1]), xytext=(-42, 10), textcoords="offset points", color=ZINC_700)
    fig.tight_layout()
    _finish(fig, output_dir / "linear_cost_convergence.png")


def plot_logistic_convergence(bundle: ExperimentBundle, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for color, (method, result) in zip(SERIES, bundle.classification.items(), strict=False):
        loss = np.asarray(result["loss_history"])
        ax.plot(np.arange(1, loss.size + 1), loss, color=color, linewidth=1.8, label=method)
    ax.set_title("Binary cross-entropy decreases for every feature subset")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.grid(alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2)
    fig.tight_layout()
    _finish(fig, output_dir / "logistic_loss_convergence.png")


def plot_roc(bundle: ExperimentBundle, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for color, (method, result) in zip(SERIES, bundle.classification.items(), strict=False):
        auc = result["scratch_metrics"]["auc"]
        ax.plot(result["fpr"], result["tpr"], color=color, linewidth=2.0, label=f"{method} · {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=ZINC_300, linestyle="--", linewidth=1.2, label="Chance")
    ax.set_title("ROC curves show robust malignant-case ranking")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _finish(fig, output_dir / "classification_roc.png")


def plot_confusion_matrix(bundle: ExperimentBundle, output_dir: Path) -> None:
    best_method, best = max(
        bundle.classification.items(), key=lambda item: item[1]["scratch_metrics"]["auc"]
    )
    matrix = np.asarray(best["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.imshow(matrix, cmap=ListedColormap([ZINC_100, CYAN]), vmin=0, vmax=max(int(matrix.max()), 1))
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=25, fontweight="bold", color=ZINC_950)
    ax.set_title(f"Confusion matrix · {best_method}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    _finish(fig, output_dir / "classification_confusion_matrix.png")


def plot_dataset_overview(bundle: ExperimentBundle, output_dir: Path) -> None:
    metadata = bundle.metadata
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(
        ["Benign", "Malignant"],
        [metadata["benign_samples"], metadata["malignant_samples"]],
        color=[CYAN, AMBER],
        width=0.58,
    )
    axes[0].set_title("Classification target distribution")
    axes[0].set_ylabel("Samples")
    axes[0].grid(axis="y")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].barh(
        ["Diabetes regression", "Tumour classification"],
        [metadata["regression_samples"], metadata["classification_samples"]],
        color=[EMERALD, VIOLET],
        height=0.48,
    )
    axes[1].set_title("Two public clinical datasets")
    axes[1].set_xlabel("Samples")
    axes[1].grid(axis="x")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _finish(fig, output_dir / "dataset_overview.png")


def generate_all_figures(bundle: ExperimentBundle, output_dir: str | Path) -> None:
    _apply_style()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    plot_dataset_overview(bundle, path)
    plot_feature_selection(bundle, path)
    plot_selection_matrix(bundle, path)
    plot_regression_fit(bundle, path)
    plot_linear_convergence(bundle, path)
    plot_logistic_convergence(bundle, path)
    plot_roc(bundle, path)
    plot_confusion_matrix(bundle, path)
