from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kriterion.ui import (
    ZINC,
    base_figure,
    classification_subsets,
    color_for,
    curve_values,
    dataset_meta,
    feature_rows,
    inject_style,
    intro,
    load_bundle,
    metric_strip,
    metric_value,
    plot,
    readable_frame,
    selected_features,
    values,
)

st.set_page_config(
    page_title="Kriterion Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded"
)
inject_style()


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def overview(bundle: dict[str, Any]) -> None:
    intro(
        "Auditable clinical prediction study",
        "Kriterion Lab",
        "A reproducible comparison of feature selection, analytical and iterative regression, and scratch logistic classification on two public clinical datasets.",
    )
    breast = dataset_meta(
        bundle, "breast_cancer", {"samples": 569, "features": 30, "target": "Malignant / benign"}
    )
    diabetes = dataset_meta(
        bundle, "diabetes", {"samples": 442, "features": 10, "target": "Disease progression"}
    )
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="study-card"><div class="step">CLASSIFICATION DATASET</div><h3>Wisconsin Diagnostic Breast Cancer</h3><div class="number">{breast.get("samples", 569)} × {breast.get("features", 30)}</div><p>{breast.get("target", "Malignant / benign")} · diagnostic morphology measurements</p></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="study-card"><div class="step">REGRESSION DATASET</div><h3>Diabetes progression</h3><div class="number">{diabetes.get("samples", 442)} × 1</div><p>BMI as the explanatory variable · continuous progression score one year after baseline</p></div>',
        unsafe_allow_html=True,
    )
    rows = feature_rows(bundle)
    subsets = classification_subsets(bundle)
    regression = bundle.get("regression", {})
    best_name = (
        max(subsets, key=lambda key: metric_value(subsets[key], "roc_auc"))
        if subsets
        else "Pending"
    )
    best_auc = metric_value(subsets.get(best_name, {}), "roc_auc") if subsets else 0.0
    gd = regression.get("gradient_descent") or regression.get("gd") or {}
    metric_strip(
        [
            ("Feature strategies", str(len(rows) or 4), "baseline included"),
            ("Best classifier", best_name, f"AUC {fmt(best_auc)}" if subsets else None),
            ("Regression R²", fmt(metric_value(gd, "r2")), "batch gradient descent"),
            (
                "Random seed",
                str(bundle.get("random_seed", bundle.get("metadata", {}).get("random_seed", 42))),
                "fixed",
            ),
        ]
    )
    st.header("One experiment, four linked views")
    flow = st.columns(4)
    stages = [
        ("01", "Select", "Compare statistical, recursive, and embedded feature evidence."),
        ("02", "Fit", "Place closed-form OLS beside a stepwise MSE optimizer."),
        ("03", "Classify", "Train sigmoid models with batch BCE updates on each subset."),
        ("04", "Audit", "Read metrics together with curves, errors, and selected variables."),
    ]
    for column, (number, title, copy) in zip(flow, stages, strict=False):
        column.markdown(
            f'<div class="study-card"><div class="step">{number}</div><h3>{title}</h3><p>{copy}</p></div>',
            unsafe_allow_html=True,
        )
    st.header("Study boundary")
    st.markdown(
        '<div class="interpretation"><b>Educational decision support, not clinical advice.</b> The app demonstrates transparent model mechanics on benchmark datasets. It does not estimate an individual patient’s diagnosis or treatment.</div>',
        unsafe_allow_html=True,
    )


def feature_selection(bundle: dict[str, Any]) -> None:
    intro(
        "Module 01 · dimensionality",
        "Feature selection changes what the model can see",
        "The same train–test split is held fixed while four input spaces are compared. Performance and retained dimensionality must be read together.",
    )
    rows = feature_rows(bundle)
    if not rows:
        st.warning("Feature-selection results are not present in the experiment bundle.")
        return
    frame = pd.DataFrame(rows)
    method_col = "method" if "method" in frame else "subset"
    methods = frame[method_col].astype(str).tolist()
    accuracy_col = "accuracy" if "accuracy" in frame else "test_accuracy"
    auc_col = "roc_auc" if "roc_auc" in frame else "auc"
    count_col = "n_features" if "n_features" in frame else "feature_count"
    figure = base_figure(460)
    figure.add_trace(
        go.Bar(
            name="Accuracy",
            x=methods,
            y=frame[accuracy_col],
            marker_color=ZINC["cyan"],
            text=[fmt(float(v)) for v in frame[accuracy_col]],
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Bar(
            name="ROC–AUC",
            x=methods,
            y=frame[auc_col],
            marker_color=ZINC["violet"],
            text=[fmt(float(v)) for v in frame[auc_col]],
            textposition="outside",
        )
    )
    figure.update_layout(
        barmode="group",
        yaxis={
            "range": [
                max(0.5, float(min(frame[accuracy_col].min(), frame[auc_col].min())) - 0.06),
                1.03,
            ],
            "title": "Score",
        },
    )
    figure.update_xaxes(title="Feature strategy")
    plot(figure, "feature-performance")
    counts = (
        frame[count_col].astype(int).tolist()
        if count_col in frame
        else [30, 10, 10, 10][: len(methods)]
    )
    best_index = int(np.argmax(frame[auc_col].astype(float).to_numpy()))
    metric_strip(
        [
            ("Highest AUC", fmt(float(frame.iloc[best_index][auc_col])), methods[best_index]),
            ("Features retained", str(counts[best_index]), f"of {max(counts)}"),
            ("Reduction", f"{100 * (1 - counts[best_index] / max(counts)):.0f}%", "at best AUC"),
        ]
    )
    tab1, tab2, tab3 = st.tabs(["Selected variables", "Method mechanics", "Result table"])
    with tab1:
        selected = selected_features(bundle)
        if selected:
            all_features = sorted(
                {feature for collection in selected.values() for feature in collection}
            )
            matrix = np.array(
                [
                    [1 if feature in selected[method] else 0 for feature in all_features]
                    for method in selected
                ]
            )
            heatmap = base_figure(max(360, 150 + len(all_features) * 13))
            heatmap.add_trace(
                go.Heatmap(
                    z=matrix.T,
                    x=list(selected),
                    y=all_features,
                    colorscale=[[0, ZINC["surface"]], [1, ZINC["cyan"]]],
                    showscale=False,
                    xgap=3,
                    ygap=2,
                    hovertemplate="%{y}<br>%{x}<extra></extra>",
                )
            )
            heatmap.update_layout(margin={"l": 150, "r": 20, "t": 25, "b": 50})
            plot(heatmap, "feature-membership")
        else:
            st.info("Selected feature names will appear after the experiment run.")
    with tab2:
        columns = st.columns(3)
        content = [
            (
                "χ² filter",
                "Ranks each non-negative feature by statistical dependence with the diagnosis. Fast and model-independent.",
            ),
            (
                "RFE wrapper",
                "Repeatedly fits a logistic estimator and removes the weakest coefficients. More expensive, model-aware.",
            ),
            (
                "RF embedded",
                "Uses impurity reduction accumulated across a forest. Captures nonlinear splits and interactions.",
            ),
        ]
        for column, (title, copy) in zip(columns, content, strict=False):
            column.markdown(
                f'<div class="study-card"><h3>{title}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )
    with tab3:
        st.dataframe(
            readable_frame(rows, [method_col, count_col, accuracy_col, auc_col]),
            width="stretch",
            hide_index=True,
        )
    st.markdown(
        '<div class="interpretation"><b>How to defend this result:</b> feature selection succeeds when a smaller subset preserves discrimination, not merely when it removes columns. The AUC gap and retained feature count should therefore be stated together.</div>',
        unsafe_allow_html=True,
    )


def regression(bundle: dict[str, Any]) -> None:
    intro(
        "Module 02 · continuous target",
        "Two routes converge on the same straight line",
        "BMI is deliberately used as a single predictor so the analytical slope and iterative gradient updates can be inspected without hiding them inside a matrix solver.",
    )
    section = bundle.get("regression", {})
    ols = section.get("ols") or section.get("closed_form") or {}
    gd = section.get("gradient_descent") or section.get("gd") or {}
    if not ols and not gd:
        st.warning("Regression results are not present in the experiment bundle.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="step">CLOSED FORM</div>', unsafe_allow_html=True)
        st.latex(r"m=\frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)},\qquad c=\bar y-m\bar x")
    with c2:
        st.markdown('<div class="step">BATCH GRADIENT DESCENT</div>', unsafe_allow_html=True)
        st.latex(
            r"m\leftarrow m-\alpha\frac{2}{n}\sum_i(\hat y_i-y_i)x_i,\quad c\leftarrow c-\alpha\frac{2}{n}\sum_i(\hat y_i-y_i)"
        )
    x_test = values(section, "x_test", "test_x") or values(ols, "x_test", "test_x", "x")
    y_test = values(section, "y_test", "test_y") or values(ols, "y_test", "test_y", "y_true")
    ols_pred = values(ols, "y_pred", "predictions", "test_predictions")
    gd_pred = values(gd, "y_pred", "predictions", "test_predictions")
    figure = base_figure(500)
    if x_test and y_test:
        figure.add_trace(
            go.Scatter(
                x=x_test,
                y=y_test,
                mode="markers",
                name="Held-out observations",
                marker={"color": ZINC["muted"], "size": 7, "opacity": 0.7},
            )
        )
    if x_test and ols_pred:
        order = np.argsort(x_test)
        figure.add_trace(
            go.Scatter(
                x=np.array(x_test)[order],
                y=np.array(ols_pred)[order],
                mode="lines",
                name="OLS",
                line={"color": ZINC["cyan"], "width": 3},
            )
        )
    if x_test and gd_pred:
        order = np.argsort(x_test)
        figure.add_trace(
            go.Scatter(
                x=np.array(x_test)[order],
                y=np.array(gd_pred)[order],
                mode="lines",
                name="Gradient descent",
                line={"color": ZINC["amber"], "width": 2, "dash": "dash"},
            )
        )
    figure.update_xaxes(title="Standardized BMI")
    figure.update_yaxes(title="Disease progression score")
    plot(figure, "regression-fit")
    methods = [("OLS", ols), ("Batch GD", gd)]
    columns = st.columns(4)
    for column, key, label in zip(
        columns,
        ["mse", "rmse", "mae", "r2"],
        ["MSE", "RMSE", "MAE", "R²"],
        strict=False,
    ):
        ols_value = metric_value(ols, key)
        gd_value = metric_value(gd, key)
        column.metric(label, fmt(gd_value), f"OLS {fmt(ols_value)}")
    history = curve_values(gd, "loss")
    if history:
        convergence = base_figure(380)
        convergence.add_trace(
            go.Scatter(
                x=list(range(1, len(history) + 1)),
                y=history,
                mode="lines",
                line={"color": ZINC["amber"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(251,191,36,.08)",
                name="MSE",
            )
        )
        convergence.update_xaxes(title="Epoch")
        convergence.update_yaxes(
            title="Training cost",
            type="log" if max(history) / max(min(history), 1e-12) > 100 else "linear",
        )
        plot(convergence, "regression-convergence")
    comparison = [
        {
            "Method": name,
            **{key.upper(): metric_value(record, key) for key in ["mse", "rmse", "mae", "r2"]},
        }
        for name, record in methods
    ]
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
    st.markdown(
        '<div class="interpretation"><b>Expected relationship:</b> with a suitable learning rate and enough epochs, batch gradient descent approaches the OLS optimum. A persistent gap usually indicates insufficient convergence or an unstable step size.</div>',
        unsafe_allow_html=True,
    )


def classification(bundle: dict[str, Any]) -> None:
    intro(
        "Module 03 · binary target",
        "A probability model built from first principles",
        "Every feature subset is passed to the same scratch logistic learner. The comparison isolates the information retained by selection from the optimizer used to learn the decision surface.",
    )
    subsets = classification_subsets(bundle)
    if not subsets:
        st.warning("Classification results are not present in the experiment bundle.")
        return
    st.markdown(
        '<div class="equation">z = Xw + b &nbsp; → &nbsp; σ(z) = 1 / (1 + e<sup>−z</sup>) &nbsp; → &nbsp; ŷ = 1[σ(z) ≥ 0.5]</div>',
        unsafe_allow_html=True,
    )
    names = list(subsets)
    selected = st.segmented_control(
        "Inspect a feature strategy",
        names,
        default=names[int(np.argmax([metric_value(subsets[name], "roc_auc") for name in names]))],
        width="stretch",
    )
    selected = selected or names[0]
    record = subsets[selected]
    metric_strip(
        [
            ("Accuracy", fmt(metric_value(record, "accuracy")), None),
            ("Precision", fmt(metric_value(record, "precision")), None),
            ("Recall", fmt(metric_value(record, "recall")), None),
            ("F1 score", fmt(metric_value(record, "f1")), None),
            ("ROC–AUC", fmt(metric_value(record, "roc_auc")), None),
        ]
    )
    left, right = st.columns([1, 1.5])
    confusion = record.get("confusion_matrix") or record.get("confusion") or [[0, 0], [0, 0]]
    confusion_array = np.asarray(confusion, dtype=float)
    with left:
        figure = base_figure(410)
        figure.add_trace(
            go.Heatmap(
                z=confusion_array,
                x=["Predicted benign", "Predicted malignant"],
                y=["Actual benign", "Actual malignant"],
                colorscale=[[0, ZINC["surface"]], [1, ZINC["cyan"]]],
                showscale=False,
                text=confusion_array.astype(int),
                texttemplate="%{text}",
                textfont={"size": 24},
                hovertemplate="%{y}<br>%{x}<br>n=%{z}<extra></extra>",
            )
        )
        figure.update_layout(title="Confusion matrix", margin={"l": 25, "r": 15, "t": 70, "b": 55})
        plot(figure, "confusion")
    with right:
        roc = base_figure(410)
        for index, name in enumerate(names):
            subset = subsets[name]
            fpr = curve_values(subset, "fpr")
            tpr = curve_values(subset, "tpr")
            if fpr and tpr:
                roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{name} · {fmt(metric_value(subset, 'roc_auc'))}",
                        line={
                            "color": color_for(name, index),
                            "width": 4 if name == selected else 2,
                        },
                        opacity=1 if name == selected else 0.58,
                    )
                )
        roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line={"color": ZINC["line"], "dash": "dot"},
            )
        )
        roc.update_xaxes(title="False-positive rate", range=[0, 1])
        roc.update_yaxes(title="True-positive rate", range=[0, 1.02])
        plot(roc, "roc-comparison")
    loss = curve_values(record, "loss")
    if loss:
        convergence = base_figure(360)
        convergence.add_trace(
            go.Scatter(
                x=list(range(1, len(loss) + 1)),
                y=loss,
                mode="lines",
                name=selected,
                line={"color": color_for(selected), "width": 2},
                fill="tozeroy",
                fillcolor="rgba(34,211,238,.06)",
            )
        )
        convergence.update_xaxes(title="Epoch")
        convergence.update_yaxes(title="Binary cross-entropy")
        plot(convergence, "classification-loss")
    rows = []
    for name, subset in subsets.items():
        rows.append(
            {
                "Feature strategy": name,
                **{
                    label: metric_value(subset, key)
                    for label, key in [
                        ("Accuracy", "accuracy"),
                        ("Precision", "precision"),
                        ("Recall", "recall"),
                        ("F1", "f1"),
                        ("AUC", "roc_auc"),
                    ]
                },
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown(
        '<div class="interpretation"><b>Clinical reading:</b> false negatives deserve explicit attention because they correspond to malignant cases predicted benign. Accuracy alone can conceal that error; recall, the confusion matrix, and the ROC curve expose it.</div>',
        unsafe_allow_html=True,
    )


def reproducibility(bundle: dict[str, Any]) -> None:
    intro(
        "Module 04 · reproducibility",
        "The result is a procedure, not a screenshot",
        "The experiment is deterministic, exportable, and structured so each visual in the app, report, and defense deck is generated from the same metrics bundle.",
    )
    seed = bundle.get("random_seed", bundle.get("metadata", {}).get("random_seed", 42))
    steps = [
        ("1", "Load", "Use scikit-learn’s bundled datasets; no remote data service is required."),
        ("2", "Split", f"Stratified 80/20 classification split and fixed random seed {seed}."),
        (
            "3",
            "Scale",
            "Min–Max values for χ²; standardized values for gradient-based optimization.",
        ),
        (
            "4",
            "Select",
            "Fit selectors on training data only, then transform held-out observations.",
        ),
        ("5", "Evaluate", "Calculate every reported metric on the untouched test partition."),
        ("6", "Export", "Write one machine-readable bundle and regenerate every visual artifact."),
    ]
    for start in range(0, len(steps), 3):
        columns = st.columns(3)
        for column, (number, title, copy) in zip(columns, steps[start : start + 3], strict=False):
            column.markdown(
                f'<div class="study-card"><div class="step">STEP {number}</div><h3>{title}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.header("Mathematical audit trail")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Linear objective")
        st.latex(r"J(m,c)=\frac{1}{n}\sum_{i=1}^{n}(mx_i+c-y_i)^2")
        st.latex(r"\frac{\partial J}{\partial m}=\frac{2}{n}\sum_i(\hat y_i-y_i)x_i")
    with right:
        st.markdown("#### Logistic objective")
        st.latex(r"\mathcal{L}=-\frac{1}{n}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]")
        st.latex(r"\nabla_w\mathcal{L}=\frac{1}{n}X^\top(p-y)")
    st.header("Experiment manifest")
    manifest = bundle.get("metadata", {})
    manifest = {"random_seed": seed, **manifest}
    st.json(manifest, expanded=True)
    st.markdown(
        '<div class="interpretation"><b>Leakage control:</b> preprocessing and feature-selection parameters are estimated from training data only. The test split is used once for the final comparison.</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="kicker">Kriterion · ML mini-project</div>', unsafe_allow_html=True)
    st.markdown("### Evidence, optimization, decisions")
    page = st.segmented_control(
        "Navigate",
        [
            "Study overview",
            "Feature selection",
            "Linear regression",
            "Logistic regression",
            "Reproducibility",
        ],
        default="Study overview",
        label_visibility="collapsed",
        width="stretch",
    )
    page = page or "Study overview"
    st.divider()
    st.caption("Divya · 00702102026")
    st.caption("Indira Gandhi Delhi Technical University for Women")
    st.caption("Public benchmark data · deterministic evaluation")


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(
        "Run the deterministic experiment pipeline before opening the app.",
        title="Experiment data unavailable",
    )
    st.code(str(exc))
    st.stop()

pages: dict[str, Any] = {
    "Study overview": overview,
    "Feature selection": feature_selection,
    "Linear regression": regression,
    "Logistic regression": classification,
    "Reproducibility": reproducibility,
}
pages[page](bundle)
