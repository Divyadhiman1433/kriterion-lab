from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
ZINC = {
    "canvas": "#09090b",
    "surface": "#18181b",
    "line": "#3f3f46",
    "muted": "#a1a1aa",
    "text": "#f4f4f5",
    "cyan": "#22d3ee",
    "amber": "#fbbf24",
    "violet": "#a78bfa",
    "rose": "#fb7185",
    "emerald": "#34d399",
}
METHOD_COLORS = {
    "All Features": ZINC["muted"],
    "Baseline": ZINC["muted"],
    "Chi-Square": ZINC["cyan"],
    "Filter": ZINC["cyan"],
    "RFE": ZINC["amber"],
    "Wrapper": ZINC["amber"],
    "Random Forest": ZINC["violet"],
    "Embedded": ZINC["violet"],
}


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        [data-testid="stAppViewContainer"] { background: #09090b; }
        [data-testid="stHeader"] { background: rgba(9,9,11,.86); }
        [data-testid="stSidebar"] { background: #0f0f12; border-right: 1px solid #27272a; }
        .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: -0.035em; }
        h1 { font-size: clamp(2.35rem, 5vw, 4.6rem) !important; line-height: .98 !important; }
        h2 { margin-top: 2.6rem !important; }
        p, li { color: #d4d4d8; }
        [data-testid="stMetric"] { background: #18181b; border: 1px solid #27272a; padding: 1rem; border-radius: .75rem; }
        [data-testid="stMetricLabel"] { color: #a1a1aa; }
        [data-testid="stMetricValue"] { color: #fafafa; letter-spacing: -.04em; }
        .kicker { color: #22d3ee; text-transform: uppercase; letter-spacing: .18em; font-size: .72rem; font-weight: 700; }
        .lede { color: #a1a1aa; font-size: 1.08rem; max-width: 760px; line-height: 1.65; }
        .rule { border-top: 1px solid #27272a; margin: 1.2rem 0 2rem; }
        .study-card { min-height: 168px; padding: 1.25rem; border: 1px solid #27272a; border-radius: .8rem; background: linear-gradient(150deg,#18181b,#101013); }
        .study-card h3 { margin: .3rem 0 .65rem; font-size: 1.25rem; }
        .study-card p { color: #a1a1aa; font-size: .9rem; }
        .study-card .number { color: #fafafa; font-size: 2rem; font-weight: 700; letter-spacing: -.05em; }
        .equation { padding: 1rem 1.2rem; border-left: 3px solid #22d3ee; background: #18181b; color: #e4e4e7; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; overflow-x: auto; }
        .interpretation { padding: 1rem 1.2rem; border: 1px solid #3f3f46; border-radius: .65rem; background: #111114; color: #d4d4d8; }
        .step { color: #71717a; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size: .78rem; letter-spacing: .08em; }
        [data-testid="stDataFrame"] { border: 1px solid #27272a; border-radius: .65rem; overflow: hidden; }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
        .stTabs [data-baseweb="tab"] { background: #18181b; border-radius: .45rem; padding: .35rem .8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _json_candidates() -> list[Path]:
    return [
        ROOT / "artifacts" / "results.json",
        ROOT / "artifacts" / "experiment_results.json",
        ROOT / "artifacts" / "metrics.json",
    ]


@st.cache_data(show_spinner="Running deterministic experiments…")
def load_bundle() -> dict[str, Any]:
    for path in _json_candidates():
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    import_attempts = [
        ("kriterion.pipeline", "run_all_experiments"),
        ("kriterion.experiments", "run_all_experiments"),
        ("kriterion", "run_all_experiments"),
        ("src.kriterion.experiments", "run_all_experiments"),
    ]
    errors = []
    for module_name, function_name in import_attempts:
        try:
            module = __import__(module_name, fromlist=[function_name])
            function = getattr(module, function_name)
            try:
                result = function(output_dir=ROOT / "artifacts")
            except TypeError:
                result = function()
            return result.to_dict() if hasattr(result, "to_dict") else result
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{function_name}: {exc}")
    raise RuntimeError("No experiment bundle was found. " + " | ".join(errors))


def base_figure(height: int = 430) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        height=height,
        paper_bgcolor=ZINC["canvas"],
        plot_bgcolor=ZINC["canvas"],
        font={"color": ZINC["text"], "family": "Inter, Arial, sans-serif"},
        margin={"l": 36, "r": 20, "t": 45, "b": 40},
        hoverlabel={"bgcolor": ZINC["surface"], "font_color": ZINC["text"]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    figure.update_xaxes(gridcolor="#27272a", zerolinecolor="#3f3f46")
    figure.update_yaxes(gridcolor="#27272a", zerolinecolor="#3f3f46")
    return figure


def plot(figure: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False}, key=key)


def feature_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    section = bundle.get("feature_selection", {})
    rows = section.get("comparison") or section.get("results") or bundle.get("feature_selection_results") or []
    if isinstance(rows, dict):
        normalized = []
        for method, value in rows.items():
            item = dict(value) if isinstance(value, dict) else {"accuracy": value}
            item.setdefault("method", method)
            normalized.append(item)
        rows = normalized
    normalized = []
    for row in rows:
        item = dict(row)
        if "scratch_accuracy" in item:
            item.setdefault("accuracy", item["scratch_accuracy"])
        if "scratch_auc" in item:
            item.setdefault("roc_auc", item["scratch_auc"])
        normalized.append(item)
    return normalized


def classification_subsets(bundle: dict[str, Any]) -> dict[str, Any]:
    section = bundle.get("classification", {})
    subsets = section.get("subsets") or section.get("results") or bundle.get("classification_results")
    if subsets is None and isinstance(section, dict):
        subsets = section
    if isinstance(subsets, list):
        return {str(item.get("method") or item.get("subset")): item for item in subsets}
    return subsets


def metric_value(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = record.get("metrics") or record.get("scratch_metrics") or record
    aliases = {
        "roc_auc": ["roc_auc", "auc"],
        "r2": ["r2", "r2_score"],
        "rmse": ["rmse"],
    }
    for candidate in aliases.get(key, [key]):
        if candidate in metrics:
            return float(metrics[candidate])
    return default


def curve_values(record: dict[str, Any], name: str) -> list[float]:
    aliases = {
        "loss": ["loss_history", "loss", "cost_history", "costs"],
        "fpr": ["fpr", "false_positive_rate"],
        "tpr": ["tpr", "true_positive_rate"],
    }
    containers = [record, record.get("roc_curve", {}), record.get("curves", {})]
    for container in containers:
        for candidate in aliases.get(name, [name]):
            value = container.get(candidate) if isinstance(container, dict) else None
            if isinstance(value, (list, tuple, np.ndarray)):
                return [float(item) for item in value]
    return []


def values(record: dict[str, Any], *keys: str) -> list[float]:
    containers = [record, record.get("predictions", {}), record.get("data", {})]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                return [float(item) for item in value]
    return []


def dataset_meta(bundle: dict[str, Any], key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    datasets = bundle.get("datasets") or bundle.get("metadata", {}).get("datasets") or {}
    item = datasets.get(key, {}) if isinstance(datasets, dict) else {}
    metadata = bundle.get("metadata", {})
    if key == "breast_cancer":
        item = {
            "samples": metadata.get("classification_samples", defaults.get("samples")),
            "features": metadata.get("classification_features", defaults.get("features")),
            "target": metadata.get("positive_class", defaults.get("target")),
            **item,
        }
    if key == "diabetes":
        item = {
            "samples": metadata.get("regression_samples", defaults.get("samples")),
            "features": metadata.get("regression_features_used", defaults.get("features")),
            **item,
        }
    return {**defaults, **item}


def color_for(method: str, index: int = 0) -> str:
    for label, color in METHOD_COLORS.items():
        if label.lower() in method.lower():
            return color
    return [ZINC["cyan"], ZINC["amber"], ZINC["violet"], ZINC["rose"]][index % 4]


def intro(kicker: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{lede}</div><div class="rule"></div>', unsafe_allow_html=True)


def metric_strip(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items, strict=False):
        column.metric(label, value, delta)


def selected_features(bundle: dict[str, Any]) -> dict[str, list[str]]:
    section = bundle.get("feature_selection", {})
    result = section.get("selected_features") or bundle.get("selected_features") or {}
    return {str(key): [str(item) for item in value] for key, value in result.items()}


def readable_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    available = [column for column in columns if column in frame.columns]
    return frame[available] if available else frame
