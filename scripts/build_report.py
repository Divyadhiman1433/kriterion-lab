from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results.json"
FIGURES = ROOT / "assets" / "figures"
LOGO = ROOT / "assets" / "brand" / "igdtuw-logo.png"
DOCX_OUT = ROOT / "artifacts" / "Kriterion_Project_Report.docx"
MD_OUT = ROOT / "docs" / "Kriterion_Academic_Report.md"

INK = RGBColor(24, 24, 27)
MUTED = RGBColor(82, 82, 91)
CYAN = RGBColor(8, 145, 178)
AMBER = RGBColor(180, 83, 9)
LINE = "D4D4D8"
FILL = "F4F4F5"


def read_results() -> dict[str, Any]:
    if not RESULTS.exists():
        raise FileNotFoundError("Run python scripts/run_experiments.py before building the report")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    element = properties.find(qn("w:shd"))
    if element is None:
        element = OxmlElement("w:shd")
        properties.append(element)
    element.set(qn("w:fill"), fill)


def borders(table) -> None:
    properties = table._tbl.tblPr
    table_borders = properties.first_child_found_in("w:tblBorders")
    if table_borders is None:
        table_borders = OxmlElement("w:tblBorders")
        properties.append(table_borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), LINE)
        table_borders.append(node)


def set_cell_margins(cell, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def configure_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    for style_name, size, before, after, color in (
        ("Heading 1", 16, 16, 8, CYAN),
        ("Heading 2", 13, 12, 6, CYAN),
        ("Heading 3", 12, 8, 4, MUTED),
    ):
        style = doc.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    caption = doc.styles["Caption"]
    caption.font.name = "Aptos"
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.font.color.rgb = MUTED
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(10)


def footer(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Kriterion · Academic ML Mini-Project")
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED


def add_cover(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if LOGO.exists():
        paragraph.add_run().add_picture(str(LOGO), width=Inches(1.15))
    institution = doc.add_paragraph()
    institution.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = institution.add_run("INDIRA GANDHI DELHI TECHNICAL UNIVERSITY FOR WOMEN")
    run.bold = True
    run.font.name = "Aptos"
    run.font.size = Pt(11)
    run.font.color.rgb = MUTED
    institution.paragraph_format.space_after = Pt(70)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Feature Selection and Scratch Optimization for Clinical Prediction")
    run.bold = True
    run.font.name = "Aptos Display"
    run.font.size = Pt(28)
    run.font.color.rgb = INK
    title.paragraph_format.space_after = Pt(16)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Kriterion — An Auditable Clinical Prediction Laboratory")
    run.font.name = "Aptos"
    run.font.size = Pt(14)
    run.font.color.rgb = CYAN
    subtitle.paragraph_format.space_after = Pt(90)
    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.add_run("Submitted by\n").bold = True
    author.add_run("Divya\nRoll number: 00702102026")
    author.paragraph_format.space_after = Pt(30)
    note = doc.add_paragraph("Machine Learning Mini-Project · Reproducible study · 2026")
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.runs[0].font.color.rgb = MUTED
    note.runs[0].font.size = Pt(9.5)
    doc.add_page_break()


def add_header(section) -> None:
    paragraph = section.header.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run("KRITERION / MACHINE LEARNING MINI-PROJECT")
    run.bold = True
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED
    paragraph.paragraph_format.space_after = Pt(4)
    properties = paragraph._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), LINE)
    border.append(bottom)
    properties.append(border)


def add_figure(doc: Document, filename: str, caption: str, width: float = 6.25) -> None:
    path = FIGURES / filename
    if not path.exists():
        return
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption, style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    borders(table)
    for index, (header, width) in enumerate(zip(headers, widths, strict=True)):
        cell = table.rows[0].cells[index]
        cell.width = Inches(width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        shade(cell, FILL)
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(header)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = table.add_row().cells
        for index, (value, width) in enumerate(zip(row, widths, strict=True)):
            cells[index].width = Inches(width)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[index])
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(value)
            run.font.size = Pt(8.5)
    doc.add_paragraph()


def add_formula(doc: Document, label: str, formula: str, meaning: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.18)
    paragraph.paragraph_format.right_indent = Inches(0.18)
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(f"{label}\n")
    run.bold = True
    run.font.color.rgb = CYAN
    equation = paragraph.add_run(formula)
    equation.font.name = "Cambria Math"
    equation.font.size = Pt(11)
    explanation = paragraph.add_run(f"\n{meaning}")
    explanation.font.size = Pt(9)
    explanation.font.color.rgb = MUTED
    properties = paragraph._p.get_or_add_pPr()
    shade_node = OxmlElement("w:shd")
    shade_node.set(qn("w:fill"), "FAFAFA")
    properties.append(shade_node)


def metric(record: dict[str, Any], name: str) -> float:
    return float(record.get("metrics", record).get(name, 0.0))


def best_classifier(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return max(data["classification"].items(), key=lambda item: float(item[1]["scratch_metrics"]["auc"]))


def build_markdown(data: dict[str, Any]) -> str:
    best_name, best = best_classifier(data)
    regression = data["regression"]
    lines = [
        "# Feature Selection and Scratch Optimization for Clinical Prediction",
        "",
        "**Kriterion — An Auditable Clinical Prediction Laboratory**  ",
        "Divya · 00702102026  ",
        "Indira Gandhi Delhi Technical University for Women",
        "",
        "## Abstract",
        "",
        "This project studies how feature selection and first-principles optimization affect two reproducible clinical prediction tasks. The classification pipeline uses the Wisconsin Diagnostic Breast Cancer dataset and compares all 30 measurements with chi-square, recursive feature elimination, and random-forest subsets. The regression pipeline uses standardized BMI to predict one-year diabetes progression and compares analytical ordinary least squares with batch gradient descent. Scratch logistic regression minimizes binary cross-entropy for every feature subset. All preprocessing is fit on training data only, all reported metrics are computed on held-out data, and the app, report, and presentation reuse one deterministic result bundle.",
        "",
        "## Measured Outcome",
        "",
        f"The highest scratch ROC-AUC was **{best['scratch_metrics']['auc']:.3f}** using **{best_name}** with {best['n_features']} features. The analytical and gradient-descent regression models produced test RMSE values of **{metric(regression['ols'], 'rmse'):.3f}** and **{metric(regression['gd'], 'rmse'):.3f}**, respectively.",
        "",
        "## Reproducibility",
        "",
        "Run `python scripts/run_experiments.py` to regenerate `artifacts/results.json`, the CSV tables, and all figures. Run `streamlit run app.py` to open the explanatory interface.",
        "",
        "## Scope",
        "",
        "The system is an educational benchmark study, not a diagnostic device. Its conclusions concern this fixed experimental split and do not establish clinical validity.",
    ]
    return "\n".join(lines) + "\n"


def build_docx(data: dict[str, Any]) -> None:
    doc = Document()
    configure_styles(doc)
    add_cover(doc)
    section = doc.sections[-1]
    add_header(section)
    footer(section)

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "This mini-project implements a reproducible end-to-end machine-learning study across feature selection, linear regression, and logistic regression. The classification task uses 569 Wisconsin Diagnostic Breast Cancer observations with 30 morphology features. A full-feature baseline is compared with chi-square filtering, recursive feature elimination, and random-forest importance, each reduced to 10 features. The regression task uses 442 diabetes observations and the standardized BMI feature to compare the closed-form ordinary least-squares solution with batch gradient descent. Logistic regression is implemented from scratch with sigmoid probabilities and binary cross-entropy. Standard library models provide reference evaluations. The final evidence includes held-out metrics, convergence traces, fit lines, confusion counts, and manually integrated ROC-AUC values."
    )
    best_name, best = best_classifier(data)
    summary = doc.add_paragraph()
    summary.add_run("Principal result. ").bold = True
    summary.add_run(
        f"The strongest scratch classifier was {best_name}, with accuracy {best['scratch_metrics']['accuracy']:.3f} and ROC-AUC {best['scratch_metrics']['auc']:.3f}, while the regression gradient descent solution numerically approached analytical OLS."
    )

    doc.add_heading("1. Problem Definition and Objectives", level=1)
    doc.add_paragraph(
        "Clinical measurements are often correlated, high-dimensional, and costly to interpret. This study asks whether a smaller subset of tumour descriptors can preserve binary discrimination and whether transparent batch optimization reaches the same result as analytical regression. The objective is explanatory rigor rather than clinical deployment."
    )
    for text in (
        "Compare filter, wrapper, embedded, and unreduced feature spaces on the same held-out classification split.",
        "Derive and implement OLS and batch-gradient linear regression for a continuous outcome.",
        "Derive and implement logistic regression with sigmoid probabilities, binary cross-entropy, and batch updates.",
        "Compute performance, error composition, ROC geometry, and convergence from one deterministic experiment bundle.",
        "Expose the same evidence in source code, a Streamlit interface, an academic report, and a title-plus-five-content-slide defense deck.",
    ):
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("2. Dataset Descriptions", level=1)
    add_table(
        doc,
        ["Dataset", "Samples", "Inputs", "Target", "Role"],
        [
            ["Wisconsin Diagnostic Breast Cancer", "569", "30 real-valued morphology measurements", "Malignant / benign", "Selection and classification"],
            ["Diabetes progression", "442", "Standardized BMI used in the mini-project", "One-year progression score", "Continuous regression"],
        ],
        [1.45, 0.55, 1.8, 1.15, 1.55],
    )
    doc.add_paragraph(
        "The classification target is encoded with malignant as the positive class. The breast-cancer split is stratified 80/20. The diabetes split is also 80/20 with the same random seed. Chi-square receives Min-Max-scaled, non-negative training values; gradient-based models receive standardized values. Every scaler and selector is fitted only on the training partition."
    )
    add_figure(doc, "dataset_overview.png", "Figure 1. Dataset size and classification target distribution.")

    doc.add_heading("3. Mathematical Formulation", level=1)
    doc.add_heading("3.1 Feature selection", level=2)
    add_formula(doc, "Chi-square score", "χ² = Σ (O − E)² / E", "A large score indicates stronger statistical dependence between a non-negative feature and the class label.")
    doc.add_paragraph(
        "Recursive feature elimination repeatedly fits a logistic estimator, removes the weakest coefficient-ranked feature, and stops at 10 retained variables. The embedded method trains a balanced random forest and ranks features by accumulated impurity reduction. These methods provide model-independent, model-aware, and nonlinear views of relevance."
    )
    doc.add_heading("3.2 Linear regression", level=2)
    add_formula(doc, "Analytical slope and intercept", "m = Cov(x,y) / Var(x),    c = ȳ − mx̄", "For one predictor, these values minimize the sum of squared residuals directly.")
    add_formula(doc, "Mean-squared objective", "J(m,c) = (1/n) Σ(mxᵢ + c − yᵢ)²", "Batch gradient descent evaluates the complete training set at each epoch.")
    add_formula(doc, "Gradients and updates", "∂J/∂m = (2/n)Σ(ŷᵢ−yᵢ)xᵢ;  ∂J/∂c = (2/n)Σ(ŷᵢ−yᵢ);  θ ← θ − α∇J", "The learning rate controls the update magnitude; the loss history diagnoses convergence.")
    doc.add_heading("3.3 Logistic regression", level=2)
    add_formula(doc, "Sigmoid probability", "pᵢ = σ(zᵢ) = 1 / (1 + exp(−zᵢ)),    z = Xw + b", "A probability at or above 0.5 is classified as malignant.")
    add_formula(doc, "Binary cross-entropy", "L = −(1/n)Σ[yᵢ log pᵢ + (1−yᵢ)log(1−pᵢ)]", "Clipped probabilities keep logarithms numerically stable.")
    add_formula(doc, "Batch gradients", "∂L/∂w = Xᵀ(p−y)/n,    ∂L/∂b = Σ(p−y)/n", "Weights and bias are updated once per full-data epoch.")
    add_formula(doc, "Manual ROC-AUC", "AUC = Σ (FPRₖ₊₁−FPRₖ)(TPRₖ₊₁+TPRₖ)/2", "The ROC points are ordered by descending score and integrated using trapezoids.")

    doc.add_heading("4. Methodology and Architecture", level=1)
    for title, body in (
        ("Load and encode", "Load bundled datasets, encode malignant as class 1, and select BMI for the interpretable one-variable regression task."),
        ("Split and fit preprocessing", "Create deterministic 80/20 partitions; fit scaling and feature selectors on training observations only."),
        ("Construct feature spaces", "Retain all 30 inputs for the baseline and select 10 for each reduced strategy."),
        ("Train scratch and reference models", "Run batch-GD logistic regression on every feature space; compare OLS and linear GD; fit scikit-learn references."),
        ("Evaluate and export", "Calculate held-out metrics, confusion counts, ROC points, manual AUC, and convergence histories, then write JSON, CSV, and figures."),
    ):
        paragraph = doc.add_paragraph()
        paragraph.add_run(f"{title}. ").bold = True
        paragraph.add_run(body)

    doc.add_heading("5. Experimental Results", level=1)
    doc.add_heading("5.1 Feature selection and classification", level=2)
    class_rows = []
    for method, record in data["classification"].items():
        metrics = record["scratch_metrics"]
        class_rows.append(
            [
                method,
                str(record["n_features"]),
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1']:.3f}",
                f"{metrics['auc']:.3f}",
            ]
        )
    add_table(doc, ["Feature space", "n", "Accuracy", "Precision", "Recall", "F1", "AUC"], class_rows, [1.55, 0.42, 0.76, 0.76, 0.7, 0.65, 0.65])
    add_figure(doc, "feature_selection_comparison.png", "Figure 2. Held-out accuracy and ROC-AUC for every feature space.")
    add_figure(doc, "feature_selection_matrix.png", "Figure 3. Membership overlap among the three reduced feature subsets.")
    doc.add_paragraph(
        f"The highest AUC was obtained by {best_name}. It retained {best['n_features']} of 30 original measurements, a {100 * (1 - best['n_features'] / 30):.0f}% reduction. The comparison demonstrates that dimensionality and discrimination must be reported together: a reduced set is useful only when its held-out loss of signal is acceptably small."
    )

    doc.add_heading("5.2 Linear regression", level=2)
    regression_rows = []
    for name, key in (("Closed-form OLS", "ols"), ("Batch gradient descent", "gd"), ("Scikit-learn reference", "sklearn")):
        record = data["regression"][key]
        values = record["metrics"]
        regression_rows.append([name, f"{values['mse']:.3f}", f"{values['rmse']:.3f}", f"{values['mae']:.3f}", f"{values['r2']:.3f}"])
    add_table(doc, ["Method", "MSE", "RMSE", "MAE", "R²"], regression_rows, [2.1, 1.0, 1.0, 1.0, 1.0])
    add_figure(doc, "regression_fit.png", "Figure 4. Held-out diabetes outcomes with analytical and gradient-descent fit lines.")
    add_figure(doc, "linear_cost_convergence.png", "Figure 5. Linear batch-gradient training cost across epochs.")
    doc.add_paragraph(
        "The OLS, batch-gradient, and library solutions agree to numerical precision after convergence. The modest R² is expected for a single explanatory measurement and is part of the design: the task makes optimizer behaviour inspectable rather than maximizing predictive complexity."
    )

    doc.add_heading("5.3 Logistic regression diagnostics", level=2)
    metrics = best["scratch_metrics"]
    add_table(
        doc,
        ["TN", "FP", "FN", "TP", "Accuracy", "AUC"],
        [[str(metrics["tn"]), str(metrics["fp"]), str(metrics["fn"]), str(metrics["tp"]), f"{metrics['accuracy']:.3f}", f"{metrics['auc']:.3f}"]],
        [0.75, 0.75, 0.75, 0.75, 1.1, 1.1],
    )
    add_figure(doc, "classification_roc.png", "Figure 6. ROC curves and manually integrated AUC values for all feature spaces.", width=5.7)
    add_figure(doc, "classification_confusion_matrix.png", f"Figure 7. Held-out confusion matrix for the best-AUC feature space: {best_name}.", width=4.9)
    add_figure(doc, "logistic_loss_convergence.png", "Figure 8. Binary cross-entropy convergence for every feature space.")
    doc.add_paragraph(
        "Accuracy is not sufficient for this task because false negatives are malignant cases predicted benign. Precision, recall, F1, the confusion counts, and ROC-AUC therefore appear together. The declining cross-entropy histories confirm that the scratch optimizer follows a stable descent path for every subset."
    )

    doc.add_heading("6. Discussion", level=1)
    doc.add_paragraph(
        "The experiment supports three technical observations. First, different selectors agree on several high-value morphology measurements but are not interchangeable: chi-square captures marginal dependence, RFE reflects a linear estimator, and forest importance can express nonlinear splits. Second, batch gradient descent converges to the analytical one-variable regression optimum when scaling and learning rate are appropriate. Third, the scratch logistic implementation is capable of competitive held-out ranking while exposing the exact loss, gradients, threshold, and error composition."
    )
    doc.add_paragraph(
        "The recommended feature space is chosen by the highest held-out AUC, with feature count used as a secondary efficiency criterion. This is an experimental recommendation only. No clinical utility claim follows from a single benchmark split, and the system must not be used for patient-level decisions."
    )

    doc.add_heading("7. Limitations and Threats to Validity", level=1)
    for text in (
        "The benchmarks are small and curated; they do not model missingness, site variation, demographic shift, or prospective clinical workflows.",
        "A single held-out split provides a clear teaching experiment but does not estimate confidence intervals or cross-site generalization.",
        "Random-forest impurity importance can favour continuous variables with many possible split points.",
        "The diabetes regression intentionally uses one feature, so its predictive ceiling is lower than a multivariable model.",
        "The 0.5 logistic threshold is conventional, not optimized for a clinical cost function.",
    ):
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("8. Conclusion", level=1)
    doc.add_paragraph(
        f"Kriterion delivers a fully reproducible study in which feature reduction, analytical regression, and scratch gradient optimization are evaluated on held-out evidence. {best_name} achieved the strongest scratch ROC-AUC ({best['scratch_metrics']['auc']:.3f}) with {best['n_features']} measurements. Linear gradient descent approached the OLS solution and produced an equivalent regression fit. The principal contribution is not algorithmic novelty; it is a compact, inspectable pipeline that connects equations, code, convergence behaviour, metrics, and visual evidence without hiding the learning process."
    )

    doc.add_heading("References", level=1)
    references = [
        "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B",
        "Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least Angle Regression. Annals of Statistics, 32(2), 407–499.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html",
        "Scikit-learn developers. load_diabetes documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "Indira Gandhi Delhi Technical University for Women. Official institutional mark. https://www.igdtuw.ac.in/igdtuw-logo.png",
    ]
    for reference in references:
        doc.add_paragraph(reference, style="List Number")

    DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX_OUT)


def main() -> None:
    data = read_results()
    MD_OUT.write_text(build_markdown(data), encoding="utf-8")
    build_docx(data)
    print(DOCX_OUT)
    print(MD_OUT)


if __name__ == "__main__":
    main()
