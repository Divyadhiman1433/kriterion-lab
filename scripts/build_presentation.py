from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "artifacts" / "Kriterion_Defense_Deck.pptx"

PAPER = RGBColor(248, 247, 243)
NAVY = RGBColor(24, 45, 66)
SLATE = RGBColor(73, 85, 96)
TEAL = RGBColor(18, 111, 112)
BURGUNDY = RGBColor(139, 44, 61)


def add_text(slide, text, x, y, w, h, size, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    return shape


def add_rule(slide, x, y, w, color=TEAL):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Pt(1.5),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_image(slide, filename, x, y, w, h):
    return slide.shapes.add_picture(
        str(ASSETS / filename),
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )


def set_background(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER


def add_heading(slide, title):
    set_background(slide)
    add_text(slide, title, 0.72, 0.38, 11.9, 0.62, 35, NAVY, True)
    add_rule(slide, 0.72, 1.12, 11.9)


def add_bullets(slide, items, x, y, w, h):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = f"•  {item}"
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(17)
        paragraph.font.color.rgb = SLATE
        paragraph.space_after = Pt(18)
    return shape


def make_deck():
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    blank = deck.slide_layouts[6]

    slide = deck.slides.add_slide(blank)
    set_background(slide)
    add_image(slide, "brand/igdtuw-logo.png", 6.16, 0.42, 1.0, 1.0)
    add_text(slide, "Indira Gandhi Delhi Technical University for Women", 1.4, 1.66, 10.53, 0.42, 20, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, "Feature Selection and Scratch Optimisation\nfor Clinical Prediction", 1.0, 2.72, 11.33, 1.48, 42, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, "Divya", 4.9, 5.35, 3.53, 0.36, 22, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, "Roll number 00702102026", 4.4, 5.92, 4.53, 0.34, 17, SLATE, False, PP_ALIGN.CENTER)

    slide = deck.slides.add_slide(blank)
    add_heading(slide, "Two datasets answer different questions")
    add_text(slide, "Classification", 0.78, 1.52, 4.9, 0.4, 24, BURGUNDY, True)
    add_text(slide, "The breast-cancer dataset contains 569 observations and 30 measurements. It is used to compare the complete feature set with three reduced subsets.", 0.78, 2.08, 4.9, 0.94, 17, SLATE)
    add_text(slide, "Regression", 7.62, 1.52, 4.9, 0.4, 24, TEAL, True)
    add_text(slide, "The diabetes dataset contains 442 observations. One BMI predictor keeps the OLS and gradient-descent solutions directly comparable.", 7.62, 2.08, 4.9, 0.94, 17, SLATE)
    add_text(slide, "The comparison follows the same sequence", 0.78, 3.58, 7.2, 0.42, 24, NAVY, True)
    add_text(slide, "1", 0.82, 4.32, 0.38, 0.34, 22, TEAL, True)
    add_text(slide, "Fit scaling and feature selection on the training data only.", 1.42, 4.32, 10.4, 0.4, 18, SLATE)
    add_text(slide, "2", 0.82, 5.02, 0.38, 0.34, 22, TEAL, True)
    add_text(slide, "Train each scratch model on the same fixed split.", 1.42, 5.02, 10.4, 0.4, 18, SLATE)
    add_text(slide, "3", 0.82, 5.72, 0.38, 0.34, 22, TEAL, True)
    add_text(slide, "Evaluate every method on untouched test data.", 1.42, 5.72, 10.4, 0.4, 18, SLATE)

    slide = deck.slides.add_slide(blank)
    add_heading(slide, "RFE retained accuracy with 20 fewer features")
    add_image(slide, "figures/feature_selection_comparison.png", 0.72, 1.42, 7.52, 4.55)
    add_text(slide, "Complete feature set", 8.72, 1.62, 3.55, 0.38, 24, NAVY, True)
    add_text(slide, "Using all 30 measurements produced 0.9737 accuracy and 0.9967 ROC–AUC.", 8.72, 2.15, 3.55, 0.9, 18, SLATE)
    add_text(slide, "RFE subset", 8.72, 3.42, 3.55, 0.38, 24, TEAL, True)
    add_text(slide, "Using 10 selected measurements preserved 0.9737 accuracy and increased ROC–AUC slightly to 0.9974.", 8.72, 3.95, 3.55, 1.12, 18, SLATE)
    add_text(slide, "For this split, RFE reduced dimensionality without reducing classification accuracy.", 0.78, 6.24, 11.6, 0.65, 18, NAVY, True)

    slide = deck.slides.add_slide(blank)
    add_heading(slide, "OLS and gradient descent produced the same line")
    add_image(slide, "figures/regression_fit.png", 0.72, 1.42, 5.84, 3.72)
    add_image(slide, "figures/linear_cost_convergence.png", 6.78, 1.42, 5.84, 3.72)
    add_text(slide, "Both methods recovered a slope of 998.58 and an intercept of 152.00, so their fitted lines overlap.", 0.78, 5.42, 5.78, 0.82, 18, NAVY, True)
    add_text(slide, "Their test results are identical: RMSE 63.73, MAE 52.26 and R² 0.233.", 6.84, 5.42, 5.48, 0.82, 18, NAVY, True)
    add_text(slide, "The moderate R² reflects the limits of BMI as a single predictor rather than a failure of gradient descent.", 0.78, 6.42, 11.55, 0.62, 17, SLATE)

    slide = deck.slides.add_slide(blank)
    add_heading(slide, "The RFE classifier retained high discrimination")
    add_text(slide, "The scratch model uses sigmoid probabilities and minimises binary cross-entropy with batch gradient descent.", 0.78, 1.32, 11.55, 0.62, 18, SLATE)
    add_image(slide, "figures/classification_roc.png", 0.72, 2.05, 6.15, 3.78)
    add_image(slide, "figures/classification_confusion_matrix.png", 7.12, 2.05, 3.55, 3.78)
    metrics = [("Accuracy", "0.9737"), ("Precision", "0.9756"), ("Recall", "0.9524"), ("F1 score", "0.9639"), ("ROC–AUC", "0.9974")]
    for index, (name, value) in enumerate(metrics):
        y = 2.08 + index * 0.72
        add_text(slide, name, 11.05, y, 1.45, 0.22, 16, SLATE)
        add_text(slide, value, 11.05, y + 0.27, 1.45, 0.25, 18, TEAL, True)
    add_text(slide, "At a threshold of 0.5, the model produced 71 true negatives, 40 true positives, one false positive and two false negatives.", 0.78, 6.25, 11.55, 0.64, 18, NAVY, True)

    slide = deck.slides.add_slide(blank)
    add_heading(slide, "The results are exploratory, not clinical")
    add_text(slide, "What the experiment supports", 0.78, 1.5, 5.35, 0.42, 24, TEAL, True)
    add_bullets(slide, ["RFE reduced the input from 30 features to 10 without reducing test accuracy.", "Batch gradient descent reproduced the analytical regression solution.", "The scratch classifier achieved high discrimination on the fixed test split."], 0.78, 2.34, 5.35, 3.15)
    add_text(slide, "What the experiment cannot establish", 7.18, 1.5, 5.35, 0.42, 24, BURGUNDY, True)
    add_bullets(slide, ["One train–test split cannot quantify performance uncertainty.", "The models were not calibrated or tested on an independent clinical dataset.", "The regression used one predictor for clarity rather than maximum accuracy."], 7.18, 2.34, 5.35, 3.15)
    add_text(slide, "The project demonstrates transparent machine-learning methods and reproducible comparisons; it is not a diagnostic system.", 0.78, 6.28, 11.55, 0.62, 18, NAVY, True)

    if len(deck.slides) != 6:
        raise RuntimeError("The presentation must contain one title slide and five content slides")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    deck.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(make_deck())
