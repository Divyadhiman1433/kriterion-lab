# Feature Selection and Scratch Optimization for Clinical Prediction

**Kriterion — An Auditable Clinical Prediction Laboratory**  
Divya · 00702102026  
Indira Gandhi Delhi Technical University for Women

## Abstract

This project studies how feature selection and first-principles optimization affect two reproducible clinical prediction tasks. The classification pipeline uses the Wisconsin Diagnostic Breast Cancer dataset and compares all 30 measurements with chi-square, recursive feature elimination, and random-forest subsets. The regression pipeline uses standardized BMI to predict one-year diabetes progression and compares analytical ordinary least squares with batch gradient descent. Scratch logistic regression minimizes binary cross-entropy for every feature subset. All preprocessing is fit on training data only, all reported metrics are computed on held-out data, and the app, report, and presentation reuse one deterministic result bundle.

## Measured Outcome

The highest scratch ROC-AUC was **0.997** using **RFE wrapper** with 10 features. The analytical and gradient-descent regression models produced test RMSE values of **63.732** and **63.732**, respectively.

## Reproducibility

Run `python scripts/run_experiments.py` to regenerate `artifacts/results.json`, the CSV tables, and all figures. Run `streamlit run app.py` to open the explanatory interface.

## Scope

The system is an educational benchmark study, not a diagnostic device. Its conclusions concern this fixed experimental split and do not establish clinical validity.
