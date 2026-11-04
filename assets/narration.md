# Presentation Narration

## Title slide

Good morning. I am Divya. My project studies feature selection and optimization methods for regression and binary classification. It combines three areas covered in the laboratory experiments: filter, wrapper, and embedded feature selection; closed-form and gradient-descent linear regression; and logistic regression implemented from scratch. The purpose is to compare these methods using a single reproducible workflow.

## Slide 1 — Two datasets answer different questions

The project uses two datasets available through scikit-learn.

For classification, I use the Wisconsin Diagnostic Breast Cancer dataset. It contains 569 observations and 30 numerical measurements. The target identifies whether a tumour is malignant or benign.

For regression, I use the diabetes progression dataset, which contains 442 observations. I intentionally use BMI as a single predictor so that the closed-form regression equation can be compared clearly with batch gradient descent.

The data is divided into training and testing sets. Scaling and feature selection are fitted only on the training data. Each model is then evaluated on untouched test data. This prevents information from the test set from influencing model training.

## Slide 2 — RFE retained accuracy with 20 fewer features

This slide compares four feature spaces: all 30 features, chi-square selection, recursive feature elimination, and random-forest importance. Each selection method retains 10 features.

The complete feature set achieved an accuracy of 0.9737 and a ROC-AUC of 0.9967.

Recursive feature elimination also achieved an accuracy of 0.9737, despite using only 10 features. Its ROC-AUC increased slightly to 0.9974.

The chi-square and random-forest subsets also maintained high AUC values, but their accuracy was lower at 0.9561.

Therefore, RFE gave the best trade-off in this experiment. It removed two-thirds of the original features without reducing test accuracy.

## Slide 3 — OLS and gradient descent produced the same line

The regression module compares two optimization approaches.

The first is ordinary least squares. Its slope is calculated by dividing the covariance of the predictor and target by the variance of the predictor. The intercept is then calculated using their mean values.

The second approach is batch gradient descent. It begins with initial slope and intercept values and repeatedly updates them to reduce mean squared error.

Both methods produced a slope of approximately 998.58 and an intercept of 152. Their fitted lines therefore overlap.

They also produced identical test results: an RMSE of 63.73, an MAE of 52.26, and an R-squared value of 0.233.

The cost curve shows that gradient descent converges toward the analytical OLS solution. The moderate R-squared value is expected because the model uses only BMI rather than all available predictors.

## Slide 4 — The RFE classifier retained high discrimination

The classification model is logistic regression implemented from scratch.

First, the model calculates a weighted sum of the input features. The sigmoid function converts this value into a probability between zero and one. A probability of at least 0.5 is classified as malignant.

The model is trained using batch gradient descent to minimize binary cross-entropy loss.

Using the 10 features selected by RFE, the model achieved an accuracy of 0.9737, precision of 0.9756, recall of 0.9524, an F1 score of 0.9639, and a ROC-AUC of 0.9974.

The confusion matrix contains 71 true negatives, 40 true positives, one false positive, and two false negatives.

The ROC curve shows strong separation between the two classes, while the loss curve confirms that binary cross-entropy decreases during training.

## Slide 5 — The results are exploratory, not clinical

The experiment supports three main conclusions.

First, RFE reduced the classification input from 30 features to 10 without reducing test accuracy.

Second, batch gradient descent reproduced the analytical regression solution. This validates the gradient calculations and update rules used in the scratch implementation.

Third, the scratch logistic classifier achieved strong test performance using the reduced feature set.

However, the project also has limitations. It uses one fixed train-test split, so it does not measure variation across different samples. The models have not been calibrated or evaluated on an independent clinical dataset. The regression task also uses only one predictor for interpretability.

Therefore, this project demonstrates feature selection, optimization, and model evaluation in a transparent and reproducible way. It should be treated as an academic machine-learning study, not as a clinical diagnostic system.
