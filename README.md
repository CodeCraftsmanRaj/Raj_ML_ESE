# ML Experiments — Theory & How to Present (Viva Prep)

This repository contains small, repeatable experiments for common machine learning algorithms. The sections below summarize the theory, key formulas, evaluation metrics, typical plots, and concise talking points you can use during a viva.

---

## How to run

Install dependencies and run all experiments:

```bash
uv add -r requirements.txt
uv run abcd.py
```

Outputs (plots & metrics) are saved in the `results/` directory:
- `results/metrics.csv` — one row per experiment, fields: `experiment`, `metrics_json`.
- `results/<safe_experiment_name>_*.png` — plots (confusion matrix, tree, scatter, decision regions, components).

---

## General evaluation terms
- MSE / RMSE: mean squared error, root mean squared error. RMSE = sqrt(mean((y - y_pred)^2)). Useful for regression error magnitude.
- R² (coefficient of determination): fraction of variance explained by model. R² = 1 - SSE/SST.
- Accuracy: fraction of correct predictions (classification).
- Confusion matrix: tabulation of true vs predicted classes — use to compute precision/recall.
- Silhouette score (clustering): measures how similar an object is to its own cluster vs other clusters; in [-1,1].
- Explained variance (PCA): fraction of total variance captured by selected principal components.

---

## 1) Linear Regression

Goal
- Predict continuous target with a linear function of inputs.

Model
- y = w^T x + b
- Fit by minimizing MSE: minimize sum_i (y_i - w^T x_i - b)^2.
- Closed-form (normal eqn): w = (X^T X)^{-1} X^T y (when invertible).

Evaluation / Plots
- RMSE, R².
- Plot: scatter of actual vs predicted; regression line on feature vs target if single-feature.

Viva tips
- Explain assumptions: linearity, independent errors, homoscedasticity, no multicollinearity.
- Mention regularized variants (Ridge, Lasso) to prevent overfitting.

---

## 2) Decision Tree (Classification / Regression)

Goal
- Learn hierarchical, interpretable rules splitting the feature space.

Concepts
- Node splits chosen by maximizing information gain or minimizing impurity.
- Common impurity (classification): Gini impurity G = 1 - sum_k p_k^2.
- For regression trees, splits reduce variance (MSE).

Evaluation / Plots
- Classification: confusion matrix, accuracy.
- Tree plot: visualize splits, feature thresholds, leaf predictions, and impurity (Gini) values.

Viva tips
- Discuss overfitting with deep trees and role of pruning / `max_depth` / `min_samples_leaf`.
- Complexity: building a tree is roughly O(n_features * n_samples * log n_samples) with heuristics.

---

## 3) K-Nearest Neighbors (KNN)

Goal
- Predict based on labels of nearest neighbors in feature space.

Concepts
- Distance metric (Euclidean common), number of neighbors `k` controls bias-variance.
- For classification: majority vote; for regression: mean of neighbors.

Evaluation / Plots
- Confusion matrix for classification.
- Explain influence of `k` and feature scaling requirement.

Viva tips
- Explain computational cost at prediction time (need to compute distances to training points); mention KD-trees or approximate neighbors.

---

## 4) Support Vector Machine (SVM)

Goal
- Find a hyperplane maximizing margin between classes (classification). Can be used for regression (SVR).

Concepts
- For linear separable case: maximize margin = 2/||w|| subject to y_i (w^T x_i + b) >= 1.
- Soft-margin allows slack variables and hinge loss.
- Kernel trick allows nonlinear decision boundaries (RBF, polynomial).

Evaluation / Plots
- Accuracy and confusion matrix for classification.
- For 2D feature cases: decision boundary plot (we provide PCA-free scatter of first-two-features and optionally decision-region grid if you use only two features).

Viva tips
- Explain kernel, C parameter (tradeoff margin vs slack), and support vectors.

---

## 5) Ensemble — Bagging (Bootstrap Aggregating)

Goal
- Reduce variance by training many base learners on bootstrap samples and averaging (or voting).

Concepts
- Common base learner: decision tree (unstable, high variance) benefits most.
- Bagging decreases variance; each model sees different data.

Evaluation / Plots
- Regression: RMSE/R².
- Optionally show distribution of predictions across estimators (not in default scripts).

Viva tips
- Contrast with boosting (bagging aims to reduce variance; boosting reduces bias by sequential corrections).

---

## 6) Ensemble — Boosting (Gradient Boosting)

Goal
- Sequentially add weak learners to correct residuals of the ensemble (e.g., Gradient Boosting, XGBoost).

Concepts
- Each new tree fits the negative gradient (residual) of the loss function.
- Hyperparameters: learning rate, n_estimators, max_depth.

Evaluation / Plots
- Regression: RMSE/R².

Viva tips
- Explain early stopping and how learning rate affects performance and overfitting.

---

## 7) K-Means Clustering

Goal
- Partition data into K clusters by minimizing within-cluster sum of squared distances.

Objective
- minimize sum_{k} sum_{x in C_k} ||x - mu_k||^2 where mu_k is cluster mean.

Evaluation / Plots
- Silhouette score.
- Decision-region plot produced over first two features showing which region maps to which cluster.

Viva tips
- Sensitivity to initialization and K; use elbow method or silhouette to select K.

---

## 8) Expectation-Maximization / Gaussian Mixture Models (GMM)

Goal
- Model data as a mixture of Gaussian components; soft clustering via responsibilities.

Concepts
- E-step: compute responsibilities gamma_{ik} = P(z_i=k | x_i, params).
- M-step: update parameters (weights, means, covariances) to maximize expected complete-data log-likelihood.

Evaluation / Plots
- Silhouette score after assigning components.
- Decision-region plot over first two features.

Viva tips
- Discuss model selection (how many components), covariance types (diag/full), and difference vs K-means (soft assignments and elliptical clusters).

---

## 9) PCA (Principal Component Analysis)

Goal
- Linear dimensionality reduction by orthogonal projection that maximizes variance.

Math
- Compute covariance S = (1/n) X^T X (assuming centered X). Compute eigen decomposition S = V Λ V^T. Principal components are eigenvectors with largest eigenvalues.
- Projection Z = X V_k (top-k eigenvectors).

Evaluation / Plots
- `explained_variance_ratio_` for components; plot first two components scatter.

Viva tips
- Distinguish PCA (unsupervised variance maximization) from LDA (supervised discrimination).

---

## 10) LDA (Linear Discriminant Analysis)

Goal
- Supervised dimensionality reduction maximizing class separability.

Math
- Solve generalized eigenproblem: S_b w = lambda S_w w where S_b is between-class scatter and S_w is within-class scatter.

Evaluation / Plots
- Projected data in reduced space (usually up to C-1 dims for C classes).

Viva tips
- Discuss LDA assumptions: gaussian class-conditional, equal covariance across classes.

---

## 11) SVD (Truncated SVD)

Goal
- Factorize matrix A ≈ U_k Σ_k V_k^T; used for dimensionality reduction and latent features.

Math
- Full SVD: A = U Σ V^T. Truncated SVD keeps k largest singular values/vectors.

Evaluation / Plots
- Similar to PCA for centered data; plot first two components.

---

## Plot conventions in this repo
- Regression: `*_actual_vs_predicted.png` (scatter of y_true vs y_pred with y=x line).
- Decision tree classifier: `*_tree.png` (top levels) and `*_confusion_matrix.png`.
- KNN/SVM classifiers: `*_confusion_matrix.png` and `*_test_scatter.png` (first-two-features scatter of test set with misclassified points indicated).
- Clustering (KMeans/GMM): `*_decision_regions.png` — grid-based regions defined by cluster assignment using first-two features.
- Dimensionality reduction: `*_components.png` showing first two components and `explained_variance` in `metrics.csv`.

---

## Short viva checklist (practice answers)
- Explain the algorithm objective in one sentence.
- State one strength and one weakness for each method.
- Describe assumptions (e.g., linearity for linear regression, equal covariance for LDA).
- Give typical hyperparameters and how they affect bias/variance.
- Describe the plots you produced and what they indicate about your model.

---

If you want, I can:
- Convert this README into a PDF/slide-style cheat sheet for quick revision.
- Add a short `viva_notes.md` with one-line answers you can memorize.

Which would you like next?
