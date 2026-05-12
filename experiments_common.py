import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
METRICS_CSV = RESULTS_DIR / "metrics.csv"

def _save_metrics_row(row: dict):
    # Store a compact CSV where each row is experiment + JSON metrics string
    file_exists = METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["experiment", "metrics_json"])
        writer.writerow([row.get("experiment", ""), json.dumps({k: v for k, v in row.items() if k != "experiment"})])

def _safe_name(name: str) -> str:
    s = name.lower()
    for ch in ["/", "(", ")", "-", ","]:
        s = s.replace(ch, "_")
    s = s.replace(" ", "_")
    # keep only alnum and underscore
    s = "".join(c for c in s if c.isalnum() or c == "_")
    # collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def _save_fig(fig, name: str):
    path = RESULTS_DIR / f"{_safe_name(name)}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def get_regression_data():
    X, y = make_regression(n_samples=300, n_features=5, noise=20, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_unsupervised_data():
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_classes=3,
        random_state=42,
    )
    return StandardScaler().fit_transform(X), y


def run_regression(model, name):
    X_train, X_test, y_train, y_test = get_regression_data()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.3f}")

    # Save metrics
    metrics = {"experiment": name, "rmse": float(rmse), "r2": float(r2)}
    _save_metrics_row(metrics)

    # Plot Actual vs Predicted
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=pred, ax=ax1)
    ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"{name} - Actual vs Predicted")
    p1 = _save_fig(fig1, f"{name}_actual_vs_predicted")

    # Residuals histogram
    resid = y_test - pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(resid, kde=True, ax=ax2)
    ax2.set_title(f"{name} - Residuals")
    _save_fig(fig2, f"{name}_residuals")

    # register plot locations
    (RESULTS_DIR / f"{_safe_name(name)}.json").write_text(json.dumps({"metrics": metrics, "plots": [str(p1)]}))


def run_clustering(model, name):
    X, y = get_unsupervised_data()
    labels = model.fit_predict(X)
    sil = float(silhouette_score(X, labels))
    print(f"{name} -> Silhouette: {sil:.3f}")

    # Save metrics
    metrics = {"experiment": name, "silhouette": sil}
    _save_metrics_row(metrics)

    # Visualize clusters in 2D (PCA)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels, palette="tab10", ax=ax, legend=False)
    ax.set_title(f"{name} - clusters (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _save_fig(fig, f"{name}_clusters_pca")

    (RESULTS_DIR / f"{_safe_name(name)}.json").write_text(json.dumps({"metrics": metrics}))


def run_classification(model, name):
    # Train/test on the classification dataset
    X, y = get_unsupervised_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from sklearn.metrics import accuracy_score, confusion_matrix

    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print(f"{name} -> Accuracy: {acc:.3f}")

    # Save metrics
    metrics = {"experiment": name, "accuracy": acc}
    _save_metrics_row(metrics)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - Confusion Matrix")
    _save_fig(fig, f"{name}_confusion_matrix")

    # If Decision Tree classifier, save tree plot and feature importances
    try:
        from sklearn import tree

        if hasattr(model, "tree_") or model.__class__.__name__.lower().find("decisiontree") >= 0:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            tree.plot_tree(model, filled=True, ax=ax2, max_depth=3)
            ax2.set_title(f"{name} - Decision Tree (top levels)")
            _save_fig(fig2, f"{name}_tree")

            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(x=list(range(len(fi))), y=fi, ax=ax3)
                ax3.set_title(f"{name} - Feature importances")
                _save_fig(fig3, f"{name}_feature_importances")
    except Exception:
        pass

    (RESULTS_DIR / f"{_safe_name(name)}.json").write_text(json.dumps({"metrics": metrics}))


def run_dim_reduction(model, name, supervised=False):
    X, y = get_unsupervised_data()
    Z = model.fit_transform(X, y) if supervised else model.fit_transform(X)
    msg = f"{name} -> shape: {Z.shape}"
    expl_var = None
    if hasattr(model, "explained_variance_ratio_"):
        expl_var = float(model.explained_variance_ratio_.sum())
        msg += f", explained variance sum: {expl_var:.3f}"
    print(msg)

    # Save metrics
    metrics = {"experiment": name, "shape0": int(Z.shape[0]), "shape1": int(Z.shape[1])}
    if expl_var is not None:
        metrics["explained_variance_sum"] = expl_var
    _save_metrics_row(metrics)

    # Plot first two components
    fig, ax = plt.subplots(figsize=(6, 5))
    if Z.shape[1] >= 2:
        hue_vals = y if supervised else None
        plot_kwargs = {"x": Z[:, 0], "y": Z[:, 1], "ax": ax}
        if hue_vals is not None:
            plot_kwargs.update({"hue": hue_vals, "palette": "tab10", "legend": False})
        sns.scatterplot(**plot_kwargs)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"{name} - first 2 components")
        _save_fig(fig, f"{name}_components")
    else:
        # fallback: plot component values
        ax.plot(Z[:, 0])
        ax.set_title(f"{name} - component 0")
        _save_fig(fig, f"{name}_component0")

    (RESULTS_DIR / f"{_safe_name(name)}.json").write_text(json.dumps({"metrics": metrics}))
