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

import json
import csv
from pathlib import Path

import numpy as np
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
    s = "".join(c for c in s if c.isalnum() or c == "_")
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

    metrics = {"experiment": name, "rmse": float(rmse), "r2": float(r2)}
    _save_metrics_row(metrics)

    # Actual vs Predicted scatter + y=x line
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=pred, ax=ax)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name} - Actual vs Predicted")
    _save_fig(fig, f"{name}_actual_vs_predicted")


def run_clustering(model, name):
    X, y = get_unsupervised_data()
    labels = model.fit_predict(X)
    sil = float(silhouette_score(X, labels))
    print(f"{name} -> Silhouette: {sil:.3f}")

    metrics = {"experiment": name, "silhouette": sil}
    _save_metrics_row(metrics)

    # Decision-region plot using first two features: create grid and predict cluster for each grid cell
    try:
        if X.shape[1] >= 2:
            X_vis = X[:, :2]
            x_min, x_max = X_vis[:, 0].min() - 1.0, X_vis[:, 0].max() + 1.0
            y_min, y_max = X_vis[:, 1].min() - 1.0, X_vis[:, 1].max() + 1.0
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]

            # Build full-dimension grid by filling other features with column means
            grid_full = np.tile(X.mean(axis=0), (grid.shape[0], 1))
            grid_full[:, 0:2] = grid

            try:
                Z = model.predict(grid_full)
            except Exception:
                Z = model.fit_predict(grid_full)
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.contourf(xx, yy, Z, alpha=0.3, cmap="Pastel1")
            sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=labels, palette="tab10", ax=ax, legend=False)
            ax.set_xlabel("feature_0")
            ax.set_ylabel("feature_1")
            ax.set_title(f"{name} - decision regions (first 2 features)")
            _save_fig(fig, f"{name}_decision_regions")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(range(X.shape[0]), X[:, 0], c=labels)
            ax.set_xlabel("index")
            ax.set_ylabel("feature_0")
            ax.set_title(f"{name} - clusters")
            _save_fig(fig, f"{name}_clusters")
    except Exception:
        pass


def run_classification(model, name):
    X, y = get_unsupervised_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from sklearn.metrics import accuracy_score, confusion_matrix

    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print(f"{name} -> Accuracy: {acc:.3f}")

    metrics = {"experiment": name, "accuracy": acc}
    _save_metrics_row(metrics)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - Confusion Matrix")
    _save_fig(fig, f"{name}_confusion_matrix")

    # Decision tree and feature importances if applicable
    try:
        from sklearn import tree

        if hasattr(model, "tree_") or "decisiontree" in model.__class__.__name__.lower():
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            tree.plot_tree(model, filled=True, ax=ax2, max_depth=3, impurity=True)
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

    # Simple scatter plot using first two original features
    try:
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        if X_test.shape[1] >= 2:
            sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, style=(preds != y_test), palette="tab10", ax=ax4)
            ax4.set_xlabel("feature_0")
            ax4.set_ylabel("feature_1")
        else:
            ax4.scatter(range(len(X_test)), X_test[:, 0], c=y_test)
            ax4.set_xlabel("index")
            ax4.set_ylabel("feature_0")
        ax4.set_title(f"{name} - Test set scatter (first 2 features)")
        _save_fig(fig4, f"{name}_test_scatter")
    except Exception:
        pass


def run_dim_reduction(model, name, supervised=False):
    X, y = get_unsupervised_data()
    Z = model.fit_transform(X, y) if supervised else model.fit_transform(X)
    msg = f"{name} -> shape: {Z.shape}"
    expl_var = None
    if hasattr(model, "explained_variance_ratio_"):
        expl_var = float(model.explained_variance_ratio_.sum())
        msg += f", explained variance sum: {expl_var:.3f}"
    print(msg)

    metrics = {"experiment": name, "shape0": int(Z.shape[0]), "shape1": int(Z.shape[1])}
    if expl_var is not None:
        metrics["explained_variance_sum"] = expl_var
    _save_metrics_row(metrics)

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
        ax.plot(Z[:, 0])
        ax.set_title(f"{name} - component 0")
        _save_fig(fig, f"{name}_component0")
