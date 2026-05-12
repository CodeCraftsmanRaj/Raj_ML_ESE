import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score


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
    print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2_score(y_test, pred):.3f}")


def run_clustering(model, name):
    X, _ = get_unsupervised_data()
    labels = model.fit_predict(X)
    print(f"{name} -> Silhouette: {silhouette_score(X, labels):.3f}")


def run_dim_reduction(model, name, supervised=False):
    X, y = get_unsupervised_data()
    Z = model.fit_transform(X, y) if supervised else model.fit_transform(X)
    msg = f"{name} -> shape: {Z.shape}"
    if hasattr(model, "explained_variance_ratio_"):
        msg += f", explained variance sum: {model.explained_variance_ratio_.sum():.3f}"
    print(msg)
