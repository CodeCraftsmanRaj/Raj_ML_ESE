from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from experiments_common import get_unsupervised_data

X, _ = get_unsupervised_data()
labels = GaussianMixture(n_components=3, random_state=42).fit_predict(X)
print(f"Expectation-Maximization (GMM) -> Silhouette: {silhouette_score(X, labels):.3f}")
