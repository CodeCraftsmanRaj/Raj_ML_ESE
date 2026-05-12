from sklearn.mixture import GaussianMixture
from experiments_common import run_clustering

run_clustering(GaussianMixture(n_components=3, random_state=42), "Expectation-Maximization (GMM)")
