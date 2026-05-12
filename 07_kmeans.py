from sklearn.cluster import KMeans
from experiments_common import run_clustering

run_clustering(KMeans(n_clusters=3, random_state=42, n_init=10), "K-Means")
