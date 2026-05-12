from sklearn.decomposition import PCA
from experiments_common import run_dim_reduction

run_dim_reduction(PCA(n_components=2, random_state=42), "PCA")
