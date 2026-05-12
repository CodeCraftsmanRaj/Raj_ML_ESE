from sklearn.decomposition import TruncatedSVD
from experiments_common import run_dim_reduction

run_dim_reduction(TruncatedSVD(n_components=2, random_state=42), "SVD")
