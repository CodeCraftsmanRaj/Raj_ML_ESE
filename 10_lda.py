from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from experiments_common import run_dim_reduction

run_dim_reduction(LinearDiscriminantAnalysis(n_components=2), "LDA", supervised=True)
