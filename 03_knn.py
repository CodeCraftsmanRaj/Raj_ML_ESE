from sklearn.neighbors import KNeighborsClassifier
from experiments_common import run_classification

run_classification(KNeighborsClassifier(n_neighbors=5), "KNN Classifier")
