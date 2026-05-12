from sklearn.svm import SVC
from experiments_common import run_classification

run_classification(SVC(kernel="rbf", probability=False), "SVM Classifier")
