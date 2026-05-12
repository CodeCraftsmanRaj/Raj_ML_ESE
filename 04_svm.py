from sklearn.svm import SVR
from experiments_common import run_regression

run_regression(SVR(kernel="rbf"), "SVM (SVR)")
