from sklearn.neighbors import KNeighborsRegressor
from experiments_common import run_regression

run_regression(KNeighborsRegressor(n_neighbors=5), "KNN")
