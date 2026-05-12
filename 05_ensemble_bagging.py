from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from experiments_common import run_regression

run_regression(
    BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42),
    "Ensemble Bagging",
)
