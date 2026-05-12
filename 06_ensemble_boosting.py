from sklearn.ensemble import GradientBoostingRegressor
from experiments_common import run_regression

run_regression(GradientBoostingRegressor(random_state=42), "Ensemble Boosting")
