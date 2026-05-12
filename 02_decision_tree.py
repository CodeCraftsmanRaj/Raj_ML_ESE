from sklearn.tree import DecisionTreeRegressor
from experiments_common import run_regression

run_regression(DecisionTreeRegressor(random_state=42), "Decision Tree")
