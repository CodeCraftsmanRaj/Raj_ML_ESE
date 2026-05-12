from sklearn.tree import DecisionTreeClassifier
from experiments_common import run_classification

run_classification(DecisionTreeClassifier(random_state=42), "Decision Tree Classifier")
