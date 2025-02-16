from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
X, y = load_your_data()  # Replace with your data loading function

# Define the model
rf = RandomForestClassifier()

# Define the Leave-One-Out cross-validation
loo = LeaveOneOut()

scores = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print(f"Average accuracy: {np.mean(scores)}")