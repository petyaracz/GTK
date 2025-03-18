# here, classes are inbalanced, so we use f score to rank grid search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.tree import plot_tree
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load data
d = pd.read_csv('https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/credit_risk_dataset.csv')

# Pipeline 1: Data preprocessing
preprocessor = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first')),
    ('variance_threshold', VarianceThreshold(0.05))
])

# Separate features and target
X = d.drop('loan_status', axis=1)
y = d['loan_status']

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Pipeline 2: Grid Search with Decision Tree
dt = DecisionTreeClassifier(random_state=42, splitter='best', criterion='gini')
param_grid = {
    'max_depth': [3, 5, 9],
    'min_samples_leaf': [0.02, 0.04, 0.08],
    'min_samples_split': [0.02, 0.04, 0.08],
    'max_leaf_nodes': [5, 10, 20],
    'min_impurity_decrease': [0.0, 0.1, 0.2]
}
gs = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
gs.fit(X_train, y_train)

# Pipeline 3: Dummy Classifier
dc = DummyClassifier(strategy='most_frequent')
dc.fit(X_train, y_train)

# Best parameters
print(gs.best_params_)
# Best score
print(gs.best_score_)
# Best estimator
print(gs.best_estimator_)

# Predictions
y_pred1 = gs.predict(X_test)
y_pred2 = dc.predict(X_test)

# ROC AUC scores
print(roc_auc_score(y_test, y_pred1))
print(roc_auc_score(y_test, y_pred2))
print(f1_score(y_test, y_pred1))

# Confusion matrix
print(confusion_matrix(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred2))

# Plot tree
plt.figure(figsize=(20, 10))
plot_tree(gs.best_estimator_, filled=True, feature_names=preprocessor.named_steps['onehot'].get_feature_names_out())
plt.show()