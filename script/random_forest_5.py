# data handling things
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn things
from sklearn.model_selection import train_test_split
# rf
from sklearn.ensemble import RandomForestClassifier
# dt
from sklearn.tree import DecisionTreeClassifier
# xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# grid search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Load the Heart Disease dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
d = pd.read_csv(url, header=None, names=column_names, na_values='?')

# Drop rows with missing values
d = d.dropna()

# Split data into features and target
X = d.drop('target', axis=1)
y = d['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Grid search for random forest on small dataset, to avoid overfitting
param_grid = {
    'max_depth': [None, 5, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Grid search for xgboost
param_grid_xgb = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Call rf
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Call dt classifier with max_features sqrt min_samples_leaf 5 min_samples_split 2
dt = DecisionTreeClassifier(max_features='sqrt', min_samples_leaf=5, min_samples_split=2)

# Grid search rank using roc_auc
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')

# XGBoost grid search rank using roc_auc
grid_search_xgb = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')

# Fit
grid_search.fit(X_train, y_train)
grid_search_xgb.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Best estimator
rf = grid_search.best_estimator_
xgb = grid_search_xgb.best_estimator_

# Parameters
print(grid_search.best_params_)
print(grid_search_xgb.best_params_)

# Accuracy
y_pred = rf.predict(X_test)
y_pred_2 = dt.predict(X_test)
y_pred_3 = xgb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred_2))
print(accuracy_score(y_test, y_pred_3))

print(confusion_matrix(y_test, y_pred))
print(confusion_matrix(y_test, y_pred_2))
print(confusion_matrix(y_test, y_pred_3))