# import stuff

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# load dataset from UCI repository
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 

# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# define a list of random integers between 0 and 48842
#random_integers = np.random.randint(0, 48842, size=1000)

# keep only the rows with the random integers in X, y
#X = X.iloc[random_integers]
#y = y.iloc[random_integers]

# dummy code X
X = pd.get_dummies(X, drop_first=True)

# dummy code y
y = pd.get_dummies(y, drop_first=True)
y.dtypes
# keep income_>50K
y = y['income_>50K']

# count y
y.value_counts()

# training test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Decision Tree

# hyperparameter tuning with many hyperparameters
param_grid_dt = {
    'max_depth': [None, 5, 15, 20],
    'min_samples_split': [2, 4, 10, 100],
    'min_samples_leaf': [5, 10, 50, 100, 200],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=3, n_jobs=-1, scoring='roc_auc')

grid_search_dt.fit(X_train, y_train)

# random forest

# parameter tuning
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 15, 20],
    'min_samples_split': [2, 4, 10, 100],
    'min_samples_leaf': [5, 10, 50, 100, 200],
    'max_features': [None, 'sqrt', 'log2'],
    'max_samples': [0.1, 0.25, 0.5, 0.75, 1],
}

grid_search_rf = GridSearchCV(RandomForestClassifier(n_estimators=100), param_grid_rf, cv=3, n_jobs=-1, scoring='roc_auc')

grid_search_rf.fit(X_train, y_train)

# xgboost

# wide parameter tuning
param_grid_xgb = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.25, 0.5, 0.7, 1],
    'colsample_bytree': [0.25, 0.5, 0.7, 1],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 5, 10]
}

grid_search_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=3, n_jobs=-1, scoring='roc_auc')

grid_search_xgb.fit(X_train, y_train)

# predict and evaluate

# Decision Tree
y_pred_dt = grid_search_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
# f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
# Random Forest
y_pred_rf = grid_search_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
# f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
# XGBoost
y_pred_xgb = grid_search_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)

# print results
print('Decision Tree')
print(grid_search_dt.best_params_)
print('ROC AUC:', roc_auc_dt)
print('Random Forest')
print(grid_search_rf.best_params_)
print('ROC AUC:', roc_auc_rf)
print('XGBoost')
print(grid_search_xgb.best_params_)
print('ROC AUC:', roc_auc_xgb)
