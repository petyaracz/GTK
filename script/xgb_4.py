
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from ucimlrepo import fetch_ucirepo 
  
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
early_stage_diabetes_risk_prediction = fetch_ucirepo(id=529) 
  
# data (as pandas dataframes) 
X = early_stage_diabetes_risk_prediction.data.features 
y = early_stage_diabetes_risk_prediction.data.targets 

# dummy code X, drop first
X = pd.get_dummies(X, drop_first=True)

# encode y as 1 if Positive and 0 if negative
label_encoder = LabelEncoder()
# Unravel y if needed and encode it
y = np.ravel(y)  # Ensure y is 1D
y = label_encoder.fit_transform(y)
  
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# load xgboost
xgb = XGBClassifier()

# hyperparameter tuning for max_depth, min_child_weight, subsample, colsample_bytree, learning_rate
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.2]
}

# grid search

grid_search = GridSearchCV(estimator=xgb, 
                           param_grid=param_grid, 
                           n_jobs=-1, 
                           cv=2, 
                           verbose=2)

grid_search.fit(X_train, y_train)

# best params
grid_search.best_params_
# pred
xgb_pred = grid_search.predict(X_test)
# accuracy
xgb_acc = accuracy_score(y_test, xgb_pred)
# roc auc
xgb_roc = roc_auc_score(y_test, xgb_pred)
# confusion matrix  
xgb_cm = confusion_matrix(y_test, xgb_pred)
xgb_acc
xgb_roc
xgb_cm

