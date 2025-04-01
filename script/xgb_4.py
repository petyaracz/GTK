
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# fetch dataset 
X, y = fetch_openml("heart-disease", return_X_y=True, target_column='target', as_frame=True)
# convert y to int
y = y.astype('int')
  
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

# plot roc curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_pred)
plt.plot(fpr, tpr, label='XGBoost (area = {:.2f})'.format(xgb_roc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()