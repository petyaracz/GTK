############
# setup
############

from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# load sklearn dt
from sklearn.tree import DecisionTreeClassifier
# load sklearn random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# acc score roc auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
  
############
# dataset
############
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets.values.ravel() # to flatten array
  
# heatmap of X
plt.figure(figsize=(10, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# count values of y
print(y.value_counts())
X.dtypes

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

############
# models
############

# Decision Tree
dt = DecisionTreeClassifier()

# grid for DT
param_grid_dt = {
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

# grid search
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5)

# rf
rf = RandomForestClassifier(n_estimators=100)

# grid for RF
param_grid_rf = {
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth' : [5, 50],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [2, 10],
    'max_samples': [0.33, 0.66, 1]  # Add this line to include max_samples
}

# grid search
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)

############
# fit
############

grid_dt.fit(X_train, y_train)
# best pars
grid_dt.best_params_
grid_rf.fit(X_train, y_train)
grid_rf.best_params_

############
# predict
############

y_pred_dt = grid_dt.predict(X_test)
y_pred_rf = grid_rf.predict(X_test)

# accuracy
accuracy_score(y_test, y_pred_dt)
accuracy_score(y_test, y_pred_rf)

# confusion matrix
confusion_matrix(y_test, y_pred_dt)
confusion_matrix(y_test, y_pred_rf)

# best params
print(grid_dt.best_params_)
print(grid_rf.best_params_)

# plot dt tree

plt.figure(figsize=(20, 20))    
plot_tree(grid_dt.best_estimator_, filled=True)
plt.show()

# plot best rf tree
plt.figure(figsize=(20, 20))
plot_tree(grid_rf.best_estimator_[0], filled=True)
plt.show()