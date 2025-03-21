# not the best data for this
# here, classes are inbalanced, so we use f score to rank grid search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# import test train split decision tree classifier, grid search, accuracy score, confusion matrix
from sklearn.tree import DecisionTreeClassifier
# import rf
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.tree import plot_tree
# load variance threshold, pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


d = pd.read_csv('https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/credit_risk_dataset.csv')

d.dtypes
# dummy code d
d = pd.get_dummies(d, drop_first=True)

# use variance threshold to remove low variance features
vt = VarianceThreshold(0.05)
vt.fit(d)
d = d[d.columns[vt.get_support()]]

# y is loan_status
# X is the rest
X = d.drop('loan_status', axis=1)
y = d['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier (best and gini are defaults btw)
dt = DecisionTreeClassifier(max_depth=20, max_leaf_nodes=200, min_samples_split=0.001, random_state=42)

# RF classifier
rf = RandomForestClassifier(random_state=42, n_estimators = 50)

# Grid Search
param_grid = {
    'max_depth': [1, 10, 20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [100, 200],
    #'min_samples_leaf': [0.001, 0.01],
    'min_samples_split': [0.0005, 0.001, 0.01],
    #'min_impurity_decrease': [0.0, 0.01],
    'max_samples': [0.8, 0.9, 1]
}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1)

dt.fit(X_train, y_train)
gs.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_rf = gs.predict(X_test)

# cor of pred_dt and _rf
print(np.corrcoef(y_pred_dt, y_pred_rf))

# with labels that say true positive, true negative, false positive, false negative
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_dt = pd.DataFrame(cm_dt, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
cm_rf = pd.DataFrame(cm_rf, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])

print(cm_dt)
print(cm_rf)
print(roc_auc_score(y_test, y_pred_dt))
print(roc_auc_score(y_test, y_pred_rf))

# parameters of dt
print(dt.get_params())
# best estimates, rf
print(gs.best_params_)
