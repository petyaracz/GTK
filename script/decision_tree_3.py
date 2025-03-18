# here, classes are inbalanced, so we use f score to rank grid search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# import test train split decision tree classifier, grid search, accuracy score, confusion matrix
from sklearn.tree import DecisionTreeClassifier
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

# heatmap of d
sns.heatmap(d.corr(), annot=False, cmap='coolwarm')
plt.show()

# histogram of every single column
d.hist(figsize=(20, 20))
plt.show()

# plot of loan_status x other columns as boxplot
# hahaha don't run this
#for col in d.columns:
#    if col != 'loan_status':
#        d.boxplot(column=col, by='loan_status')
#        plt.title(col)
#        plt.show()

# use variance threshold to remove low variance features
vt = VarianceThreshold(0.05)
vt.fit(d)
d2 = d[d.columns[vt.get_support()]]

# nrow d2 vs nrow d
d.shape
d2.shape
# columns in d not in d2
set(d.columns) - set(d2.columns)

# y is loan_status
# X is the rest
X = d2.drop('loan_status', axis=1)
y = d2['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier (best and gini are defaults btw)
dt = DecisionTreeClassifier(random_state=42, splitter='best', criterion='gini')

# Grid Search
param_grid = {
    'max_depth': [3, 5, 9, 15],
    'min_samples_leaf': [0.01, 0.02, 0.04, 0.08],
    'min_samples_split': [0.01, 0.02, 0.04, 0.08],
    'max_leaf_nodes': [5, 10, 20, 100, 200],
    'min_impurity_decrease': [0.0, 0.1, 0.2]
}

# gs w/ roc auc
gs = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1)

gs.fit(X_train, y_train)

# dummy classifier
dc = DummyClassifier(strategy='most_frequent')
dc.fit(X_train, y_train)

# Best parameters
print(gs.best_params_)
# Best score
print(gs.best_score_)
# Best estimator
print(gs.best_estimator_)
# roc auc on test
y_pred1 = gs.predict(X_test)
y_pred2 = dc.predict(X_test)
# ROC AUC scores

print(roc_auc_score(y_test, y_pred1))
print(roc_auc_score(y_test, y_pred2))
print(f1_score(y_test, y_pred1))
print(f1_score(y_test, y_pred2))

# make ROC AUC plot
# ROC AUC plot
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, gs.predict_proba(X_test)[:, 1])
fpr2, tpr2, thresholds2 = roc_curve(y_test, dc.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label='Decision Tree')
plt.show()

# Confusion matrix
print(confusion_matrix(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred2))
# with labels that say true positive, true negative, false positive, false negative
cm = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
cm2 = pd.DataFrame(cm2, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])


plt.figure(figsize=(20, 10))
plot_tree(gs.best_estimator_, filled=True, feature_names=X.columns)
plt.show()