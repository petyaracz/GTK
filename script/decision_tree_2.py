url = 'https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/diabetes_dataset.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# import test train split decision tree classifier, grid search, accuracy score, confusion matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
# import pipeline
from sklearn.pipeline import Pipeline

d = pd.read_csv(url)

# draw correlation matrix for all columns
plt.figure(figsize=(10,10))
sns.heatmap(d.corr(), annot=True, cmap='coolwarm')
plt.show()

# draw hist for all columns
plt.figure(figsize=(10,10))
d.hist()
plt.show()

# use variance threshold to remove columns with low variance

# define the threshold
vt = VarianceThreshold(threshold=0.1)

# fit the threshold to the data
vt.fit(d)

# get the indices of the features that are being kept
feat_var_threshold = d.columns[vt.get_support()]
# get proportion of features that are being kept
len(feat_var_threshold) / len(d.columns)
# subset data
d = d[feat_var_threshold]

# define y as Outcome
y = d['Outcome']
# define X as all columns except Outcome
X = d.drop('Outcome', axis=1)

# nrow data
d.shape

# test training split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# set up hyperparameter grid
param_grid = {
'criterion' : ['gini'],
'splitter' : ['best'],
'max_depth' : [1,2,5,10], 
'min_samples_split' : [1,10,20,100],
'min_samples_leaf' : [1,5,50,100],
'max_features' : ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
'max_leaf_nodes' : [1,3,5,10],
'min_impurity_decrease' : [0,0.01,0.05,0.1]
}

# set up decision tree model
dt = DecisionTreeClassifier()

# set up grid search with training eval split
grid_search = GridSearchCV(dt, param_grid, cv=3)

# fit grid search
# measure duration of fit
import time
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()
print(end - start)

# training accuracy
grid_search.best_score_

# test accuracy
accuracy_score(y_test, grid_search.predict(X_test))

# confusion matrix
confusion_matrix(y_test, grid_search.predict(X_test))

# grab best tree
best_tree = grid_search.best_estimator_

# draw best tree with labels
plt.figure(figsize=(12, 8))
from sklearn.tree import plot_tree
plot_tree(best_tree, feature_names=X.columns, filled=True)
plt.show()

# feature importance
best_tree.feature_importances_
