# we are fitting a decision tree on our data using grid search CV

import pandas as pd
import numpy as np
# load sklearn for preprocessing, model selection, and metrics, knn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# load data

d = pd.read_csv('dat/titanic.csv')

# list columns
d.columns

# one-hot encode sex
d = pd.get_dummies(d, columns=['Sex'])

# keep columns Survived, Pclass, Sex, Age
d = d[['Survived', 'Pclass', 'Sex_female', 'Age']]

# drop missing values

d = d.dropna()

# randomise order of rows
d = d.sample(frac=1)

# training and eval and test split: 80% training, 20% test
train, test = train_test_split(d, test_size=0.2)

# set up hyperparameter grid
criteria = ['gini', 'entropy']
splitter = ['best', 'random']
max_depth = np.arange(1, 10)
min_samples_split = np.arange(2, 10)
min_samples_leaf = np.arange(1, 10)
max_features = ['auto', 'sqrt', 'log2']
max_leaf_nodes = np.arange(2, 10)
min_impurity_decrease = np.arange(0, 0.1, 0.01)

param_grid = {'criterion': criteria, 'splitter': splitter, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_impurity_decrease': min_impurity_decrease}

# set up decision tree model
dt = DecisionTreeClassifier()

# set up grid search with training eval split

# grid_search = GridSearchCV(dt, param_grid, cv=5)

# fit grid search
# grid_search.fit(train[['Pclass', 'Age']], train['Survived'])
random_search = RandomizedSearchCV(dt, param_grid, cv=5, n_iter=1000, random_state=42)

random_search.fit(train[['Pclass', 'Sex_female', 'Age']], train['Survived'])

# best hyperparameter
random_search.best_params_

# best score
random_search.best_score_

# best tree
best_tree = random_search.best_estimator_

# varimp best tree
best_tree.feature_importances_

feature_names = ['Pclass', 'Sex_female', 'Age']
importance_labels = dict(zip(feature_names, best_tree.feature_importances_))
print(importance_labels)

# plot tree

plt.figure(figsize=(12, 8))
plot_tree(
    best_tree,
    feature_names=['Pclass', 'Sex_female', 'Age'],
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()

# best tree confusion matrix
pred = random_search.predict(test[['Pclass', 'Sex_female', 'Age']])
confusion_matrix(test['Survived'], pred)