import pandas as pd
import numpy as np
# load sklearn random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# load RandomizedSearchCV
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

# load data

data = pd.read_csv('https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/credit_risk_dataset.csv')

# sample data down to 30%
data = data.sample(frac=0.3, random_state=42)

# nrow data
print(data.shape)

data.dtypes

# if column object dummy code column
data = pd.get_dummies(data, drop_first=True)

# drop rows with missing values
data = data.dropna()

# split data into features and target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# training test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create random forest classifier
clf = RandomForestClassifier()

# create parameter grid
param_grid = {
    'n_estimators': [500, 700],
    # 'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [5, 10],
    # 'max_features': ['auto', 'sqrt', 'log2', None],
    'max_leaf_nodes': [None, 3, 5, 10],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
    # 'class_weight': [None, 'balanced', 'balanced_subsample']
}

# create grid searches

## using accuracy and grid search
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=2)

# fit grid search
grid_search.fit(X_train, y_train)

# get confusion matrix on training data from best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)
confusion_matrix(y, y_pred)
# get accuracy
accuracy_score(y, y_pred)
# get model attr
best_model.get_params()

# get confusion matrix on test data from best model
y_pred = best_model.predict(X_test)
confusion_matrix(y_test, y_pred)
# get accuracy
accuracy_score(y_test, y_pred)
# nice