import pandas as pd
import numpy as np
# load sklearn random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# load data

data = pd.read_csv('https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/accident.csv')

# nrow data
print(data.shape)

data.dtypes

# if column object dummy code column
data = pd.get_dummies(data, drop_first=True)

# drop rows with missing values
data = data.dropna()

# split data into features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# summary of y
y.value_counts()

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create random forest classifier
clf = RandomForestClassifier()

# create parameter grid
param_grid = {
    'n_estimators': [100, 500, 800],
    'max_depth': [None, 10, 20],
    'min_samples_split': [1, 5, 10],
    'min_samples_leaf': [1, 2, 10],
    'max_samples': [None, 0.5, .8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# create grid searches

## using accuracy
grid_search_1 = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)

## using f score
grid_search_2 = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2, scoring='f1')

# fit grid searches
grid_search_1.fit(X_train, y_train)

grid_search_2.fit(X_train, y_train)

# get accuracies on test
y_pred_1 = grid_search_1.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred_1)
print(f"RF1 accuracy: {accuracy_1}")

y_pred_2 = grid_search_2.predict(X_test)
accuracy_2 = accuracy_score(y_test, y_pred_2)
print(f"RF2 accuracy: {accuracy_2}")

# get confusion matrices
conf_matrix_1 = confusion_matrix(y_test, y_pred_1)
print(conf_matrix_1)  
conf_matrix_2 = confusion_matrix(y_test, y_pred_2)
print(conf_matrix_2)  
