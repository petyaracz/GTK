# load numpy pandas sklearn knn
import numpy as np
import pandas as pd
# load sklearn for preprocessing, model selection, and metrics, knn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# load data
d = pd.read_csv('https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/titanic.csv')

# list columns
d.columns

# one hot encode Pclass Sex
d = pd.get_dummies(d, columns=['Sex'])

# keep relevant cols

d = d[['Survived', 'Pclass', 'Sex_female', 'Age', 'SibSp', 'Parch', 'Fare']]

# drop missing values
d = d.dropna()

# training and eval and test split: 80% training, 20% test
train, test = train_test_split(d, test_size=0.2)

# set up hyperparameter grid
k = np.arange(1, 10)
param_grid = {'n_neighbors': k}

# set up knn model
knn = KNeighborsClassifier()

# set up grid search with training eval split
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(knn, param_grid, cv=5)

# fit grid search
grid_search.fit(train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], train['Survived'])

# best hyperparameter
grid_search.best_params_

# best score
grid_search.best_score_

# predict on test set
pred = grid_search.predict(test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

# accuracy
accuracy_score(test['Survived'], pred)

# confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(test['Survived'], pred)

# save predictions
test['pred'] = pred

# save test set
# not run:
# test.to_csv('dat/titanic_test.csv', index=False)