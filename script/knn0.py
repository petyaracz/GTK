# load numpy pandas sklearn knn
import numpy as np
import pandas as pd
# load sklearn for preprocessing, model selection, and metrics, knn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# load data
d = pd.read_csv('https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/titanic.csv')

# one hot encode Pclass Sex
d = pd.get_dummies(d, columns=['Sex'])

# keep relevant cols

d = d[['Survived', 'Pclass', 'Sex_female', 'Age']]

# drop missing values
d = d.dropna()

# training and eval and test split: 80% training, 20% test
train, test = train_test_split(d, test_size=0.2)

# set up KNN model with k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# fit model
knn.fit(train[['Pclass', 'Age', 'Sex_female']], train['Survived'])

# predict on test set
pred = knn.predict(test[['Pclass', 'Age', 'Sex_female']])

# accuracy
accuracy_score(test['Survived'], pred)

# confusion matrix
confusion_matrix(test['Survived'], pred)

# knn2 with k = 5
knn2 = KNeighborsClassifier(n_neighbors=5)

# fit model
knn2.fit(train[['Pclass', 'Age', 'Sex_female']], train['Survived'])

# predict on test set
pred2 = knn2.predict(test[['Pclass', 'Age', 'Sex_female']])

# accuracy
accuracy_score(test['Survived'], pred2)
# previous accuracy
accuracy_score(test['Survived'], pred)