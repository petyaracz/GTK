import pandas as pd

url = 'https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/accident.csv'

df = pd.read_csv(url)

# drop rows with missing values
df = df.dropna()

# one hot encode Sex column
df = pd.get_dummies(df, columns=["Gender","Helmet_Used","Seatbelt_Used"], drop_first=True)

## setup

# split the data into X and y
X = df.drop(columns=["Survived"])
y = df["Survived"]

# split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## model

# KNN classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# grid search CV

from sklearn.model_selection import GridSearchCV

# define the model

model = KNeighborsClassifier()

# define the grid

grid = {
    "n_neighbors": [3, 5, 7, 9, 11]
}   

# define the grid search

grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring="accuracy")

# fit the grid search

grid_search.fit(X_train, y_train)

# get the best model

best_model = grid_search.best_estimator_

# evaluate the best model

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# best model k
print(f"Best model k: {best_model.n_neighbors}")

# dummy model

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dummy accuracy: {accuracy:.2f}")