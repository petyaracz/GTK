# ctrl shift p :D
# print hello vilag

print("Hello vilag")

# toltsd be a tablazatot errol a linkrol https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/titanic.csv

import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/petyaracz/GTK/refs/heads/main/dat/titanic.csv")

df

# list column names for df
df.columns

# keep Survived Pclass Sex Age columns

df = df[[
"Survived",
"Pclass",
"Sex",
"Age"]]

df

# remove rows with na
df = df.dropna()

# one hot encoding for Sex column, change to male
df = pd.get_dummies(df, columns=["Sex"], drop_first=True)

df

# split df to training and test data
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a KNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# predict on test data
y_pred = knn.predict(X_test)

# calculate accuracy
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# calculate confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

# define knn2 with k = 7
knn2 = KNeighborsClassifier(n_neighbors=7)
knn2.fit(X_train, y_train)
# predict
y_pred2 = knn2.predict(X_test)
# accuracy
accuracy_score(y_test, y_pred2)

# create column y_miss which is 1 if y_test != y_pred else 0
X_test["y_miss"] = np.where(y_test != y_pred, 1, 0)
import matplotlib.pyplot as plt

#  plot Age vs y_miss using a boxplot
plot1 = X_test.boxplot(column="Age", by="y_miss")
plt.show()

# plot Male vs y_miss using barplot
plot2 = X_test.groupby("y_miss")["Sex_male"].value_counts().unstack().plot(kind="bar")
plt.show()