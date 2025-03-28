from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# dummy code X, drop first
import pandas as pd
X = pd.get_dummies(X, drop_first=True)

# unravel y
y = y.values.ravel()
# if y is "<=50K", then 0, else 1
y = [0 if i == '<=50K' else 1 for i in y]

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

xgb = XGBClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# fit models
xgb.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# predict
xgb_pred = xgb.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# accuracy
from sklearn.metrics import accuracy_score
# roc auc
from sklearn.metrics import roc_auc_score
# confusion matrix
from sklearn.metrics import confusion_matrix

# accuracy
xgb_acc = accuracy_score(y_test, xgb_pred)
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
# roc auc
xgb_roc = roc_auc_score(y_test, xgb_pred)
dt_roc = roc_auc_score(y_test, dt_pred)
rf_roc = roc_auc_score(y_test, rf_pred)
# confusion matrix
xgb_cm = confusion_matrix(y_test, xgb_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
# print values for xgb 
xgb_acc, xgb_roc, xgb_cm
# print values for dt
dt_acc, dt_roc, dt_cm
# print values for rf
rf_acc, rf_roc, rf_cm

# varimp with labels
importances = xgb.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
for i in range(X.shape[1]):
    print(f'{features[i]}: {importances[i]}')