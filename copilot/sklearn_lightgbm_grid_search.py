import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, confusion_matrix, classification_report

# Load your data
d_sccs = pd.read_csv('d_sccs.csv')  # replace with your actual data loading
d = pd.read_csv('d.csv')  # replace with your actual data loading

# Define features and target
x = d_sccs.drop(columns=['possession_trance_present'])
y = d_sccs['possession_trance_present']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

# Define hyperparameters grid
param_grid = {
    'learning_rate': [0.01, 0.02, 0.03, 0.1],
    'max_depth': [1, 3, 5, 7],
    'subsample': [0.4, 0.7, 0.9, 0.95],
    'colsample_bytree': [0.1, 0.2, 0.3],
    'min_split_gain': [1e-3, 1e-5],
    'n_estimators': [100]
}

# Initialize LightGBM classifier
lgbm = LGBMClassifier(objective='binary', random_state=1)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation log loss: {-grid_search.best_score_}")

# Retrieve the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation data
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
log_loss_val = log_loss(y_val, y_pred_proba)
print(f"Validation log loss with best parameters: {log_loss_val}")

# Visualize feature importance
import matplotlib.pyplot as plt
import lightgbm as lgb

lgb.plot_importance(best_model, max_num_features=10)
plt.title("Feature Importances")
plt.show()

# Generate confusion matrix
y_pred = best_model.predict(X_val)
conf_matrix = confusion_matrix(y_val, y_pred)
print(f"Confusion matrix:\n{conf_matrix}")

# Generate classification report
print(classification_report(y_val, y_pred))

# Predict on "test data" (d)
X_test = d.drop(columns=['possession_trance_present'])
y_test = d['possession_trance_present']

y_test_pred = best_model.predict(X_test)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print(f"Test data confusion matrix:\n{test_conf_matrix}")
print(classification_report(y_test, y_test_pred))