# load the packages
import os
import pickle
import inspect
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
cwd        = os.getcwd()
data       = pd.read_csv(cwd + f"\\Data\\histopathological_data.csv")
train_data = pd.read_csv(cwd + f"\\Data\\training_histopathological_data.csv")
test_data  = pd.read_csv(cwd + f"\\Data\\testing_histopathological_data.csv")
data       = data[data.columns[2:].values] # Exclude the directory path and files columns
train_data = train_data[train_data.columns[1:].values] # Exclude the directory path and files columns
test_data  = test_data[test_data.columns[1:].values] # Exclude the directory path and files columns
data.head()
train_data.head()
test_data.head()

X_train = train_data.drop(['PatientID', 'cancer'], axis = 1) # Dropping both PatientID and the target variable (cancer)
y_train = train_data['cancer']
X_test  = test_data.drop(['PatientID', 'cancer'], axis = 1) # Dropping both PatientID and the target variable (cancer)
y_test  = test_data['cancer']

X = data.drop(['cancer'], axis = 1)  # Dropping both PatientID and the target variable (cancer)
y = data['cancer']

# Build a Random Forest model
rf_model = RandomForestClassifier(n_estimators = 200, random_state = 7, min_samples_split = 4,
                                  min_samples_leaf = 1, max_depth = 30, criterion = 'entropy',
                                  bootstrap = False)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Plot confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize = (8, 6))
sns.heatmap(conf_matrix, annot = True,
            fmt = "d", cmap = "Blues", cbar = False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification report
print(classification_report(y_test, y_pred_test))

# Define the number of folds for grouped cross-validation
n_splits = 5

# Initialize GroupKFold with the number of folds
group_kfold = GroupKFold(n_splits = n_splits)

# Perform grouped cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv = group_kfold, groups = X['PatientID'])

# Print mean cross-validation accuracy
print("Mean CV Accuracy (Grouped Cross-Validation):", cv_scores.mean())

# Get the parameters of the trained Random Forest model
rf_params = rf_model.get_params()

# Print the parameters
print("Parameters of the Random Forest model:")
for param_name, param_value in rf_params.items():
    print(param_name, ":", param_value)

# Get feature importances from the trained Random Forest model
feature_importance = rf_model.feature_importances_

# Get the names of the features
feature_names = X.columns

# Sort feature importances in descending order
indices = feature_importance.argsort()[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance Plot")
plt.bar(range(X.shape[1]), feature_importance[indices], color = "b", align = "center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation = 90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

### Save to Pickle File
def is_serializable(obj):
    """
    Check if an object is serializable.
    """
    try:
        pickle.dumps(obj)
        return True
    except:
        return False

def save_locals_to_pickle(filename):
    # Get local variables
    caller_locals = inspect.currentframe().f_back.f_locals.copy()

    # Filter out non-serializable objects
    serializable_locals = {k: v for k, v in caller_locals.items() if is_serializable(v)}

    # Store local variables in a pickle file
    try:
        with open(filename, 'wb') as f:
            pickle.dump(serializable_locals, f)
            print("Local variables saved to", filename)
    except Exception as e:
        print("Error occurred while saving:", e)

# Example usage
filename = 'RFModel.pkl'
save_locals_to_pickle(filename)
print('')

#######################################################################################################################

# Define the model
rf = RandomForestClassifier(random_state = 7)

# Define the parameter grid
param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Create the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator = rf,
    param_distributions = param_dist,
    n_iter = 50,  # Number of random iterations
    scoring = 'accuracy',
    cv = 3,  # Number of cross-validation folds
    random_state = 7,
    n_jobs = -1  # Use all processors
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score())
