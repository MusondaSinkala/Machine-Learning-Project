# load the packages
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
cwd = os.getcwd()
data = pd.read_csv(cwd + f"\\Data\\histopathological_data.csv")
data = data[data.columns[2:].values] # Exclude the directory path and files columns
data.head()

# Random, small sample from the data
sample_data = data#.sample(n = 300000, random_state = 7)

# Extract unique PatientID values
unique_patient_ids = sample_data['PatientID'].unique()

# Randomly shuffle the unique PatientID values
random.shuffle(unique_patient_ids)

# Split the shuffled PatientID values into training and test sets (e.g., 80% train, 20% test)
train_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size = 0.2, random_state = 7)

# Filter the original dataset to create the training and test sets
train_data = data[data['PatientID'].isin(train_patient_ids)]
test_data = data[data['PatientID'].isin(test_patient_ids)]

X_train = train_data.drop(['PatientID', 'cancer'], axis = 1) # Dropping both PatientID and the target variable (cancer)
y_train = train_data['cancer']
X_test = test_data.drop(['PatientID', 'cancer'], axis = 1) # Dropping both PatientID and the target variable (cancer)
y_test = test_data['cancer']

# Exclude the PatientID feature
X = sample_data.drop(['cancer'], axis = 1)  # Dropping both PatientID and the target variable (cancer)
y = sample_data['cancer']

# Build a Random Forest model
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 7)
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