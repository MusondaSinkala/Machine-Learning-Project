# load the packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# load the data
cwd = os.getcwd()
data = pd.read_csv(cwd + f"\\Data\\histopathological_data.csv")
data = data[data.columns[2:].values] # Exclude the directory path and files columns
data.head()

# Random, small sample from the data
sample_data = data#.sample(n = 200000, random_state = 7)

# Exclude the PatientID feature
X = sample_data.drop(['PatientID', 'cancer'], axis = 1)  # Dropping both PatientID and the target variable (cancer)
y = sample_data['cancer']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 7)

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Normalize the input data using StandardScaler
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define input dimension, hidden layer dimension, and output dimension
input_dim = X_train.shape[1]  # Input dimension based on the number of features
hidden_dim = 50
output_dim = 1  # Assuming it's a binary classification task

# Instantiate the model
model = FeedforwardNN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Convert your data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)  # Squeeze the output to match the shape of y_train_tensor

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Forward pass on test data
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = torch.sigmoid(outputs).round()  # Round the output to 0 or 1 for binary classification

# Convert predictions to numpy array
predictions = predictions.numpy()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_tensor, predictions)
precision = precision_score(y_test_tensor, predictions)
recall = recall_score(y_test_tensor, predictions)
f1 = f1_score(y_test_tensor, predictions)
conf_matrix = confusion_matrix(y_test_tensor, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Get the parameters of the trained neural network model
model_params = model.state_dict()

# Print the parameters
print("Parameters of the trained neural network model:")
for param_name, param_value in model_params.items():
    print(param_name, ":", param_value)