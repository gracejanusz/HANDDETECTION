import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.chdir("./classifier")

# Load data from CSV
def load_mnist_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separate features and labels
    X_train = train_data.iloc[:, 1:].values  # Features
    y_train = train_data.iloc[:, 0].values  # Labels
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    return X_train, y_train, X_test, y_test

# Preprocess data
def preprocess_data(X_train, y_train, X_test, y_test):
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape X for LSTM input (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
        label_encoder,
    )

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # Increased dropout rate
        self.fc1 = nn.Linear(hidden_size, 16)  # Reduced number of units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take the last output of the LSTM
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)

# Paths to MNIST data
train_path = "mnist/sign_mnist_train/sign_mnist_train.csv"
test_path = "mnist/sign_mnist_test/sign_mnist_test.csv"

# Load and preprocess data
X_train, y_train, X_test, y_test, label_encoder = preprocess_data(
    *load_mnist_data(train_path, test_path)
)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model parameters
input_size = X_train.shape[2]
hidden_size = 32  # Reduced hidden size
output_size = len(torch.unique(y_train))

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay

# Training loop with early stopping
epochs = 20
patience = 5
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_lstm_model.pth")  # Save the best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(torch.load("best_lstm_model.pth"))

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model and label encoder
torch.save(model.state_dict(), "lstm_model.pth")
np.save("label_encoder_classes.npy", label_encoder.classes_)
print("Model and label encoder saved.")