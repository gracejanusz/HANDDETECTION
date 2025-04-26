# train_model24.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt  # For plotting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# 1. Load and prepare data
df = pd.read_csv("data.csv")

# Filter out 'J' and 'Z'
df = df[~df['label'].isin(['J', 'Z'])]

# Remove classes with only 1 sample
class_counts = df['label'].value_counts()
df = df[df['label'].isin(class_counts[class_counts > 1].index)]

# Properly parse features
features = []
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        split_data = df[col].str.split(",", expand=True).astype(float)
        features.append(split_data)
    else:
        features.append(df[[col]])

# Build features and labels
X = pd.concat(features, axis=1).values
y = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Create Dataset and DataLoader
class HandGestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HandGestureDataset(X_train, y_train)
test_dataset = HandGestureDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Define the Model
class HandGestureModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X.shape[1]  # Should be 63
num_classes = len(np.unique(y))  # Should be 24

model = HandGestureModel(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model and plot live loss
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    loss_history = []

    plt.ion()  # Turn on interactive mode for live updates
    fig, ax = plt.subplots()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Plot
        ax.clear()
        ax.plot(loss_history, label="Training Loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        plt.pause(0.1)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    plt.ioff()
    plt.show()

train_model(model, train_loader, criterion, optimizer, epochs=20)

# 5. Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")

evaluate_model(model, test_loader)

# 6. Save model and preprocessing tools
torch.save(model.state_dict(), "hand_gesture_model.pth")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model, label encoder, and scaler saved successfully!")
