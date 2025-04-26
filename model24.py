# model24.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Define the improved neural network model
class HandGestureModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        print(f"Input size: {input_size}")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Custom dataset class
class HandGestureDataset(Dataset):
    def __init__(self, csv_file, scaler=None, label_encoder=None, fit_scaler_encoder=True):
        self.data = pd.read_csv(csv_file, header=None)
        self.data = self.data.rename(columns={0: 'label'})
        self.data = self.data.drop(index=0).reset_index(drop=True)

        X = self.data.iloc[:, 1:].astype(float).values
        y = self.data['label'].values

        if fit_scaler_encoder:
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self.X = self.scaler.fit_transform(X)
            self.y = self.label_encoder.fit_transform(y)
        else:
            self.scaler = scaler
            self.label_encoder = label_encoder
            self.X = self.scaler.transform(X)
            self.y = self.label_encoder.transform(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label

# Load and preprocess data
def load_data(csv_file):
    full_dataset = HandGestureDataset(csv_file)
    train_data, test_data = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(full_dataset, train_data)
    test_subset = torch.utils.data.Subset(full_dataset, test_data)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    return train_loader, test_loader, full_dataset.scaler, full_dataset.label_encoder

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=80):
    model.train()
    loss_history = []

    plt.ion()
    fig, ax = plt.subplots()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        ax.clear()
        ax.plot(loss_history, label="Training Loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        plt.pause(0.1)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    plt.ioff()
    plt.show()

# Testing loop
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Main
def main():
    csv_file = 'data.csv'

    train_loader, test_loader, scaler, label_encoder = load_data(csv_file)

    input_size = 63  # 21 landmarks × 3 (x, y, z)
    num_classes = 24  # A–I, K–Y (no J, no Z)

    model = HandGestureModel(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=80)
    test_model(model, test_loader)

    # Save everything
    torch.save(model.state_dict(), 'hand_gesture_model.pth')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model, scaler, and label encoder saved successfully!")

if __name__ == "__main__":
    main()
