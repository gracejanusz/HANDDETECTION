import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

os.chdir("./classifier")

# Load data
train_df = pd.read_csv('mnist/sign_mnist_train.csv')
test_df = pd.read_csv('mnist/sign_mnist_test.csv')

# Separate features and labels
train_labels = train_df['label'].values
train_images = train_df.drop(['label'], axis=1).values.reshape(-1, 28, 28).astype(np.float32)

test_labels = test_df['label'].values
test_images = test_df.drop(['label'], axis=1).values.reshape(-1, 28, 28).astype(np.float32)

# Normalize images
train_images /= 255.0
test_images /= 255.0

# Verify and remap labels
train_labels = np.where(train_labels == 24, 23, train_labels)  # Example: Map 24 to 23
test_labels = np.where(test_labels == 24, 23, test_labels)

# Check unique labels after remapping
print(np.unique(train_labels))
print(np.unique(test_labels))

# Custom Dataset
class SignLanguageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Create datasets and dataloaders
train_dataset = SignLanguageDataset(train_images, train_labels, transform=transform)
test_dataset = SignLanguageDataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.flatten = nn.Flatten()


        # Calculate the flattened size dynamically
        dummy_input = torch.zeros(1, 1, 28, 28)  # Batch size = 1, 1 channel, 28x28 image
        flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 24)  # 24 classes (A-Y, excluding J)

    def _get_flattened_size(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 35
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"MODEL ACCURACY = {accuracy:.2f}%")