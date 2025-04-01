import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the neural network model
class HandGestureModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, num_classes) # Output layer (num_classes is 3 for A, B, C)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)              # Output layer
        return x

# Custom dataset to load the hand gesture data
class HandGestureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Map string labels to integers: A -> 0, B -> 1, C -> 2
        self.label_map = {'a': 0, 'b': 1, 'c': 2}
        
        self.features = self.data.iloc[:, 1:].values  # All columns except the label
        self.labels = self.data.iloc[:, 0].apply(lambda x: self.label_map[x]).values  # Convert labels to integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Labels as integers
        return features, label

# Load and preprocess the dataset
def load_data(csv_file):
    dataset = HandGestureDataset(csv_file)
    # Split the dataset into training and validation sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, test_loader

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss every epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Test the model
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
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

# Main function to run the program
def main():
    # Path to your CSV file
    csv_file = 'data.csv'

    # Load data
    train_loader, test_loader = load_data(csv_file)

    # Initialize the model
    input_size = 63  # 21 landmarks (x, y, z) * 3 (for each hand landmark)
    num_classes = 3  # A, B, C
    model = HandGestureModel(input_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    # Test the model
    test_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'hand_gesture_model.pth')
    print("Model saved as 'hand_gesture_model.pth'")

if __name__ == "__main__":
    main()
