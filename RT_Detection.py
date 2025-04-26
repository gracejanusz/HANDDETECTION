import cv2
import mediapipe as mp
import torch
import numpy as np

# Define the hand gesture model (same as the training model)
class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = torch.nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = torch.nn.Linear(64, num_classes) # Output layer (num_classes is 3 for A, B, C)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)              # Output layer
        return x

# Load the trained model
def load_model(model_path='hand_gesture_model.pth'):
    input_size = 63  # 21 landmarks (x, y, z) * 3 (for each hand landmark)
    num_classes = 3  # A, B, C
    model = HandGestureModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the hand landmarks into the format the model expects
def preprocess_landmarks(landmarks):
    # Flatten the landmarks into a 1D tensor (21 landmarks * 3 coordinates)
    flattened = []
    for lm in landmarks:
        flattened.extend([lm.x, lm.y, lm.z])  # Add x, y, z for each landmark
    return torch.tensor(flattened, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Class labels (A, B, C)
labels = ['A', 'B', 'C']

# Start video capture
cap = cv2.VideoCapture(0)

# Load the trained model
model = load_model()

# Real-time detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Draw hand landmarks on the frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess the hand landmarks and make a prediction
            landmarks = preprocess_landmarks(hand_landmarks.landmark)
            with torch.no_grad():
                outputs = model(landmarks)
                _, predicted = torch.max(outputs, 1)
                predicted_label = labels[predicted.item()]

            # Display the predicted gesture on the screen
            cv2.putText(frame, f'Gesture: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow("Hand Gesture Detection", frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
