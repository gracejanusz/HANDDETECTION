import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

os.chdir("./classifier")

# Define the LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 16)  # Match the saved model
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)  # Match the saved model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take the last output of the LSTM
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)

# Load the trained LSTM model
model_path = "best_lstm_model.pth"
label_encoder_path = "label_encoder_classes.npy"

# Define model parameters (must match training)
input_size = 784  # 21 landmarks * 3 coordinates (x, y, z)
hidden_size = 32  # Match the saved model
output_size = 24  # Number of classes (e.g., A-Z)

# Initialize the model and load the state dictionary
model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the label encoder
label_encoder = np.load(label_encoder_path, allow_pickle=True)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Draw hand landmarks and predict sign
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert landmarks to numpy array and reshape for LSTM input
            landmarks = np.array(landmarks).reshape(1, 1, -1)  # (batch_size, timesteps, features)

            # Predict the sign using LSTM
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
            outputs = model(landmarks_tensor)
            _, predicted = torch.max(outputs, 1)

            # Map the predicted index to the corresponding alphabet letter
            predicted_sign = label_encoder[predicted.item()]

            # Display the predicted sign on the frame
            cv2.putText(frame, f"Predicted Sign: {chr(predicted_sign + 65)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Real-Time Sign Language Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()