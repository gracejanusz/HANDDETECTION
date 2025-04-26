# RT_Detection24.py

import cv2
import mediapipe as mp
import torch
import numpy as np
import joblib

# Define the same model structure
class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model, label encoder, and scaler
def load_model(model_path, encoder_path, scaler_path, input_size, num_classes):
    model = HandGestureModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    label_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    return model, label_encoder, scaler

# Preprocess landmarks
def preprocess_landmarks(landmarks, scaler):
    flattened = []
    for lm in landmarks:
        flattened.extend([lm.x, lm.y, lm.z])
    flattened = np.array(flattened).reshape(1, -1)
    flattened = scaler.transform(flattened)
    return torch.tensor(flattened, dtype=torch.float32)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Model info
input_size = 63
num_classes = 24

# Load model, encoder, scaler
model, label_encoder, scaler = load_model(
    model_path="hand_gesture_model.pth",
    encoder_path="label_encoder.pkl",
    scaler_path="scaler.pkl",
    input_size=input_size,
    num_classes=num_classes
)

# Start webcam
cap = cv2.VideoCapture(1)  # or 0 depending on your system

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = preprocess_landmarks(hand_landmarks.landmark, scaler)
            with torch.no_grad():
                outputs = model(landmarks)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                confidence = confidence.item()

            if confidence > 0.80:
                predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
                cv2.putText(frame, f'Gesture: {predicted_label} ({confidence*100:.1f}%)', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'Gesture: ...', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
