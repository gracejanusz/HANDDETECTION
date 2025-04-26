import cv2
import mediapipe as mp
import numpy as np
import joblib  
import os

os.chdir("./classifier")

# Load the trained SVM model
svm_model = joblib.load('svm_classifier.joblib')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract landmarks as features
def extract_features(landmarks):
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
        features.append(lm.z)
    return np.array(features).reshape(1, -1)

# Map numeric predictions to letters (e.g., 0 = 'A', 1 = 'B', ..., 25 = 'Z')
def map_prediction_to_letter(prediction):
    return chr(prediction + ord('A'))  # Convert 0 -> 'A', 1 -> 'B', etc.

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features and predict the letter
            features = extract_features(hand_landmarks)
            prediction = svm_model.predict(features)
            predicted_letter = map_prediction_to_letter(prediction[0])

            # Display the predicted letter on the frame
            cv2.putText(frame, f'Letter: {predicted_letter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Hand Sign Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()