import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# CSV file path
csv_filename = "data.csv"

# Ensure CSV file exists and has headers
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["label"] + [f"x{i},y{i},z{i}" for i in range(21)]
        writer.writerow(header)

# Start video capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame
    cv2.imshow("Hand Landmark Capture", frame)
    
    # Capture key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    
    # If a letter key (a-z) is pressed, store landmarks
    if result.multi_hand_landmarks and 'a' <= chr(key).lower() <= 'z':
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [chr(key).lower()]  # Store lowercase letter as label
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Write to CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
            print(f"Saved hand landmarks for letter: {chr(key).lower()}")

cap.release()
cv2.destroyAllWindows()
