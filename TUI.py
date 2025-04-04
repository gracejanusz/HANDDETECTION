import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from time import sleep
from PIL import Image

# === Load model and utilities ===

class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_model(model_path='hand_gesture_model.pth', num_classes=3):
    input_size = 63  # 21 x (x, y, z)
    model = HandGestureModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# === Hand detection & prediction ===

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
labels = ['a', 'b', 'c']  # Adjust according to your trained model

def classify_hand_from_frame(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if not result.multi_hand_landmarks:
        return None  # No hand detected

    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
    return labels[prediction.item()]

# === CLI TUI Interaction ===

def show_asl_image(letter):
    path = f'asl_images/{letter}.png'
    if not os.path.exists(path):
        print(f"[!] Missing image: {path}")
        return
    img = Image.open(path)
    img.show()

def asl_trainer(target_letter='a'):
    print("Welcome to the ASL Sign Trainer!")
    print(f"Your task: Make the ASL sign for the letter: '{target_letter}'")
    print("An image will pop up showing how to do it.")
    sleep(1)

    show_asl_image(target_letter)
    model = load_model()

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        cv2.imshow("Live Feed - Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            print("Exiting...")
            break
        elif key == 32:  # SPACE key to capture
            print("[*] Capturing...")
            predicted = classify_hand_from_frame(frame, model)
            if predicted is None:
                print("No hand detected. Try again.\n")
                continue
            print(f"[Model Prediction]: {predicted}")
            if predicted == target_letter.upper():
                print("✅ Nice job! You got it right.")
                break
            else:
                print("❌ That doesn't look quite right. Try again!\n")

    cap.release()
    cv2.destroyAllWindows()

# === Run trainer ===

if __name__ == "__main__":
    asl_trainer(target_letter='A')
