import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
from PIL import Image

# === Model definition and utilities ===

class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_model(model_path='hand_gesture_model.pth', num_classes=3):
    input_size = 63
    model = HandGestureModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# === Hand detection & prediction ===

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
labels = ['a', 'b', 'c']

def classify_hand_from_frame(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None
    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
    return labels[prediction.item()]

# === GUI App ===

class ASLTrainerApp(QWidget):
    def __init__(self, target_letter='a'):
        super().__init__()
        self.target_letter = target_letter.lower()
        self.model = load_model()
        self.cap = cv2.VideoCapture(1)
        self.setWindowTitle("ASL Sign Trainer")

        self.image_label = QLabel()
        self.live_feed = QLabel()
        self.capture_button = QPushButton("Capture Sign")
        self.capture_button.clicked.connect(self.capture_frame)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Make the ASL sign for the letter: '{self.target_letter.upper()}'"))
        layout.addWidget(self.image_label)
        layout.addWidget(self.live_feed)
        layout.addWidget(self.capture_button)
        self.setLayout(layout)

        self.load_asl_image()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def load_asl_image(self):
        path = f'asl_images/{self.target_letter}.png'
        try:
            img = Image.open(path)
            img = img.convert('RGB')
            img = img.resize((150, 150))
            qimage = QImage(img.tobytes(), img.size[0], img.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
        except FileNotFoundError:
            self.image_label.setText(f"[!] Missing image: {path}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.live_feed.setPixmap(QPixmap.fromImage(qimg))

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Error", "Camera error.")
            return
        prediction = classify_hand_from_frame(frame, self.model)
        if prediction is None:
            QMessageBox.warning(self, "Try Again", "No hand detected.")
        elif prediction == self.target_letter:
            QMessageBox.information(self, "Success", f"✅ Nice job! You signed '{self.target_letter.upper()}' correctly.")
        else:
            QMessageBox.warning(self, "Try Again", f"❌ That doesn't look like '{self.target_letter.upper()}'. Try again!")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLTrainerApp(target_letter='a')
    window.resize(400, 600)
    window.show()
    sys.exit(app.exec())
