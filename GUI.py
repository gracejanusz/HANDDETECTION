# asl_gui_trainer.py

import sys
import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image
import joblib

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import QTimer, Qt

class ASLTrainerApp(QWidget):
    def __init__(self, letters=None):
        super().__init__()

        # === Setup ===
        self.letters = letters if letters else list("abcdefghiklmnopqrstuvwxy")  # Skipping J and Z
        self.current_index = 0
        self.model, self.label_encoder, self.scaler = self.load_model()
        self.cap = cv2.VideoCapture(1)

        self.setWindowTitle("ASL Sign Trainer")
        self.setFixedSize(960, 720)

        # === Layout ===
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Instruction
        self.instruction = QLabel()
        self.instruction.setAlignment(Qt.AlignCenter)
        self.instruction.setFont(QFont("Arial", 24))
        self.layout.addWidget(self.instruction)

        # ASL Image
        self.asl_image = QLabel()
        self.asl_image.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.asl_image)

        # Webcam Feed
        self.live_feed = QLabel()
        self.live_feed.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.live_feed)

        # Button Row
        self.button_row = QHBoxLayout()
        self.layout.addLayout(self.button_row)

        # Capture Button
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setFont(QFont("Arial", 18))
        self.capture_btn.clicked.connect(self.capture_frame)
        self.button_row.addWidget(self.capture_btn)

        # Next Button
        self.next_btn = QPushButton("Next")
        self.next_btn.setFont(QFont("Arial", 18))
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.go_to_next_letter)
        self.button_row.addWidget(self.next_btn)

        # Timer for webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

        self.load_current_letter()

    def load_model(self, model_path='hand_gesture_model.pth', encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
        # Must match your improved model (with 4 layers)
        class HandGestureModel(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.fc3 = torch.nn.Linear(128, 64)
                self.fc4 = torch.nn.Linear(64, num_classes)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        input_size = 63
        num_classes = 24
        model = HandGestureModel(input_size, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)

        return model, label_encoder, scaler

    def load_current_letter(self):
        letter = self.letters[self.current_index]
        self.instruction.setText(f"Show the ASL sign for: '{letter.upper()}'")
        self.next_btn.setEnabled(False)
        self.load_asl_image(letter)

    def load_asl_image(self, letter):
        path = os.path.join("asl_images", f"{letter}.png")
        if os.path.exists(path):
            try:
                img = Image.open(path).resize((300, 300))
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                data = img.tobytes("raw", "RGBA")
                qimg = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
                self.asl_image.setPixmap(QPixmap.fromImage(qimg))
            except Exception as e:
                self.asl_image.setText(f"[!] Failed to load image:\n{e}")
        else:
            self.asl_image.setText(f"[!] Missing ASL image for '{letter.upper()}'")

    def classify_hand_from_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None

        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, -1)
        landmarks = self.scaler.transform(landmarks)  # Normalize using saved scaler
        input_tensor = torch.tensor(landmarks, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, prediction = torch.max(output, 1)

        predicted_index = prediction.item()
        predicted_letter = self.label_encoder.inverse_transform([predicted_index])[0]
        return predicted_letter

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 360))
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.live_feed.setPixmap(QPixmap.fromImage(qimg))

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Error", "Camera error.")
            return

        prediction = self.classify_hand_from_frame(frame)
        correct_letter = self.letters[self.current_index]

        if prediction is None:
            QMessageBox.information(self, "Try Again", "No hand detected.")
        elif prediction.lower() == correct_letter:
            QMessageBox.information(self, "Success", f"✅ Correct! You signed '{prediction.upper()}'.")
            self.next_btn.setEnabled(True)
        else:
            QMessageBox.information(self, "Try Again", f"❌ That looked like '{prediction.upper()}'. Try again.")

    def go_to_next_letter(self):
        self.current_index += 1
        if self.current_index >= len(self.letters):
            QMessageBox.information(self, "Done!", "🎉 You've finished all letters!")
            self.close()
            return
        self.load_current_letter()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# === Launch ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLTrainerApp(letters=list("abcdefghiklmnopqrstuvwxy"))  # 24 letters, no j, z
    window.show()
    sys.exit(app.exec())
