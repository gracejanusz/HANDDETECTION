# === live_sign_match.py ===
import os
import csv
import numpy as np
import cv2
import mediapipe as mp
from match_sign_to_gloss import load_all_glosses, match_gloss

# Setup MediaPipe Pose + Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Temporary folder to save live recordings
TEMP_DIR = "data/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_keypoints(results_pose, results_hands):
    keypoints = []
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * (33*3))

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * (21*3))

    return keypoints

def record_temp_sequence(seconds=3, fps=10):
    cap = cv2.VideoCapture(0)
    sequence = []
    frame_count = 0
    target_frames = seconds * fps

    print("[*] Recording your sign... Hold steady!")

    while cap.isOpened() and frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        keypoints = extract_keypoints(results_pose, results_hands)
        sequence.append(keypoints)

        cv2.putText(frame, f"Recording ({frame_count+1}/{target_frames})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('Sign Recorder', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    temp_path = os.path.join(TEMP_DIR, "live_sign.csv")
    with open(temp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sequence)

    print(f"[+] Saved live sign to {temp_path}")
    return temp_path

def load_sequence(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        sequence = np.array(list(reader)).astype(np.float32)
    return sequence

if __name__ == "__main__":
    # Step 1: Record user live sign
    live_csv_path = record_temp_sequence()

    # Step 2: Load live sequence
    input_sequence = load_sequence(live_csv_path)

    # Step 3: Load database
    print("[*] Loading database...")
    database = load_all_glosses()

    # Step 4: Match!
    print("[*] Matching your sign...")
    best_label, best_score = match_gloss(input_sequence, database)

    print("==============================")
    print(f"🖐 Best matched gloss: {best_label.upper()} (Score: {best_score:.4f})")
    print("==============================")