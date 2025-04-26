# === record_user_input.py ===
import cv2
import mediapipe as mp
import csv
import os
import google.generativeai as genai  # <-- New Gemini import
from config import GEMINI_API_KEY  # Import API key from config.py

# Setup Gemini API Key

genai.configure(api_key=GEMINI_API_KEY)

# Setup MediaPipe Pose + Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Output CSV directory
OUTPUT_DIR = "data/collected_keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def record_sequence(gloss_label, seconds=3, fps=10):
    cap = cv2.VideoCapture(1)
    sequence = []
    frame_count = 0
    target_frames = seconds * fps

    print(f"[*] Recording sign for: {gloss_label} ({seconds}s)")

    while cap.isOpened() and frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        keypoints = extract_keypoints(results_pose, results_hands)
        sequence.append(keypoints)

        cv2.putText(frame, f"Recording {gloss_label} ({frame_count+1}/{target_frames})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    out_path = os.path.join(OUTPUT_DIR, f"{gloss_label}.csv")
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sequence)

    print(f"[+] Saved sequence to {out_path}")

def gloss_to_english(gloss_text):
    model = genai.GenerativeModel("gemini-1.5-pro")
    convo = model.start_chat()
    convo.send_message(f"Convert the following ASL gloss into natural English sentences:\n\nGloss: {gloss_text}\nEnglish:")
    english_text = convo.last.text.strip()
    return english_text

def english_to_gloss(english_text):
    model = genai.GenerativeModel("gemini-1.5-pro")
    convo = model.start_chat()
    convo.send_message(f"Convert the following English sentence into ASL gloss (keywords, no extra words):\n\nEnglish: {english_text}\nGloss:")
    gloss_text = convo.last.text.strip()
    return gloss_text

if __name__ == "__main__":
    mode = input("Record new sign (r) or Test gloss translation (t)? ").lower()

    if mode == 'r':
        label = input("Enter gloss label for recording: ")
        record_sequence(label)

    elif mode == 't':
        choice = input("Translate Gloss->English (g) or English->Gloss (e)? ").lower()
        if choice == 'g':
            gloss = input("Enter ASL Gloss: ")
            print("English Translation:", gloss_to_english(gloss))
        elif choice == 'e':
            english = input("Enter English sentence: ")
            print("ASL Gloss Translation:", english_to_gloss(english))

    else:
        print("Invalid option!")