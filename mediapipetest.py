import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))


def calculate_confidence(gesture, landmarks):
    """Assign a confidence score based on hand shape similarity to a given ASL gesture."""
    confidence = 0

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    thumb_base = landmarks[mp_hands.HandLandmark.THUMB_IP]
    index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_base = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_base = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_base = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    palm_base = landmarks[mp_hands.HandLandmark.WRIST]

    if gesture == "peace sign":
        if (index_tip.y < palm_base.y and middle_tip.y < palm_base.y and  # Index & middle up
            ring_tip.y > middle_base.y and pinky_tip.y > middle_base.y and  # Ring & pinky curled
            distance(index_tip, middle_tip) > 0.05):  # Fingers should be apart
            confidence = 1.0
            
    # A in ASL: Fist with thumb on the side
    elif gesture == "A in ASL":
        if (index_tip.y > index_base.y and middle_tip.y > middle_base.y and
            ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y and  # Fingers are curled
            distance(thumb_tip, index_base) < 0.05 and
            thumb_tip.x < index_base.x and thumb_tip.y < palm_base.y):  # Thumb rests against index base
            confidence = 0.98
        else:
            confidence = 0.3

    # B in ASL: All fingers extended, thumb across palm
    elif gesture == "B in ASL":
        if (index_tip.y < palm_base.y and middle_tip.y < palm_base.y and
            ring_tip.y < palm_base.y and pinky_tip.y < palm_base.y and  # All fingers extended
            distance(thumb_tip, palm_base) < 0.1 and
            thumb_tip.y > index_base.y):  # Thumb rests across the palm
            confidence = 0.95
        else:
            confidence = 0.4

    # C in ASL: Curved fingers forming a "C" shape
    elif gesture == "C in ASL":
        if (distance(index_tip, palm_base) > 0.08 and
            distance(thumb_tip, index_tip) > 0.05 and
            distance(pinky_tip, palm_base) > 0.08 and
            index_tip.x > palm_base.x and pinky_tip.x < palm_base.x):  # Circular shape check
            confidence = 0.96
        else:
            confidence = 0.35

    return confidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Calculate confidence for each gesture
            gesture_confidences = {gesture: calculate_confidence(gesture, landmarks)
                                   for gesture in ["peace sign", "A in ASL", "B in ASL", "C in ASL"]}

            # Find the best gesture
            best_gesture = max(gesture_confidences, key=gesture_confidences.get)
            best_confidence = gesture_confidences[best_gesture]

            # Threshold to ensure valid detection
            if best_confidence > 0.75:
                print(f"Detected {best_gesture} with confidence: {best_confidence:.2f}")
            else:
                print(f"Low confidence gesture detected: {best_gesture} with confidence: {best_confidence:.2f}")

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
