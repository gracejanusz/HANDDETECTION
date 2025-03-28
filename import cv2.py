import cv2
import pyautogui
import mediapipe as mp

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get positions of the key landmarks for each finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
            ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]

            # Check for Peace Sign: index and middle fingers extended, others curled
            if (index_tip.y < thumb_base.y and  # index is above thumb base
                middle_tip.y < thumb_base.y and  # middle is above thumb base
                pinky_tip.y > ring_tip.y and  # pinky is lower than ring
                ring_tip.y > thumb_base.y):  # ring is lower than thumb
                hand_gesture = 'peace sign'
            # Check for A in ASL (fist with thumb on the side)
            elif (index_tip.y > index_base.y and  # index extended above base
                  middle_tip.y > middle_base.y and  # middle curled down
                  ring_tip.y > ring_base.y and  # ring curled down
                  pinky_tip.y > pinky_base.y and  # pinky curled down
                  abs(thumb_base.x - index_base.x) < 0.2):  # thumb on the side
                hand_gesture = 'A in ASL'
            # Check for B in ASL (fingers extended, thumb across palm)
            elif (index_tip.y < thumb_base.y and  # index above thumb base
                  middle_tip.y < thumb_base.y and  # middle above thumb base
                  ring_tip.y < thumb_base.y and  # ring above thumb base
                  pinky_tip.y < thumb_base.y and  # pinky above thumb base
                  abs(index_tip.x - middle_tip.x) < 0.2 and  # index and middle close
                  abs(ring_tip.x - pinky_tip.x) < 0.2):  # ring and pinky close
                hand_gesture = 'B in ASL'
            # Check for C in ASL (curved fingers, index and thumb separated)
            elif (abs(index_tip.x - thumb_tip.x) > 0.2 and  # index and thumb separated
                  abs(index_tip.y - thumb_tip.y) > 0.2 and  # index and thumb not aligned
                  abs(pinky_tip.x - thumb_tip.x) > 0.2):  # pinky and thumb separated
                hand_gesture = 'C in ASL'
            else:
                hand_gesture = 'other'

            # Perform actions based on gesture
            if hand_gesture == 'peace sign':
                print("PEACEEEE")
            elif hand_gesture == 'A in ASL':
                print("A in ASL")
            elif hand_gesture == 'B in ASL':
                print("B in ASL")
            elif hand_gesture == 'C in ASL':
                print("C in ASL")
            else:
                print("Not detected")

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
