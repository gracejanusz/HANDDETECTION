import os
import csv
import mediapipe as mp
import cv2

# Paths
dataset_path = '/Users/efeucer/Desktop/hacktech/HANDDETECTION/classifier/asl_dataset'
output_csv = '/Users/efeucer/Desktop/hacktech/HANDDETECTION/classifier/landmarks.csv'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Prepare CSV file
header_written = False

with open(output_csv, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Iterate through each folder (0-9, A-Z)
    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        # Iterate through each image in the folder
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe Hands
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Extract landmarks from the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Write header if not already written
                if not header_written:
                    header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
                    csv_writer.writerow(header)
                    header_written = True

                # Write the row to the CSV
                csv_writer.writerow([label] + landmarks)

print(f"Landmark data has been saved to {output_csv}")
hands.close()