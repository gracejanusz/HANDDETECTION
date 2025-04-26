import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Paths to the dataset
train_csv_path = "/Users/efeucer/Desktop/hacktech/HANDDETECTION/classifier/mnist/sign_mnist_train/sign_mnist_train.csv"
test_csv_path = "/Users/efeucer/Desktop/hacktech/HANDDETECTION/classifier/mnist/sign_mnist_test/sign_mnist_test.csv"

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Output file for landmarks
output_csv_path = "/Users/efeucer/Desktop/hacktech/HANDDETECTION/classifier/landmarks.csv"

# Load data from CSV
def load_sign_mnist_data(csv_path):
    data = pd.read_csv(csv_path)
    labels = data.iloc[:, 0].values  # First column is the label
    images = data.iloc[:, 1:].values  # Remaining columns are pixel values
    images = images.reshape(-1, 28, 28)  # Reshape to 28x28 images
    return images, labels

# Process a single image
def process_image(image, label):
    # Convert image to uint8
    image = image.astype(np.uint8)
    
    # Convert grayscale image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Process the image with Mediapipe
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Flatten landmarks into a single row
            flattened_landmarks = []
            for lm in hand_landmarks.landmark:
                flattened_landmarks.extend([lm.x, lm.y, lm.z])
            return [label] + flattened_landmarks  # Include label as the first column
    return None  # Return None if no landmarks are detected

# Process the dataset
def process_dataset(csv_path, dataset_name):
    images, labels = load_sign_mnist_data(csv_path)
    processed_data = []
    for i, (image, label) in enumerate(zip(images, labels)):
        result = process_image(image, label)
        if result:
            processed_data.append(result)
        if i % 100 == 0:
            print(f"Processed {i}/{len(images)} images from {dataset_name} dataset.")
    return processed_data

# Process train and test datasets
train_data = process_dataset(train_csv_path, "train")
test_data = process_dataset(test_csv_path, "test")

# Combine and save the data to a single CSV file
all_data = train_data + test_data
columns = ["label"] + [f"landmark_{i}" for i in range(len(all_data[0]) - 1)]  # Generate column names
df = pd.DataFrame(all_data, columns=columns)
df.to_csv(output_csv_path, index=False)
print(f"Saved all landmarks to '{output_csv_path}'.")