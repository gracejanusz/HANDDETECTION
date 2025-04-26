# === match_sign_to_gloss.py ===
import os
import csv
import numpy as np
from scipy.spatial.distance import cdist

# Path where all recorded signs are saved
DATA_DIR = "data/collected_keypoints"

# Function to load a saved CSV sequence
def load_sequence(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        sequence = np.array(list(reader)).astype(np.float32)
    return sequence

# Function to load all gloss sequences into memory
def load_all_glosses():
    database = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            label = filename[:-4]  # Remove .csv extension
            path = os.path.join(DATA_DIR, filename)
            database[label] = load_sequence(path)
    return database

# Compare two sequences (simple DTW-like alignment)
def compare_sequences(seq1, seq2):
    # Resize both to same length for simplicity
    min_len = min(len(seq1), len(seq2))
    seq1 = seq1[:min_len]
    seq2 = seq2[:min_len]

    # Compute average L2 distance per frame
    distances = np.linalg.norm(seq1 - seq2, axis=1)
    return np.mean(distances)

# Match an input sequence to the closest gloss

def match_gloss(input_sequence, database):
    best_label = None
    best_score = float('inf')

    for label, sequence in database.items():
        score = compare_sequences(input_sequence, sequence)
        if score < best_score:
            best_score = score
            best_label = label

    return best_label, best_score

if __name__ == "__main__":
    # Example usage
    input_path = input("Enter path to the CSV file you want to match: ")
    input_sequence = load_sequence(input_path)

    print("[*] Loading database...")
    database = load_all_glosses()

    print("[*] Matching...")
    best_label, best_score = match_gloss(input_sequence, database)

    print(f"Best match: {best_label} (Score: {best_score:.4f})")