import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import joblib

os.chdir("./classifier")

# Load data from CSV
def load_mnist_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separate features and labels
    X_train = train_data.iloc[:, 1:].values  # Features
    y_train = train_data.iloc[:, 0].values  # Labels
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    return X_train, y_train, X_test, y_test

# Paths to MNIST data
train_path = "mnist/sign_mnist_train/sign_mnist_train.csv"
test_path = "mnist/sign_mnist_test/sign_mnist_test.csv"

# Load and preprocess data
X_train, y_train, X_test, y_test = load_mnist_data(train_path, test_path)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Reduce features to 63 using PCA
pca = PCA(n_components=63)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Path to save the trained model
model_path = "svm_classifier.joblib"

# Check if the model already exists
if os.path.exists(model_path):
    # Load the trained model
    svm_clf = joblib.load(model_path)
    print("Loaded trained SVM classifier from disk.")
else:
    # Initialize and train the SVM classifier
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_clf.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(svm_clf, model_path)
    print("Trained and saved SVM classifier.")

# Predict on test set
y_pred = svm_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy:", accuracy)
print(report)

# Save the label encoder
label_encoder_path = "label_encoder_classes.npy"
np.save(label_encoder_path, label_encoder.classes_)
print("Label encoder saved.")