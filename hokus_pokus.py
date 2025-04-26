import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

os.chdir("./classifier")

# Load data from CSV
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values  # Features (landmarks)
    y = data.iloc[:, 0].values  # Labels (letters)
    return X, y

# Load data
csv_file_path = "../data.csv"
X, y = load_data_from_csv(csv_file_path)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Path to save/load the trained model
model_path = "svm_classifier.joblib"

# Check if the model already exists
if os.path.exists(model_path):
    # Load the trained model
    svm_clf = joblib.load(model_path)
    print("Loaded trained SVM classifier from disk.")
else:
    # Initialize and train the SVM classifier
    svm_clf = SVC(kernel='poly', C=1.0, gamma='scale', random_state=42)
    svm_clf.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(svm_clf, model_path)
    print("Trained and saved SVM classifier.")

# Predict on test set
y_pred = svm_clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)