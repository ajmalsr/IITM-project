"""
HOG + SVM Classifier for Image Classification
------------------------------------------------
This script trains an SVM classifier using Histogram of Oriented Gradients (HOG) features.
"""

import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

# Set dataset path
DATASET_PATH = "datas"

# Lists to store extracted features and labels
hog_features = []
labels = []

# Read and process images from dataset
categories = os.listdir(DATASET_PATH)
print(f"Categories found: {categories}")

for category in categories:
    category_path = os.path.join(DATASET_PATH, category)
    image_files = os.listdir(category_path)

    for image_file in image_files:
        image_path = os.path.join(category_path, image_file)

        # Read and preprocess image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (64, 32))  # Standardized size

        # Extract Histogram of Oriented Gradients (HOG) features
        hog_descriptor = feature.hog(resized_image, 
                                     orientations=9, 
                                     pixels_per_cell=(10, 10),
                                     cells_per_block=(2, 2), 
                                     transform_sqrt=True, 
                                     block_norm="L1")

        hog_features.append(hog_descriptor)
        labels.append(1 if category == "vehicles" else 0)  # Binary labels (1 = Vehicles, 0 = Others)

# Convert lists to NumPy arrays
hog_features = np.array(hog_features)
labels = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Train SVM model
print(f"Training on {len(y_train)} images...")
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(X_train, y_train)

# Evaluate model
accuracy = svm_model.score(X_test, y_test)
print(f"SVM Model Accuracy: {accuracy:.4f}")

# Predict on test set
y_pred = svm_model.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute and plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], 'y--', label="Random Guess")
plt.plot(fpr, tpr, marker='.', label="SVM Model")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save the trained model
joblib.dump(svm_model, "hog_svm_model.pkl")
print("Model saved as 'hog_svm_model.pkl'.")
