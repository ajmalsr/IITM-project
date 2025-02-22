# 🚗 Vehicle Detection using HOG + SVM

This repository contains **vehicle detection scripts** using **HOG (Histogram of Oriented Gradients) features** and **SVM (Support Vector Machine)**.\
The scripts allow detection on **images, videos, and live streams**, with optional **motion tracking**.

---

## 📂 Contents
| Script Name                   | Description |
|--------------------------------|------------|
| **`hog_svm_classifier.py`**    | Trains an SVM model using HOG features on a dataset. |
| **`hog_svm_video_detector.py`** | Detects vehicles in a video using HOG + SVM with Sliding Window & NMS. |
| **`hog_svm_video_tracker.py`** | Adds motion tracking using OpenCV’s Background Subtraction (MOG2) to filter moving objects. |
| **`hog_svm_image_test.py`** | Tests the trained model on a folder of test images. |
| **`hog_svm_video_test.py`** | Detects objects in video but without motion tracking. |

---

## 📂 1️⃣ `hog_svm_classifier.py` - Train HOG + SVM Model

### 🎯 Functionality

- Extracts **HOG features** from images.
- Trains a **Linear SVM** classifier for vehicle classification.
- Saves the trained model as `hog_svm_model.pkl`.

### ⚙️ Dependencies

```sh
pip install numpy scikit-image opencv-python joblib scikit-learn
```

### 🛠 How to Train the Model

1. Place training images inside a folder, e.g., `datas/vehicles` and `datas/non-vehicles`.
2. Run:
   ```sh
   python hog_svm_classifier.py
   ```
3. The model is saved as `hog_svm_model.pkl`.

---

## 🎥 2️⃣ `hog_svm_video_detector.py` - Video Detection

### 🎯 Functionality

- Detects vehicles in a **video** using **HOG + SVM**.
- Uses **Sliding Window & Image Pyramid** for multi-scale detection.
- Applies **Non-Maximum Suppression (NMS)** to filter overlapping detections.

### 🛠 How to Run

```sh
python hog_svm_video_detector.py
```

**⚠ Requires:** `hog_svm_model.pkl` and a video file (e.g., `cam2.mkv`).

---

## 🚗 3️⃣ `hog_svm_video_tracker.py` - Video Detection with Motion Tracking

### 🎯 Functionality

- Adds **motion tracking** using **OpenCV’s Background Subtraction (MOG2)**.
- Detects **only moving vehicles** (reduces false positives).
- Uses **sliding window, image pyramids, and NMS**.

### 🛠 How to Run

```sh
python hog_svm_video_tracker.py
```

**⚠ Requires:** `hog_svm_model.pkl` and a video file.

---

## 🖼 4️⃣ `hog_svm_image_test.py` - Image Testing

### 🎯 Functionality

- Loads the trained SVM model and classifies **test images**.
- Extracts HOG features and predicts object type.
- Displays **predictions with HOG visualization**.

### 🛠 How to Run

```sh
python hog_svm_image_test.py
```

**⚠ Requires:** A folder containing test images (e.g., `TEST_IMAGE/`).

---

## 📺 5️⃣ `hog_svm_video_test.py` - Video Testing without Motion Tracking

### 🎯 Functionality

- Detects objects in **video** using HOG + SVM.
- **No motion tracking** (compares performance with tracker version).

### 🛠 How to Run

```sh
python hog_svm_video_test.py
```

**⚠ Requires:** `hog_svm_model.pkl` and a video file.

---

## 🛠 Dependencies

Before running any script, install the required dependencies:

```sh
pip install numpy opencv-python scikit-image scikit-learn joblib imutils
```

---

## 🚀 Next Steps

- **Convert this for mobile deployment (TFLite, CoreML, ONNX)?**
- **Improve accuracy using Deep Learning (YOLO, Faster R-CNN)?**
- **Integrate tracking with object IDs (SORT, DeepSORT, Kalman Filter)?**

