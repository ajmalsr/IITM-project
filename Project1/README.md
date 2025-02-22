# 🚦 Traffic Light Detection from Video

This script detects traffic light changes (Red, Yellow, Green) in a video using **OpenCV**. It processes each frame, applies **thresholding** to identify illuminated traffic lights, and logs the timestamp when each light turns on.

---

## 📌 Features
- **Detects traffic light changes** (Red, Yellow, Green) in a video.
- Uses **Region of Interest (ROI)** to analyze specific areas where lights appear.
- Applies **grayscale conversion & thresholding** for light detection.
- Outputs timestamps when a light turns on.

---

## 🛠 Dependencies
Before running the script, install the required libraries:
```sh
pip install opencv-python numpy
```
---

## 🚀 How to Run
1. **Place a video file** in the same directory as the script.
2. **Update the video path** in the script:
   ```python
   video_path = 'path_of_the_video.mp4'
   ```
3. **Run the script:**
   ```sh
   python traffic_light_detection.py
   ```
4. **Output Example:**
   ```sh
   Green is ON at: 2.50 sec
   Yellow is ON at: 5.30 sec
   Red is ON at: 6.90 sec
   ```

---

## 🎯 How It Works
1. Reads video frame-by-frame.
2. Extracts **Regions of Interest (ROIs)** for Red, Yellow, and Green lights.
3. Converts them to **grayscale** and applies **thresholding** to identify illumination.
4. Compares with the **previous frame** to detect state changes.
5. Logs **timestamp of state changes**.

---

## 📌 Next Steps
- **Enhance detection** by adding support for dynamic traffic light positions.
- **Improve accuracy** by implementing deep learning for better classification.
- **Deploy on real-time video streams** from cameras.

🚀 **Let's build smarter traffic detection!**

