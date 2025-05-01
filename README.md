# 🧠 Face Recognition with Emotion & Liveness Detection

A Python Flask-based real-time face recognition web app that detects known faces, classifies facial expressions, and performs liveness (anti-spoofing) detection using ONNX models.

---

## 🔍 Features

- ✅ Real-time webcam-based face recognition
- 😀 Emotion classification with `FER`
- 🛡️ Anti-spoofing detection using ONNX model (`Silent-Face-Anti-Spoofing`)
- 📸 Snapshot saving for known identities
- 🖼️ Snapshot gallery modal UI
- 🌐 Modern frontend using HTML, CSS, JavaScript

---

## 🏗️ Tech Stack

- Python 3.9+
- Flask
- OpenCV
- face_recognition (uses `dlib`, requires Docker deployment)
- onnxruntime
- fer (for emotion detection)

---

## 🚀 How to Run (Docker)

```bash
# Clone this repo
git clone https://github.com/Lelekhoa1812/Facial-Recognition-Web-app.git
cd Facial-Recognition-Web-app

# Build and run the container
docker build -t face-app .
docker run -p 7860:7860 face-app
```
=> Then visit: `http://localhost:7860`

---

## 📁 Project Structure
```plaintext
├── app.py                # Main Flask app
├── known_faces/          # Folder of registered face images
├── snapshots/            # Saved face snapshots
├── models/               # ONNX anti-spoofing model
├── static/
│   └── index.html        # UI layout
│   ├── styles.css        # CSS styles
│   └── script.js         # Modal gallery logic
├── requirements.txt
└── Dockerfile
```

---

## 📦 Model Sources

- **Anti-Spoofing ONNX**: [Silent-Face-Anti-Spoofing](https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite)
- **Emotion Detection**: [`fer`](https://github.com/justinshenk/fer)
