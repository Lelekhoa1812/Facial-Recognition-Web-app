# Access site: https://binkhoale1812-facial-recognition.hf.space/
import os, cv2, glob, base64
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime
import mediapipe as mp
import onnxruntime as ort
import torch
from torchvision import transforms
from mobilefacenet import MobileFaceNet

# ───── Setup ─────
LIVENESS_MODEL = "models/model1.onnx"
DETECTION_MODEL = "models/mobilefacenet.pt"

ROOT_DIR = "/data"
KNOWN_DIR = os.path.join(ROOT_DIR, "known_faces")
SNAP_DIR  = os.path.join(ROOT_DIR, "snapshots")
def ensure_dirs():
    try:
        os.makedirs(KNOWN_DIR, exist_ok=True)
        os.makedirs(SNAP_DIR, exist_ok=True)
    except Exception as e:
        print("Error when creating img directories", e)

# ───── Init MediaPipe Face Detection ─────
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ───── Init ONNX models ─────
live_sess = ort.InferenceSession(LIVENESS_MODEL, providers=["CPUExecutionProvider"])

# ───── Init PT models ─────
device = torch.device("cpu")
embed_model = MobileFaceNet().to(device)
embed_model.load_state_dict(torch.load(DETECTION_MODEL, map_location=device))
embed_model.eval()

# ───── Storage ─────
known_names, known_feats = [], []

# ───── Feature Embedding ─────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
def get_embedding(face_img):
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(face_tensor).cpu().numpy()
    return emb[0] / np.linalg.norm(emb[0])


# ───── Liveness Detection ─────
def is_live_face(img):
    try:
        resized = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        input_blob = resized.transpose(2, 0, 1)[np.newaxis]
        spoof, live = live_sess.run(None, {"input": input_blob})[0][0]
        return live > spoof
    except Exception as e:
        print("Liveness error:", e)
        return False

# ───── Load Known Faces ─────
for fn in os.listdir(KNOWN_DIR):
    path = os.path.join(KNOWN_DIR, fn)
    img = cv2.imread(path)
    if img is not None:
        results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            bboxC = results.detections[0].location_data.relative_bounding_box
            h, w = img.shape[:2]
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face = img[y:y+bh, x:x+bw]
            known_feats.append(get_embedding(face))
            known_names.append(os.path.splitext(fn)[0])

# ───── Similarity ─────
def cosine_sim(a, b): return np.dot(a, b)

# ───── Snapshot ─────
def save_snapshot(name, img):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(SNAP_DIR, f"{name}_{ts}.jpg")
    cv2.imwrite(fp, img)

# ───── Flask App ─────
app = Flask(__name__)
saved_ids = set()

def gen_frames():
    if 'cam' not in app.config:
        app.config['cam'] = cv2.VideoCapture(0)
    cam = app.config['cam']

    while True:
        ret, frame = cam.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x, y, bw, bh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                face = frame[y:y+bh, x:x+bw]
                emb = get_embedding(face)

                # Match
                name, best_sim = "Unknown", 0
                for kname, kfeat in zip(known_names, known_feats):
                    sim = cosine_sim(emb, kfeat)
                    if sim > 0.6 and sim > best_sim:
                        name, best_sim = kname, sim

                # Liveness
                live = is_live_face(face)
                label = f"{name}"
                color = (0,255,0) if live else (0,0,255)

                cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if live and name == "Unknown":
                    _, buf_crop = cv2.imencode('.jpg', face)
                    b64_crop = base64.b64encode(buf_crop).decode('utf-8')
                    app.config['last_unknown_face'] = f"data:image/jpeg;base64,{b64_crop}"
                    cv2.putText(frame, "[Register?]", (x, y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

                if live and name != "Unknown" and name not in saved_ids:
                    save_snapshot(name, face)
                    saved_ids.add(name)

        ret, buf = cv2.imencode('.jpg', frame)
        yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

@app.route("/")
def home(): return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshots")
def snapshots():
    pics = sorted(glob.glob(f"{SNAP_DIR}/*.jpg"), reverse=True)
    return jsonify([p.replace("\\", "/") for p in pics])

@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    name, img_data = data.get("name"), data.get("image")
    if not name or not img_data:
        return jsonify({"error": "Missing name/image"}), 400

    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(KNOWN_DIR, f"{name}.jpg"), img)
    emb = get_embedding(img)
    known_names.append(name)
    known_feats.append(emb)
    return jsonify({"success": True}), 200

if __name__ == "__main__":
    ensure_dirs()  # Ensure folder are uploaded
    app.run(host="0.0.0.0", port=7860)
