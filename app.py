import os, cv2, glob, base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from insightface.app import FaceAnalysis
import onnxruntime as ort

# ───── Setup ─────
KNOWN_DIR, SNAP_DIR = "known_faces", "snapshots"
MODEL_PATH = "models/model1.onnx"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# ───── Init FaceAnalysis (InsightFace) ─────
faceapp = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

# ───── Init Anti-Spoof Model ─────
liveness_sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ───── Load known faces ─────
known_names, known_feats = [], []
for fn in os.listdir(KNOWN_DIR):
    path = os.path.join(KNOWN_DIR, fn)
    img = cv2.imread(path)
    faces = faceapp.get(img)
    if faces:
        known_names.append(os.path.splitext(fn)[0])
        known_feats.append(faces[0].normed_embedding)

# ───── Helpers ─────
def cosine_sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_live_face(img):
    try:
        inp = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]
        spoil, live = liveness_sess.run(None, {"input": inp})[0][0]
        return live > spoil
    except Exception as e:
        print("Liveness error:", e)
        return False

def save_snapshot(name, img):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(SNAP_DIR, f"{name}_{ts}.jpg")
    cv2.imwrite(fp, img)
    print("📸 Saved:", fp)

# ───── Flask ─────
app = Flask(__name__)
saved_ids = set()

def gen_frames():
    if 'cam' not in app.config:
        app.config['cam'] = cv2.VideoCapture(0)
    cam = app.config['cam']

    while True:
        ret, frame = cam.read()
        if not ret: break

        faces = faceapp.get(frame)
        for face in faces:
            box = face.bbox.astype(int)
            top, right, bottom, left = box[1], box[2], box[3], box[0]
            crop = frame[top:bottom, left:right]
            name = "Unknown"
            best_sim = 0

            for feat, kname in zip(known_feats, known_names):
                sim = cosine_sim(face.normed_embedding, feat)
                if sim > 0.6 and sim > best_sim:
                    name, best_sim = kname, sim

            # Anti-spoofing
            live = is_live_face(crop)
            label = f"{name}"
            color = (0, 255, 0) if live else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if live and name == "Unknown":
                _, buf_crop = cv2.imencode('.jpg', crop)
                b64_crop = base64.b64encode(buf_crop).decode('utf-8')
                app.config['last_unknown_face'] = f"data:image/jpeg;base64,{b64_crop}"
                cv2.putText(frame, "[Unknown - click to register]", (left, bottom + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

            if live and name != "Unknown" and name not in saved_ids:
                save_snapshot(name, crop); saved_ids.add(name)

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
        return jsonify({"error": "Name and image required"}), 400

    img_bytes = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    save_path = os.path.join(KNOWN_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, img)

    faces = faceapp.get(img)
    if faces:
        known_feats.append(faces[0].normed_embedding)
        known_names.append(name)
        print(f"✅ Registered new face: {name}")
        return jsonify({"success": True}), 200
    else:
        return jsonify({"error": "Encoding failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
