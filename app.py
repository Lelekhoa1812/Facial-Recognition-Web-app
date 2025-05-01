import os, cv2, glob
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
import face_recognition
from fer import FER
import onnxruntime as ort

# ──────────────────────── Config paths ────────────────────────
KNOWN_DIR    = "known_faces"
SNAP_DIR     = "snapshots"
MODEL_PATH   = "models/model1.onnx" # from https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# ──────────────────────── Load known faces ────────────────────
known_encs, known_names = [], []
for fn in os.listdir(KNOWN_DIR):
    if fn.lower().endswith(('.jpg', '.png')):
        img = face_recognition.load_image_file(os.path.join(KNOWN_DIR, fn))
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encs.append(enc[0])
            known_names.append(os.path.splitext(fn)[0])

# ──────────────────── Load emotion & liveness model ───────────
emotion_det = FER(mtcnn=True)
liveness_sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def is_live_face(img):
    """Return True if live, False if spoof."""
    try:
        inp = cv2.resize(img, (224,224)).astype(np.float32) / 255.0
        inp = inp.transpose(2,0,1)[None]               # 1×3×224×224
        spoil, live = liveness_sess.run(None, {"input": inp})[0][0]
        return live > spoil                            # live prob higher
    except Exception as e:
        print("Liveness error:", e)
        return False

def detect_emotion(img):
    try:
        emo, _ = emotion_det.top_emotion(img)
        return emo.capitalize() if emo else "Neutral"
    except:
        return "Neutral"

def save_snapshot(name, img):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(SNAP_DIR, f"{name}_{ts}.jpg")
    cv2.imwrite(fp, img);  print("📸", fp)

# ───────────────────────── Flask app ──────────────────────────
app = Flask(__name__)
saved_ids = set()   # avoid duplicate snapshots in one run

# ——— Video generator
def gen_frames():
    cam = cv2.VideoCapture(0) # Start streaming
    while True:
        ok, frame = cam.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        # Process image and make recognition if heuristic distance is by within
        for (top,right,bottom,left), enc in zip(locs, encs):
            name = "Unknown"
            dist = face_recognition.face_distance(known_encs, enc)
            if len(dist) and dist.min() < 0.5: # heuristic distance adjust if needed
                idx = dist.argmin(); name = known_names[idx]
            # Crop down the frame consist of the face
            crop = frame[top:bottom, left:right]
            live    = is_live_face(crop)
            emotion = detect_emotion(crop)
            label   = f"{name} - {emotion}"
            color   = (0,255,0) if live else (0,0,255)
            # Draw labels
            cv2.rectangle(frame,(left,top),(right,bottom),color,2)
            cv2.rectangle(frame,(left,bottom-20),(right,bottom),color,cv2.FILLED)
            cv2.putText(frame,label,(left+5,bottom-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1)
            # Unknown face can be registered
            if live and name == "Unknown":
                # Export for JS
                _, buf_crop = cv2.imencode('.jpg', crop)
                b64_crop = base64.b64encode(buf_crop).decode('utf-8')
                frame = cv2.putText(frame, "[Unknown - click to register]", (left, top-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                # Overlay base64 face snapshot in frame (sent as metadata? or client polls for latest?)
                app.config['last_unknown_face'] = f"data:image/jpeg;base64,{b64_crop}"
            # Save a snapshot for known face id
            if live and name!="Unknown" and name not in saved_ids:
                save_snapshot(name, crop);  saved_ids.add(name)
        # Save encode img as in .jpg + the frame
        ret, buf = cv2.imencode('.jpg', frame)
        yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

# ——— Routes
@app.route("/")
def home(): return render_template("index.html")

# Streaming video
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Save a snapshot but only for known faces
@app.route("/snapshots")
def snapshots():
    pics = sorted(glob.glob(f"{SNAP_DIR}/*.jpg"), reverse=True)
    pics = [p.replace("\\","/") for p in pics]   # for Windows paths
    return jsonify(pics)

# ——— Register unknown faces
from flask import request
import base64
@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    name = data.get("name")
    img_data = data.get("image")
    # Error log for corrupted data
    if not name or not img_data:
        return jsonify({"error": "Name and image required"}), 400
    # Decode image from base64
    img_bytes = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Save to known_faces/
    save_path = os.path.join(KNOWN_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, img)
    # Re-encode and update memory
    new_enc = face_recognition.face_encodings(img)
    if new_enc:
        known_encs.append(new_enc[0])
        known_names.append(name)
        print(f"✅ Registered new face: {name}")
        return jsonify({"success": True}), 200
    else:
        return jsonify({"error": "Face encoding failed"}), 500

# ────────────────────────── Run local ─────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
