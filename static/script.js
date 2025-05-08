// =======================
// 📸 Webcam + Frame Capture
// =======================
const video = document.getElementById("cam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let latestUnknownFace = null;

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// Periodically send frame to backend
setInterval(async () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  const base64 = canvas.toDataURL("image/jpeg");
  // Handshake with FastAPI
  try {
    const res = await fetch("/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 })
    });
    const faces = await res.json();
    // Redraw base image
    ctx.drawImage(video, 0, 0);
    // Iterate between faces
    faces.forEach(face => {
      const [x, y, bw, bh] = face.box;
      ctx.strokeStyle = face.live ? "lime" : "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, bw, bh);

      ctx.fillStyle = face.live ? "rgba(0,255,0,0.6)" : "rgba(255,0,0,0.6)";
      ctx.fillRect(x, y - 20, bw, 20);

      ctx.fillStyle = "#000";
      ctx.font = "14px sans-serif";
      ctx.fillText(face.name, x + 5, y - 5);
    });
    // Show register button only for live unknowns
    const unknown = faces.find(f => f.name === "Unknown" && f.live);
    if (unknown) {
      latestUnknownFace = base64;
      document.getElementById("registerBtn").style.display = "inline-block";
    } else {
      latestUnknownFace = null;
      document.getElementById("registerBtn").style.display = "none";
    }
  } catch (e) {
    console.error("Failed to send frame:", e);
  }
}, 2000); // Every 2 seconds


// =======================
// 🖼️ Snapshot Gallery Modal
// =======================
const modal = document.getElementById("modal");
const closeBtn = document.querySelector(".close");
document.getElementById("showSnaps").onclick = async () => {
  const res = await fetch("/snapshots");
  const pics = await res.json();
  const gal = document.getElementById("gallery");
  gal.innerHTML = '';
  pics.forEach(p => {
    const img = document.createElement("img");
    img.src = p;
    gal.appendChild(img);
  });
  modal.style.display = "block";
};
closeBtn.onclick = () => modal.style.display = "none";
window.onclick = e => { if (e.target === modal) modal.style.display = "none"; };

// =======================
// 🧍 Unknown Face Registration
// =======================
const regModal = document.getElementById("registerModal");
const regBtn = document.getElementById("registerBtn");
const regClose = document.querySelector(".closeReg");
const facePreview = document.getElementById("facePreview");
const confirmReg = document.getElementById("confirmReg");

// Show registration modal
regBtn.onclick = () => {
  if (!latestUnknownFace) return alert("No unknown face detected!");
  facePreview.src = latestUnknownFace;
  facePreview.dataset.base64 = latestUnknownFace;
  regModal.style.display = "block";
};

// Close registration modal
regClose.onclick = () => regModal.style.display = "none";

// Confirm registration
confirmReg.onclick = async () => {
  const name = document.getElementById("newName").value.trim();
  const base64img = facePreview.dataset.base64;
  if (!name) return alert("Please enter a name");

  const res = await fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, image: base64img })
  });

  const json = await res.json();
  alert(json.success ? "✅ Face registered!" : "❌ Error: " + json.error);
  regModal.style.display = "none";
};
