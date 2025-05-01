// --- Gallery ---
const modal = document.getElementById('modal');
const closeBtn = document.querySelector('.close');
document.getElementById('showSnaps').onclick = async ()=>{
  const res = await fetch('/snapshots'); const pics = await res.json();
  const gal = document.getElementById('gallery'); gal.innerHTML='';
  pics.forEach(p=>{const img=document.createElement('img');img.src=p;gal.appendChild(img);});
  modal.style.display='block';
}
closeBtn.onclick = ()=> modal.style.display='none';
window.onclick = e=>{if(e.target==modal) modal.style.display='none';}

// --- Unknown faces registration ---
const regModal = document.getElementById('registerModal');
const regBtn = document.getElementById('registerBtn');
const regClose = document.querySelector('.closeReg');
const facePreview = document.getElementById('facePreview');
const confirmReg = document.getElementById('confirmReg');
// Poll last unknown face snapshot from backend
async function fetchLastUnknown() {
  const img = await fetch("/video_feed", { method: 'HEAD' });
  const preview = appConfig.last_unknown_face; // bind this from /register
  return preview;
}
// Handle none 
regBtn.onclick = async () => {
  const preview = await fetch("/last_unknown"); // optional if exposed
  if (!preview) return alert("No unknown face detected!");
  facePreview.src = preview;
  facePreview.dataset.base64 = preview;
  regModal.style.display = 'block';
};
// Close registration modal
regClose.onclick = () => regModal.style.display = 'none';
// Confirm btn trigger new id (name) saving
confirmReg.onclick = async () => {
  const name = document.getElementById('newName').value.trim();
  const base64img = facePreview.dataset.base64;
  if (!name) return alert("Please enter a name");
  const res = await fetch('/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, image: base64img })
  });
  const json = await res.json();
  alert(json.success ? "✅ Face registered!" : "❌ Error: " + json.error);
  regModal.style.display = 'none';
};
