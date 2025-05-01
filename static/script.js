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
