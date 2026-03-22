"""Slide-scan UI: simple video-based row-at-a-time scanning.

Flow: For each of 3 rows, user records a short video sliding across the row.
Client extracts 3 evenly-spaced frames per video, uploads to /slide-scan/identify.

Screens:
  1. Before scan: camera preview + "Start Scanning" button
  2. During scan: recording indicator + timer + "Done" button
  3. Row result: 3 card thumbnails with names + prices + "Next Row" button
  4. All done: 3x3 grid of all 9 cards with total value

Integration:
    GET  /slide-scan            -> serve this HTML
    POST /slide-scan/identify   -> receive card_0..card_8 images, identify
"""

SLIDE_SCAN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Slide Scan</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#000;color:#fff;font-family:-apple-system,system-ui,sans-serif;
  height:100dvh;width:100vw;display:flex;flex-direction:column;overflow:hidden}
video#cam{width:100%;flex:1;object-fit:cover}
#hud{position:absolute;top:0;left:0;right:0;padding:16px 20px;
  background:linear-gradient(rgba(0,0,0,.7),transparent);z-index:10;
  display:flex;justify-content:space-between;align-items:center}
#rowText{font-size:22px;font-weight:700}
#recDot{width:14px;height:14px;border-radius:50%;background:#f44;display:none}
#recDot.on{display:inline-block;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.3}}
#timer{font-size:16px;font-family:monospace;display:none}
#btnWrap{position:absolute;bottom:0;left:0;right:0;padding:24px 20px;
  background:linear-gradient(transparent,rgba(0,0,0,.7));z-index:10}
#btn{width:100%;padding:18px;font-size:20px;font-weight:700;border:none;
  border-radius:14px;cursor:pointer;transition:all .15s}
#btn:active{transform:scale(.97)}
.btn-green{background:#22c55e;color:#000}
.btn-red{background:#ef4444;color:#fff}
.btn-blue{background:#3b82f6;color:#fff}
#resultScreen{position:absolute;inset:0;z-index:20;background:#000;
  display:none;flex-direction:column;overflow-y:auto;padding:20px}
#resultScreen.show{display:flex}
.card-row{display:flex;gap:10px;justify-content:center;margin-bottom:8px}
.card-thumb{flex:0 0 30%;text-align:center}
.card-thumb img{width:100%;border-radius:6px;aspect-ratio:5/7;object-fit:cover;
  background:#222}
.card-thumb .name{font-size:12px;margin-top:4px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.card-thumb .price{font-size:14px;font-weight:700;color:#22c55e}
#finalGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
#totalValue{font-size:22px;font-weight:700;color:#22c55e;text-align:center;
  margin:16px 0}
h2{text-align:center;margin-bottom:12px}
#modeToggle{display:flex;align-items:center;gap:6px;font-size:12px;opacity:.9}
#modeToggle label{cursor:pointer}
#modeToggle input{display:none}
#modeToggle .toggle{width:36px;height:20px;border-radius:10px;background:#555;
  position:relative;display:inline-block;vertical-align:middle;transition:background .2s;cursor:pointer}
#modeToggle .toggle::after{content:'';position:absolute;left:2px;top:2px;width:16px;height:16px;
  border-radius:50%;background:#fff;transition:transform .2s}
#modeToggle input:checked+.toggle{background:#22c55e}
#modeToggle input:checked+.toggle::after{transform:translateX(16px)}
</style>
</head>
<body>

<video id="cam" autoplay playsinline muted></video>

<div id="hud">
  <span id="rowText">Row 1 of 3</span>
  <span style="display:flex;align-items:center;gap:8px">
    <span id="modeToggle">
      <label><input type="checkbox" id="fastMode" checked><span class="toggle"></span></label>
      <span id="modeLabel">Fast</span>
    </span>
    <span id="recDot"></span>
    <span id="timer"></span>
  </span>
</div>

<div id="btnWrap">
  <button id="btn" class="btn-green" onclick="onBtn()">Start Scanning</button>
</div>

<div id="resultScreen"></div>

<script>
const cam = document.getElementById('cam');
const rowText = document.getElementById('rowText');
const recDot = document.getElementById('recDot');
const timerEl = document.getElementById('timer');
const btn = document.getElementById('btn');
const resultScreen = document.getElementById('resultScreen');
const fastModeEl = document.getElementById('fastMode');
const modeLabelEl = document.getElementById('modeLabel');

fastModeEl.addEventListener('change', () => {
  modeLabelEl.textContent = fastModeEl.checked ? 'Fast' : 'Full';
});

function getEndpoint() {
  return fastModeEl.checked ? '/slide-scan/fast' : '/slide-scan/identify';
}

let stream = null;
let recorder = null;
let chunks = [];
let row = 0;           // 0,1,2
let recording = false;
let timerStart = 0;
let timerRaf = null;
let allCards = [];      // accumulates across rows
const MAX_SEC = 5;

// --- Camera ---
async function initCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode:'environment', width:{ideal:1920}, height:{ideal:1080} },
      audio: false
    });
    cam.srcObject = stream;
  } catch(e) { rowText.textContent = 'Camera: ' + e.message; }
}
initCam();

// --- Button handler ---
function onBtn() {
  if (!recording) startRecording();
  else stopRecording();
}

function startRecording() {
  if (!stream) return;
  recording = true;
  chunks = [];

  const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9'
    : MediaRecorder.isTypeSupported('video/webm')
      ? 'video/webm'
      : 'video/mp4';
  recorder = new MediaRecorder(stream, { mimeType });
  recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  recorder.onstop = () => processVideo();
  recorder.start();

  // UI
  rowText.textContent = 'Slide across Row ' + (row+1) + '...';
  recDot.classList.add('on');
  timerEl.style.display = 'inline';
  btn.textContent = 'Done';
  btn.className = 'btn-red';
  timerStart = performance.now();
  tickTimer();

  // Auto-stop after MAX_SEC
  setTimeout(() => { if (recording) stopRecording(); }, MAX_SEC * 1000);
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  recorder.stop();
  recDot.classList.remove('on');
  timerEl.style.display = 'none';
  cancelAnimationFrame(timerRaf);
  btn.textContent = 'Processing...';
  btn.disabled = true;
}

function tickTimer() {
  if (!recording) return;
  const s = ((performance.now() - timerStart) / 1000).toFixed(1);
  timerEl.textContent = s + 's';
  timerRaf = requestAnimationFrame(tickTimer);
}

// --- Extract 3 frames from recorded video ---
async function processVideo() {
  const blob = new Blob(chunks, { type: recorder.mimeType });
  const url = URL.createObjectURL(blob);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  v.src = url;

  await new Promise((res, rej) => {
    v.onloadedmetadata = res;
    v.onerror = rej;
  });

  const dur = v.duration;
  if (!dur || dur < 0.3) {
    showRowResult([], 'Video too short');
    return;
  }

  // Seek to 3 evenly spaced points (25%, 50%, 75%)
  const times = [dur*0.25, dur*0.5, dur*0.75];
  const frames = [];
  const canvas = document.createElement('canvas');

  for (const t of times) {
    v.currentTime = t;
    await new Promise(r => { v.onseeked = r; });
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(v, 0, 0);
    const frameBlob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.92));
    frames.push(frameBlob);
  }

  URL.revokeObjectURL(url);
  uploadRow(frames);
}

// --- Upload 3 frames for this row ---
async function uploadRow(frames) {
  const form = new FormData();
  for (let i = 0; i < frames.length; i++) {
    const pos = row * 3 + i;
    form.append('card_' + pos, frames[i], 'card_' + pos + '.jpg');
  }

  try {
    const resp = await fetch(getEndpoint(), { method:'POST', body: form });
    const data = await resp.json();
    if (data.error) { showRowResult([], data.error); return; }
    const rowCards = (data.cards || []).slice(0, 3);
    allCards.push(...rowCards);
    showRowResult(rowCards);
  } catch(e) {
    showRowResult([], 'Network error: ' + e.message);
  }
}

// --- Screen 3: Row result ---
function showRowResult(cards, error) {
  let h = '<h2>Row ' + (row+1) + ' Result</h2>';
  if (error) {
    h += '<p style="text-align:center;color:#f44;margin:20px">' + error + '</p>';
  } else {
    h += '<div class="card-row">';
    for (const c of cards) {
      const img = c.local_image_url || c.segment_image_url || '';
      const name = c.card_name || 'Unknown';
      const price = c.variant_price || c.market_price;
      h += '<div class="card-thumb">';
      if (img) h += '<img src="' + img + '">';
      h += '<div class="name">' + name + '</div>';
      if (price) h += '<div class="price">$' + price.toFixed(2) + '</div>';
      h += '</div>';
    }
    h += '</div>';
  }

  row++;
  if (row < 3) {
    h += '<div style="padding:20px"><button class="btn-green" style="width:100%;padding:16px;font-size:18px;font-weight:700;border:none;border-radius:14px;cursor:pointer" onclick="nextRow()">Next Row</button></div>';
  } else {
    h += '<div style="padding:20px"><button class="btn-blue" style="width:100%;padding:16px;font-size:18px;font-weight:700;border:none;border-radius:14px;cursor:pointer" onclick="showFinal()">View All Cards</button></div>';
  }

  resultScreen.innerHTML = h;
  resultScreen.classList.add('show');
}

function nextRow() {
  resultScreen.classList.remove('show');
  rowText.textContent = 'Row ' + (row+1) + ' of 3';
  btn.textContent = 'Start Scanning';
  btn.className = 'btn-green';
  btn.disabled = false;
}

// --- Screen 4: Final grid ---
function showFinal() {
  const total = allCards.reduce((s,c) => s + (c.variant_price || c.market_price || 0), 0);
  let h = '<h2>Page Complete</h2>';
  h += '<div id="totalValue">$' + total.toFixed(2) + ' total</div>';
  h += '<div id="finalGrid">';
  for (const c of allCards) {
    const img = c.local_image_url || c.segment_image_url || '';
    const name = c.card_name || 'Unknown';
    const price = c.variant_price || c.market_price;
    h += '<div class="card-thumb">';
    if (img) h += '<img src="' + img + '">';
    h += '<div class="name">' + name + '</div>';
    if (price) h += '<div class="price">$' + price.toFixed(2) + '</div>';
    h += '</div>';
  }
  h += '</div>';
  h += '<div style="padding:20px"><button class="btn-green" style="width:100%;padding:16px;font-size:18px;font-weight:700;border:none;border-radius:14px;cursor:pointer" onclick="resetAll()">Scan Again</button></div>';

  resultScreen.innerHTML = h;
}

function resetAll() {
  row = 0;
  allCards = [];
  resultScreen.classList.remove('show');
  resultScreen.innerHTML = '';
  rowText.textContent = 'Row 1 of 3';
  btn.textContent = 'Start Scanning';
  btn.className = 'btn-green';
  btn.disabled = false;
}
</script>
</body>
</html>
"""
