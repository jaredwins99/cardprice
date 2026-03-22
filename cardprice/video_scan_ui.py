"""Video-based slide-scan UI: record video, server extracts cards.

Flow:
    1. User taps "Scan Row" -> starts MediaRecorder video recording
    2. User slides phone across row (~3-5 seconds)
    3. Taps button again to stop (or auto-stops after 5s)
    4. Video uploaded to server -> server extracts 3 card frames
    5. After 3 rows (9 cards), auto-submits for identification

All detection logic is server-side. Client only records and uploads video.

Integration:
    GET  /video-scan              -> serve this HTML
    POST /video-scan/extract      -> receive video, extract cards, return images
    POST /slide-scan/identify     -> reuse existing identify endpoint
"""

VIDEO_SCAN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Video Scan</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a0a1a; color: #fff; font-family: -apple-system, system-ui, sans-serif;
       overflow: hidden; height: 100dvh; width: 100vw; display: flex; flex-direction: column;
       -webkit-user-select: none; user-select: none; }

#video { width: 100%; flex: 1; object-fit: cover; background: #000; }

#topBar { position: absolute; top: 0; left: 0; right: 0; padding: 12px 16px;
          display: flex; justify-content: space-between; align-items: center;
          background: linear-gradient(rgba(0,0,0,.7), transparent); z-index: 10; }
#rowLabel { font-size: 20px; font-weight: 700; }
#status { font-size: 14px; opacity: .8; }

#thumbStrip { position: absolute; top: 60px; left: 0; right: 0; display: flex;
              justify-content: center; gap: 4px; padding: 8px; z-index: 10; }
.thumb { width: 36px; height: 50px; border: 2px solid rgba(255,255,255,.25);
         border-radius: 4px; background: rgba(0,0,0,.4); object-fit: cover; }
.thumb.filled { border-color: #4ecca3; }
.thumb.active { border-color: #ff0; box-shadow: 0 0 8px #ff0; }
.thumb.uploading { border-color: #f80; animation: thumbPulse 0.6s infinite; }
@keyframes thumbPulse { 50% { opacity: .5; } }

#timerBar { position: absolute; bottom: 90px; left: 16px; right: 16px; height: 6px;
            background: rgba(255,255,255,.15); border-radius: 3px; z-index: 10;
            display: none; }
#timerFill { height: 100%; width: 0%; background: #f44; border-radius: 3px;
             transition: width 100ms linear; }

#bottomBar { position: absolute; bottom: 0; left: 0; right: 0; padding: 20px;
             display: flex; justify-content: center; z-index: 10;
             background: linear-gradient(transparent, rgba(0,0,0,.7)); }
#scanBtn { padding: 16px 48px; font-size: 20px; font-weight: 700; border: none;
           border-radius: 50px; background: #4ecca3; color: #000; cursor: pointer;
           transition: all .15s; }
#scanBtn:active { transform: scale(.95); }
#scanBtn:disabled { background: #444; color: #888; }
#scanBtn.recording { background: #f44; animation: recPulse 1s infinite; }
@keyframes recPulse { 50% { opacity: .7; } }

/* Recording indicator */
#recDot { display: none; position: absolute; top: 16px; left: 50%;
          transform: translateX(-50%); z-index: 20; }
#recDot.show { display: flex; align-items: center; gap: 8px; }
#recDot .dot { width: 12px; height: 12px; border-radius: 50%; background: #f44;
               animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: .3; } }
#recDot .label { font-size: 14px; font-weight: 600; color: #f44; }

/* Overlay screens */
#overlay { position: absolute; inset: 0; display: none; z-index: 30;
           background: rgba(0,0,0,.9); justify-content: center; align-items: center;
           flex-direction: column; gap: 16px; padding: 20px; }
#overlay.show { display: flex; }
#overlay .msg { font-size: 22px; font-weight: 700; text-align: center; }
#overlay .sub { font-size: 14px; opacity: .7; text-align: center; }
.spinner { width: 40px; height: 40px; border: 4px solid rgba(255,255,255,.2);
           border-top-color: #4ecca3; border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Flash on capture */
.flash { position: absolute; inset: 0; background: #fff; z-index: 15;
         animation: flashAnim .2s forwards; pointer-events: none; }
@keyframes flashAnim { from { opacity: .5; } to { opacity: 0; } }

/* Guide arrows */
#slideGuide { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
              z-index: 11; display: none; pointer-events: none; }
#slideGuide.show { display: block; }
#slideGuide svg { width: 200px; height: 60px; }
</style>
</head>
<body>
<video id="video" autoplay playsinline muted></video>

<div id="topBar">
  <span id="rowLabel">Row 1 / 3</span>
  <span id="status">Ready</span>
</div>

<div id="thumbStrip"></div>

<div id="recDot">
  <div class="dot"></div>
  <span class="label">REC</span>
</div>

<div id="slideGuide">
  <svg viewBox="0 0 200 60" fill="none">
    <path d="M30 30 L170 30" stroke="rgba(255,255,255,.4)" stroke-width="2" stroke-dasharray="8 4"/>
    <polygon points="170,20 190,30 170,40" fill="rgba(255,255,255,.4)"/>
    <text x="100" y="55" text-anchor="middle" fill="rgba(255,255,255,.5)" font-size="12">slide across row</text>
  </svg>
</div>

<div id="timerBar"><div id="timerFill"></div></div>

<div id="bottomBar">
  <button id="scanBtn" onclick="toggleRecording()">Scan Row 1</button>
</div>

<div id="overlay">
  <div class="spinner"></div>
  <div class="msg" id="overlayMsg">Processing...</div>
  <div class="sub" id="overlaySub"></div>
</div>

<script>
const V = document.getElementById('video');
const thumbStrip = document.getElementById('thumbStrip');
const scanBtn = document.getElementById('scanBtn');
const rowLabel = document.getElementById('rowLabel');
const statusEl = document.getElementById('status');
const recDot = document.getElementById('recDot');
const slideGuide = document.getElementById('slideGuide');
const timerBar = document.getElementById('timerBar');
const timerFill = document.getElementById('timerFill');
const overlay = document.getElementById('overlay');

// Config
const MAX_RECORD_SECS = 6;
const CARDS_PER_ROW = 3;
const TOTAL_ROWS = 3;
const TOTAL_CARDS = CARDS_PER_ROW * TOTAL_ROWS;

// State
let currentRow = 0;
let recording = false;
let recorder = null;
let recordChunks = [];
let timerInterval = null;
let recordStartTime = 0;
let stream = null;
let cardImages = new Array(TOTAL_CARDS).fill(null);  // blob URLs
let cardBlobs = new Array(TOTAL_CARDS).fill(null);    // actual blobs

// Build thumbnail strip
for (let i = 0; i < TOTAL_CARDS; i++) {
  const img = document.createElement('img');
  img.className = 'thumb';
  img.id = 'thumb_' + i;
  img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  thumbStrip.appendChild(img);
}
updateThumbs();

// Camera init
async function initCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false
    });
    V.srcObject = stream;
    await V.play();
  } catch (e) {
    statusEl.textContent = 'Camera error: ' + e.message;
  }
}
initCamera();

function updateThumbs() {
  for (let i = 0; i < TOTAL_CARDS; i++) {
    const t = document.getElementById('thumb_' + i);
    const isCurrentRow = Math.floor(i / CARDS_PER_ROW) === currentRow;
    t.classList.toggle('active', isCurrentRow && recording);
  }
}

function toggleRecording() {
  if (recording) {
    stopRecording();
  } else {
    startRecording();
  }
}

function startRecording() {
  if (!stream) { statusEl.textContent = 'No camera'; return; }
  if (recording) return;

  recording = true;
  recordChunks = [];

  // Pick a supported mime type
  let mimeType = 'video/webm;codecs=vp8';
  if (!MediaRecorder.isTypeSupported(mimeType)) {
    mimeType = 'video/webm';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'video/mp4';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = '';  // let browser pick
      }
    }
  }

  const opts = mimeType ? { mimeType: mimeType } : {};
  recorder = new MediaRecorder(stream, opts);

  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordChunks.push(e.data);
  };

  recorder.onstop = () => {
    const blob = new Blob(recordChunks, { type: recorder.mimeType || 'video/webm' });
    uploadVideo(blob);
  };

  recorder.start(100);  // collect data every 100ms for smoother chunks

  // UI updates
  scanBtn.textContent = 'Stop';
  scanBtn.classList.add('recording');
  recDot.classList.add('show');
  slideGuide.classList.add('show');
  statusEl.textContent = 'Slide across row ' + (currentRow + 1) + '...';
  updateThumbs();

  // Timer bar
  recordStartTime = Date.now();
  timerBar.style.display = 'block';
  timerFill.style.width = '0%';
  timerInterval = setInterval(() => {
    const elapsed = (Date.now() - recordStartTime) / 1000;
    const pct = Math.min(100, (elapsed / MAX_RECORD_SECS) * 100);
    timerFill.style.width = pct + '%';
    timerFill.style.background = pct > 80 ? '#f44' : '#f80';

    if (elapsed >= MAX_RECORD_SECS) {
      stopRecording();
    }
  }, 100);
}

function stopRecording() {
  if (!recording || !recorder) return;
  recording = false;
  clearInterval(timerInterval);

  recorder.stop();

  // UI reset
  scanBtn.classList.remove('recording');
  scanBtn.disabled = true;
  scanBtn.textContent = 'Processing...';
  recDot.classList.remove('show');
  slideGuide.classList.remove('show');
  timerBar.style.display = 'none';
  statusEl.textContent = 'Uploading video...';

  // Mark current row thumbs as uploading
  for (let i = currentRow * CARDS_PER_ROW; i < (currentRow + 1) * CARDS_PER_ROW; i++) {
    document.getElementById('thumb_' + i).classList.add('uploading');
  }
}

async function uploadVideo(blob) {
  statusEl.textContent = 'Extracting cards from video...';

  const form = new FormData();
  const ext = (blob.type || '').includes('mp4') ? 'mp4' : 'webm';
  form.append('video', blob, 'row_' + currentRow + '.' + ext);
  form.append('num_cards', CARDS_PER_ROW.toString());
  form.append('row', currentRow.toString());

  try {
    const resp = await fetch('/video-scan/extract', { method: 'POST', body: form });
    const data = await resp.json();

    if (data.error) {
      statusEl.textContent = 'Error: ' + data.error;
      scanBtn.disabled = false;
      scanBtn.textContent = 'Retry Row ' + (currentRow + 1);
      clearUploadingThumbs();
      return;
    }

    // Server returns { cards: [ { index, image_url, image_data_b64 }, ... ] }
    const cards = data.cards || [];
    for (const card of cards) {
      const pos = currentRow * CARDS_PER_ROW + card.index;
      if (pos >= TOTAL_CARDS) continue;

      // Convert base64 to blob
      const b64 = card.image_data;
      const byteStr = atob(b64);
      const ab = new Uint8Array(byteStr.length);
      for (let j = 0; j < byteStr.length; j++) ab[j] = byteStr.charCodeAt(j);
      const imgBlob = new Blob([ab], { type: 'image/jpeg' });

      cardBlobs[pos] = imgBlob;
      const url = URL.createObjectURL(imgBlob);
      cardImages[pos] = url;

      const t = document.getElementById('thumb_' + pos);
      t.src = url;
      t.classList.add('filled');
      t.classList.remove('uploading', 'active');
    }

    // Flash effect
    const flash = document.createElement('div');
    flash.className = 'flash';
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 250);

    // Advance to next row
    currentRow++;
    clearUploadingThumbs();

    if (currentRow >= TOTAL_ROWS) {
      // All rows done - submit for identification
      scanBtn.disabled = true;
      scanBtn.textContent = 'Identifying...';
      rowLabel.textContent = 'Done!';
      statusEl.textContent = 'Identifying ' + cardBlobs.filter(Boolean).length + ' cards...';
      submitCards();
    } else {
      rowLabel.textContent = 'Row ' + (currentRow + 1) + ' / ' + TOTAL_ROWS;
      scanBtn.disabled = false;
      scanBtn.textContent = 'Scan Row ' + (currentRow + 1);
      statusEl.textContent = 'Row ' + currentRow + ' done! ' + cards.length + ' cards extracted.';
    }
  } catch (e) {
    statusEl.textContent = 'Network error: ' + e.message;
    scanBtn.disabled = false;
    scanBtn.textContent = 'Retry Row ' + (currentRow + 1);
    clearUploadingThumbs();
  }
}

function clearUploadingThumbs() {
  for (let i = 0; i < TOTAL_CARDS; i++) {
    document.getElementById('thumb_' + i).classList.remove('uploading');
  }
}

async function submitCards() {
  overlay.classList.add('show');
  document.getElementById('overlayMsg').textContent = 'Identifying...';
  document.getElementById('overlaySub').textContent = cardBlobs.filter(Boolean).length + ' cards';

  const form = new FormData();
  for (let i = 0; i < TOTAL_CARDS; i++) {
    if (cardBlobs[i]) {
      form.append('card_' + i, cardBlobs[i], 'card_' + i + '.jpg');
    }
  }

  try {
    const resp = await fetch('/slide-scan/identify?variants=true', { method: 'POST', body: form });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('overlayMsg').textContent = 'Error';
      document.getElementById('overlaySub').textContent = data.error;
      setTimeout(() => { overlay.classList.remove('show'); resetAll(); }, 3000);
      return;
    }

    showResults(data);
  } catch (e) {
    document.getElementById('overlayMsg').textContent = 'Network Error';
    document.getElementById('overlaySub').textContent = e.message;
    setTimeout(() => { overlay.classList.remove('show'); resetAll(); }, 3000);
  }
}

function showResults(data) {
  const cards = data.cards || [];
  const total = data.total_value ? '$' + data.total_value.toFixed(2) : '';

  let html = '<div style="width:100%;max-height:80vh;overflow-y:auto;padding:16px">';
  html += '<div style="text-align:center;margin-bottom:12px">';
  html += '<div style="font-size:24px;font-weight:700">Page Scanned</div>';
  if (total) html += '<div style="font-size:18px;color:#4ecca3;margin-top:4px">' + total + ' total</div>';
  html += '</div>';

  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">';
  for (const card of cards) {
    const price = card.variant_price || card.market_price;
    const name = card.card_name || 'Unknown';
    const imgSrc = card.local_image_url || card.segment_image_url || '';
    html += '<div style="text-align:center;background:rgba(255,255,255,.1);border-radius:8px;padding:6px">';
    if (imgSrc) html += '<img src="' + imgSrc + '" style="width:100%;border-radius:4px;aspect-ratio:5/7;object-fit:cover">';
    html += '<div style="font-size:11px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + name + '</div>';
    if (price) html += '<div style="font-size:13px;font-weight:700;color:#4ecca3">$' + price.toFixed(2) + '</div>';
    if (card.detected_variant && card.detected_variant !== 'normal')
      html += '<div style="font-size:10px;color:#ff0">' + card.detected_variant + '</div>';
    html += '</div>';
  }
  html += '</div>';

  html += '<div style="display:flex;gap:12px;margin-top:16px;justify-content:center">';
  html += '<button onclick="resetAll();overlay.classList.remove(\'show\')" style="padding:12px 32px;font-size:16px;border:none;border-radius:25px;background:#4ecca3;color:#000;font-weight:700;cursor:pointer">Scan Next Page</button>';
  html += '</div></div>';

  overlay.innerHTML = html;
}

function resetAll() {
  currentRow = 0;
  recording = false;
  recorder = null;
  recordChunks = [];
  cardImages.fill(null);
  cardBlobs.fill(null);

  rowLabel.textContent = 'Row 1 / ' + TOTAL_ROWS;
  scanBtn.textContent = 'Scan Row 1';
  scanBtn.disabled = false;
  scanBtn.classList.remove('recording');
  statusEl.textContent = 'Ready';
  recDot.classList.remove('show');
  slideGuide.classList.remove('show');
  timerBar.style.display = 'none';

  for (let i = 0; i < TOTAL_CARDS; i++) {
    const t = document.getElementById('thumb_' + i);
    if (t) {
      t.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
      t.classList.remove('filled', 'active', 'uploading');
    }
  }

  overlay.innerHTML = '<div class="spinner"></div><div class="msg" id="overlayMsg">Processing...</div><div class="sub" id="overlaySub"></div>';
  overlay.classList.remove('show');
}
</script>
</body>
</html>
"""
