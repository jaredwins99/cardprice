"""Slide-scan UI: 3-tap scanning for 9-card binder pages.

Flow: User taps "Scan Row" for each of 3 rows. Slides phone across row.
System detects 3 card transitions via brightness peaks and captures automatically.
After 3 rows (9 cards), auto-submits to /slide-scan/identify.

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
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #000; color: #fff; font-family: -apple-system, system-ui, sans-serif;
       overflow: hidden; height: 100dvh; width: 100vw; display: flex; flex-direction: column; }
#video { width: 100%; flex: 1; object-fit: cover; }
#topBar { position: absolute; top: 0; left: 0; right: 0; padding: 12px 16px;
          display: flex; justify-content: space-between; align-items: center;
          background: linear-gradient(rgba(0,0,0,.6), transparent); z-index: 10; }
#rowLabel { font-size: 20px; font-weight: 700; }
#status { font-size: 14px; opacity: .8; }
#thumbStrip { position: absolute; top: 60px; left: 0; right: 0; display: flex;
              justify-content: center; gap: 4px; padding: 8px; z-index: 10; }
.thumb { width: 36px; height: 50px; border: 2px solid rgba(255,255,255,.3);
         border-radius: 4px; background: rgba(0,0,0,.4); object-fit: cover; }
.thumb.filled { border-color: #4f4; }
.thumb.active { border-color: #ff0; box-shadow: 0 0 8px #ff0; }
#bottomBar { position: absolute; bottom: 0; left: 0; right: 0; padding: 20px;
             display: flex; justify-content: center; z-index: 10;
             background: linear-gradient(transparent, rgba(0,0,0,.7)); }
#scanBtn { padding: 16px 48px; font-size: 20px; font-weight: 700; border: none;
           border-radius: 50px; background: #4f4; color: #000; cursor: pointer;
           transition: all .15s; }
#scanBtn:active { transform: scale(.95); }
#scanBtn:disabled { background: #555; color: #999; }
#scanBtn.scanning { background: #f44; animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity: .7; } }
#brightBar { position: absolute; bottom: 90px; left: 16px; right: 16px; height: 4px;
             background: rgba(255,255,255,.15); border-radius: 2px; z-index: 10; }
#brightFill { height: 100%; width: 50%; background: #4f4; border-radius: 2px;
              transition: width 50ms; }
canvas { display: none; }
#overlay { position: absolute; inset: 0; display: none; z-index: 20;
           background: rgba(0,0,0,.85); justify-content: center; align-items: center;
           flex-direction: column; gap: 16px; }
#overlay.show { display: flex; }
#overlay .msg { font-size: 22px; font-weight: 700; }
#overlay .sub { font-size: 14px; opacity: .7; }
.flash { position: absolute; inset: 0; background: #fff; z-index: 15;
         animation: flashAnim .15s forwards; pointer-events: none; }
@keyframes flashAnim { from { opacity: .6; } to { opacity: 0; } }
</style>
</head>
<body>
<video id="video" autoplay playsinline muted></video>
<canvas id="canvas"></canvas>

<div id="topBar">
  <span id="rowLabel">Row 1 / 3</span>
  <span id="status">Ready</span>
</div>

<div id="thumbStrip"></div>

<div id="brightBar"><div id="brightFill"></div></div>

<div id="bottomBar">
  <button id="scanBtn" onclick="startRow()">Scan Row 1</button>
</div>

<div id="overlay">
  <div class="msg" id="overlayMsg">Submitting...</div>
  <div class="sub" id="overlaySub">Identifying 9 cards</div>
</div>

<script>
const V = document.getElementById('video');
const C = document.getElementById('canvas');
const ctx = C.getContext('2d', { willReadFrequently: true });
const thumbStrip = document.getElementById('thumbStrip');
const scanBtn = document.getElementById('scanBtn');
const rowLabel = document.getElementById('rowLabel');
const statusEl = document.getElementById('status');
const brightFill = document.getElementById('brightFill');
const overlay = document.getElementById('overlay');

// State
let currentRow = 0;        // 0, 1, 2
let scanning = false;
let captures = new Array(9).fill(null);  // blob URLs
let captureBlobs = new Array(9).fill(null);
let rowCaptures = 0;        // how many cards captured in current row scan
let rafId = null;

// Brightness peak detection state
let brightHistory = [];     // rolling window of brightness values
let lastPeakTime = 0;       // prevent double-captures
let inGutter = true;        // start assuming we're in a gutter
let peakBrightness = 0;     // track peak within current card region
let peakFrame = null;       // store the frame at peak brightness

const HISTORY_LEN = 30;     // ~1 second at 30fps
const MIN_PEAK_GAP_MS = 400; // minimum ms between captures
const GUTTER_THRESHOLD = 0.92; // ratio below peak to consider "gutter" (relative)

// Build thumbnail strip
for (let i = 0; i < 9; i++) {
  const img = document.createElement('img');
  img.className = 'thumb';
  img.id = 'thumb_' + i;
  img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  thumbStrip.appendChild(img);
}
updateActiveThumb();

// Camera init
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false
    });
    V.srcObject = stream;
    await V.play();
    C.width = V.videoWidth;
    C.height = V.videoHeight;
  } catch (e) {
    statusEl.textContent = 'Camera error: ' + e.message;
  }
}
initCamera();

function updateActiveThumb() {
  for (let i = 0; i < 9; i++) {
    const t = document.getElementById('thumb_' + i);
    t.classList.toggle('active', i === currentRow * 3 + rowCaptures && scanning);
  }
}

function startRow() {
  if (scanning) {
    // Cancel current scan
    stopScanning();
    return;
  }
  scanning = true;
  rowCaptures = 0;
  brightHistory = [];
  lastPeakTime = 0;
  inGutter = true;
  peakBrightness = 0;
  peakFrame = null;

  scanBtn.textContent = 'Cancel';
  scanBtn.classList.add('scanning');
  statusEl.textContent = 'Slide across row ' + (currentRow + 1) + '...';
  updateActiveThumb();

  rafId = requestAnimationFrame(scanLoop);
}

function stopScanning() {
  scanning = false;
  cancelAnimationFrame(rafId);
  scanBtn.classList.remove('scanning');
  if (currentRow < 3) {
    scanBtn.textContent = 'Scan Row ' + (currentRow + 1);
  }
  statusEl.textContent = 'Ready';
  updateActiveThumb();
}

function scanLoop() {
  if (!scanning || rowCaptures >= 3) return;

  // Draw current frame
  ctx.drawImage(V, 0, 0, C.width, C.height);

  // Sample brightness from center vertical strip (middle 30% width, full height)
  const stripX = Math.floor(C.width * 0.35);
  const stripW = Math.floor(C.width * 0.3);
  const stripH = C.height;
  const imgData = ctx.getImageData(stripX, 0, stripW, stripH);
  const d = imgData.data;

  // Fast brightness: sample every 16th pixel
  let sum = 0, count = 0;
  for (let i = 0; i < d.length; i += 64) { // 64 = 16 pixels * 4 channels
    sum += d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114;
    count++;
  }
  const brightness = sum / count / 255; // normalized 0-1

  // Update brightness bar
  brightFill.style.width = (brightness * 100) + '%';

  // Add to history
  brightHistory.push({ b: brightness, t: performance.now() });
  if (brightHistory.length > HISTORY_LEN) brightHistory.shift();

  // Need at least a few frames to detect patterns
  if (brightHistory.length < 5) {
    rafId = requestAnimationFrame(scanLoop);
    return;
  }

  // Compute running average and detect peaks
  const now = performance.now();
  const recentAvg = brightHistory.slice(-5).reduce((s, x) => s + x.b, 0) / 5;
  const windowAvg = brightHistory.reduce((s, x) => s + x.b, 0) / brightHistory.length;

  // Adaptive threshold: cards are brighter than average
  // We detect: rising into card territory -> peak -> falling into gutter
  const isAboveAvg = brightness > windowAvg * 1.02;

  if (inGutter && isAboveAvg) {
    // Entering a card region
    inGutter = false;
    peakBrightness = brightness;
    peakFrame = ctx.getImageData(0, 0, C.width, C.height);
  } else if (!inGutter && isAboveAvg) {
    // Still on a card - track peak
    if (brightness > peakBrightness) {
      peakBrightness = brightness;
      peakFrame = ctx.getImageData(0, 0, C.width, C.height);
    }
  } else if (!inGutter && !isAboveAvg) {
    // Falling into gutter - capture at peak if enough time has passed
    if (now - lastPeakTime > MIN_PEAK_GAP_MS && peakFrame) {
      captureFrame(peakFrame);
      lastPeakTime = now;
    }
    inGutter = true;
    peakBrightness = 0;
    peakFrame = null;
  }

  rafId = requestAnimationFrame(scanLoop);
}

function captureFrame(imageData) {
  if (rowCaptures >= 3) return;

  const pos = currentRow * 3 + rowCaptures;

  // Create a temp canvas to convert imageData to blob
  const tc = document.createElement('canvas');
  tc.width = C.width;
  tc.height = C.height;
  const tctx = tc.getContext('2d');

  if (imageData instanceof ImageData) {
    tctx.putImageData(imageData, 0, 0);
  } else {
    tctx.drawImage(V, 0, 0, C.width, C.height);
  }

  // Flash effect
  const flash = document.createElement('div');
  flash.className = 'flash';
  document.body.appendChild(flash);
  setTimeout(() => flash.remove(), 200);

  tc.toBlob(blob => {
    if (!blob) return;
    captureBlobs[pos] = blob;
    const url = URL.createObjectURL(blob);
    captures[pos] = url;

    const thumb = document.getElementById('thumb_' + pos);
    thumb.src = url;
    thumb.classList.add('filled');

    rowCaptures++;
    statusEl.textContent = rowCaptures + '/3 cards captured';
    updateActiveThumb();

    if (rowCaptures >= 3) {
      // Row done
      scanning = false;
      cancelAnimationFrame(rafId);
      scanBtn.classList.remove('scanning');
      currentRow++;

      if (currentRow >= 3) {
        // All 9 cards captured - submit
        scanBtn.disabled = true;
        scanBtn.textContent = 'Submitting...';
        rowLabel.textContent = 'Done!';
        statusEl.textContent = 'Identifying cards...';
        submitCards();
      } else {
        rowLabel.textContent = 'Row ' + (currentRow + 1) + ' / 3';
        scanBtn.textContent = 'Scan Row ' + (currentRow + 1);
        statusEl.textContent = 'Row ' + currentRow + ' done!';
      }
    }
  }, 'image/jpeg', 0.92);
}

// Manual capture fallback: tap during scan to force-capture current frame
V.addEventListener('click', () => {
  if (!scanning || rowCaptures >= 3) return;
  captureFrame(null); // null = grab live frame
});

async function submitCards() {
  overlay.classList.add('show');
  document.getElementById('overlayMsg').textContent = 'Identifying...';
  document.getElementById('overlaySub').textContent = captureBlobs.filter(Boolean).length + ' cards';

  const form = new FormData();
  for (let i = 0; i < 9; i++) {
    if (captureBlobs[i]) {
      form.append('card_' + i, captureBlobs[i], 'card_' + i + '.jpg');
    }
  }

  try {
    const resp = await fetch('/slide-scan/identify', { method: 'POST', body: form });
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
  if (total) html += '<div style="font-size:18px;color:#4f4;margin-top:4px">' + total + ' total</div>';
  html += '</div>';

  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">';
  for (const card of cards) {
    const price = card.variant_price || card.market_price;
    const name = card.card_name || 'Unknown';
    const imgSrc = card.local_image_url || card.segment_image_url || '';
    html += '<div style="text-align:center;background:rgba(255,255,255,.1);border-radius:8px;padding:6px">';
    if (imgSrc) html += '<img src="' + imgSrc + '" style="width:100%;border-radius:4px;aspect-ratio:5/7;object-fit:cover">';
    html += '<div style="font-size:11px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + name + '</div>';
    if (price) html += '<div style="font-size:13px;font-weight:700;color:#4f4">$' + price.toFixed(2) + '</div>';
    if (card.detected_variant && card.detected_variant !== 'normal')
      html += '<div style="font-size:10px;color:#ff0">' + card.detected_variant + '</div>';
    html += '</div>';
  }
  html += '</div>';

  html += '<div style="display:flex;gap:12px;margin-top:16px;justify-content:center">';
  html += '<button onclick="resetAll();overlay.classList.remove(\'show\')" style="padding:12px 32px;font-size:16px;border:none;border-radius:25px;background:#4f4;color:#000;font-weight:700;cursor:pointer">Scan Next Page</button>';
  html += '</div></div>';

  overlay.innerHTML = html;
}

function resetAll() {
  currentRow = 0;
  rowCaptures = 0;
  scanning = false;
  captures.fill(null);
  captureBlobs.fill(null);
  brightHistory = [];
  inGutter = true;
  peakBrightness = 0;
  peakFrame = null;

  rowLabel.textContent = 'Row 1 / 3';
  scanBtn.textContent = 'Scan Row 1';
  scanBtn.disabled = false;
  scanBtn.classList.remove('scanning');
  statusEl.textContent = 'Ready';

  for (let i = 0; i < 9; i++) {
    const t = document.getElementById('thumb_' + i);
    if (t) {
      t.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
      t.classList.remove('filled', 'active');
    }
  }

  // Rebuild overlay structure
  overlay.innerHTML = '<div class="msg" id="overlayMsg">Submitting...</div><div class="sub" id="overlaySub">Identifying 9 cards</div>';
  overlay.classList.remove('show');
}
</script>
</body>
</html>
"""
