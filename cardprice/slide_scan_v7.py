"""Slide-scan UI v7: ZERO-TAP production grade.

Phase 1 POC: Prove camera → capture → server works.
- Camera starts immediately
- Captures a frame every 2 seconds (dead simple interval)
- Shows thumbnail + sends to server
- Debug readout shows camera state

Once POC works, layer on smart detection.

Integration into server.py:
    elif self.path == "/slide-scan-v7":
        from cardprice.slide_scan_v7 import SLIDE_SCAN_V7_HTML
        self._send_html(SLIDE_SCAN_V7_HTML)
"""

SLIDE_SCAN_V7_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Card Scanner</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background: #000; color: #fff;
    height: 100vh; height: 100dvh;
    overflow: hidden;
}
video { width: 100%; height: 70vh; object-fit: cover; }
#info {
    padding: 12px; background: #111; font-size: 13px; font-family: monospace;
    color: #4ecca3; min-height: 60px; overflow-y: auto;
}
#thumbs { display: flex; gap: 4px; padding: 8px; flex-wrap: wrap; background: #111; }
#thumbs img { width: 50px; height: 70px; object-fit: cover; border: 2px solid #333; border-radius: 4px; }
#thumbs img.latest { border-color: #4ecca3; }
.results-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 25;
    background: #1a1a2e; display: none; flex-direction: column; align-items: center;
    overflow-y: auto; padding: 30px 20px;
}
.results-overlay.visible { display: flex; }
.results-overlay h2 { font-size: 24px; color: #4ecca3; margin-bottom: 16px; }
.result-card {
    display: flex; gap: 12px; align-items: center; background: rgba(255,255,255,0.05);
    border-radius: 8px; padding: 10px; margin-bottom: 8px; width: 100%; max-width: 400px;
}
.result-card img { width: 60px; height: 80px; object-fit: cover; border-radius: 4px; }
.result-card .info { flex: 1; }
.result-card .name { font-weight: 600; font-size: 15px; }
.result-card .detail { color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 2px; }
.reset-btn {
    margin-top: 16px; padding: 14px 36px; background: #333; color: #fff;
    border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer;
}
</style>
</head>
<body>
<video id="cam" autoplay playsinline muted></video>
<div id="info">Starting...</div>
<div id="thumbs"></div>
<div class="results-overlay" id="resultsOverlay">
    <h2>Results</h2>
    <div id="resultsList"></div>
    <button class="reset-btn" onclick="resetAll()">Scan Again</button>
</div>

<script>
const TOTAL = 9;
const V = document.getElementById('cam');
const info = document.getElementById('info');
const thumbs = document.getElementById('thumbs');
let captures = [];
let phase = 'starting'; // starting, ready, capturing, submitting, done

function log(msg) {
    info.textContent = msg;
    console.log('[scanner]', msg);
}

// ===== PHASE 1: Start camera =====
async function init() {
    if (!navigator.mediaDevices) { log('ERROR: No camera API. Need HTTPS.'); return; }

    log('Requesting camera...');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false
        });
        V.srcObject = stream;
        await V.play();
        log('Camera active. Waiting for frames...');

        // Wait for real video dimensions
        for (let i = 0; i < 50; i++) {
            if (V.videoWidth > 0) break;
            await new Promise(r => setTimeout(r, 100));
        }

        if (!V.videoWidth) { log('ERROR: Camera has no frames after 5s.'); return; }

        log('Camera: ' + V.videoWidth + 'x' + V.videoHeight + '. Scanning starts NOW. Point at cards.');
        phase = 'ready';
        startCapturing();
    } catch(e) {
        log('Camera error: ' + e.name + ' - ' + e.message);
    }
}

// ===== PHASE 2: Capture frames =====
// Dead simple: capture a frame when we detect CHANGE from previous frame.
// This means: when you slide to a new card, the image changes → we capture.
let prevGray = null;
let frameDiffHistory = [];
let lastCaptureTime = 0;
const CAPTURE_COOLDOWN = 800; // ms between captures
const DIFF_WINDOW = 10;       // frames to track

function getGrayCenter() {
    // Grab center 50% of frame as small grayscale array
    const c = document.createElement('canvas');
    const sw = 80, sh = 60;
    c.width = sw; c.height = sh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    // Draw center crop
    const sx = V.videoWidth * 0.25, sy = V.videoHeight * 0.25;
    const sWidth = V.videoWidth * 0.5, sHeight = V.videoHeight * 0.5;
    ctx.drawImage(V, sx, sy, sWidth, sHeight, 0, 0, sw, sh);
    const d = ctx.getImageData(0, 0, sw, sh).data;
    const gray = new Float32Array(sw * sh);
    for (let i = 0; i < gray.length; i++) {
        gray[i] = 0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2];
    }
    return gray;
}

function frameDiff(a, b) {
    if (!a || !b || a.length !== b.length) return 999;
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
}

function startCapturing() {
    phase = 'capturing';

    // State: track motion phases
    // STILL → MOVING → STILL = card transition → capture on 2nd STILL
    // Ultra-simple: track consecutive low-diff frames. Capture when stable for ~1s.
    // After capture, require some movement (high diff) before next capture.
    let stillFrames = 0;
    let needsMovement = false;  // after capture, must see movement before next capture
    let sawMovement = false;
    const STILL_THRESHOLD = 5.0;   // generous — phone cameras are noisy
    const MOVE_THRESHOLD = 8.0;    // must see this much diff to count as "moved to new card"
    const STILL_NEEDED = 8;        // ~0.5s at ~15fps on phone

    function tick() {
        if (phase !== 'capturing') return;

        const gray = getGrayCenter();
        const diff = frameDiff(gray, prevGray);
        prevGray = gray;

        const now = Date.now();
        const cooldownOk = (now - lastCaptureTime) > CAPTURE_COOLDOWN;

        // Track movement
        if (diff > MOVE_THRESHOLD) {
            sawMovement = true;
            stillFrames = 0;
        } else if (diff < STILL_THRESHOLD) {
            stillFrames++;
        } else {
            // In between — don't reset stillFrames, just don't increment
        }

        // Capture logic:
        // - First capture: just need stillness (no movement required)
        // - Subsequent: need movement then stillness (so we don't double-capture same card)
        const canCapture = cooldownOk && stillFrames >= STILL_NEEDED;
        if (canCapture) {
            if (captures.length === 0 || sawMovement) {
                doCapture();
                stillFrames = 0;
                sawMovement = false;
            }
        }

        // Debug info
        const stateStr = diff > MOVE_THRESHOLD ? 'MOVING' : (stillFrames >= STILL_NEEDED ? 'READY' : 'SETTLING');
        log(captures.length + '/' + TOTAL + ' | ' +
            stateStr + ' | diff=' + diff.toFixed(1) +
            ' | still=' + stillFrames + '/' + STILL_NEEDED +
            (sawMovement ? ' | moved' : '') +
            (captures.length >= TOTAL ? ' | SUBMITTING...' : ''));

        if (captures.length >= TOTAL) {
            phase = 'submitting';
            submitAll();
            return;
        }

        requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
}

function doCapture() {
    const c = document.createElement('canvas');
    c.width = V.videoWidth; c.height = V.videoHeight;
    c.getContext('2d').drawImage(V, 0, 0);

    // Flash
    V.style.opacity = '0.3';
    setTimeout(() => V.style.opacity = '1', 100);
    if (navigator.vibrate) navigator.vibrate(50);

    const dataUrl = c.toDataURL('image/jpeg', 0.92);
    c.toBlob(blob => {
        if (!blob) return;
        captures.push({ dataUrl, blob });

        // Add thumbnail
        const img = document.createElement('img');
        img.src = dataUrl;
        if (thumbs.lastChild) thumbs.lastChild.classList.remove('latest');
        img.classList.add('latest');
        thumbs.appendChild(img);
    }, 'image/jpeg', 0.92);

    lastCaptureTime = Date.now();
}

// ===== PHASE 3: Submit =====
async function submitAll() {
    log('Uploading ' + captures.length + ' cards...');
    try {
        const fd = new FormData();
        for (let i = 0; i < captures.length; i++)
            fd.append('card_' + i, captures[i].blob, 'card_' + i + '.jpg');

        const resp = await fetch('/slide-scan/identify', { method: 'POST', body: fd });
        const result = await resp.json();
        showResults(result);
    } catch(e) {
        log('Upload error: ' + e.message);
        phase = 'capturing';
        startCapturing();
    }
}

function showResults(result) {
    phase = 'done';
    const ro = document.getElementById('resultsOverlay');
    const list = document.getElementById('resultsList');
    list.innerHTML = '';
    const cards = result.cards || result.results || [];
    if (!cards.length) {
        list.innerHTML = '<p style="color:#888">No cards identified.</p>';
    }
    for (const card of cards) {
        const div = document.createElement('div');
        div.className = 'result-card';
        const nm = card.name || card.card_name || 'Unknown';
        const dt = card.set || card.detail || '';
        const pr = card.price ? ('$' + parseFloat(card.price).toFixed(2)) : '';
        const ts = card.thumb || card.image || '';
        div.innerHTML = (ts ? '<img src="'+ts+'">' : '') +
            '<div class="info"><div class="name">'+nm+'</div>' +
            '<div class="detail">'+dt + (pr ? ' — '+pr : '') +'</div></div>';
        list.appendChild(div);
    }
    if (result.error) list.innerHTML += '<p style="color:#e74c3c;">'+result.error+'</p>';
    ro.classList.add('visible');
}

function resetAll() {
    captures = [];
    thumbs.innerHTML = '';
    prevGray = null;
    lastCaptureTime = 0;
    document.getElementById('resultsOverlay').classList.remove('visible');
    phase = 'ready';
    startCapturing();
}

init();
</script>
</body>
</html>
"""
