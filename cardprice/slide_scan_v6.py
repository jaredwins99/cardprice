"""Slide-scan UI v6: camera code copied from condition_camera_ui.py.

Scans a 3x3 binder page one row at a time. User slides the phone across
each row; the UI auto-captures 3 frames at 1-second intervals per row.
After all 3 rows (9 images), submits to /slide-scan/identify.

Integration into server.py:
    elif self.path == "/slide-scan-v6":
        from cardprice.slide_scan_v6 import SLIDE_SCAN_V6_HTML
        self._send_html(SLIDE_SCAN_V6_HTML)
"""

SLIDE_SCAN_V6_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Slide Scan</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background: #000;
    color: #fff;
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
}

/* ---- Camera container ---- */
.camera-wrap {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

video {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: 1;
}

/* ---- Overlay canvas (template + guides) ---- */
canvas#overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    pointer-events: none;
}

/* ---- Hidden capture canvas ---- */
canvas#capture {
    display: none;
}

/* ---- Flash effect ---- */
.flash {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: #fff;
    z-index: 20;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.1s;
}
.flash.active {
    opacity: 0.8;
    transition: none;
}

/* ---- Top bar ---- */
.top-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%);
    padding: 16px 20px 30px;
}
.row-label {
    font-size: 22px;
    font-weight: 700;
}
.row-instruction {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    margin-top: 4px;
}
.row-dots {
    display: flex;
    gap: 8px;
    margin-top: 12px;
}
.row-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: rgba(255,255,255,0.3);
    transition: all 0.3s;
}
.row-dot.active { background: #4ecca3; box-shadow: 0 0 8px rgba(78,204,163,0.5); }
.row-dot.done { background: #4ecca3; }

/* ---- Bottom bar ---- */
.bottom-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%);
    padding: 30px 20px 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}
.status-text {
    font-size: 15px;
    color: rgba(255,255,255,0.8);
    text-align: center;
    min-height: 20px;
}
.scan-btn {
    padding: 16px 40px;
    background: #4ecca3;
    color: #1a1a2e;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    min-width: 200px;
}
.scan-btn:disabled {
    opacity: 0.5;
    cursor: default;
}
.scan-btn.capturing {
    background: #e74c3c;
}

/* ---- Thumbnails strip ---- */
.thumbs-strip {
    display: flex;
    gap: 6px;
    justify-content: center;
    flex-wrap: wrap;
    max-width: 100%;
}
.thumb-img {
    width: 55px;
    height: 55px;
    object-fit: cover;
    border-radius: 4px;
    border: 2px solid #333;
}
.thumb-img.current-row {
    border-color: #4ecca3;
}

/* ---- Results overlay ---- */
.results-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 25;
    background: #1a1a2e;
    display: none;
    flex-direction: column;
    align-items: center;
    overflow-y: auto;
    padding: 30px 20px;
}
.results-overlay.visible { display: flex; }
.results-overlay h2 {
    font-size: 24px;
    color: #4ecca3;
    margin-bottom: 16px;
}
.result-card {
    display: flex;
    gap: 12px;
    align-items: center;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    width: 100%;
    max-width: 400px;
}
.result-card img {
    width: 60px;
    height: 60px;
    object-fit: cover;
    border-radius: 4px;
}
.result-card .info {
    flex: 1;
    font-size: 14px;
}
.result-card .name {
    font-weight: 600;
    font-size: 15px;
}
.result-card .detail {
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    margin-top: 2px;
}
</style>
</head>
<body>

<div class="camera-wrap">
    <video id="cam" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <canvas id="capture"></canvas>
    <div class="flash" id="flash"></div>

    <div class="top-bar">
        <div class="row-label" id="rowLabel">Row 1 of 3</div>
        <div class="row-instruction" id="rowInstr">Tap "Scan Row" then slowly slide across the row</div>
        <div class="row-dots" id="rowDots"></div>
    </div>

    <div class="bottom-bar">
        <div class="thumbs-strip" id="thumbs"></div>
        <div class="status-text" id="status">Starting camera...</div>
        <button class="scan-btn" id="scanBtn" onclick="onScanBtn()">Scan Row 1</button>
    </div>

    <div class="results-overlay" id="resultsOverlay">
        <h2>Identification Results</h2>
        <div id="resultsList"></div>
    </div>
</div>

<script>
// ===== State =====
const ROWS = 3;
const FRAMES_PER_ROW = 3;
const FRAME_INTERVAL_MS = 1000;

let currentRow = 0;       // 0-based
let captures = [];         // all captured {dataUrl, blob} objects (up to 9)
let capturing = false;
let captureTimer = null;
let framesThisRow = 0;
let stream = null;

// ===== Camera setup (IDENTICAL to condition_camera_ui.py) =====
const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const captureCanvas = document.getElementById('capture');
const ctx = overlay.getContext('2d');
const capCtx = captureCanvas.getContext('2d');

async function startCamera() {
    // Check if getUserMedia is available at all
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        const isHTTP = location.protocol === 'http:';
        if (isHTTP) {
            setStatus(isIOS
                ? 'Camera requires HTTPS on iOS. Ask your admin to enable HTTPS, or open this page on a computer.'
                : 'Camera unavailable over HTTP. Try HTTPS, or use Chrome on Android (allows HTTP on local network).');
        } else {
            setStatus('Camera not available on this device/browser.');
        }
        document.getElementById('scanBtn').disabled = true;
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1920 },
                height: { ideal: 1080 },
            },
            audio: false,
        });
        video.srcObject = stream;

        // Wait for actual video frames to be available (not just metadata).
        // 'loadeddata' fires when the first frame is ready, which guarantees
        // videoWidth/videoHeight are set and the video can be drawn to canvas.
        // On iOS Safari, 'loadedmetadata' fires too early (before frames exist).
        await new Promise((resolve, reject) => {
            video.addEventListener('loadeddata', resolve, { once: true });
            video.addEventListener('error', reject, { once: true });
            // Timeout after 10s in case the event never fires
            setTimeout(() => resolve(), 10000);
        });

        await video.play();

        // Delay resizeOverlay slightly so the layout has settled on mobile.
        // On some Android devices, clientWidth/Height are still 0 immediately
        // after play() resolves.
        await new Promise(r => setTimeout(r, 100));
        resizeOverlay();

        // Verify canvas got real dimensions; retry if not
        if (!overlay.width || !overlay.height) {
            await new Promise(r => setTimeout(r, 300));
            resizeOverlay();
        }

        setStatus('Ready. Tap "Scan Row 1" to begin.');
    } catch (e) {
        console.error('Camera error:', e);
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        const isHTTP = location.protocol === 'http:';

        if (e.name === 'NotAllowedError') {
            setStatus('Camera access denied. Please allow camera permission and reload.');
        } else if (e.name === 'NotFoundError') {
            setStatus('No camera found on this device.');
        } else if (e.name === 'NotReadableError' || e.name === 'AbortError') {
            setStatus('Camera is in use by another app. Close it and reload.');
        } else if (isHTTP && isIOS) {
            setStatus('Camera requires HTTPS on iOS Safari. Use Chrome on Android, or enable HTTPS on the server.');
        } else if (isHTTP) {
            setStatus('Camera failed over HTTP. Try HTTPS, or use Chrome on Android (allows HTTP on local network).');
        } else {
            setStatus('Camera error: ' + e.message);
        }
        document.getElementById('scanBtn').disabled = true;
    }
}

function resizeOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const w = overlay.clientWidth;
    const h = overlay.clientHeight;
    // Guard against 0 dimensions (layout not settled yet on mobile)
    if (w === 0 || h === 0) return;
    overlay.width = w * dpr;
    overlay.height = h * dpr;
    // Reset transform to avoid accumulated scaling on repeated calls
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', () => {
    // Delay slightly so mobile browsers finish layout before we read dimensions
    setTimeout(resizeOverlay, 50);
});
// Also handle orientation changes (some mobile browsers don't fire resize)
window.addEventListener('orientationchange', () => {
    setTimeout(resizeOverlay, 200);
});

// ===== UI helpers =====
function setStatus(text) {
    document.getElementById('status').textContent = text;
}

function updateUI() {
    const label = document.getElementById('rowLabel');
    const instr = document.getElementById('rowInstr');
    const dots = document.getElementById('rowDots');
    const btn = document.getElementById('scanBtn');

    if (currentRow >= ROWS) {
        label.textContent = 'All rows captured';
        instr.textContent = 'Review thumbnails, then submit.';
        btn.textContent = 'Submit';
        btn.classList.remove('capturing');
    } else {
        label.textContent = `Row ${currentRow + 1} of ${ROWS}`;
        instr.textContent = 'Tap button then slowly slide across the row';
        btn.textContent = capturing ? `Capturing...` : `Scan Row ${currentRow + 1}`;
        btn.classList.toggle('capturing', capturing);
    }
    btn.disabled = capturing;

    // Dots
    dots.innerHTML = '';
    for (let i = 0; i < ROWS; i++) {
        const dot = document.createElement('div');
        dot.className = 'row-dot' + (i < currentRow ? ' done' : i === currentRow ? ' active' : '');
        dots.appendChild(dot);
    }

    // Thumbnails
    renderThumbs();
}

function renderThumbs() {
    const container = document.getElementById('thumbs');
    container.innerHTML = '';
    for (let i = 0; i < captures.length; i++) {
        const img = document.createElement('img');
        img.src = captures[i].dataUrl;
        img.className = 'thumb-img';
        const row = Math.floor(i / FRAMES_PER_ROW);
        if (row === currentRow) img.classList.add('current-row');
        container.appendChild(img);
    }
}

// ===== Capture =====
function captureFrame() {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    capCtx.drawImage(video, 0, 0);

    // Flash
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 150);

    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);
    captureCanvas.toBlob(blob => {
        captures.push({ dataUrl, blob });
        framesThisRow++;
        setStatus(`Captured ${framesThisRow}/${FRAMES_PER_ROW} for row ${currentRow + 1}`);
        renderThumbs();

        if (framesThisRow >= FRAMES_PER_ROW) {
            // Row done
            capturing = false;
            if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
            currentRow++;
            updateUI();
        }
    }, 'image/jpeg', 0.92);
}

function onScanBtn() {
    if (currentRow >= ROWS) {
        // Submit
        submitImages();
        return;
    }

    if (capturing) return;

    capturing = true;
    framesThisRow = 0;
    updateUI();
    setStatus('Slide slowly across the row...');

    // Capture first frame immediately
    captureFrame();

    // Then capture remaining frames at intervals
    captureTimer = setInterval(() => {
        if (framesThisRow < FRAMES_PER_ROW) {
            captureFrame();
        } else {
            clearInterval(captureTimer);
            captureTimer = null;
        }
    }, FRAME_INTERVAL_MS);
}

// ===== Submit =====
async function submitImages() {
    const btn = document.getElementById('scanBtn');
    btn.disabled = true;
    btn.textContent = 'Identifying...';

    try {
        const formData = new FormData();
        for (let i = 0; i < captures.length; i++) {
            formData.append('image_' + i, captures[i].blob, 'frame_' + i + '.jpg');
        }

        const resp = await fetch('/slide-scan/identify', {
            method: 'POST',
            body: formData,
        });
        const result = await resp.json();
        showResults(result);
    } catch (e) {
        console.error('Submit error:', e);
        setStatus('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = 'Retry Submit';
    }
}

function showResults(result) {
    const overlay = document.getElementById('resultsOverlay');
    const list = document.getElementById('resultsList');
    list.innerHTML = '';

    const cards = result.cards || result.results || [];
    if (cards.length === 0) {
        list.innerHTML = '<p style="color:rgba(255,255,255,0.6);">No cards identified.</p>';
    } else {
        for (const card of cards) {
            const div = document.createElement('div');
            div.className = 'result-card';
            const thumbSrc = card.thumb || card.image || '';
            const name = card.name || card.card_name || 'Unknown';
            const detail = card.set || card.detail || '';
            div.innerHTML = `
                ${thumbSrc ? '<img src="' + thumbSrc + '">' : ''}
                <div class="info">
                    <div class="name">${name}</div>
                    <div class="detail">${detail}</div>
                </div>
            `;
            list.appendChild(div);
        }
    }

    if (result.error) {
        list.innerHTML += '<p style="color:#e74c3c;margin-top:12px;">' + result.error + '</p>';
    }

    overlay.classList.add('visible');
}

// ===== Init =====
updateUI();
startCamera();
</script>
</body>
</html>
"""
