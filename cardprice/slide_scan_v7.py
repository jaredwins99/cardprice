"""Slide-scan UI v7: auto-detection + overlay, relaxed thresholds.

Scans a 3x3 binder page one row at a time. User slides the phone across
each row; the UI auto-detects card presence via edge density and sharpness,
then auto-captures when a card is stable. After all 3 rows (9 images),
submits to /slide-scan/identify with field names card_0..card_8.

Camera init is identical to slide_scan_v6.py.
Overlay canvas uses condition_camera_ui.py's setup.

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
    overflow: hidden; touch-action: none;
    -webkit-user-select: none; user-select: none;
}
.camera-wrap { position: relative; width: 100%; height: 100%; display: flex; flex-direction: column; }
video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1; }
canvas#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none; }
canvas#capture { display: none; }
.flash { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: #fff; z-index: 20; opacity: 0; pointer-events: none; transition: opacity 0.1s; }
.flash.active { opacity: 0.8; transition: none; }
.top-bar {
    position: absolute; top: 0; left: 0; right: 0; z-index: 10;
    background: linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%);
    padding: 16px 20px 30px; display: flex; justify-content: space-between; align-items: flex-start;
}
.row-label { font-size: 22px; font-weight: 700; }
.capture-count { font-size: 18px; font-weight: 600; color: #4ecca3; }
.row-dots { display: flex; gap: 8px; margin-top: 8px; }
.row-dot { width: 10px; height: 10px; border-radius: 50%; background: rgba(255,255,255,0.3); transition: all 0.3s; }
.row-dot.active { background: #4ecca3; box-shadow: 0 0 8px rgba(78,204,163,0.5); }
.row-dot.done { background: #4ecca3; }
.bottom-bar {
    position: absolute; bottom: 0; left: 0; right: 0; z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%);
    padding: 30px 20px; display: flex; flex-direction: column; align-items: center; gap: 12px;
}
.status-text { font-size: 15px; color: rgba(255,255,255,0.8); text-align: center; min-height: 20px; }
.scan-btn {
    padding: 16px 40px; background: #4ecca3; color: #1a1a2e; border: none; border-radius: 10px;
    font-size: 18px; font-weight: 700; cursor: pointer; min-width: 200px;
}
.scan-btn:disabled { opacity: 0.5; cursor: default; }
.scan-btn.scanning { background: #e74c3c; }
.thumb-strip { display: flex; gap: 6px; justify-content: center; flex-wrap: wrap; max-width: 100%; }
.thumb-img { width: 55px; height: 55px; object-fit: cover; border-radius: 4px; border: 2px solid #333; }
.thumb-img.current-row { border-color: #4ecca3; }
.results-overlay {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: 25;
    background: #1a1a2e; display: none; flex-direction: column; align-items: center;
    overflow-y: auto; padding: 30px 20px;
}
.results-overlay.visible { display: flex; }
.results-overlay h2 { font-size: 24px; color: #4ecca3; margin-bottom: 16px; }
.result-card {
    display: flex; gap: 12px; align-items: center; background: rgba(255,255,255,0.05);
    border-radius: 8px; padding: 10px; margin-bottom: 8px; width: 100%; max-width: 400px;
}
.result-card img { width: 60px; height: 60px; object-fit: cover; border-radius: 4px; }
.result-card .info { flex: 1; font-size: 14px; }
.result-card .name { font-weight: 600; font-size: 15px; }
.result-card .detail { color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 2px; }
.reset-btn {
    margin-top: 16px; padding: 14px 36px; background: #333; color: #fff;
    border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer;
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
        <div>
            <div class="row-label" id="rowLabel">Row 1 of 3</div>
            <div class="row-dots" id="rowDots"></div>
        </div>
        <div class="capture-count" id="captureCount">0/3</div>
    </div>

    <div class="bottom-bar">
        <div class="thumb-strip" id="thumbStrip"></div>
        <div class="status-text" id="status">Starting camera...</div>
        <button class="scan-btn" id="scanBtn" onclick="toggleScanning()">Scan Row 1</button>
    </div>

    <div class="results-overlay" id="resultsOverlay">
        <h2>Identification Results</h2>
        <div id="resultsList"></div>
        <button class="reset-btn" onclick="resetAll()">Scan Again</button>
    </div>
</div>

<script>
// ===== Constants =====
const ROWS = 3, CARDS_PER_ROW = 3, TOTAL_CARDS = 9;
const EDGE_THRESH = 25;         // Sobel magnitude to count as edge
const MIN_EDGE_DENSITY = 0.02;  // relaxed: fraction of region with edges
const MIN_SHARPNESS = 8;        // relaxed laplacian variance
const STABLE_NEEDED = 4;        // consecutive good frames before capture
const GUTTER_NEEDED = 3;        // frames of low-edge before re-arming
const ANALYSIS_W = 240;         // downsample width for speed

// ===== State =====
let currentRow = 0, captures = [], scanning = false, stream = null;
const S = { IDLE: 0, DETECT: 1, STABLE: 2, CAPTURED: 3 };
let state = S.IDLE, stableN = 0, gutterN = 0;
let lastCapturePixels = null; // for frame diff after capture
const DIFF_THRESHOLD = 25; // mean pixel diff needed to consider "new card"

// ===== Camera setup (IDENTICAL to slide_scan_v6.py) =====
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

// ===== Reusable analysis canvas =====
let _ac = null, _actx = null;
function acCanvas() {
    if (!_ac) { _ac = document.createElement('canvas'); _actx = _ac.getContext('2d', { willReadFrequently: true }); }
    return { c: _ac, x: _actx };
}

// ===== Frame analysis =====
function analyzeFrame() {
    if (!video.videoWidth || !video.videoHeight)
        return { cardPresent: false, sharp: false, hint: 'Waiting for camera...' };

    const { c: ac, x: actx } = acCanvas();
    const scale = ANALYSIS_W / video.videoWidth;
    const ah = Math.round(video.videoHeight * scale);
    ac.width = ANALYSIS_W; ac.height = ah;
    actx.drawImage(video, 0, 0, ANALYSIS_W, ah);

    // ROI: center 70%
    const rx = Math.round(ANALYSIS_W * 0.15), ry = Math.round(ah * 0.15);
    const rw = Math.round(ANALYSIS_W * 0.7), rh = Math.round(ah * 0.7);
    const img = actx.getImageData(rx, ry, rw, rh);
    const d = img.data, n = rw * rh;

    // Grayscale + brightness
    const g = new Uint8Array(n);
    let bsum = 0;
    for (let i = 0; i < n; i++) { const j = i*4; g[i] = (0.299*d[j] + 0.587*d[j+1] + 0.114*d[j+2]) | 0; bsum += g[i]; }
    const avgB = bsum / n;
    if (avgB < 30) return { cardPresent: false, sharp: false, hint: 'Too dark' };
    if (avgB > 240) return { cardPresent: false, sharp: false, hint: 'Too bright' };

    // Sobel edges + Laplacian sharpness in one pass
    let edgePx = 0, lapSum = 0, lapN = 0;
    for (let y = 1; y < rh - 1; y++) {
        for (let x = 1; x < rw - 1; x++) {
            const i = y * rw + x;
            const gx = -g[i-rw-1]+g[i-rw+1] -2*g[i-1]+2*g[i+1] -g[i+rw-1]+g[i+rw+1];
            const gy = -g[i-rw-1]-2*g[i-rw]-g[i-rw+1] +g[i+rw-1]+2*g[i+rw]+g[i+rw+1];
            if (Math.sqrt(gx*gx + gy*gy) > EDGE_THRESH) edgePx++;
            const lap = g[i-rw]+g[i+rw]+g[i-1]+g[i+1]-4*g[i];
            lapSum += lap*lap; lapN++;
        }
    }

    const edgeDensity = edgePx / n;
    const sharpness = Math.sqrt(lapSum / Math.max(lapN, 1));
    const cardPresent = edgeDensity >= MIN_EDGE_DENSITY;
    const sharp = sharpness >= MIN_SHARPNESS;

    let hint = '';
    if (!cardPresent) hint = 'Slide to next card...';
    else if (!sharp) hint = 'Hold steady — blurry';

    return { cardPresent, sharp, edgeDensity, sharpness, hint };
}

// ===== UI =====
function setStatus(t) { document.getElementById('status').textContent = t; }

function updateUI() {
    const rowCap = captures.length - currentRow * CARDS_PER_ROW;
    const btn = document.getElementById('scanBtn');
    const lbl = document.getElementById('rowLabel');
    const cnt = document.getElementById('captureCount');
    const dots = document.getElementById('rowDots');

    if (currentRow >= ROWS) {
        lbl.textContent = 'All rows captured';
        cnt.textContent = captures.length + '/' + TOTAL_CARDS;
        btn.textContent = 'Submit'; btn.classList.remove('scanning'); btn.disabled = false;
    } else {
        lbl.textContent = 'Row ' + (currentRow+1) + ' of ' + ROWS;
        cnt.textContent = Math.max(0, rowCap) + '/' + CARDS_PER_ROW;
        if (scanning) { btn.textContent = 'Stop'; btn.classList.add('scanning'); btn.disabled = false; }
        else { btn.textContent = 'Scan Row ' + (currentRow+1); btn.classList.remove('scanning'); btn.disabled = false; }
    }

    dots.innerHTML = '';
    for (let i = 0; i < ROWS; i++) {
        const d = document.createElement('div');
        d.className = 'row-dot' + (i < currentRow ? ' done' : i === currentRow ? ' active' : '');
        dots.appendChild(d);
    }
    renderThumbs();
}

function renderThumbs() {
    const c = document.getElementById('thumbStrip');
    c.innerHTML = '';
    for (let i = 0; i < captures.length; i++) {
        const img = document.createElement('img');
        img.src = captures[i].dataUrl;
        img.className = 'thumb-img';
        if (Math.floor(i / CARDS_PER_ROW) === currentRow) img.classList.add('current-row');
        c.appendChild(img);
    }
}

// ===== Overlay drawing =====
function drawOverlay() {
    const w = overlay.clientWidth, h = overlay.clientHeight;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    if (!scanning) return;

    // Guide rect (card-shaped, center)
    const cardAR = 63/88, gH = h * 0.6, gW = gH * cardAR;
    const gX = (w - gW) / 2, gY = (h - gH) / 2 - h * 0.02;

    // Dim surround
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0, 0, w, gY);
    ctx.fillRect(0, gY, gX, gH);
    ctx.fillRect(gX + gW, gY, w - gX - gW, gH);
    ctx.fillRect(0, gY + gH, w, h - gY - gH);

    // Border color by state
    let bc = 'rgba(255,255,255,0.4)';
    if (state === S.STABLE) { const p = stableN / STABLE_NEEDED; bc = p > 0.6 ? '#4ecca3' : '#f1c40f'; }
    else if (state === S.CAPTURED) bc = '#4ecca3';

    ctx.strokeStyle = bc; ctx.lineWidth = 3;
    ctx.beginPath(); roundRect(ctx, gX, gY, gW, gH, 8); ctx.stroke();

    // Corner brackets
    const bL = 22; ctx.strokeStyle = bc; ctx.lineWidth = 4; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(gX, gY+bL); ctx.lineTo(gX, gY); ctx.lineTo(gX+bL, gY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gX+gW-bL, gY); ctx.lineTo(gX+gW, gY); ctx.lineTo(gX+gW, gY+bL); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gX, gY+gH-bL); ctx.lineTo(gX, gY+gH); ctx.lineTo(gX+bL, gY+gH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gX+gW-bL, gY+gH); ctx.lineTo(gX+gW, gY+gH); ctx.lineTo(gX+gW, gY+gH-bL); ctx.stroke();

    // Row capture counter above guide
    const rc = captures.length - currentRow * CARDS_PER_ROW;
    ctx.font = '600 16px -apple-system, sans-serif'; ctx.fillStyle = bc; ctx.textAlign = 'right';
    ctx.fillText(rc + '/' + CARDS_PER_ROW, gX + gW - 8, gY - 8);
}

function roundRect(c, x, y, w, h, r) {
    c.moveTo(x+r, y); c.lineTo(x+w-r, y); c.quadraticCurveTo(x+w, y, x+w, y+r);
    c.lineTo(x+w, y+h-r); c.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
    c.lineTo(x+r, y+h); c.quadraticCurveTo(x, y+h, x, y+h-r);
    c.lineTo(x, y+r); c.quadraticCurveTo(x, y, x+r, y);
}

// ===== Scanning control =====
function toggleScanning() {
    if (currentRow >= ROWS) { submitAll(); return; }
    if (scanning) stopScanning(); else startScanning();
}

function startScanning() {
    scanning = true; state = S.DETECT; stableN = 0; gutterN = 0;
    updateUI(); setStatus('Slide slowly across the row...');
    requestAnimationFrame(scanLoop);
}

function stopScanning() {
    scanning = false; state = S.IDLE;
    const rc = captures.length - currentRow * CARDS_PER_ROW;
    if (rc >= CARDS_PER_ROW) {
        currentRow++;
        if (currentRow >= ROWS) setStatus('All rows captured! Tap Submit.');
        else setStatus('Row done. Tap to scan next row.');
    } else {
        setStatus('Stopped. ' + rc + '/' + CARDS_PER_ROW + ' captured.');
    }
    updateUI();
    const w = overlay.clientWidth, h = overlay.clientHeight;
    if (w && h) ctx.clearRect(0, 0, w, h);
}

// ===== Scan loop =====
let _skip = 0;
function scanLoop() {
    if (!scanning) return;
    if ((captures.length - currentRow * CARDS_PER_ROW) >= CARDS_PER_ROW) { stopScanning(); return; }

    drawOverlay();

    _skip++;
    if (_skip % 2 === 0) {
        const a = analyzeFrame();
        switch (state) {
            case S.DETECT:
                if (a.cardPresent && a.sharp) { state = S.STABLE; stableN = 1; setStatus('Card detected...'); }
                else setStatus(a.hint || 'Slide to next card...');
                break;
            case S.STABLE:
                if (a.cardPresent && a.sharp) {
                    stableN++;
                    if (stableN >= STABLE_NEEDED) {
                        doCapture(); state = S.CAPTURED; setStatus('Captured! Slide to next...');
                    } else setStatus('Steady... (' + Math.round(stableN/STABLE_NEEDED*100) + '%)');
                } else {
                    stableN = Math.max(0, stableN - 2);
                    if (stableN === 0) { state = S.DETECT; setStatus(a.hint || 'Lost — reposition'); }
                }
                break;
            case S.CAPTURED:
                // After capture, wait briefly then go back to detecting.
                // Simple timer approach — avoids complex gutter/frame-diff detection.
                // At ~30fps, 15 frames ≈ 0.5s cooldown before looking for next card.
                gutterN++;
                if (gutterN >= 15) {
                    state = S.DETECT; stableN = 0; gutterN = 0;
                    setStatus('Slide to next card...');
                }
                break;
        }
    }
    requestAnimationFrame(scanLoop);
}

// ===== Capture =====
function doCapture() {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    capCtx.drawImage(video, 0, 0);

    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 150);
    if (navigator.vibrate) navigator.vibrate(50);

    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);
    captureCanvas.toBlob(blob => { captures.push({ dataUrl, blob }); updateUI(); }, 'image/jpeg', 0.92);
}

// ===== Submit =====
async function submitAll() {
    const btn = document.getElementById('scanBtn');
    btn.disabled = true; btn.textContent = 'Identifying...';
    setStatus('Uploading ' + captures.length + ' images...');

    try {
        const fd = new FormData();
        for (let i = 0; i < captures.length; i++)
            fd.append('card_' + i, captures[i].blob, 'card_' + i + '.jpg');

        const resp = await fetch('/slide-scan/identify', { method: 'POST', body: fd });
        const result = await resp.json();
        showResults(result);
    } catch (e) {
        console.error('Submit error:', e);
        setStatus('Error: ' + e.message);
        btn.disabled = false; btn.textContent = 'Retry Submit';
    }
}

function showResults(result) {
    const ro = document.getElementById('resultsOverlay');
    const list = document.getElementById('resultsList');
    list.innerHTML = '';

    const cards = result.cards || result.results || [];
    if (cards.length === 0) {
        list.innerHTML = '<p style="color:rgba(255,255,255,0.6);">No cards identified.</p>';
    } else {
        for (const card of cards) {
            const div = document.createElement('div');
            div.className = 'result-card';
            const ts = card.thumb || card.image || '';
            const nm = card.name || card.card_name || 'Unknown';
            const dt = card.set || card.detail || '';
            div.innerHTML = (ts ? '<img src="'+ts+'">' : '') +
                '<div class="info"><div class="name">'+nm+'</div><div class="detail">'+dt+'</div></div>';
            list.appendChild(div);
        }
    }
    if (result.error) list.innerHTML += '<p style="color:#e74c3c;margin-top:12px;">'+result.error+'</p>';
    ro.classList.add('visible');
}

function resetAll() {
    captures = []; currentRow = 0; scanning = false;
    state = S.IDLE; stableN = 0; gutterN = 0;
    document.getElementById('resultsOverlay').classList.remove('visible');
    updateUI(); setStatus('Ready. Tap "Scan Row 1" to begin.');
}

// ===== Init =====
updateUI();
startCamera();
</script>
</body>
</html>
"""
