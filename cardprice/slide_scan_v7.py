"""Slide-scan UI v7: smart auto-capture with quality gates.

Camera code copied character-for-character from slide_scan_v6.py (confirmed
working on Brave via Cloudflare tunnel).

Behavior:
  1. Idle: live camera feed + card guide rectangle + "Scan Row N" button.
  2. Scanning: analyse every frame via requestAnimationFrame:
     - Card fill check (Sobel edges -> rectangle 35-70% of frame)
     - Sharpness check (Laplacian variance > 30)
     - Single card check (exactly 1 card-like rectangle)
     - 3 consecutive good frames -> AUTO-CAPTURE
     - Wait for card to EXIT before looking for next card
     - After 3 captures -> row done
  3. Row transition: "Row N done!" then "Scan Row N+1" button
  4. After 3 rows: 3x3 grid preview + Submit -> POST /slide-scan/identify

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
<title>Slide Scan v7</title>
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

/* ---- Hidden analysis canvas ---- */
canvas#analysis {
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
.capture-count {
    position: absolute;
    top: 16px; right: 20px;
    font-size: 28px;
    font-weight: 700;
    color: #4ecca3;
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
.scan-btn.hidden {
    display: none;
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
    animation: thumbSlideIn 0.3s ease-out;
}
.thumb-img.current-row {
    border-color: #4ecca3;
}
@keyframes thumbSlideIn {
    from { transform: scale(0.5); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
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
.grid-preview {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 4px;
    max-width: 320px;
    margin-bottom: 20px;
}
.grid-preview img {
    width: 100%;
    aspect-ratio: 0.716;
    object-fit: cover;
    border-radius: 4px;
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
    <canvas id="analysis"></canvas>
    <div class="flash" id="flash"></div>

    <div class="top-bar">
        <div class="row-label" id="rowLabel">Row 1 of 3</div>
        <div class="capture-count" id="captureCount"></div>
        <div class="row-dots" id="rowDots"></div>
    </div>

    <div class="bottom-bar">
        <div class="thumbs-strip" id="thumbs"></div>
        <div class="status-text" id="status">Starting camera...</div>
        <button class="scan-btn" id="scanBtn" onclick="onScanBtn()">Scan Row 1</button>
    </div>

    <div class="results-overlay" id="resultsOverlay">
        <h2>Scan Complete</h2>
        <div class="grid-preview" id="gridPreview"></div>
        <div id="resultsList"></div>
        <button class="scan-btn" id="submitBtn" onclick="submitImages()" style="margin-top:16px;">Submit for Identification</button>
    </div>
</div>

<script>
// ===== Constants =====
const ROWS = 3;
const CARDS_PER_ROW = 3;
const CARD_ASPECT = 0.716;          // width / height of Pokemon card
const MIN_FILL = 0.35;
const MAX_FILL = 0.70;
const SHARPNESS_THRESHOLD = 30;
const CONSECUTIVE_GOOD_NEEDED = 3;
const ANALYSIS_SIZE = 160;          // downscale for speed

// ===== State =====
let currentRow = 0;
let captures = [];
let scanning = false;
let cardsThisRow = 0;
let consecutiveGood = 0;
let waitingForExit = false;
let animFrameId = null;
let stream = null;

// ===== Camera setup (IDENTICAL to condition_camera_ui.py) =====
const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const captureCanvas = document.getElementById('capture');
const analysisCanvas = document.getElementById('analysis');
const ctx = overlay.getContext('2d');
const capCtx = captureCanvas.getContext('2d');
const anaCtx = analysisCanvas.getContext('2d');

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
        drawOverlay();
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
    const dots = document.getElementById('rowDots');
    const btn = document.getElementById('scanBtn');
    const countEl = document.getElementById('captureCount');

    if (currentRow >= ROWS) {
        label.textContent = 'All rows captured!';
        btn.classList.add('hidden');
        countEl.textContent = '';
        showGridPreview();
    } else {
        label.textContent = `Row ${currentRow + 1} of ${ROWS}`;
        if (scanning) {
            btn.classList.add('hidden');
            countEl.textContent = `${cardsThisRow}/${CARDS_PER_ROW}`;
        } else {
            btn.classList.remove('hidden');
            btn.textContent = `Scan Row ${currentRow + 1}`;
            btn.disabled = false;
            countEl.textContent = cardsThisRow > 0 ? `${cardsThisRow}/${CARDS_PER_ROW}` : '';
        }
    }

    // Dots
    dots.innerHTML = '';
    for (let i = 0; i < ROWS; i++) {
        const dot = document.createElement('div');
        dot.className = 'row-dot' + (i < currentRow ? ' done' : i === currentRow ? ' active' : '');
        dots.appendChild(dot);
    }

    renderThumbs();
}

function renderThumbs() {
    const container = document.getElementById('thumbs');
    container.innerHTML = '';
    for (let i = 0; i < captures.length; i++) {
        const img = document.createElement('img');
        img.src = captures[i].dataUrl;
        img.className = 'thumb-img';
        const row = Math.floor(i / CARDS_PER_ROW);
        if (row === currentRow) img.classList.add('current-row');
        container.appendChild(img);
    }
}

// ===== Guide rectangle dimensions =====
function getGuideRect(w, h) {
    // Card guide: ~50% of frame height, card aspect ratio
    const guideH = h * 0.50;
    const guideW = guideH * CARD_ASPECT;
    const gx = (w - guideW) / 2;
    const gy = (h - guideH) / 2;
    return { x: gx, y: gy, w: guideW, h: guideH };
}

// ===== Overlay drawing =====
function drawOverlay(statusMsg) {
    const w = overlay.clientWidth;
    const h = overlay.clientHeight;
    if (!w || !h) return;

    ctx.clearRect(0, 0, w, h);

    const guide = getGuideRect(w, h);

    // Draw guide rectangle
    if (!scanning) {
        // Idle: bright green solid
        ctx.strokeStyle = '#4ecca3';
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
    } else if (waitingForExit) {
        // Waiting for card to leave
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([8, 8]);
    } else if (consecutiveGood >= 1) {
        // Getting close to capture
        ctx.strokeStyle = '#f1c40f';
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
    } else {
        // Scanning, looking for card
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 8]);
    }

    ctx.strokeRect(guide.x, guide.y, guide.w, guide.h);
    ctx.setLineDash([]);

    // Corner accents
    const cornerLen = 20;
    ctx.lineWidth = 4;
    ctx.strokeStyle = scanning && consecutiveGood >= 2 ? '#4ecca3' : (scanning ? 'rgba(255,255,255,0.5)' : '#4ecca3');
    // Top-left
    ctx.beginPath();
    ctx.moveTo(guide.x, guide.y + cornerLen); ctx.lineTo(guide.x, guide.y); ctx.lineTo(guide.x + cornerLen, guide.y);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(guide.x + guide.w - cornerLen, guide.y); ctx.lineTo(guide.x + guide.w, guide.y); ctx.lineTo(guide.x + guide.w, guide.y + cornerLen);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(guide.x, guide.y + guide.h - cornerLen); ctx.lineTo(guide.x, guide.y + guide.h); ctx.lineTo(guide.x + cornerLen, guide.y + guide.h);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(guide.x + guide.w - cornerLen, guide.y + guide.h); ctx.lineTo(guide.x + guide.w, guide.y + guide.h); ctx.lineTo(guide.x + guide.w, guide.y + guide.h - cornerLen);
    ctx.stroke();

    // Status text at bottom
    if (statusMsg) {
        ctx.font = '600 16px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#fff';
        ctx.fillText(statusMsg, w / 2, h - 80);
    }

    // Slide direction arrow during scanning
    if (scanning && !waitingForExit) {
        ctx.font = '24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.fillText('\u2192', w / 2, guide.y - 20);
    }
}

// ===== Image analysis (runs on every frame during scanning) =====

function getFrameData() {
    // Draw video to small analysis canvas for speed
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    const scale = ANALYSIS_SIZE / Math.max(vw, vh);
    const aw = Math.round(vw * scale);
    const ah = Math.round(vh * scale);
    analysisCanvas.width = aw;
    analysisCanvas.height = ah;
    anaCtx.drawImage(video, 0, 0, aw, ah);
    return anaCtx.getImageData(0, 0, aw, ah);
}

function toGrayscale(imageData) {
    const { data, width, height } = imageData;
    const gray = new Float32Array(width * height);
    for (let i = 0; i < gray.length; i++) {
        const j = i * 4;
        gray[i] = 0.299 * data[j] + 0.587 * data[j+1] + 0.114 * data[j+2];
    }
    return { gray, width, height };
}

// Laplacian variance for sharpness
function laplacianVariance(grayObj) {
    const { gray, width, height } = grayObj;
    // Sample center 50% region for speed
    const x0 = Math.floor(width * 0.25);
    const x1 = Math.floor(width * 0.75);
    const y0 = Math.floor(height * 0.25);
    const y1 = Math.floor(height * 0.75);

    let sum = 0, sum2 = 0, n = 0;
    for (let y = y0 + 1; y < y1 - 1; y++) {
        for (let x = x0 + 1; x < x1 - 1; x++) {
            const idx = y * width + x;
            const lap = gray[idx - width] + gray[idx + width] + gray[idx - 1] + gray[idx + 1] - 4 * gray[idx];
            sum += lap;
            sum2 += lap * lap;
            n++;
        }
    }
    if (n === 0) return 0;
    const mean = sum / n;
    return (sum2 / n) - (mean * mean);
}

// Sobel edge magnitude
function sobelEdges(grayObj) {
    const { gray, width, height } = grayObj;
    const edges = new Float32Array(width * height);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            const gx = -gray[idx-width-1] + gray[idx-width+1]
                       -2*gray[idx-1] + 2*gray[idx+1]
                       -gray[idx+width-1] + gray[idx+width+1];
            const gy = -gray[idx-width-1] - 2*gray[idx-width] - gray[idx-width+1]
                       +gray[idx+width-1] + 2*gray[idx+width] + gray[idx+width+1];
            edges[idx] = Math.sqrt(gx*gx + gy*gy);
        }
    }
    return edges;
}

// Find card-like rectangle from edge image
function findCardRect(edges, width, height) {
    // Threshold edges
    const threshold = 40;
    let minX = width, maxX = 0, minY = height, maxY = 0;
    let edgeCount = 0;

    for (let y = 2; y < height - 2; y++) {
        for (let x = 2; x < width - 2; x++) {
            if (edges[y * width + x] > threshold) {
                edgeCount++;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (edgeCount < 50) return null;

    const rectW = maxX - minX;
    const rectH = maxY - minY;
    if (rectW < 10 || rectH < 10) return null;

    const aspect = rectW / rectH;
    const fillRatio = (rectW * rectH) / (width * height);

    return { x: minX, y: minY, w: rectW, h: rectH, aspect, fillRatio, edgeCount };
}

// Count distinct edge clusters (to detect multiple cards)
function countEdgeClusters(edges, width, height) {
    // Simple: check if edge density in left third and right third are both high
    // which would indicate two cards side by side
    const threshold = 40;
    const third = Math.floor(width / 3);
    let leftEdges = 0, rightEdges = 0, centerEdges = 0;
    const centerY0 = Math.floor(height * 0.2);
    const centerY1 = Math.floor(height * 0.8);

    for (let y = centerY0; y < centerY1; y++) {
        for (let x = 2; x < third; x++) {
            if (edges[y * width + x] > threshold) leftEdges++;
        }
        for (let x = third; x < 2 * third; x++) {
            if (edges[y * width + x] > threshold) centerEdges++;
        }
        for (let x = 2 * third; x < width - 2; x++) {
            if (edges[y * width + x] > threshold) rightEdges++;
        }
    }

    // If both left and right have significant edges and center has a gap-like dip,
    // there might be two cards
    const avgSide = (leftEdges + rightEdges) / 2;
    if (leftEdges > 30 && rightEdges > 30 && centerEdges < avgSide * 0.5) {
        return 2;
    }
    return 1;
}

// Main analysis function returns { cardFill, sharpness, singleCard, feedback }
function analyzeFrame() {
    const imageData = getFrameData();
    if (!imageData) return { cardFill: false, sharpness: false, singleCard: false, feedback: 'No video' };

    const grayObj = toGrayscale(imageData);
    const { width, height } = grayObj;

    // 1. Sharpness
    const lapVar = laplacianVariance(grayObj);
    const sharpness = lapVar > SHARPNESS_THRESHOLD;

    // 2. Edge detection
    const edges = sobelEdges(grayObj);
    const rect = findCardRect(edges, width, height);

    if (!rect) {
        return { cardFill: false, sharpness, singleCard: true, feedback: 'No card detected', fillRatio: 0, lapVar };
    }

    // 3. Card fill check
    const cardFill = rect.fillRatio >= MIN_FILL && rect.fillRatio <= MAX_FILL;

    // 4. Aspect ratio check (allow some tolerance for Pokemon cards)
    const aspectOk = rect.aspect > 0.5 && rect.aspect < 0.95;

    // 5. Single card check
    const clusters = countEdgeClusters(edges, width, height);
    const singleCard = clusters === 1;

    // Build feedback message
    let feedback = '';
    if (!cardFill && rect.fillRatio < MIN_FILL) {
        feedback = 'Move closer';
    } else if (!cardFill && rect.fillRatio > MAX_FILL) {
        feedback = 'Move back a bit';
    } else if (!sharpness) {
        feedback = 'Too blurry - hold steady';
    } else if (!singleCard) {
        feedback = 'Center one card only';
    } else if (!aspectOk) {
        feedback = 'Center the card';
    } else {
        feedback = 'Hold steady...';
    }

    return {
        cardFill: cardFill && aspectOk,
        sharpness,
        singleCard,
        feedback,
        fillRatio: rect.fillRatio,
        lapVar,
    };
}

// Check if card has exited frame (for preventing double captures)
function isCardPresent() {
    const imageData = getFrameData();
    if (!imageData) return false;
    const grayObj = toGrayscale(imageData);
    const edges = sobelEdges(grayObj);
    const rect = findCardRect(edges, grayObj.width, grayObj.height);
    // Card is "present" if fill > 25%
    return rect && rect.fillRatio > 0.25;
}

// ===== Scanning loop =====
function scanLoop() {
    if (!scanning) return;

    const w = overlay.clientWidth;
    const h = overlay.clientHeight;

    if (waitingForExit) {
        // Wait for card to leave the frame
        const present = isCardPresent();
        if (!present) {
            waitingForExit = false;
            consecutiveGood = 0;
            setStatus('Slide to next card...');
        }
        drawOverlay(present ? 'Remove card from frame...' : 'Slide to next card...');
        animFrameId = requestAnimationFrame(scanLoop);
        return;
    }

    const result = analyzeFrame();
    const allGood = result.cardFill && result.sharpness && result.singleCard;

    if (allGood) {
        consecutiveGood++;
    } else {
        consecutiveGood = 0;
    }

    // Draw overlay with feedback
    let overlayMsg = result.feedback;
    if (consecutiveGood >= 1 && consecutiveGood < CONSECUTIVE_GOOD_NEEDED) {
        overlayMsg = `Hold steady... (${consecutiveGood}/${CONSECUTIVE_GOOD_NEEDED})`;
    }
    drawOverlay(overlayMsg);

    if (consecutiveGood >= CONSECUTIVE_GOOD_NEEDED) {
        // AUTO-CAPTURE!
        captureCard();
        consecutiveGood = 0;
        waitingForExit = true;

        if (cardsThisRow >= CARDS_PER_ROW) {
            // Row done
            scanning = false;
            if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
            currentRow++;
            setStatus(`Row ${currentRow} done!`);
            setTimeout(() => updateUI(), 500);
            return;
        }
    }

    animFrameId = requestAnimationFrame(scanLoop);
}

// ===== Capture =====
function captureCard() {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    capCtx.drawImage(video, 0, 0);

    // Flash effect
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 150);

    // Haptic feedback
    if (navigator.vibrate) navigator.vibrate(50);

    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);
    captureCanvas.toBlob(blob => {
        captures.push({ dataUrl, blob });
        cardsThisRow++;
        setStatus(`Captured ${cardsThisRow}/${CARDS_PER_ROW} for row ${currentRow + 1}`);
        document.getElementById('captureCount').textContent = `${cardsThisRow}/${CARDS_PER_ROW}`;
        renderThumbs();
    }, 'image/jpeg', 0.92);
}

// ===== Button handler =====
function onScanBtn() {
    if (currentRow >= ROWS) {
        showGridPreview();
        return;
    }
    if (scanning) return;

    scanning = true;
    cardsThisRow = 0;
    consecutiveGood = 0;
    waitingForExit = false;
    updateUI();
    setStatus(`Scanning Row ${currentRow + 1}... slide cards through the guide`);
    scanLoop();
}

// ===== Grid preview =====
function showGridPreview() {
    const overlay = document.getElementById('resultsOverlay');
    const grid = document.getElementById('gridPreview');
    const list = document.getElementById('resultsList');
    grid.innerHTML = '';
    list.innerHTML = '';

    for (let i = 0; i < captures.length; i++) {
        const img = document.createElement('img');
        img.src = captures[i].dataUrl;
        grid.appendChild(img);
    }

    overlay.classList.add('visible');
}

// ===== Submit =====
async function submitImages() {
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.textContent = 'Identifying...';

    try {
        const formData = new FormData();
        for (let i = 0; i < captures.length; i++) {
            formData.append('card_' + i, captures[i].blob, 'card_' + i + '.jpg');
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

    const btn = document.getElementById('submitBtn');
    btn.textContent = 'Done';
    btn.disabled = true;
}

// ===== Init =====
updateUI();
startCamera();
</script>
</body>
</html>
"""
