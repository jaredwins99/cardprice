"""Slide-scan binder camera UI for real-time card detection.

User holds phone ~15cm from binder and slowly slides across each row.
The camera detects individual cards as they pass through a central detection
zone, auto-capturing at peak sharpness/alignment. All detection runs
client-side in JS — no server round-trips during scanning.

Integration into server.py:
    elif self.path == "/slide-scan":
        from cardprice.slide_scan_ui import SLIDE_SCAN_HTML
        self._send_html(SLIDE_SCAN_HTML)
"""

SLIDE_SCAN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Slide Scan</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
}

video {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: 1;
}

/* Hidden canvases for processing */
canvas.hidden-canvas { display: none; }

/* ---- Detection zone overlay ---- */
canvas#overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    pointer-events: none;
}

/* ---- Capture strip (thumbnails at top) ---- */
.capture-strip {
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to bottom, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.6) 80%, transparent 100%);
    padding: 8px 10px 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    pointer-events: none;
}

.strip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    pointer-events: auto;
}

.strip-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
}

.strip-counter {
    font-size: 13px;
    color: #4ecca3;
    font-weight: 700;
}

.strip-thumbs {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.strip-thumbs::-webkit-scrollbar { display: none; }

.strip-thumb {
    flex-shrink: 0;
    width: 52px;
    height: 72px;
    border-radius: 4px;
    border: 2px solid #333;
    object-fit: cover;
    opacity: 0;
    transform: scale(0.7);
    transition: all 0.3s ease;
}
.strip-thumb.visible {
    opacity: 1;
    transform: scale(1);
}
.strip-thumb.latest {
    border-color: #4ecca3;
    box-shadow: 0 0 8px rgba(78, 204, 163, 0.5);
}

/* Placeholder slots */
.strip-slot {
    flex-shrink: 0;
    width: 52px;
    height: 72px;
    border-radius: 4px;
    border: 2px dashed rgba(255,255,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: rgba(255,255,255,0.3);
}

/* ---- Row progress indicator ---- */
.row-progress {
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 10;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: none;
}

.row-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgba(255,255,255,0.25);
    transition: all 0.3s;
}
.row-dot.active {
    background: #4ecca3;
    box-shadow: 0 0 8px rgba(78, 204, 163, 0.5);
    transform: scale(1.3);
}
.row-dot.done {
    background: #4ecca3;
}

/* ---- Bottom status bar ---- */
.bottom-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.5) 70%, transparent 100%);
    padding: 30px 20px 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

.status-text {
    font-size: 15px;
    color: rgba(255,255,255,0.8);
    text-align: center;
    min-height: 20px;
    transition: color 0.2s;
}
.status-text.ready {
    color: #4ecca3;
    font-weight: 600;
}

.detection-meter {
    width: 200px;
    height: 4px;
    background: rgba(255,255,255,0.15);
    border-radius: 2px;
    overflow: hidden;
}
.detection-fill {
    height: 100%;
    width: 0%;
    background: #4ecca3;
    border-radius: 2px;
    transition: width 0.15s;
}

.bottom-btns {
    display: flex;
    gap: 16px;
    align-items: center;
}

.btn {
    padding: 12px 24px;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
}
.btn:active { transform: scale(0.95); }

.btn-manual {
    background: rgba(255,255,255,0.15);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.3);
}

.btn-next-row {
    background: #4ecca3;
    color: #1a1a2e;
    display: none;
}

.btn-done {
    background: #e94560;
    color: #fff;
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
    transition: opacity 0.15s;
}
.flash.active {
    opacity: 0.6;
    transition: none;
}

/* ---- Row transition overlay ---- */
.row-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 15;
    background: rgba(0,0,0,0.85);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
}
.row-overlay.visible { display: flex; }
.row-overlay h2 {
    font-size: 22px;
    color: #4ecca3;
}
.row-overlay p {
    font-size: 15px;
    color: rgba(255,255,255,0.7);
    text-align: center;
    padding: 0 30px;
}
.row-overlay .btn {
    margin-top: 10px;
}

/* ---- Done overlay ---- */
.done-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 25;
    background: #1a1a2e;
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px;
}
.done-overlay.visible { display: flex; }
.done-overlay h2 {
    font-size: 24px;
    color: #4ecca3;
    margin-bottom: 10px;
}
.done-overlay p {
    color: rgba(255,255,255,0.7);
    margin-bottom: 16px;
    text-align: center;
}

.done-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    max-width: 300px;
    margin-bottom: 20px;
}
.done-grid img {
    width: 100%;
    aspect-ratio: 63/88;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #333;
}

.done-overlay .btn-submit {
    padding: 16px 40px;
    background: #4ecca3;
    color: #1a1a2e;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
}
.done-overlay .btn-submit:disabled {
    opacity: 0.5;
}
.upload-status {
    margin-top: 12px;
    font-size: 14px;
    color: rgba(255,255,255,0.6);
    min-height: 20px;
}

/* ---- Settings gear ---- */
.settings-btn {
    position: absolute;
    top: 8px;
    right: 10px;
    z-index: 11;
    background: none;
    border: none;
    color: rgba(255,255,255,0.6);
    font-size: 22px;
    cursor: pointer;
    padding: 6px;
}

.settings-panel {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 30;
    background: rgba(0,0,0,0.92);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 20px;
    padding: 30px;
}
.settings-panel.visible { display: flex; }
.settings-panel h3 {
    font-size: 20px;
    color: #4ecca3;
}
.setting-row {
    display: flex;
    align-items: center;
    gap: 16px;
    width: 100%;
    max-width: 300px;
    justify-content: space-between;
}
.setting-row label {
    font-size: 15px;
    color: rgba(255,255,255,0.8);
}
.setting-row select, .setting-row input {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #555;
    background: #222;
    color: #fff;
    font-size: 15px;
    width: 80px;
    text-align: center;
}

/* ---- Debug info (hidden by default) ---- */
.debug-info {
    position: absolute;
    bottom: 120px;
    right: 10px;
    z-index: 10;
    font-size: 10px;
    color: rgba(255,255,255,0.4);
    text-align: right;
    line-height: 1.4;
    pointer-events: none;
    display: none;
}
.debug-info.visible { display: block; }
</style>
</head>
<body>

<div class="camera-wrap">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <canvas id="detect" class="hidden-canvas"></canvas>
    <canvas id="capture" class="hidden-canvas"></canvas>

    <!-- Capture strip -->
    <div class="capture-strip" id="captureStrip">
        <div class="strip-header">
            <span class="strip-title" id="stripTitle">Row 1</span>
            <span class="strip-counter" id="stripCounter">0 / 9</span>
        </div>
        <div class="strip-thumbs" id="stripThumbs"></div>
    </div>

    <!-- Row progress dots -->
    <div class="row-progress" id="rowProgress"></div>

    <!-- Bottom bar -->
    <div class="bottom-bar">
        <div class="status-text" id="statusText">Starting camera...</div>
        <div class="detection-meter">
            <div class="detection-fill" id="detectionFill"></div>
        </div>
        <div class="bottom-btns">
            <button class="btn btn-manual" id="btnManual" onclick="manualCapture()">Capture</button>
            <button class="btn btn-next-row" id="btnNextRow" onclick="nextRow()">Next Row</button>
            <button class="btn btn-done" id="btnDone" onclick="showDone()">Done</button>
        </div>
    </div>

    <!-- Flash -->
    <div class="flash" id="flash"></div>

    <!-- Row transition overlay -->
    <div class="row-overlay" id="rowOverlay">
        <h2 id="rowOverlayTitle">Row 1 Complete</h2>
        <p id="rowOverlayText">Move your phone to the start of the next row, then tap Continue.</p>
        <button class="btn btn-next-row" style="display:inline-block" onclick="continueNextRow()">Continue Scanning</button>
    </div>

    <!-- Done overlay -->
    <div class="done-overlay" id="doneOverlay">
        <h2>Scan Complete</h2>
        <p id="doneSummary">9 cards captured</p>
        <div class="done-grid" id="doneGrid"></div>
        <button class="btn-submit" id="btnSubmit" onclick="submitCards()">Identify Cards</button>
        <div class="upload-status" id="uploadStatus"></div>
    </div>

    <!-- Settings -->
    <button class="settings-btn" id="settingsBtn" onclick="toggleSettings()">&#9881;</button>
    <div class="settings-panel" id="settingsPanel">
        <h3>Scan Settings</h3>
        <div class="setting-row">
            <label>Columns</label>
            <input type="number" id="setCols" value="3" min="1" max="6">
        </div>
        <div class="setting-row">
            <label>Rows</label>
            <input type="number" id="setRows" value="3" min="1" max="6">
        </div>
        <div class="setting-row">
            <label>Debug</label>
            <select id="setDebug">
                <option value="0">Off</option>
                <option value="1">On</option>
            </select>
        </div>
        <button class="btn" style="background:#4ecca3;color:#1a1a2e;margin-top:10px" onclick="applySettings()">Apply</button>
    </div>

    <!-- Debug info -->
    <div class="debug-info" id="debugInfo"></div>
</div>

<script>
// ============================================================
// Configuration
// ============================================================
let CFG = {
    cols: 3,
    rows: 3,
    get total() { return this.cols * this.rows; },

    // Detection zone: center portion of frame
    zoneW: 0.50,   // width fraction of frame
    zoneH: 0.70,   // height fraction of frame

    // Thresholds
    edgeDensityMin: 0.12,       // fraction of edge pixels in zone
    sharpnessMin: 15.0,         // Laplacian variance threshold
    contrastMin: 10,            // brightness diff card vs background
    readyFramesNeeded: 3,       // consecutive "ready" frames to trigger capture
    frameSampleInterval: 3,     // process every Nth frame

    // Detection canvas width (for speed)
    detectWidth: 320,

    debug: false,
};

// ============================================================
// State
// ============================================================
let video, overlay, detectCanvas, captureCanvas;
let overlayCtx, detectCtx, captureCtx;
let captures = [];         // array of {dataUrl, row, col}
let currentRow = 0;        // 0-indexed
let currentColInRow = 0;   // cards captured in current row
let readyCount = 0;        // consecutive ready frames
let cooldown = false;       // after capture, wait for card to leave
let cooldownFrames = 0;     // count frames during cooldown
let frameCount = 0;
let scanning = true;
let animFrameId = null;

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', init);

async function init() {
    video = document.getElementById('video');
    overlay = document.getElementById('overlay');
    detectCanvas = document.getElementById('detect');
    captureCanvas = document.getElementById('capture');

    overlayCtx = overlay.getContext('2d');
    detectCtx = detectCanvas.getContext('2d', { willReadFrequently: true });
    captureCtx = captureCanvas.getContext('2d');

    buildUI();

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1920 },
                height: { ideal: 1080 },
            },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setStatus('Slide across cards slowly...');

        // Match overlay to video display size
        resizeOverlay();
        window.addEventListener('resize', resizeOverlay);

        // Start detection loop
        animFrameId = requestAnimationFrame(detectionLoop);
    } catch (err) {
        setStatus('Camera error: ' + err.message);
    }
}

function resizeOverlay() {
    const r = video.getBoundingClientRect();
    overlay.width = r.width;
    overlay.height = r.height;
}

// ============================================================
// UI builders
// ============================================================
function buildUI() {
    buildThumbs();
    buildRowDots();
}

function buildThumbs() {
    const container = document.getElementById('stripThumbs');
    container.innerHTML = '';
    for (let i = 0; i < CFG.total; i++) {
        if (captures[i]) {
            const img = document.createElement('img');
            img.className = 'strip-thumb visible' + (i === captures.length - 1 ? ' latest' : '');
            img.src = captures[i].dataUrl;
            container.appendChild(img);
        } else {
            const slot = document.createElement('div');
            slot.className = 'strip-slot';
            slot.textContent = i + 1;
            container.appendChild(slot);
        }
    }
    document.getElementById('stripTitle').textContent = 'Row ' + (currentRow + 1);
    document.getElementById('stripCounter').textContent = captures.length + ' / ' + CFG.total;
}

function buildRowDots() {
    const container = document.getElementById('rowProgress');
    container.innerHTML = '';
    for (let r = 0; r < CFG.rows; r++) {
        const dot = document.createElement('div');
        dot.className = 'row-dot';
        if (r < currentRow) dot.classList.add('done');
        if (r === currentRow) dot.classList.add('active');
        container.appendChild(dot);
    }
}

function setStatus(msg, ready) {
    const el = document.getElementById('statusText');
    el.textContent = msg;
    el.className = 'status-text' + (ready ? ' ready' : '');
}

function setMeter(fraction) {
    document.getElementById('detectionFill').style.width = (fraction * 100) + '%';
}

function updateDebug(info) {
    if (!CFG.debug) return;
    const el = document.getElementById('debugInfo');
    el.textContent = Object.entries(info).map(([k,v]) => k + ': ' + (typeof v === 'number' ? v.toFixed(2) : v)).join('\n');
}

// ============================================================
// Detection loop
// ============================================================
function detectionLoop(timestamp) {
    if (!scanning) return;

    frameCount++;
    drawOverlay();

    if (frameCount % CFG.frameSampleInterval === 0) {
        processFrame();
    }

    animFrameId = requestAnimationFrame(detectionLoop);
}

// ============================================================
// Draw overlay (detection zone rectangle)
// ============================================================
function drawOverlay() {
    const W = overlay.width;
    const H = overlay.height;
    overlayCtx.clearRect(0, 0, W, H);

    // Card-shaped detection zone in center
    const zw = W * CFG.zoneW;
    const zh = H * CFG.zoneH;
    const zx = (W - zw) / 2;
    const zy = (H - zh) / 2;

    // Dim outside zone
    overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    // Top
    overlayCtx.fillRect(0, 0, W, zy);
    // Bottom
    overlayCtx.fillRect(0, zy + zh, W, H - zy - zh);
    // Left
    overlayCtx.fillRect(0, zy, zx, zh);
    // Right
    overlayCtx.fillRect(zx + zw, zy, W - zx - zw, zh);

    // Zone border
    const borderColor = cooldown ? 'rgba(255, 200, 50, 0.6)' :
                         readyCount >= CFG.readyFramesNeeded ? 'rgba(78, 204, 163, 0.9)' :
                         readyCount > 0 ? 'rgba(78, 204, 163, 0.5)' :
                         'rgba(255, 255, 255, 0.4)';
    overlayCtx.strokeStyle = borderColor;
    overlayCtx.lineWidth = readyCount > 0 ? 3 : 2;
    overlayCtx.setLineDash(readyCount > 0 ? [] : [8, 6]);

    // Rounded rect
    const r = 8;
    overlayCtx.beginPath();
    overlayCtx.moveTo(zx + r, zy);
    overlayCtx.lineTo(zx + zw - r, zy);
    overlayCtx.arcTo(zx + zw, zy, zx + zw, zy + r, r);
    overlayCtx.lineTo(zx + zw, zy + zh - r);
    overlayCtx.arcTo(zx + zw, zy + zh, zx + zw - r, zy + zh, r);
    overlayCtx.lineTo(zx + r, zy + zh);
    overlayCtx.arcTo(zx, zy + zh, zx, zy + zh - r, r);
    overlayCtx.lineTo(zx, zy + r);
    overlayCtx.arcTo(zx, zy, zx + r, zy, r);
    overlayCtx.closePath();
    overlayCtx.stroke();
    overlayCtx.setLineDash([]);

    // Corner brackets for alignment guidance
    const bLen = 20;
    overlayCtx.strokeStyle = borderColor;
    overlayCtx.lineWidth = 3;
    // top-left
    overlayCtx.beginPath();
    overlayCtx.moveTo(zx, zy + bLen); overlayCtx.lineTo(zx, zy); overlayCtx.lineTo(zx + bLen, zy);
    overlayCtx.stroke();
    // top-right
    overlayCtx.beginPath();
    overlayCtx.moveTo(zx + zw - bLen, zy); overlayCtx.lineTo(zx + zw, zy); overlayCtx.lineTo(zx + zw, zy + bLen);
    overlayCtx.stroke();
    // bottom-left
    overlayCtx.beginPath();
    overlayCtx.moveTo(zx, zy + zh - bLen); overlayCtx.lineTo(zx, zy + zh); overlayCtx.lineTo(zx + bLen, zy + zh);
    overlayCtx.stroke();
    // bottom-right
    overlayCtx.beginPath();
    overlayCtx.moveTo(zx + zw - bLen, zy + zh); overlayCtx.lineTo(zx + zw, zy + zh); overlayCtx.lineTo(zx + zw, zy + zh - bLen);
    overlayCtx.stroke();

    // Direction arrow (slide right indicator)
    if (!cooldown && readyCount === 0 && captures.length < CFG.total) {
        overlayCtx.save();
        overlayCtx.globalAlpha = 0.3 + 0.15 * Math.sin(Date.now() / 400);
        overlayCtx.fillStyle = '#fff';
        overlayCtx.font = '28px sans-serif';
        overlayCtx.textAlign = 'center';
        overlayCtx.fillText('\u25B6', W / 2, zy + zh + 30);
        overlayCtx.restore();
    }
}

// ============================================================
// Frame processing — all detection logic
// ============================================================
function processFrame() {
    if (!video.videoWidth || cooldown) {
        if (cooldown) {
            cooldownFrames++;
            // Check if card has left the frame (edge density drops)
            const info = analyzeZone();
            if (info && info.edgeDensity < CFG.edgeDensityMin * 0.5) {
                // Card has left — ready for next detection
                cooldown = false;
                cooldownFrames = 0;
                setStatus('Slide to next card...', false);
                setMeter(0);
            } else if (cooldownFrames > 30) {
                // Timeout: force reset after ~1.5s even if card hasn't fully left
                cooldown = false;
                cooldownFrames = 0;
                setStatus('Slide to next card...', false);
                setMeter(0);
            }
            updateDebug({ state: 'cooldown', frames: cooldownFrames, edgeDensity: info ? info.edgeDensity : 0 });
        }
        return;
    }

    const info = analyzeZone();
    if (!info) return;

    const isReady = info.edgeDensity >= CFG.edgeDensityMin
                 && info.sharpness >= CFG.sharpnessMin
                 && info.contrast >= CFG.contrastMin;

    updateDebug({
        edge: info.edgeDensity,
        sharp: info.sharpness,
        contrast: info.contrast,
        ready: readyCount,
        state: isReady ? 'READY' : 'scanning',
    });

    if (isReady) {
        readyCount++;
        setMeter(Math.min(readyCount / CFG.readyFramesNeeded, 1));

        if (readyCount === 1) {
            setStatus('Card detected...', false);
        } else if (readyCount >= 2) {
            setStatus('Hold steady...', true);
        }

        if (readyCount >= CFG.readyFramesNeeded) {
            captureCard();
        }
    } else {
        if (readyCount > 0) readyCount = Math.max(0, readyCount - 1);
        setMeter(Math.min(readyCount / CFG.readyFramesNeeded, 1));
        if (readyCount === 0 && !cooldown) {
            if (captures.length < CFG.total) {
                setStatus('Slide across cards slowly...', false);
            }
        }
    }
}

// ============================================================
// Analyze the detection zone on a downsampled frame
// ============================================================
function analyzeZone() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    // Downsample
    const scale = CFG.detectWidth / vw;
    const dw = CFG.detectWidth;
    const dh = Math.round(vh * scale);
    detectCanvas.width = dw;
    detectCanvas.height = dh;
    detectCtx.drawImage(video, 0, 0, dw, dh);

    // Zone bounds in detect coords
    const zx = Math.round(dw * (1 - CFG.zoneW) / 2);
    const zy = Math.round(dh * (1 - CFG.zoneH) / 2);
    const zw = Math.round(dw * CFG.zoneW);
    const zh = Math.round(dh * CFG.zoneH);

    if (zw <= 0 || zh <= 0) return null;

    const imgData = detectCtx.getImageData(zx, zy, zw, zh);
    const px = imgData.data;
    const total = zw * zh;

    // Convert to grayscale
    const gray = new Float32Array(total);
    for (let i = 0; i < total; i++) {
        const off = i * 4;
        gray[i] = 0.299 * px[off] + 0.587 * px[off + 1] + 0.114 * px[off + 2];
    }

    // ---- Edge density via Sobel ----
    let edgeCount = 0;
    const edgeThreshold = 60;
    for (let y = 1; y < zh - 1; y++) {
        for (let x = 1; x < zw - 1; x++) {
            const idx = y * zw + x;
            // Sobel X
            const gx = -gray[idx - zw - 1] + gray[idx - zw + 1]
                       -2 * gray[idx - 1] + 2 * gray[idx + 1]
                       -gray[idx + zw - 1] + gray[idx + zw + 1];
            // Sobel Y
            const gy = -gray[idx - zw - 1] - 2 * gray[idx - zw] - gray[idx - zw + 1]
                       +gray[idx + zw - 1] + 2 * gray[idx + zw] + gray[idx + zw + 1];
            const mag = Math.sqrt(gx * gx + gy * gy);
            if (mag > edgeThreshold) edgeCount++;
        }
    }
    const edgeDensity = edgeCount / total;

    // ---- Sharpness via Laplacian variance ----
    let lapSum = 0, lapSumSq = 0;
    let lapCount = 0;
    for (let y = 1; y < zh - 1; y++) {
        for (let x = 1; x < zw - 1; x++) {
            const idx = y * zw + x;
            const lap = gray[idx - zw] + gray[idx + zw] + gray[idx - 1] + gray[idx + 1] - 4 * gray[idx];
            lapSum += lap;
            lapSumSq += lap * lap;
            lapCount++;
        }
    }
    const lapMean = lapSum / lapCount;
    const sharpness = (lapSumSq / lapCount) - (lapMean * lapMean);

    // ---- Contrast: center vs border brightness ----
    // Center 40% of zone
    const cx1 = Math.round(zw * 0.3);
    const cy1 = Math.round(zh * 0.3);
    const cx2 = Math.round(zw * 0.7);
    const cy2 = Math.round(zh * 0.7);
    let centerSum = 0, centerCount = 0;
    let borderSum = 0, borderCount = 0;
    for (let y = 0; y < zh; y++) {
        for (let x = 0; x < zw; x++) {
            const v = gray[y * zw + x];
            if (x >= cx1 && x < cx2 && y >= cy1 && y < cy2) {
                centerSum += v;
                centerCount++;
            } else {
                borderSum += v;
                borderCount++;
            }
        }
    }
    const centerMean = centerCount ? centerSum / centerCount : 128;
    const borderMean = borderCount ? borderSum / borderCount : 128;
    const contrast = Math.abs(centerMean - borderMean);

    return { edgeDensity, sharpness, contrast };
}

// ============================================================
// Capture at full resolution
// ============================================================
function captureCard() {
    if (captures.length >= CFG.total) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Capture the detection zone at full resolution
    const zx = Math.round(vw * (1 - CFG.zoneW) / 2);
    const zy = Math.round(vh * (1 - CFG.zoneH) / 2);
    const zw = Math.round(vw * CFG.zoneW);
    const zh = Math.round(vh * CFG.zoneH);

    captureCanvas.width = zw;
    captureCanvas.height = zh;
    captureCtx.drawImage(video, zx, zy, zw, zh, 0, 0, zw, zh);

    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);
    const col = currentColInRow;
    const row = currentRow;

    captures.push({ dataUrl, row, col });
    currentColInRow++;

    // Flash
    triggerFlash();

    // Update UI
    buildThumbs();

    // Scroll strip to latest
    const thumbsEl = document.getElementById('stripThumbs');
    thumbsEl.scrollLeft = thumbsEl.scrollWidth;

    // Haptic feedback if available
    if (navigator.vibrate) navigator.vibrate(50);

    // Check row completion
    if (currentColInRow >= CFG.cols) {
        if (currentRow >= CFG.rows - 1) {
            // All rows done
            setStatus('All cards captured!', true);
            document.getElementById('btnDone').style.display = 'inline-block';
            document.getElementById('btnManual').style.display = 'none';
            scanning = false;
        } else {
            // Row complete — prompt for next
            showRowComplete();
        }
    }

    // Cooldown — prevent double capture
    readyCount = 0;
    cooldown = true;
    cooldownFrames = 0;
    setMeter(0);
}

function triggerFlash() {
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 120);
}

// ============================================================
// Manual capture
// ============================================================
function manualCapture() {
    if (captures.length >= CFG.total) return;
    // Force capture regardless of detection
    readyCount = CFG.readyFramesNeeded;
    captureCard();
}

// ============================================================
// Row transitions
// ============================================================
function showRowComplete() {
    scanning = false;
    const ovl = document.getElementById('rowOverlay');
    document.getElementById('rowOverlayTitle').textContent = 'Row ' + (currentRow + 1) + ' Complete';

    if (currentRow >= CFG.rows - 1) {
        document.getElementById('rowOverlayText').textContent = 'All rows scanned!';
    } else {
        document.getElementById('rowOverlayText').textContent =
            'Move your phone to the start of row ' + (currentRow + 2) + ', then tap Continue.';
    }
    ovl.classList.add('visible');
}

function continueNextRow() {
    document.getElementById('rowOverlay').classList.remove('visible');
    currentRow++;
    currentColInRow = 0;
    readyCount = 0;
    cooldown = false;
    cooldownFrames = 0;
    scanning = true;
    buildUI();
    setStatus('Slide across row ' + (currentRow + 1) + '...', false);
    setMeter(0);
    animFrameId = requestAnimationFrame(detectionLoop);
}

function nextRow() {
    showRowComplete();
}

// ============================================================
// Done / submit
// ============================================================
function showDone() {
    scanning = false;
    const ovl = document.getElementById('doneOverlay');
    document.getElementById('doneSummary').textContent = captures.length + ' cards captured';

    const grid = document.getElementById('doneGrid');
    grid.innerHTML = '';
    // Display in grid layout
    for (let i = 0; i < captures.length; i++) {
        const img = document.createElement('img');
        img.src = captures[i].dataUrl;
        img.onclick = () => {
            // Tap to remove & re-scan
            if (confirm('Remove card ' + (i + 1) + ' and re-scan?')) {
                captures.splice(i, 1);
                // Recalculate row/col positions
                recalcPositions();
                showDone();
            }
        };
        grid.appendChild(img);
    }
    // Adjust grid columns to match CFG.cols
    grid.style.gridTemplateColumns = 'repeat(' + CFG.cols + ', 1fr)';
    ovl.classList.add('visible');
}

function recalcPositions() {
    for (let i = 0; i < captures.length; i++) {
        captures[i].row = Math.floor(i / CFG.cols);
        captures[i].col = i % CFG.cols;
    }
    currentRow = captures.length > 0 ? Math.floor((captures.length - 1) / CFG.cols) : 0;
    currentColInRow = captures.length % CFG.cols;
}

async function submitCards() {
    const btn = document.getElementById('btnSubmit');
    const statusEl = document.getElementById('uploadStatus');
    btn.disabled = true;
    statusEl.textContent = 'Identifying ' + captures.length + ' cards...';

    try {
        // Send individual card images to /slide-scan/identify
        const formData = new FormData();
        for (let i = 0; i < captures.length; i++) {
            const cap = captures[i];
            // Convert dataUrl to blob
            const resp0 = await fetch(cap.dataUrl);
            const blob = await resp0.blob();
            // Use grid position as field name: card_0 through card_8
            const pos = cap.row * CFG.cols + cap.col;
            formData.append('card_' + pos, blob, 'card_' + pos + '.jpg');
        }

        const resp = await fetch('/slide-scan/identify', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Identification failed: ' + resp.status);
        const data = await resp.json();

        if (data.error) {
            statusEl.textContent = 'Error: ' + data.error;
            btn.disabled = false;
            return;
        }

        // Show results inline in the done overlay
        displayResults(data, statusEl);
    } catch (err) {
        statusEl.textContent = 'Error: ' + err.message;
        btn.disabled = false;
    }
}

function displayResults(data, statusEl) {
    const grid = document.getElementById('doneGrid');
    grid.innerHTML = '';

    // Show total value
    const totalNM = data.total_value || 0;
    const totalMP = data.total_mp || 0;
    statusEl.innerHTML = '<span style="color:#4ecca3;font-size:20px;font-weight:700">NM $' +
        totalNM.toFixed(2) + '</span>' +
        (totalMP > 0 ? ' &nbsp; <span style="color:#ccc;font-size:16px">MP $' + totalMP.toFixed(2) + '</span>' : '') +
        '<br><span style="font-size:12px;color:rgba(255,255,255,0.5)">' +
        data.total_cards + ' cards identified</span>';

    // Display result cards in grid
    (data.cards || []).forEach((card, i) => {
        const el = document.createElement('div');
        el.style.cssText = 'position:relative;text-align:center;';

        const imgSrc = card.local_image_url || card.image_url || '';
        const price = card.variant_price || card.market_price;
        const name = card.card_name || card.card_id || '???';
        const variant = card.detected_variant && card.detected_variant !== 'normal'
            ? card.detected_variant.replace(/_/g, ' ') : '';

        el.innerHTML =
            (imgSrc ? '<img src="' + imgSrc + '" style="width:100%;aspect-ratio:63/88;object-fit:cover;border-radius:4px;border:1px solid #333">' : '') +
            '<div style="font-size:10px;margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + name + '</div>' +
            (variant ? '<div style="font-size:9px;color:#e94560">' + variant + '</div>' : '') +
            (price ? '<div style="font-size:11px;color:#4ecca3;font-weight:700">$' + price.toFixed(2) + '</div>' : '') +
            '<div style="font-size:9px;color:rgba(255,255,255,0.4)">' + (card.confidence * 100).toFixed(0) + '%</div>';
        grid.appendChild(el);
    });

    // Adjust grid columns
    grid.style.gridTemplateColumns = 'repeat(' + CFG.cols + ', 1fr)';
    grid.style.maxWidth = '400px';
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

// ============================================================
// Settings
// ============================================================
function toggleSettings() {
    document.getElementById('settingsPanel').classList.toggle('visible');
}

function applySettings() {
    CFG.cols = parseInt(document.getElementById('setCols').value) || 3;
    CFG.rows = parseInt(document.getElementById('setRows').value) || 3;
    CFG.debug = document.getElementById('setDebug').value === '1';

    document.getElementById('debugInfo').className = 'debug-info' + (CFG.debug ? ' visible' : '');

    // Reset scan
    captures = [];
    currentRow = 0;
    currentColInRow = 0;
    readyCount = 0;
    cooldown = false;
    cooldownFrames = 0;

    buildUI();
    toggleSettings();
    setStatus('Slide across cards slowly...', false);
    setMeter(0);

    // Restart scanning if stopped
    if (!scanning) {
        scanning = true;
        document.getElementById('doneOverlay').classList.remove('visible');
        document.getElementById('rowOverlay').classList.remove('visible');
        document.getElementById('btnDone').style.display = 'none';
        document.getElementById('btnManual').style.display = 'inline-block';
        animFrameId = requestAnimationFrame(detectionLoop);
    }
}

// ============================================================
// Prevent pinch zoom on iOS
// ============================================================
document.addEventListener('gesturestart', e => e.preventDefault());
document.addEventListener('touchmove', e => {
    if (e.touches.length > 1) e.preventDefault();
}, { passive: false });

</script>
</body>
</html>
"""
