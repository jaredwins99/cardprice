"""Manual-capture card scanning UI with live card detection overlay.

User holds phone over a single card at a time. Live video shows a card outline
(green = good framing, red = not aligned). User taps the big shutter button
to capture. Frame freezes showing the detected card region. User taps Accept
to save or Retake to try again. Auto-crops to card edges.

Row/slot guidance:
    - Grid shows which slot is being captured next
    - Filmstrip with thumbnails of captured cards
    - Completion screen with 3x3 grid + submit

Integration into server.py:
    GET  /slide-scan       -> serve this HTML
    POST /slide-scan       -> receive 9 card images, identify, return JSON
"""

SLIDE_SCAN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Card Scanner</title>
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

/* Detection zone overlay canvas */
canvas#overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    pointer-events: none;
}

/* Frozen frame canvas (shown during freeze) */
canvas#freezeCanvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: 3;
    display: none;
}

/* ================================================================ */
/*  TOP HUD: slot indicator + card counter                           */
/* ================================================================ */
.top-hud {
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to bottom, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.55) 80%, transparent 100%);
    padding: 10px 14px 24px;
    pointer-events: none;
}

.slot-label {
    font-size: 16px;
    font-weight: 700;
    text-align: center;
    color: rgba(255,255,255,0.95);
    margin-bottom: 6px;
}

.card-counter {
    text-align: center;
    font-size: 13px;
    color: rgba(255,255,255,0.65);
}

/* Slot grid mini-map */
.slot-grid {
    display: flex;
    justify-content: center;
    gap: 4px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.slot-grid-cell {
    width: 28px;
    height: 38px;
    border-radius: 3px;
    border: 2px solid rgba(255,255,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 8px;
    color: rgba(255,255,255,0.3);
    transition: all 0.3s;
}
.slot-grid-cell.active {
    border-color: #e94560;
    border-style: solid;
    background: rgba(233, 69, 96, 0.2);
    color: #e94560;
    font-weight: 700;
}
.slot-grid-cell.filled {
    border-color: #4ecca3;
    background: rgba(78, 204, 163, 0.2);
    color: #4ecca3;
}

/* ================================================================ */
/*  FILMSTRIP: bottom thumbnail bar                                  */
/* ================================================================ */
.filmstrip {
    position: absolute;
    bottom: 110px; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%);
    padding: 20px 10px 8px;
    pointer-events: none;
}

.strip-thumbs {
    display: flex;
    gap: 4px;
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
    justify-content: center;
}
.strip-thumbs::-webkit-scrollbar { display: none; }

.strip-slot {
    flex-shrink: 0;
    width: 36px;
    height: 50px;
    border-radius: 3px;
    border: 1.5px dashed rgba(255,255,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    color: rgba(255,255,255,0.3);
}
.strip-slot.active-slot {
    border-color: #e94560;
    border-style: solid;
}

.strip-thumb {
    flex-shrink: 0;
    width: 36px;
    height: 50px;
    border-radius: 3px;
    border: 1.5px solid #4ecca3;
    object-fit: cover;
    cursor: pointer;
    pointer-events: auto;
}

/* ================================================================ */
/*  SHUTTER BUTTON AREA                                              */
/* ================================================================ */
.shutter-area {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 110px;
    z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.6) 70%, transparent 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
}

.shutter-btn {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    border: 4px solid #fff;
    background: transparent;
    cursor: pointer;
    position: relative;
    transition: all 0.15s;
    -webkit-tap-highlight-color: transparent;
}
.shutter-btn::after {
    content: '';
    position: absolute;
    top: 4px; left: 4px; right: 4px; bottom: 4px;
    border-radius: 50%;
    background: #fff;
    transition: all 0.15s;
}
.shutter-btn:active {
    transform: scale(0.9);
}
.shutter-btn:active::after {
    background: #ccc;
}
.shutter-btn.disabled {
    opacity: 0.3;
    pointer-events: none;
}

/* Skip / Done small buttons flanking shutter */
.shutter-side-btn {
    padding: 8px 14px;
    border: 1px solid rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.7);
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    min-width: 56px;
    text-align: center;
}
.shutter-side-btn:active {
    background: rgba(255,255,255,0.2);
}
.shutter-side-btn.hidden { visibility: hidden; }

/* ================================================================ */
/*  FREEZE-FRAME CONFIRM OVERLAY                                     */
/* ================================================================ */
.freeze-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 15;
    display: none;
    flex-direction: column;
    pointer-events: none;
}
.freeze-overlay.visible {
    display: flex;
    pointer-events: auto;
}

.freeze-top-spacer { flex: 1; }

.freeze-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 40px;
    padding: 20px 0 40px;
    background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.6) 70%, transparent 100%);
}

.freeze-btn {
    padding: 14px 28px;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.1s;
}
.freeze-btn:active { transform: scale(0.95); }

.freeze-btn-retake {
    background: rgba(255,255,255,0.15);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.3);
}
.freeze-btn-accept {
    background: #4ecca3;
    color: #1a1a2e;
}

/* ================================================================ */
/*  GREEN FLASH (capture feedback)                                   */
/* ================================================================ */
.flash {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(78, 204, 163, 0.35);
    z-index: 20;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
}
.flash.active {
    opacity: 1;
    transition: none;
}

/* ================================================================ */
/*  DONE / COMPLETION OVERLAY                                        */
/* ================================================================ */
.done-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 25;
    background: #1a1a2e;
    display: none;
    flex-direction: column;
    align-items: center;
    overflow-y: auto;
    padding: 20px 16px 40px;
}
.done-overlay.visible { display: flex; }
.done-overlay h2 {
    font-size: 22px;
    color: #4ecca3;
    margin-bottom: 4px;
}
.done-overlay .done-sub {
    color: rgba(255,255,255,0.6);
    font-size: 13px;
    margin-bottom: 12px;
}

.done-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    max-width: 320px;
    width: 100%;
    margin-bottom: 16px;
}
.done-cell {
    position: relative;
    cursor: pointer;
}
.done-cell img {
    width: 100%;
    aspect-ratio: 63/88;
    object-fit: cover;
    border-radius: 6px;
    border: 2px solid #4ecca3;
}
.done-cell .cell-label {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: rgba(0,0,0,0.65);
    color: #fff;
    font-size: 10px;
    text-align: center;
    padding: 2px 0;
    border-radius: 0 0 6px 6px;
}
.done-cell .cell-rescan {
    position: absolute;
    top: 4px; right: 4px;
    background: rgba(233, 69, 96, 0.85);
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 9px;
    padding: 2px 6px;
    cursor: pointer;
    font-weight: 600;
    opacity: 0;
    transition: opacity 0.2s;
}
.done-cell:active .cell-rescan { opacity: 1; }

.btn-submit {
    padding: 16px 40px;
    background: #4ecca3;
    color: #1a1a2e;
    border: none;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 700;
    cursor: pointer;
    width: 100%;
    max-width: 320px;
}
.btn-submit:active { transform: scale(0.97); }
.btn-submit:disabled { opacity: 0.5; cursor: default; }

.btn-rescan-all {
    padding: 10px 20px;
    background: transparent;
    color: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
    margin-top: 8px;
    width: 100%;
    max-width: 320px;
    text-align: center;
}

/* ================================================================ */
/*  IDENTIFYING OVERLAY (spinner + timer)                            */
/* ================================================================ */
.id-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 30;
    background: rgba(26, 26, 46, 0.95);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.id-overlay.visible { display: flex; }

.spinner-ring {
    width: 48px;
    height: 48px;
    border: 4px solid rgba(255,255,255,0.15);
    border-top-color: #e94560;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.id-timer {
    font-size: 28px;
    font-weight: 700;
    color: #fff;
}
.id-label {
    font-size: 14px;
    color: rgba(255,255,255,0.55);
}

/* ================================================================ */
/*  RESULTS OVERLAY                                                  */
/* ================================================================ */
.results-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 35;
    background: #1a1a2e;
    display: none;
    flex-direction: column;
    overflow-y: auto;
    padding: 16px 12px 40px;
}
.results-overlay.visible { display: flex; }

.results-header {
    text-align: center;
    margin-bottom: 10px;
}
.results-header h2 {
    font-size: 20px;
    color: #e94560;
}

.results-summary {
    background: #16213e;
    border-radius: 12px;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.results-summary .total-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
}
.results-summary .total-value {
    font-size: 22px;
    font-weight: 700;
    color: #4ecca3;
}
.results-summary .card-count {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
}

.result-card {
    background: #16213e;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 6px;
    display: flex;
    gap: 10px;
    align-items: center;
}
.result-card .rc-imgs {
    display: flex;
    gap: 4px;
    flex-shrink: 0;
}
.result-card .rc-img {
    width: 46px;
    height: 64px;
    border-radius: 4px;
    object-fit: cover;
    background: #0f1629;
}
.result-card .rc-info {
    flex: 1;
    min-width: 0;
}
.result-card .rc-name {
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.result-card .rc-name a {
    color: #eee;
    text-decoration: none;
}
.result-card .rc-set {
    font-size: 11px;
    color: rgba(255,255,255,0.5);
}
.result-card .rc-meta {
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    margin-top: 1px;
}
.result-card .rc-price {
    font-size: 15px;
    font-weight: 700;
    color: #4ecca3;
    flex-shrink: 0;
}
.result-card .rc-price.no-price {
    color: rgba(255,255,255,0.25);
}

.variant-badge {
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 4px;
    border-radius: 3px;
    margin-left: 4px;
    vertical-align: middle;
    background: #444;
    color: #fff;
}
.variant-badge.first-edition  { background: #b8860b; }
.variant-badge.reverse-holo   { background: #2e86de; }
.variant-badge.stamped         { background: #8e44ad; }
.variant-badge.promo           { background: #e67e22; }
.variant-badge.prerelease      { background: #16a085; }
.variant-badge.shadowless      { background: #6a5acd; }

.btn-new-scan {
    display: block;
    width: 100%;
    padding: 14px;
    font-size: 15px;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    background: #e94560;
    color: #fff;
    cursor: pointer;
    margin-top: 12px;
}
.btn-new-scan:active {
    background: #c23152;
}

/* ================================================================ */
/*  SETTINGS                                                         */
/* ================================================================ */
.settings-btn {
    position: absolute;
    top: 8px;
    right: 10px;
    z-index: 11;
    background: none;
    border: none;
    color: rgba(255,255,255,0.5);
    font-size: 20px;
    cursor: pointer;
    padding: 6px;
    pointer-events: auto;
}
.settings-panel {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 40;
    background: rgba(0,0,0,0.92);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 18px;
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
    max-width: 280px;
    justify-content: space-between;
}
.setting-row label {
    font-size: 15px;
    color: rgba(255,255,255,0.8);
}
.setting-row input, .setting-row select {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #555;
    background: #222;
    color: #fff;
    font-size: 15px;
    width: 72px;
    text-align: center;
}

/* Debug info */
.debug-info {
    position: absolute;
    top: 50%;
    right: 10px;
    z-index: 10;
    font-size: 9px;
    color: rgba(255,255,255,0.35);
    text-align: right;
    line-height: 1.4;
    pointer-events: none;
    display: none;
    font-family: monospace;
}
.debug-info.visible { display: block; }
</style>
</head>
<body>

<div class="camera-wrap">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <canvas id="freezeCanvas"></canvas>
    <canvas id="detect" class="hidden-canvas"></canvas>
    <canvas id="capture" class="hidden-canvas"></canvas>

    <!-- ======== TOP HUD ======== -->
    <div class="top-hud" id="topHud">
        <div class="slot-label" id="slotLabel">Card 1 of 9</div>
        <div class="slot-grid" id="slotGrid"></div>
        <div class="card-counter" id="cardCounter">0 captured</div>
    </div>

    <!-- ======== FILMSTRIP ======== -->
    <div class="filmstrip" id="filmstrip">
        <div class="strip-thumbs" id="stripThumbs"></div>
    </div>

    <!-- ======== SHUTTER AREA ======== -->
    <div class="shutter-area" id="shutterArea">
        <button class="shutter-side-btn" id="btnSkip" onclick="skipSlot()">Skip</button>
        <button class="shutter-btn" id="shutterBtn" onclick="onShutterTap()"></button>
        <button class="shutter-side-btn" id="btnDone" onclick="showDone()">Done</button>
    </div>

    <!-- ======== FREEZE-FRAME CONFIRM ======== -->
    <div class="freeze-overlay" id="freezeOverlay">
        <div class="freeze-top-spacer"></div>
        <div class="freeze-controls">
            <button class="freeze-btn freeze-btn-retake" onclick="retakeCapture()">Retake</button>
            <button class="freeze-btn freeze-btn-accept" onclick="acceptCapture()">Accept</button>
        </div>
    </div>

    <!-- ======== GREEN FLASH ======== -->
    <div class="flash" id="flash"></div>

    <!-- ======== COMPLETION / REVIEW OVERLAY ======== -->
    <div class="done-overlay" id="doneOverlay">
        <h2>Review Cards</h2>
        <p class="done-sub" id="doneSummary">9 cards captured &mdash; tap any to re-scan</p>
        <div class="done-grid" id="doneGrid"></div>
        <button class="btn-submit" id="btnSubmit" onclick="submitCards()">Submit for Identification</button>
        <button class="btn-rescan-all" onclick="restartScan()">Start Over</button>
    </div>

    <!-- ======== IDENTIFYING OVERLAY ======== -->
    <div class="id-overlay" id="idOverlay">
        <div class="spinner-ring"></div>
        <div class="id-timer" id="idTimer">0s</div>
        <div class="id-label">Identifying cards...</div>
    </div>

    <!-- ======== RESULTS OVERLAY ======== -->
    <div class="results-overlay" id="resultsOverlay">
        <div class="results-header"><h2>Scan Results</h2></div>
        <div class="results-summary" id="resultsSummary">
            <div>
                <div class="total-label">Page Total</div>
                <div class="total-value" id="resTotalValue">$0.00</div>
            </div>
            <div style="text-align:right">
                <div class="card-count" id="resCardCount">0 cards</div>
            </div>
        </div>
        <div id="resultsCards"></div>
        <button class="btn-new-scan" onclick="restartFull()">Scan Another Page</button>
    </div>

    <!-- ======== SETTINGS ======== -->
    <button class="settings-btn" onclick="toggleSettings()">&#9881;</button>
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
            <select id="setDebug"><option value="0">Off</option><option value="1">On</option></select>
        </div>
        <button class="freeze-btn freeze-btn-accept" style="margin-top:10px" onclick="applySettings()">Apply</button>
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

    // Detection zone: card-shaped region in center of frame
    zoneW: 0.55,
    zoneH: 0.65,

    // Card detection thresholds (for edge-finding auto-crop)
    edgeDensityMin: 0.08,
    cannyHigh: 100,

    // Detection canvas width (for speed)
    detectWidth: 320,

    debug: false,
};

// ============================================================
// State
// ============================================================
let video, overlay, freezeCanvas, detectCanvas, captureCanvas;
let overlayCtx, freezeCtx, detectCtx, captureCtx;
let captures = [];           // array of {dataUrl, row, col} indexed by grid position
let activeSlot = 0;          // which slot we're capturing into next
let scanning = true;
let frozen = false;          // freeze-frame active
let animFrameId = null;
let pendingCropData = null;  // dataUrl waiting for accept/retake
let cardRect = null;         // detected card rect in video coords {x,y,w,h} or null
let frameCount = 0;

// Audio context for shutter sound
let audioCtx = null;

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', init);

async function init() {
    video = document.getElementById('video');
    overlay = document.getElementById('overlay');
    freezeCanvas = document.getElementById('freezeCanvas');
    detectCanvas = document.getElementById('detect');
    captureCanvas = document.getElementById('capture');

    overlayCtx = overlay.getContext('2d');
    freezeCtx = freezeCanvas.getContext('2d');
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

        await new Promise((resolve, reject) => {
            video.addEventListener('loadeddata', resolve, { once: true });
            video.addEventListener('error', reject, { once: true });
            setTimeout(() => resolve(), 10000);
        });

        await video.play();

        setTimeout(() => {
            resizeOverlay();
            window.addEventListener('resize', resizeOverlay);
            animFrameId = requestAnimationFrame(liveLoop);
        }, 200);
    } catch (err) {
        document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#e74c3c;font-size:18px;">Camera Error<br><br><span style="font-size:14px;color:#888;">' + err.message + '<br><br>Make sure you allow camera access.<br>On some devices, HTTPS is required.</span></div>';
    }
}

function resizeOverlay() {
    const r = video.getBoundingClientRect();
    overlay.width = r.width;
    overlay.height = r.height;
    freezeCanvas.width = r.width;
    freezeCanvas.height = r.height;
}

// ============================================================
// UI builders
// ============================================================
function buildUI() {
    buildHud();
    buildThumbs();
    updateShutterState();
}

function buildHud() {
    const captured = captures.filter(Boolean).length;
    document.getElementById('slotLabel').textContent =
        'Card ' + (activeSlot + 1) + ' of ' + CFG.total;
    document.getElementById('cardCounter').textContent =
        captured + ' captured';

    // Slot grid mini-map
    const grid = document.getElementById('slotGrid');
    grid.innerHTML = '';
    for (let i = 0; i < CFG.total; i++) {
        const cell = document.createElement('div');
        cell.className = 'slot-grid-cell';
        if (captures[i]) {
            cell.classList.add('filled');
        } else if (i === activeSlot && scanning) {
            cell.classList.add('active');
        }
        const r = Math.floor(i / CFG.cols) + 1;
        const c = (i % CFG.cols) + 1;
        cell.textContent = r + '.' + c;
        grid.appendChild(cell);
    }
}

function buildThumbs() {
    const container = document.getElementById('stripThumbs');
    container.innerHTML = '';

    for (let i = 0; i < CFG.total; i++) {
        const cap = captures[i];
        if (cap) {
            const img = document.createElement('img');
            img.className = 'strip-thumb';
            img.src = cap.dataUrl;
            img.onclick = () => rescanSlot(i);
            container.appendChild(img);
        } else {
            const slot = document.createElement('div');
            slot.className = 'strip-slot';
            if (i === activeSlot && scanning) {
                slot.classList.add('active-slot');
            }
            const r = Math.floor(i / CFG.cols) + 1;
            const c = (i % CFG.cols) + 1;
            slot.textContent = r + '.' + c;
            container.appendChild(slot);
        }
    }
}

function updateShutterState() {
    const allDone = captures.filter(Boolean).length >= CFG.total;
    const shutterBtn = document.getElementById('shutterBtn');
    shutterBtn.className = 'shutter-btn' + (allDone ? ' disabled' : '');
}

// ============================================================
// Live detection loop: draws card outline on video
// ============================================================
function liveLoop() {
    if (!scanning || frozen) return;

    frameCount++;
    // Only run edge detection every 3rd frame for performance
    if (frameCount % 3 === 0) {
        detectCardEdges();
    }
    drawOverlay();

    animFrameId = requestAnimationFrame(liveLoop);
}

// ============================================================
// Card edge detection via Sobel + projection histograms
// ============================================================
function detectCardEdges() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) { cardRect = null; return; }

    const scale = CFG.detectWidth / vw;
    const dw = CFG.detectWidth;
    const dh = Math.round(vh * scale);
    detectCanvas.width = dw;
    detectCanvas.height = dh;
    detectCtx.drawImage(video, 0, 0, dw, dh);

    // Analyze center detection zone
    const zx = Math.round(dw * (1 - CFG.zoneW) / 2);
    const zy = Math.round(dh * (1 - CFG.zoneH) / 2);
    const zw = Math.round(dw * CFG.zoneW);
    const zh = Math.round(dh * CFG.zoneH);
    if (zw <= 0 || zh <= 0) { cardRect = null; return; }

    const imgData = detectCtx.getImageData(zx, zy, zw, zh);
    const px = imgData.data;
    const total = zw * zh;

    // Convert to grayscale
    const gray = new Uint8Array(total);
    for (let i = 0; i < total; i++) {
        const off = i * 4;
        gray[i] = Math.round(0.299 * px[off] + 0.587 * px[off + 1] + 0.114 * px[off + 2]);
    }

    // Sobel edge detection
    const edges = new Uint8Array(total);
    let edgeCount = 0;
    for (let y = 1; y < zh - 1; y++) {
        for (let x = 1; x < zw - 1; x++) {
            const idx = y * zw + x;
            const gx = -gray[idx - zw - 1] + gray[idx - zw + 1]
                       -2 * gray[idx - 1] + 2 * gray[idx + 1]
                       -gray[idx + zw - 1] + gray[idx + zw + 1];
            const gy = -gray[idx - zw - 1] - 2 * gray[idx - zw] - gray[idx - zw + 1]
                       +gray[idx + zw - 1] + 2 * gray[idx + zw] + gray[idx + zw + 1];
            const mag = Math.sqrt(gx * gx + gy * gy);
            edges[idx] = mag > CFG.cannyHigh ? 255 : 0;
            if (edges[idx]) edgeCount++;
        }
    }

    const edgeDensity = edgeCount / total;
    let hasEdges = edgeDensity >= CFG.edgeDensityMin;

    if (hasEdges) {
        // Use projection histograms to find card boundaries
        const colHist = new Float32Array(zw);
        const rowHist = new Float32Array(zh);
        for (let y = 1; y < zh - 1; y++) {
            for (let x = 1; x < zw - 1; x++) {
                if (edges[y * zw + x]) {
                    colHist[x]++;
                    rowHist[y]++;
                }
            }
        }

        const colThresh = zh * 0.03;
        const rowThresh = zw * 0.03;

        let minX = zw, maxX = 0, minY = zh, maxY = 0;

        for (let x = 0; x < zw; x++) {
            if (colHist[x] >= colThresh) { minX = x; break; }
        }
        for (let x = zw - 1; x >= 0; x--) {
            if (colHist[x] >= colThresh) { maxX = x; break; }
        }
        for (let y = 0; y < zh; y++) {
            if (rowHist[y] >= rowThresh) { minY = y; break; }
        }
        for (let y = zh - 1; y >= 0; y--) {
            if (rowHist[y] >= rowThresh) { maxY = y; break; }
        }

        const bw = maxX - minX;
        const bh = maxY - minY;
        if (bw > 20 && bh > 20) {
            const aspect = bw / bh;
            if (aspect > 0.4 && aspect < 1.1) {
                const invScale = 1 / scale;
                cardRect = {
                    x: (zx + minX) * invScale,
                    y: (zy + minY) * invScale,
                    w: bw * invScale,
                    h: bh * invScale,
                    quality: Math.min(1, edgeDensity / 0.2),
                    aspect: aspect,
                };
            } else {
                cardRect = null;
            }
        } else {
            cardRect = null;
        }
    } else {
        cardRect = null;
    }

    if (CFG.debug) {
        updateDebug({
            edges: edgeDensity,
            hasCard: !!cardRect,
            aspect: cardRect ? cardRect.aspect : 0,
            quality: cardRect ? cardRect.quality : 0,
        });
    }
}

// ============================================================
// Draw overlay: card outline on live video
// ============================================================
function drawOverlay() {
    const W = overlay.width;
    const H = overlay.height;
    overlayCtx.clearRect(0, 0, W, H);

    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;

    // Determine object-fit: cover scaling
    const videoAspect = vw / vh;
    const dispAspect = W / H;
    let sx, sy;
    if (videoAspect > dispAspect) {
        sy = H / vh;
        sx = sy;
    } else {
        sx = W / vw;
        sy = sx;
    }
    const offsetX = (W - vw * sx) / 2;
    const offsetY = (H - vh * sy) / 2;

    if (cardRect) {
        // Draw detected card outline
        const rx = cardRect.x * sx + offsetX;
        const ry = cardRect.y * sy + offsetY;
        const rw = cardRect.w * sx;
        const rh = cardRect.h * sy;

        const good = cardRect.quality > 0.3;
        const color = good ? 'rgba(78, 204, 163, 0.9)' : 'rgba(233, 69, 96, 0.7)';

        // Semi-transparent dim outside the card
        overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        overlayCtx.fillRect(0, 0, W, ry);
        overlayCtx.fillRect(0, ry + rh, W, H - ry - rh);
        overlayCtx.fillRect(0, ry, rx, rh);
        overlayCtx.fillRect(rx + rw, ry, W - rx - rw, rh);

        // Card outline
        overlayCtx.strokeStyle = color;
        overlayCtx.lineWidth = 3;
        roundRect(overlayCtx, rx, ry, rw, rh, 8);
        overlayCtx.stroke();

        // Corner brackets
        const bLen = 24;
        overlayCtx.lineWidth = 4;
        overlayCtx.strokeStyle = color;

        overlayCtx.beginPath();
        overlayCtx.moveTo(rx, ry + bLen); overlayCtx.lineTo(rx, ry); overlayCtx.lineTo(rx + bLen, ry);
        overlayCtx.stroke();
        overlayCtx.beginPath();
        overlayCtx.moveTo(rx + rw - bLen, ry); overlayCtx.lineTo(rx + rw, ry); overlayCtx.lineTo(rx + rw, ry + bLen);
        overlayCtx.stroke();
        overlayCtx.beginPath();
        overlayCtx.moveTo(rx, ry + rh - bLen); overlayCtx.lineTo(rx, ry + rh); overlayCtx.lineTo(rx + bLen, ry + rh);
        overlayCtx.stroke();
        overlayCtx.beginPath();
        overlayCtx.moveTo(rx + rw - bLen, ry + rh); overlayCtx.lineTo(rx + rw, ry + rh); overlayCtx.lineTo(rx + rw, ry + rh - bLen);
        overlayCtx.stroke();

        // Status label
        overlayCtx.font = '600 13px -apple-system, sans-serif';
        overlayCtx.textAlign = 'center';
        overlayCtx.fillStyle = color;
        overlayCtx.fillText(good ? 'Card detected' : 'Align card', rx + rw / 2, ry - 10);

    } else {
        // No card detected: show guide zone with dashed outline
        const zw = W * CFG.zoneW;
        const zh = H * CFG.zoneH;
        const zx = (W - zw) / 2;
        const zy = (H - zh) / 2;

        // Light dim outside
        overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        overlayCtx.fillRect(0, 0, W, zy);
        overlayCtx.fillRect(0, zy + zh, W, H - zy - zh);
        overlayCtx.fillRect(0, zy, zx, zh);
        overlayCtx.fillRect(zx + zw, zy, W - zx - zw, zh);

        // Dashed guide rectangle
        overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        overlayCtx.lineWidth = 2;
        overlayCtx.setLineDash([8, 6]);
        roundRect(overlayCtx, zx, zy, zw, zh, 8);
        overlayCtx.stroke();
        overlayCtx.setLineDash([]);

        // Corner brackets
        const bLen = 20;
        overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        overlayCtx.lineWidth = 3;
        overlayCtx.beginPath();
        overlayCtx.moveTo(zx, zy + bLen); overlayCtx.lineTo(zx, zy); overlayCtx.lineTo(zx + bLen, zy);
        overlayCtx.stroke();
        overlayCtx.beginPath();
        overlayCtx.moveTo(zx + zw - bLen, zy); overlayCtx.lineTo(zx + zw, zy); overlayCtx.lineTo(zx + zw, zy + bLen);
        overlayCtx.stroke();
        overlayCtx.beginPath();
        overlayCtx.moveTo(zx, zy + zh - bLen); overlayCtx.lineTo(zx, zy + zh); overlayCtx.lineTo(zx + bLen, zy + zh);
        overlayCtx.stroke();
        overlayCtx.beginPath();
        overlayCtx.moveTo(zx + zw - bLen, zy + zh); overlayCtx.lineTo(zx + zw, zy + zh); overlayCtx.lineTo(zx + zw, zy + zh - bLen);
        overlayCtx.stroke();

        // Instruction text
        overlayCtx.font = '600 14px -apple-system, sans-serif';
        overlayCtx.textAlign = 'center';
        overlayCtx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        overlayCtx.fillText('Center card in frame', W / 2, zy - 12);
    }
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

// ============================================================
// Shutter tap: freeze frame
// ============================================================
function onShutterTap() {
    if (frozen || !scanning) return;
    if (captures.filter(Boolean).length >= CFG.total) return;

    // Play shutter sound
    playShutterSound();

    // Haptic feedback
    if (navigator.vibrate) navigator.vibrate(40);

    // Flash
    triggerFlash();

    // Freeze the video frame onto the freeze canvas
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    const dispW = freezeCanvas.width;
    const dispH = freezeCanvas.height;

    // Calculate cover-fit source rect
    const videoAspect = vw / vh;
    const dispAspect = dispW / dispH;
    let srcX = 0, srcY = 0, srcW = vw, srcH = vh;
    if (videoAspect > dispAspect) {
        srcW = vh * dispAspect;
        srcX = (vw - srcW) / 2;
    } else {
        srcH = vw / dispAspect;
        srcY = (vh - srcH) / 2;
    }

    freezeCtx.drawImage(video, srcX, srcY, srcW, srcH, 0, 0, dispW, dispH);

    // Run one more edge detection on the frozen frame to get best crop
    detectCardEdges();

    // Draw the card detection overlay on the freeze canvas
    drawFreezeOverlay();

    // Crop to card region (or detection zone fallback)
    let cropX, cropY, cropW, cropH;
    if (cardRect) {
        const margin = 0.02;
        cropX = Math.max(0, Math.round(cardRect.x - cardRect.w * margin));
        cropY = Math.max(0, Math.round(cardRect.y - cardRect.h * margin));
        cropW = Math.min(vw - cropX, Math.round(cardRect.w * (1 + 2 * margin)));
        cropH = Math.min(vh - cropY, Math.round(cardRect.h * (1 + 2 * margin)));
    } else {
        cropX = Math.round(vw * (1 - CFG.zoneW) / 2);
        cropY = Math.round(vh * (1 - CFG.zoneH) / 2);
        cropW = Math.round(vw * CFG.zoneW);
        cropH = Math.round(vh * CFG.zoneH);
    }

    // Capture at full resolution
    captureCanvas.width = cropW;
    captureCanvas.height = cropH;
    captureCtx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    pendingCropData = captureCanvas.toDataURL('image/jpeg', 0.92);

    // Show freeze canvas
    freezeCanvas.style.display = 'block';
    frozen = true;

    // Show accept/retake controls
    document.getElementById('freezeOverlay').classList.add('visible');
    document.getElementById('shutterArea').style.display = 'none';
}

function drawFreezeOverlay() {
    const W = freezeCanvas.width;
    const H = freezeCanvas.height;
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;

    const videoAspect = vw / vh;
    const dispAspect = W / H;
    let sx, sy;
    if (videoAspect > dispAspect) {
        sy = H / vh; sx = sy;
    } else {
        sx = W / vw; sy = sx;
    }
    const offsetX = (W - vw * sx) / 2;
    const offsetY = (H - vh * sy) / 2;

    if (cardRect) {
        const rx = cardRect.x * sx + offsetX;
        const ry = cardRect.y * sy + offsetY;
        const rw = cardRect.w * sx;
        const rh = cardRect.h * sy;

        // Dim outside card
        freezeCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        freezeCtx.fillRect(0, 0, W, ry);
        freezeCtx.fillRect(0, ry + rh, W, H - ry - rh);
        freezeCtx.fillRect(0, ry, rx, rh);
        freezeCtx.fillRect(rx + rw, ry, W - rx - rw, rh);

        freezeCtx.strokeStyle = '#4ecca3';
        freezeCtx.lineWidth = 3;
        roundRect(freezeCtx, rx, ry, rw, rh, 8);
        freezeCtx.stroke();

        freezeCtx.font = '700 15px -apple-system, sans-serif';
        freezeCtx.textAlign = 'center';
        freezeCtx.fillStyle = '#4ecca3';
        freezeCtx.fillText('Card detected - cropped to edges', rx + rw / 2, ry - 12);
    } else {
        const zw = W * CFG.zoneW;
        const zh = H * CFG.zoneH;
        const zx = (W - zw) / 2;
        const zy = (H - zh) / 2;

        freezeCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        freezeCtx.fillRect(0, 0, W, zy);
        freezeCtx.fillRect(0, zy + zh, W, H - zy - zh);
        freezeCtx.fillRect(0, zy, zx, zh);
        freezeCtx.fillRect(zx + zw, zy, W - zx - zw, zh);

        freezeCtx.strokeStyle = 'rgba(233, 69, 96, 0.8)';
        freezeCtx.lineWidth = 3;
        roundRect(freezeCtx, zx, zy, zw, zh, 8);
        freezeCtx.stroke();

        freezeCtx.font = '700 14px -apple-system, sans-serif';
        freezeCtx.textAlign = 'center';
        freezeCtx.fillStyle = 'rgba(233, 69, 96, 0.9)';
        freezeCtx.fillText('No card edges found - using center crop', W / 2, zy - 12);
    }
}

// ============================================================
// Accept / Retake
// ============================================================
function acceptCapture() {
    if (!pendingCropData) return;

    const slotIdx = activeSlot;
    const row = Math.floor(slotIdx / CFG.cols);
    const col = slotIdx % CFG.cols;
    captures[slotIdx] = { dataUrl: pendingCropData, row: row, col: col };

    advanceToNextSlot();
    unfreezeVideo();
    buildUI();

    if (captures.filter(Boolean).length >= CFG.total) {
        setTimeout(() => showDone(), 400);
    }
}

function retakeCapture() {
    pendingCropData = null;
    unfreezeVideo();
}

function unfreezeVideo() {
    frozen = false;
    pendingCropData = null;
    freezeCanvas.style.display = 'none';
    document.getElementById('freezeOverlay').classList.remove('visible');
    document.getElementById('shutterArea').style.display = 'flex';
    animFrameId = requestAnimationFrame(liveLoop);
}

function advanceToNextSlot() {
    for (let i = 1; i <= CFG.total; i++) {
        const idx = (activeSlot + i) % CFG.total;
        if (!captures[idx]) {
            activeSlot = idx;
            return;
        }
    }
    activeSlot = CFG.total;
}

// ============================================================
// Skip slot
// ============================================================
function skipSlot() {
    advanceToNextSlot();
    buildUI();
}

// ============================================================
// Audio / haptic feedback
// ============================================================
function playShutterSound() {
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.type = 'square';
        osc.frequency.setValueAtTime(1200, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(400, audioCtx.currentTime + 0.06);

        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);

        osc.start(audioCtx.currentTime);
        osc.stop(audioCtx.currentTime + 0.08);
    } catch (e) {
        // Audio not available
    }
}

function triggerFlash() {
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 150);
}

// ============================================================
// Debug
// ============================================================
function updateDebug(info) {
    if (!CFG.debug) return;
    const el = document.getElementById('debugInfo');
    el.textContent = Object.entries(info).map(([k,v]) =>
        k + ': ' + (typeof v === 'number' ? v.toFixed(3) : v)).join('\n');
}

// ============================================================
// Completion / review screen
// ============================================================
function showDone() {
    scanning = false;
    const ovl = document.getElementById('doneOverlay');
    const count = captures.filter(Boolean).length;
    document.getElementById('doneSummary').textContent =
        count + ' cards captured \u2014 tap any to re-scan';

    const grid = document.getElementById('doneGrid');
    grid.innerHTML = '';

    for (let i = 0; i < CFG.total; i++) {
        const cell = document.createElement('div');
        cell.className = 'done-cell';

        if (captures[i]) {
            const img = document.createElement('img');
            img.src = captures[i].dataUrl;
            cell.appendChild(img);
        } else {
            const placeholder = document.createElement('div');
            placeholder.style.cssText =
                'width:100%;aspect-ratio:63/88;background:#16213e;border-radius:6px;' +
                'border:2px dashed rgba(255,255,255,0.2);display:flex;align-items:center;' +
                'justify-content:center;color:rgba(255,255,255,0.3);font-size:12px';
            placeholder.textContent = 'Empty';
            cell.appendChild(placeholder);
        }

        const label = document.createElement('div');
        label.className = 'cell-label';
        label.textContent = 'R' + (Math.floor(i / CFG.cols) + 1) +
                            ' C' + ((i % CFG.cols) + 1);
        cell.appendChild(label);

        const rescanBtn = document.createElement('button');
        rescanBtn.className = 'cell-rescan';
        rescanBtn.textContent = 'Re-scan';
        cell.appendChild(rescanBtn);

        ((idx) => {
            cell.onclick = () => rescanSlot(idx);
        })(i);

        grid.appendChild(cell);
    }

    grid.style.gridTemplateColumns = 'repeat(' + CFG.cols + ', 1fr)';
    document.getElementById('btnSubmit').disabled = (count === 0);

    ovl.classList.add('visible');
}

// ============================================================
// Re-scan a specific slot
// ============================================================
function rescanSlot(idx) {
    document.getElementById('doneOverlay').classList.remove('visible');
    activeSlot = idx;
    scanning = true;
    frozen = false;
    buildUI();
    animFrameId = requestAnimationFrame(liveLoop);
}

// ============================================================
// Submit cards for identification
// ============================================================
async function submitCards() {
    const count = captures.filter(Boolean).length;
    if (count === 0) return;

    const btn = document.getElementById('btnSubmit');
    btn.disabled = true;

    document.getElementById('doneOverlay').classList.remove('visible');
    const idOvl = document.getElementById('idOverlay');
    idOvl.classList.add('visible');

    const timerStart = Date.now();
    const timerEl = document.getElementById('idTimer');
    timerEl.textContent = '0s';
    const timerInterval = setInterval(() => {
        timerEl.textContent = Math.round((Date.now() - timerStart) / 1000) + 's';
    }, 1000);

    try {
        const formData = new FormData();
        for (let i = 0; i < CFG.total; i++) {
            if (!captures[i]) continue;
            const resp0 = await fetch(captures[i].dataUrl);
            const blob = await resp0.blob();
            formData.append('card_' + i, blob, 'card_' + i + '.jpg');
        }

        const resp = await fetch('/slide-scan/identify', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Server error: ' + resp.status);
        const data = await resp.json();

        clearInterval(timerInterval);
        idOvl.classList.remove('visible');

        if (data.error) {
            btn.disabled = false;
            document.getElementById('doneOverlay').classList.add('visible');
            alert('Error: ' + data.error);
            return;
        }

        showResults(data);
    } catch (err) {
        clearInterval(timerInterval);
        idOvl.classList.remove('visible');
        btn.disabled = false;
        document.getElementById('doneOverlay').classList.add('visible');
        alert('Error: ' + err.message);
    }
}

// ============================================================
// Results display
// ============================================================
function showResults(data) {
    const ovl = document.getElementById('resultsOverlay');
    const cards = data.cards || [];
    let total = 0;
    let identified = 0;

    const container = document.getElementById('resultsCards');
    container.innerHTML = '';

    for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        if (card.market_price) total += Number(card.market_price);
        if (card.card_id) identified++;

        const row = document.createElement('div');
        row.className = 'result-card';

        const imgs = document.createElement('div');
        imgs.className = 'rc-imgs';

        const slotIdx = card.position != null ? card.position : i;
        if (captures[slotIdx]) {
            const segImg = document.createElement('img');
            segImg.className = 'rc-img';
            segImg.src = captures[slotIdx].dataUrl;
            imgs.appendChild(segImg);
        }

        const refSrc = card.local_image_url || card.image_url;
        if (refSrc) {
            const refImg = document.createElement('img');
            refImg.className = 'rc-img';
            refImg.src = refSrc;
            imgs.appendChild(refImg);
        }
        row.appendChild(imgs);

        const info = document.createElement('div');
        info.className = 'rc-info';

        const nameDiv = document.createElement('div');
        nameDiv.className = 'rc-name';
        if (card.tcgplayer_url) {
            const link = document.createElement('a');
            link.href = card.tcgplayer_url;
            link.target = '_blank';
            link.rel = 'noopener';
            link.textContent = card.card_name || 'Unknown';
            nameDiv.appendChild(link);
        } else {
            nameDiv.textContent = card.card_name || 'Unknown';
        }
        if (card.detected_variant && card.detected_variant !== 'normal') {
            const badge = document.createElement('span');
            badge.className = 'variant-badge ' +
                card.detected_variant.replace(/_/g, '-');
            badge.textContent = card.detected_variant.replace(/_/g, ' ');
            nameDiv.appendChild(badge);
        }
        info.appendChild(nameDiv);

        const setDiv = document.createElement('div');
        setDiv.className = 'rc-set';
        setDiv.textContent = card.set_name || '';
        info.appendChild(setDiv);

        if (card.confidence || card.method) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'rc-meta';
            const parts = [];
            if (card.confidence) parts.push(Math.round(card.confidence * 100) + '%');
            if (card.method) parts.push(card.method);
            parts.push('R' + (Math.floor(slotIdx / CFG.cols) + 1) +
                        'C' + ((slotIdx % CFG.cols) + 1));
            metaDiv.textContent = parts.join(' \u00b7 ');
            info.appendChild(metaDiv);
        }
        row.appendChild(info);

        const priceDiv = document.createElement('div');
        priceDiv.className = 'rc-price';
        if (card.market_price) {
            priceDiv.textContent = '$' + Number(card.market_price).toFixed(2);
        } else {
            priceDiv.textContent = '--';
            priceDiv.className += ' no-price';
        }
        row.appendChild(priceDiv);

        container.appendChild(row);
    }

    document.getElementById('resTotalValue').textContent = '$' + total.toFixed(2);
    document.getElementById('resCardCount').textContent =
        identified + '/' + cards.length + ' identified';

    ovl.classList.add('visible');
}

// ============================================================
// Restart
// ============================================================
function restartScan() {
    document.getElementById('doneOverlay').classList.remove('visible');
    captures = [];
    activeSlot = 0;
    scanning = true;
    frozen = false;
    pendingCropData = null;
    cardRect = null;

    buildUI();
    animFrameId = requestAnimationFrame(liveLoop);
}

function restartFull() {
    document.getElementById('resultsOverlay').classList.remove('visible');
    restartScan();
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

    document.getElementById('debugInfo').className =
        'debug-info' + (CFG.debug ? ' visible' : '');

    toggleSettings();
    restartScan();
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
