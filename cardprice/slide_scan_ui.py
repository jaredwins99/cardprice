"""Manual-capture card scanning UI with premium visual feedback.

User points phone at one card at a time. Live video shows real-time detection
overlay with card outline tracking, center crosshair, sharpness meter, and
progress ring. Tap shutter to capture. System auto-crops the card, user
reviews and accepts or retakes. Thumbnail strip shows progress (9 slots).

Visual feedback features:
    - Card outline tracker: green/yellow/red rectangle tracking detected edges
    - Center crosshair: pulses green when card center aligns with frame center
    - Sharpness meter: top bar showing real-time frame sharpness (red->green)
    - Progress ring: circular fill around shutter button showing alignment quality
    - State-driven text hints: context-aware instructions based on detection state
    - Card slot indicator: highlights current R#C# slot in thumbnail strip
    - Capture toast with "Captured!" and "Move to next card" hint

Integration into server.py:
    GET  /slide-scan            -> serve this HTML
    POST /slide-scan/identify   -> receive card images, identify, return JSON
"""

SLIDE_SCAN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Slide Scan</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --accent: #4ecca3;
    --accent-dim: rgba(78, 204, 163, 0.3);
    --warn: #ffc048;
    --danger: #e94560;
    --bg: #1a1a2e;
    --bg-dark: #0f0f1a;
    --text: #fff;
    --text-dim: rgba(255,255,255,0.55);
    --safe-bottom: env(safe-area-inset-bottom, 0px);
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-dark);
    color: var(--text);
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
}

/* ================================================================ */
/*  SCREEN: CAMERA                                                   */
/* ================================================================ */
#screen-camera {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* -- Thumbnail strip at top -- */
.thumb-strip {
    background: var(--bg);
    padding: 10px 12px 8px;
    z-index: 10;
    flex-shrink: 0;
}
.thumb-strip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.thumb-counter {
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
}
.thumb-counter span { color: var(--accent); }

.thumb-slots {
    display: flex;
    gap: 6px;
    justify-content: center;
}
.thumb-slot {
    width: 50px;
    height: 70px;
    border-radius: 6px;
    border: 2px dashed rgba(255,255,255,0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    position: relative;
    cursor: pointer;
    transition: all 0.3s;
    overflow: hidden;
    flex-shrink: 0;
}
.thumb-slot.next-slot {
    border-color: var(--accent);
    border-style: solid;
    box-shadow: 0 0 12px rgba(78,204,163,0.25);
    animation: slotPulse 2s ease-in-out infinite;
}
@keyframes slotPulse {
    0%,100% { box-shadow: 0 0 8px rgba(78,204,163,0.2); }
    50% { box-shadow: 0 0 16px rgba(78,204,163,0.45); }
}
.thumb-slot .slot-num {
    font-size: 12px;
    color: rgba(255,255,255,0.2);
    font-weight: 600;
}
.thumb-slot.next-slot .slot-num {
    color: var(--accent);
    font-weight: 700;
}
.thumb-slot .slot-label {
    font-size: 7px;
    color: rgba(255,255,255,0.15);
    margin-top: 1px;
    font-family: monospace;
}
.thumb-slot.next-slot .slot-label {
    color: rgba(78,204,163,0.6);
}
.thumb-slot img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 4px;
}
.thumb-slot.filled {
    border: 2px solid var(--accent);
}
.thumb-slot.filled:active { transform: scale(0.92); }
.thumb-slot .redo-badge {
    position: absolute;
    top: -4px; right: -4px;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--danger);
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 2;
}
.thumb-slot.filled .redo-badge { display: flex; }

/* -- Camera viewport -- */
.camera-area {
    flex: 1;
    position: relative;
    overflow: hidden;
    background: #000;
}
.camera-area video {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
}
.camera-area canvas#detect-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 3;
    pointer-events: none;
}

/* ---- SHARPNESS METER ---- */
.sharpness-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: rgba(0,0,0,0.5);
    z-index: 5;
    pointer-events: none;
}
.sharpness-bar-fill {
    height: 100%;
    width: 0%;
    border-radius: 0 2px 2px 0;
    transition: width 0.12s ease-out, background 0.3s;
    background: var(--danger);
}
.sharpness-label {
    position: absolute;
    top: 6px; right: 8px;
    font-size: 10px;
    font-weight: 700;
    color: rgba(255,255,255,0.5);
    z-index: 5;
    pointer-events: none;
    font-family: monospace;
    text-shadow: 0 1px 4px rgba(0,0,0,0.8);
    letter-spacing: 0.5px;
}

/* ---- STATUS HINT ---- */
.status-hint {
    position: absolute;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 5;
    pointer-events: none;
    text-align: center;
    width: 85%;
}
.status-hint-text {
    font-size: 16px;
    font-weight: 700;
    color: rgba(255,255,255,0.8);
    text-shadow: 0 2px 10px rgba(0,0,0,0.8);
    transition: color 0.25s;
    padding: 6px 16px;
    background: rgba(0,0,0,0.45);
    border-radius: 20px;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    display: inline-block;
}
.status-hint-text.state-ready { color: var(--accent); }
.status-hint-text.state-warn { color: var(--warn); }
.status-hint-text.state-error { color: var(--danger); }
.status-hint-text.state-success { color: var(--accent); font-size: 20px; }

/* Flash */
.flash-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(78,204,163,0.3);
    z-index: 20;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
}
.flash-overlay.active { opacity: 1; transition: none; }

/* Capture toast */
.capture-toast {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%,-50%) scale(0.7);
    z-index: 22;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s, transform 0.2s;
    text-align: center;
}
.capture-toast.show {
    opacity: 1;
    transform: translate(-50%,-50%) scale(1);
}
.capture-toast .toast-check {
    font-size: 56px;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(78,204,163,0.5);
}
.capture-toast .toast-label {
    font-size: 18px; font-weight: 700;
    color: #fff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.8);
    margin-top: 4px;
}
.capture-toast .toast-next {
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    margin-top: 6px;
}

/* -- Bottom bar with shutter -- */
.bottom-bar {
    background: var(--bg);
    padding: 12px 16px calc(12px + var(--safe-bottom));
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    z-index: 10;
    flex-shrink: 0;
    position: relative;
}

/* Shutter with progress ring */
.shutter-wrap {
    position: relative;
    width: 76px; height: 76px;
}
.shutter-ring-svg {
    position: absolute;
    top: 0; left: 0;
    width: 76px; height: 76px;
    transform: rotate(-90deg);
    pointer-events: none;
}
.shutter-ring-bg {
    fill: none;
    stroke: rgba(255,255,255,0.08);
    stroke-width: 3;
}
.shutter-ring-fill {
    fill: none;
    stroke: var(--accent);
    stroke-width: 3;
    stroke-linecap: round;
    stroke-dasharray: 207.3;
    stroke-dashoffset: 207.3;
    transition: stroke-dashoffset 0.15s ease-out, stroke 0.2s;
}
.shutter-btn {
    position: absolute;
    top: 4px; left: 4px;
    width: 68px; height: 68px;
    border-radius: 50%;
    border: 4px solid var(--accent);
    background: transparent;
    cursor: pointer;
    transition: transform 0.1s, border-color 0.2s, box-shadow 0.2s;
    -webkit-tap-highlight-color: transparent;
}
.shutter-btn:active { transform: scale(0.9); }
.shutter-btn::after {
    content: '';
    position: absolute;
    top: 5px; left: 5px; right: 5px; bottom: 5px;
    border-radius: 50%;
    background: var(--accent);
    transition: background 0.15s;
}
.shutter-btn:active::after { background: #3bb88e; }
.shutter-btn:disabled { opacity: 0.3; pointer-events: none; }
.shutter-btn.pulse-ready {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(78,204,163,0.4);
}

.submit-btn {
    position: absolute;
    right: 16px;
    padding: 10px 18px;
    border: none;
    border-radius: 10px;
    background: var(--danger);
    color: #fff;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    display: none;
    transition: transform 0.1s;
}
.submit-btn:active { transform: scale(0.95); }
.submit-btn.visible { display: block; }

canvas.proc-canvas { display: none; }

/* ================================================================ */
/*  SCREEN: REVIEW                                                   */
/* ================================================================ */
#screen-review {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 30;
    background: var(--bg-dark);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
}
#screen-review.visible { display: flex; }
.review-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
.review-image {
    max-width: 70%; max-height: 55vh;
    border-radius: 10px;
    border: 3px solid var(--accent);
    object-fit: contain;
    margin-bottom: 8px;
}
.review-status { font-size: 14px; color: var(--text-dim); margin-bottom: 20px; min-height: 20px; }
.review-buttons { display: flex; gap: 16px; }
.review-btn {
    padding: 14px 32px; border: none; border-radius: 12px;
    font-size: 16px; font-weight: 700; cursor: pointer;
    transition: transform 0.1s;
}
.review-btn:active { transform: scale(0.93); }
.review-btn.keep { background: var(--accent); color: var(--bg); }
.review-btn.retake {
    background: rgba(255,255,255,0.12);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.2);
}

/* ================================================================ */
/*  SCREEN: PREVIEW                                                  */
/* ================================================================ */
#screen-preview {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 30;
    background: var(--bg);
    display: none;
    flex-direction: column;
    align-items: center;
    overflow-y: auto;
    padding: 20px 16px 40px;
}
#screen-preview.visible { display: flex; }
.preview-title { font-size: 22px; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
.preview-sub { font-size: 13px; color: var(--text-dim); margin-bottom: 16px; }
.preview-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    max-width: 320px;
    width: 100%;
    margin-bottom: 20px;
}
.preview-cell {
    aspect-ratio: 2.5/3.5;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    cursor: pointer;
}
.preview-cell img { width: 100%; height: 100%; object-fit: cover; }
.preview-cell .cell-empty { font-size: 12px; color: rgba(255,255,255,0.2); }
.preview-cell .cell-num {
    position: absolute; top: 4px; left: 6px;
    font-size: 11px; font-weight: 700;
    color: rgba(255,255,255,0.5);
    background: rgba(0,0,0,0.5);
    padding: 1px 5px; border-radius: 4px;
}
.preview-buttons { display: flex; gap: 12px; margin-top: 4px; }
.preview-btn {
    padding: 14px 28px; border: none; border-radius: 12px;
    font-size: 16px; font-weight: 700; cursor: pointer;
    transition: transform 0.1s;
}
.preview-btn:active { transform: scale(0.93); }
.preview-btn.go { background: var(--accent); color: var(--bg); }
.preview-btn.back {
    background: rgba(255,255,255,0.12);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.2);
}

/* ================================================================ */
/*  SCREEN: UPLOADING                                                */
/* ================================================================ */
#screen-uploading {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 35;
    background: var(--bg-dark);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
}
#screen-uploading.visible { display: flex; }
.spinner {
    width: 48px; height: 48px;
    border: 4px solid rgba(255,255,255,0.1);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.upload-text { font-size: 16px; color: var(--text-dim); }

/* ================================================================ */
/*  TOAST                                                            */
/* ================================================================ */
.toast {
    position: fixed;
    bottom: 120px; left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: rgba(233,69,96,0.92);
    color: #fff;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 14px; font-weight: 600;
    z-index: 50;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s, transform 0.25s;
    text-align: center;
    max-width: 85%;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>

<div id="screen-camera">
    <div class="thumb-strip">
        <div class="thumb-strip-header">
            <div class="thumb-counter"><span id="capture-count">0</span> of 9 captured</div>
        </div>
        <div class="thumb-slots" id="thumb-slots"></div>
    </div>

    <div class="camera-area">
        <video id="cam-video" autoplay playsinline muted></video>
        <canvas id="detect-overlay"></canvas>
        <div class="sharpness-bar">
            <div class="sharpness-bar-fill" id="sharpness-fill"></div>
        </div>
        <div class="sharpness-label" id="sharpness-label">SHARP</div>
        <div class="status-hint">
            <span class="status-hint-text" id="status-hint">Center a card in frame</span>
        </div>
        <div class="flash-overlay" id="flash"></div>
        <div class="capture-toast" id="capture-toast">
            <div class="toast-check">&#10003;</div>
            <div class="toast-label">Captured!</div>
            <div class="toast-next" id="toast-next">Move to next card &rarr;</div>
        </div>
    </div>

    <div class="bottom-bar">
        <div class="shutter-wrap">
            <svg class="shutter-ring-svg" viewBox="0 0 76 76">
                <circle class="shutter-ring-bg" cx="38" cy="38" r="33"></circle>
                <circle class="shutter-ring-fill" id="shutter-ring" cx="38" cy="38" r="33"></circle>
            </svg>
            <button class="shutter-btn" id="shutter-btn" aria-label="Capture"></button>
        </div>
        <button class="submit-btn" id="submit-btn">Submit</button>
    </div>

    <canvas class="proc-canvas" id="proc-canvas"></canvas>
    <canvas class="proc-canvas" id="detect-canvas"></canvas>
</div>

<div id="screen-review">
    <div class="review-title" id="review-title">Card Preview</div>
    <img class="review-image" id="review-img" alt="Captured card">
    <div class="review-status" id="review-status"></div>
    <div class="review-buttons">
        <button class="review-btn retake" id="btn-retake">Retake</button>
        <button class="review-btn keep" id="btn-keep">Keep</button>
    </div>
</div>

<div id="screen-preview">
    <div class="preview-title" id="preview-title">Ready to Identify</div>
    <div class="preview-sub" id="preview-sub">9 cards captured</div>
    <div class="preview-grid" id="preview-grid"></div>
    <div class="preview-buttons">
        <button class="preview-btn back" id="preview-back">Back</button>
        <button class="preview-btn go" id="preview-go">Identify Cards</button>
    </div>
</div>

<div id="screen-uploading">
    <div class="spinner"></div>
    <div class="upload-text">Identifying cards...</div>
</div>

<div class="toast" id="toast"></div>

<script>
(function() {
    'use strict';

    var TOTAL_SLOTS = 9;
    var COLS = 3;
    var RING_CIRC = 2 * Math.PI * 33;

    // ---- State ----
    var captures = new Array(TOTAL_SLOTS).fill(null);
    var nextSlot = 0;
    var reviewSlot = -1;
    var reviewBlob = null;
    var reviewDataUrl = null;
    var cameraStream = null;

    // Detection state
    var detectState = 'idle';
    var smoothSharpness = 0;
    var prevFrameGray = null;
    var motionMagnitude = 0;
    var lastAnalysis = null;
    var readyScore = 0;
    var detectAnimId = null;
    var detectFrameCount = 0;

    // Detection config
    var DETECT_W = 320;
    var SHARPNESS_MAX = 80;
    var EDGE_MIN = 0.10;
    var SHARPNESS_MIN = 12;
    var CONTRAST_MIN = 8;
    var MOTION_THRESH = 8.0;

    // ---- DOM refs ----
    var video = document.getElementById('cam-video');
    var canvas = document.getElementById('proc-canvas');
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    var detectCanvas = document.getElementById('detect-canvas');
    var detectCtx = detectCanvas.getContext('2d', { willReadFrequently: true });
    var overlayCanvas = document.getElementById('detect-overlay');
    var overlayCtx = overlayCanvas.getContext('2d');
    var shutterBtn = document.getElementById('shutter-btn');
    var submitBtn = document.getElementById('submit-btn');
    var flash = document.getElementById('flash');
    var countEl = document.getElementById('capture-count');
    var thumbSlotsEl = document.getElementById('thumb-slots');
    var sharpnessFill = document.getElementById('sharpness-fill');
    var sharpnessLabel = document.getElementById('sharpness-label');
    var statusHint = document.getElementById('status-hint');
    var shutterRing = document.getElementById('shutter-ring');
    var captureToastEl = document.getElementById('capture-toast');
    var toastNextEl = document.getElementById('toast-next');
    var screenReview = document.getElementById('screen-review');
    var screenPreview = document.getElementById('screen-preview');
    var screenUploading = document.getElementById('screen-uploading');
    var reviewImg = document.getElementById('review-img');
    var reviewTitleEl = document.getElementById('review-title');
    var reviewStatus = document.getElementById('review-status');
    var btnKeep = document.getElementById('btn-keep');
    var btnRetake = document.getElementById('btn-retake');
    var previewTitleEl = document.getElementById('preview-title');
    var previewSub = document.getElementById('preview-sub');
    var previewGrid = document.getElementById('preview-grid');
    var previewBack = document.getElementById('preview-back');
    var previewGo = document.getElementById('preview-go');
    var toastEl = document.getElementById('toast');
    var captureToastTimer = null;

    // ============================================================
    // Thumbnail strip with R#C# slot labels
    // ============================================================
    function buildSlots() {
        thumbSlotsEl.innerHTML = '';
        for (var i = 0; i < TOTAL_SLOTS; i++) {
            var slot = document.createElement('div');
            slot.className = 'thumb-slot' + (i === nextSlot ? ' next-slot' : '');
            slot.dataset.idx = i;
            if (captures[i]) {
                slot.classList.add('filled');
                var img = document.createElement('img');
                img.src = captures[i].dataUrl;
                slot.appendChild(img);
                var badge = document.createElement('div');
                badge.className = 'redo-badge';
                badge.textContent = '\u21BB';
                slot.appendChild(badge);
            } else {
                var num = document.createElement('div');
                num.className = 'slot-num';
                num.textContent = i + 1;
                slot.appendChild(num);
                var lbl = document.createElement('div');
                lbl.className = 'slot-label';
                lbl.textContent = 'R' + (Math.floor(i/COLS)+1) + 'C' + ((i%COLS)+1);
                slot.appendChild(lbl);
            }
            slot.addEventListener('click', (function(idx) {
                return function() { onSlotTap(idx); };
            })(i));
            thumbSlotsEl.appendChild(slot);
        }
    }

    function updateUI() {
        var filled = captures.filter(Boolean).length;
        countEl.textContent = filled;
        submitBtn.classList.toggle('visible', filled > 0);
        buildSlots();
    }

    function computeNextSlot() {
        for (var i = 0; i < TOTAL_SLOTS; i++) {
            if (!captures[i]) return i;
        }
        return TOTAL_SLOTS;
    }

    // ============================================================
    // Sharpness meter
    // ============================================================
    function updateSharpness(val) {
        smoothSharpness = smoothSharpness * 0.7 + val * 0.3;
        var f = Math.min(smoothSharpness / SHARPNESS_MAX, 1);
        sharpnessFill.style.width = (f * 100) + '%';
        if (f > 0.5) {
            sharpnessFill.style.background = '#4ecca3';
            sharpnessLabel.textContent = 'SHARP';
            sharpnessLabel.style.color = '#4ecca3';
        } else if (f > 0.25) {
            sharpnessFill.style.background = '#ffc048';
            sharpnessLabel.textContent = 'OK';
            sharpnessLabel.style.color = '#ffc048';
        } else {
            sharpnessFill.style.background = '#e94560';
            sharpnessLabel.textContent = 'BLURRY';
            sharpnessLabel.style.color = '#e94560';
        }
    }

    // ============================================================
    // Progress ring around shutter
    // ============================================================
    function setRing(frac) {
        shutterRing.style.strokeDashoffset = RING_CIRC * (1 - Math.min(frac, 1));
        if (frac >= 0.8) {
            shutterRing.style.stroke = '#4ecca3';
            shutterBtn.classList.add('pulse-ready');
        } else if (frac > 0.3) {
            shutterRing.style.stroke = '#ffc048';
            shutterBtn.classList.remove('pulse-ready');
        } else {
            shutterRing.style.stroke = 'rgba(255,255,255,0.3)';
            shutterBtn.classList.remove('pulse-ready');
        }
    }

    // ============================================================
    // Status hint
    // ============================================================
    function setHint(msg, state) {
        statusHint.textContent = msg;
        statusHint.className = 'status-hint-text';
        if (state) statusHint.classList.add('state-' + state);
    }

    // ============================================================
    // Overlay resize
    // ============================================================
    function resizeOverlay() {
        var r = video.getBoundingClientRect();
        if (r.width > 0) { overlayCanvas.width = r.width; overlayCanvas.height = r.height; }
    }

    // ============================================================
    // Draw detection overlay
    // ============================================================
    function drawOverlay() {
        var W = overlayCanvas.width, H = overlayCanvas.height;
        if (!W || !H) return;
        overlayCtx.clearRect(0, 0, W, H);

        // Guide zone matching card aspect
        var gw = W * 0.62;
        var gh = gw * (3.5 / 2.5);
        if (gh > H * 0.75) { gh = H * 0.75; gw = gh * (2.5 / 3.5); }
        var gx = (W - gw) / 2;
        var gy = (H - gh) / 2;

        // Colors from detection state
        var bc, bw, dash;
        if (detectState === 'steady') { bc = '#4ecca3'; bw = 3; dash = false; }
        else if (detectState === 'detected' || detectState === 'centering') { bc = '#ffc048'; bw = 2.5; dash = false; }
        else if (detectState === 'blurry') { bc = '#e94560'; bw = 2.5; dash = [6,4]; }
        else { bc = 'rgba(78,204,163,0.35)'; bw = 1.5; dash = [8,6]; }

        // Dim outside
        overlayCtx.fillStyle = 'rgba(0,0,0,0.35)';
        overlayCtx.fillRect(0, 0, W, gy);
        overlayCtx.fillRect(0, gy + gh, W, H - gy - gh);
        overlayCtx.fillRect(0, gy, gx, gh);
        overlayCtx.fillRect(gx + gw, gy, W - gx - gw, gh);

        // Rounded rect border
        var r = 10;
        overlayCtx.strokeStyle = bc;
        overlayCtx.lineWidth = bw;
        overlayCtx.setLineDash(dash || []);
        overlayCtx.beginPath();
        overlayCtx.moveTo(gx+r, gy);
        overlayCtx.lineTo(gx+gw-r, gy);
        overlayCtx.arcTo(gx+gw, gy, gx+gw, gy+r, r);
        overlayCtx.lineTo(gx+gw, gy+gh-r);
        overlayCtx.arcTo(gx+gw, gy+gh, gx+gw-r, gy+gh, r);
        overlayCtx.lineTo(gx+r, gy+gh);
        overlayCtx.arcTo(gx, gy+gh, gx, gy+gh-r, r);
        overlayCtx.lineTo(gx, gy+r);
        overlayCtx.arcTo(gx, gy, gx+r, gy, r);
        overlayCtx.closePath();
        overlayCtx.stroke();
        overlayCtx.setLineDash([]);

        // Corner brackets
        var bl = 28;
        overlayCtx.strokeStyle = bc;
        overlayCtx.lineWidth = 4;
        overlayCtx.lineCap = 'round';
        overlayCtx.beginPath(); overlayCtx.moveTo(gx, gy+bl); overlayCtx.lineTo(gx, gy); overlayCtx.lineTo(gx+bl, gy); overlayCtx.stroke();
        overlayCtx.beginPath(); overlayCtx.moveTo(gx+gw-bl, gy); overlayCtx.lineTo(gx+gw, gy); overlayCtx.lineTo(gx+gw, gy+bl); overlayCtx.stroke();
        overlayCtx.beginPath(); overlayCtx.moveTo(gx, gy+gh-bl); overlayCtx.lineTo(gx, gy+gh); overlayCtx.lineTo(gx+bl, gy+gh); overlayCtx.stroke();
        overlayCtx.beginPath(); overlayCtx.moveTo(gx+gw-bl, gy+gh); overlayCtx.lineTo(gx+gw, gy+gh); overlayCtx.lineTo(gx+gw, gy+gh-bl); overlayCtx.stroke();
        overlayCtx.lineCap = 'butt';

        // Center crosshair
        var cx = W/2, cy = H/2, cl = 18, cg = 7;
        var cc = 'rgba(255,255,255,0.2)', cw = 1;
        if (detectState === 'steady') {
            var p = 0.6 + 0.4 * Math.sin(Date.now()/200);
            cc = 'rgba(78,204,163,' + p + ')';
            cw = 2;
        } else if (detectState === 'detected' || detectState === 'centering') {
            cc = 'rgba(255,192,72,0.45)';
            cw = 1.5;
        }
        overlayCtx.strokeStyle = cc;
        overlayCtx.lineWidth = cw;
        overlayCtx.beginPath(); overlayCtx.moveTo(cx-cl-cg, cy); overlayCtx.lineTo(cx-cg, cy); overlayCtx.stroke();
        overlayCtx.beginPath(); overlayCtx.moveTo(cx+cg, cy); overlayCtx.lineTo(cx+cl+cg, cy); overlayCtx.stroke();
        overlayCtx.beginPath(); overlayCtx.moveTo(cx, cy-cl-cg); overlayCtx.lineTo(cx, cy-cg); overlayCtx.stroke();
        overlayCtx.beginPath(); overlayCtx.moveTo(cx, cy+cg); overlayCtx.lineTo(cx, cy+cl+cg); overlayCtx.stroke();

        // Center dot when steady
        if (detectState === 'steady') {
            overlayCtx.fillStyle = cc;
            overlayCtx.beginPath();
            overlayCtx.arc(cx, cy, 3, 0, Math.PI*2);
            overlayCtx.fill();
        }

        // Glow effect when steady
        if (detectState === 'steady') {
            var ga = 0.08 + 0.05 * Math.sin(Date.now()/300);
            overlayCtx.save();
            overlayCtx.strokeStyle = 'rgba(78,204,163,' + ga*3 + ')';
            overlayCtx.lineWidth = 10;
            overlayCtx.beginPath();
            overlayCtx.moveTo(gx+r, gy);
            overlayCtx.lineTo(gx+gw-r, gy);
            overlayCtx.arcTo(gx+gw, gy, gx+gw, gy+r, r);
            overlayCtx.lineTo(gx+gw, gy+gh-r);
            overlayCtx.arcTo(gx+gw, gy+gh, gx+gw-r, gy+gh, r);
            overlayCtx.lineTo(gx+r, gy+gh);
            overlayCtx.arcTo(gx, gy+gh, gx, gy+gh-r, r);
            overlayCtx.lineTo(gx, gy+r);
            overlayCtx.arcTo(gx, gy, gx+r, gy, r);
            overlayCtx.closePath();
            overlayCtx.stroke();
            overlayCtx.restore();
        }
    }

    // ============================================================
    // Frame analysis (edge density, sharpness, contrast, motion)
    // ============================================================
    function analyzeFrame() {
        var vw = video.videoWidth, vh = video.videoHeight;
        if (!vw || !vh) return null;

        var sc = DETECT_W / vw;
        var dw = DETECT_W, dh = Math.round(vh * sc);
        detectCanvas.width = dw;
        detectCanvas.height = dh;
        detectCtx.drawImage(video, 0, 0, dw, dh);

        // Center zone matching guide
        var zw = Math.round(dw * 0.62);
        var zh = Math.round(zw * (3.5/2.5));
        if (zh > dh * 0.75) { zh = Math.round(dh * 0.75); zw = Math.round(zh * (2.5/3.5)); }
        var zx = Math.round((dw - zw) / 2);
        var zy = Math.round((dh - zh) / 2);
        if (zw <= 4 || zh <= 4) return null;

        var imgData = detectCtx.getImageData(zx, zy, zw, zh);
        var px = imgData.data;
        var total = zw * zh;

        var gray = new Float32Array(total);
        for (var i = 0; i < total; i++) {
            var o = i * 4;
            gray[i] = 0.299*px[o] + 0.587*px[o+1] + 0.114*px[o+2];
        }

        // Motion detection
        if (prevFrameGray && prevFrameGray.length === total) {
            var ds = 0, sc2 = 0;
            for (var i = 0; i < total; i += 4) { ds += Math.abs(gray[i] - prevFrameGray[i]); sc2++; }
            motionMagnitude = sc2 > 0 ? ds / sc2 : 0;
        } else { motionMagnitude = 0; }
        prevFrameGray = new Float32Array(gray);

        // Edge density
        var ec = 0;
        for (var y = 1; y < zh-1; y++) {
            for (var x = 1; x < zw-1; x++) {
                var idx = y*zw+x;
                var gx2 = -gray[idx-zw-1]+gray[idx-zw+1]-2*gray[idx-1]+2*gray[idx+1]-gray[idx+zw-1]+gray[idx+zw+1];
                var gy2 = -gray[idx-zw-1]-2*gray[idx-zw]-gray[idx-zw+1]+gray[idx+zw-1]+2*gray[idx+zw]+gray[idx+zw+1];
                if (Math.sqrt(gx2*gx2+gy2*gy2) > 60) ec++;
            }
        }
        var edgeDensity = ec / total;

        // Sharpness (Laplacian variance)
        var ls = 0, lss = 0, lc = 0;
        for (var y = 1; y < zh-1; y++) {
            for (var x = 1; x < zw-1; x++) {
                var idx = y*zw+x;
                var lap = gray[idx-zw]+gray[idx+zw]+gray[idx-1]+gray[idx+1]-4*gray[idx];
                ls += lap; lss += lap*lap; lc++;
            }
        }
        var sharpness = (lss/lc) - (ls/lc)*(ls/lc);

        // Contrast
        var cx1=Math.round(zw*0.3),cy1=Math.round(zh*0.3),cx2=Math.round(zw*0.7),cy2=Math.round(zh*0.7);
        var cs=0,cc2=0,bs=0,bc2=0;
        for (var y = 0; y < zh; y++) {
            for (var x = 0; x < zw; x++) {
                var v = gray[y*zw+x];
                if (x>=cx1&&x<cx2&&y>=cy1&&y<cy2) { cs+=v; cc2++; } else { bs+=v; bc2++; }
            }
        }
        var contrast = Math.abs((cc2?cs/cc2:128) - (bc2?bs/bc2:128));

        return { edgeDensity: edgeDensity, sharpness: sharpness, contrast: contrast };
    }

    // ============================================================
    // Detection loop
    // ============================================================
    function detectionLoop() {
        detectFrameCount++;
        if (detectFrameCount % 3 === 0 && video.videoWidth) {
            var info = analyzeFrame();
            if (info) {
                lastAnalysis = info;
                updateSharpness(info.sharpness);

                var blurry = motionMagnitude > MOTION_THRESH;
                var hasCard = info.edgeDensity >= EDGE_MIN * 0.6;
                var isGood = info.edgeDensity >= EDGE_MIN
                          && info.sharpness >= SHARPNESS_MIN
                          && info.contrast >= CONTRAST_MIN
                          && !blurry;

                var es = Math.min(info.edgeDensity / (EDGE_MIN * 1.5), 1);
                var ss = Math.min(info.sharpness / (SHARPNESS_MIN * 3), 1);
                var cs = Math.min(info.contrast / (CONTRAST_MIN * 2), 1);
                var mp = blurry ? 0.3 : 1.0;
                readyScore = readyScore * 0.6 + (es*0.4 + ss*0.35 + cs*0.25) * mp * 0.4;
                setRing(readyScore);

                if (isGood) {
                    detectState = 'steady';
                    setHint('Hold steady \u2014 tap to capture', 'ready');
                } else if (blurry && hasCard) {
                    detectState = 'blurry';
                    setHint('Too blurry \u2014 slow down', 'error');
                } else if (hasCard) {
                    detectState = 'centering';
                    setHint('Center the card', 'warn');
                } else {
                    detectState = 'idle';
                    setHint('Move closer to cards', '');
                }
            }
        }
        drawOverlay();
        detectAnimId = requestAnimationFrame(detectionLoop);
    }

    function startDetection() {
        if (detectAnimId) return;
        setTimeout(function() {
            resizeOverlay();
            window.addEventListener('resize', resizeOverlay);
            detectAnimId = requestAnimationFrame(detectionLoop);
        }, 300);
    }

    function stopDetection() {
        if (detectAnimId) { cancelAnimationFrame(detectAnimId); detectAnimId = null; }
    }

    // ============================================================
    // Camera
    // ============================================================
    function startCamera() {
        navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false
        }).then(function(stream) {
            cameraStream = stream;
            video.srcObject = stream;
            video.play();
            startDetection();
        }).catch(function(err) {
            showToast('Camera access denied.');
            console.error('Camera error:', err);
        });
    }

    // ============================================================
    // Capture + crop pipeline
    // ============================================================
    function captureFrame() {
        if (!video.videoWidth) return null;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        return canvas;
    }

    var CARD_W = 420, CARD_H = 586;

    function cropCardFromFrame(src) {
        var sw = src.width, sh = src.height;
        if (sw < 100 || sh < 100) return null;
        var pw = 480, ps = pw/sw, ph = Math.round(sh*ps);
        var pc = document.createElement('canvas');
        pc.width = pw; pc.height = ph;
        var pctx = pc.getContext('2d', { willReadFrequently: true });
        pctx.drawImage(src, 0, 0, pw, ph);
        var id = pctx.getImageData(0, 0, pw, ph);
        var px = id.data, tp = pw*ph;

        var gray = new Uint8Array(tp);
        for (var i=0;i<tp;i++) { var o=i*4; gray[i]=Math.round(0.299*px[o]+0.587*px[o+1]+0.114*px[o+2]); }

        var blurred = new Uint8Array(tp);
        for (var y=1;y<ph-1;y++) for (var x=1;x<pw-1;x++) {
            var idx=y*pw+x;
            blurred[idx]=(gray[idx-pw-1]+2*gray[idx-pw]+gray[idx-pw+1]+2*gray[idx-1]+4*gray[idx]+2*gray[idx+1]+gray[idx+pw-1]+2*gray[idx+pw]+gray[idx+pw+1])>>4;
        }

        var eb = new Uint8Array(tp);
        for (var y=1;y<ph-1;y++) for (var x=1;x<pw-1;x++) {
            var idx=y*pw+x;
            var gx=-blurred[idx-pw-1]+blurred[idx-pw+1]-2*blurred[idx-1]+2*blurred[idx+1]-blurred[idx+pw-1]+blurred[idx+pw+1];
            var gy=-blurred[idx-pw-1]-2*blurred[idx-pw]-blurred[idx-pw+1]+blurred[idx+pw-1]+2*blurred[idx+pw]+blurred[idx+pw+1];
            eb[idx]=(Math.sqrt(gx*gx+gy*gy)>50)?255:0;
        }

        var dil = new Uint8Array(tp);
        for (var y=1;y<ph-1;y++) for (var x=1;x<pw-1;x++) {
            var idx=y*pw+x;
            if (eb[idx]||eb[idx-1]||eb[idx+1]||eb[idx-pw]||eb[idx+pw]||eb[idx-pw-1]||eb[idx-pw+1]||eb[idx+pw-1]||eb[idx+pw+1]) dil[idx]=255;
        }

        var contours = findContours(dil, pw, ph);
        var fcx=pw/2,fcy=ph/2,fa=pw*ph;
        var bestQ=null,bestD=Infinity;

        for (var ci=0;ci<contours.length;ci++) {
            var c=contours[ci],per=0;
            for (var pi=0;pi<c.length;pi++) { var pj=(pi+1)%c.length; var dx=c[pj][0]-c[pi][0],dy=c[pj][1]-c[pi][1]; per+=Math.sqrt(dx*dx+dy*dy); }
            if (per<80) continue;
            var ap=rdpSimplify(c, 0.02*per);
            if (ap.length!==4||!isConvex(ap)) continue;
            var ar=polygonArea(ap);
            if (ar<fa*0.15||ar>fa*0.80) continue;
            var xs=ap.map(function(p){return p[0];}),ys=ap.map(function(p){return p[1];});
            var bw2=Math.max.apply(null,xs)-Math.min.apply(null,xs),bh2=Math.max.apply(null,ys)-Math.min.apply(null,ys);
            if (!bw2||!bh2) continue;
            var asp=Math.min(bw2,bh2)/Math.max(bw2,bh2);
            if (asp<0.55||asp>0.85) continue;
            var tb=false;
            for (var qi=0;qi<4;qi++) { if (ap[qi][0]<=3||ap[qi][1]<=3||ap[qi][0]>=pw-3||ap[qi][1]>=ph-3) { tb=true; break; } }
            if (tb) continue;
            var qcx=(ap[0][0]+ap[1][0]+ap[2][0]+ap[3][0])/4,qcy=(ap[0][1]+ap[1][1]+ap[2][1]+ap[3][1])/4;
            var d=Math.sqrt((qcx-fcx)*(qcx-fcx)+(qcy-fcy)*(qcy-fcy));
            if (d<bestD) { bestD=d; bestQ=ap; }
        }

        if (!bestQ) return null;
        var ord=orderCorners(bestQ);
        var sc2=ord.map(function(pt){return [pt[0]/ps, pt[1]/ps];});
        return perspectiveWarp(src, sc2, CARD_W, CARD_H);
    }

    function findContours(bin, w, h) {
        var vis=new Uint8Array(w*h), out=[];
        for (var y=1;y<h-1;y++) for (var x=1;x<w-1;x++) {
            var idx=y*w+x;
            if (bin[idx]!==255||vis[idx]) continue;
            var comp=[],stk=[[x,y]]; vis[idx]=1;
            while (stk.length) {
                var cur=stk.pop(); comp.push(cur);
                for (var dy=-1;dy<=1;dy++) for (var dx=-1;dx<=1;dx++) {
                    if (!dx&&!dy) continue;
                    var nx=cur[0]+dx,ny=cur[1]+dy;
                    if (nx<0||ny<0||nx>=w||ny>=h) continue;
                    var ni=ny*w+nx;
                    if (bin[ni]===255&&!vis[ni]) { vis[ni]=1; stk.push([nx,ny]); }
                }
            }
            if (comp.length<40) continue;
            var bdr=[];
            for (var bi=0;bi<comp.length;bi++) {
                var bx=comp[bi][0],by=comp[bi][1],ob=false;
                for (var dy=-1;dy<=1&&!ob;dy++) for (var dx=-1;dx<=1&&!ob;dx++) {
                    if (!dx&&!dy) continue;
                    var nx=bx+dx,ny=by+dy;
                    if (nx<0||ny<0||nx>=w||ny>=h||bin[ny*w+nx]===0) ob=true;
                }
                if (ob) bdr.push([bx,by]);
            }
            if (bdr.length>=40) {
                var bcx=0,bcy=0;
                for (var bi=0;bi<bdr.length;bi++){bcx+=bdr[bi][0];bcy+=bdr[bi][1];}
                bcx/=bdr.length;bcy/=bdr.length;
                bdr.sort(function(a,b){return Math.atan2(a[1]-bcy,a[0]-bcx)-Math.atan2(b[1]-bcy,b[0]-bcx);});
                out.push(bdr);
            }
        }
        return out;
    }

    function rdpSimplify(pts, eps) {
        if (pts.length<=2) return pts.slice();
        var md=0,mi=0,s=pts[0],e=pts[pts.length-1];
        for (var i=1;i<pts.length-1;i++){var d=ptLineDist(pts[i],s,e);if(d>md){md=d;mi=i;}}
        if (md>eps){var l=rdpSimplify(pts.slice(0,mi+1),eps),r=rdpSimplify(pts.slice(mi),eps);return l.slice(0,-1).concat(r);}
        return [s,e];
    }

    function ptLineDist(p,a,b) {
        var dx=b[0]-a[0],dy=b[1]-a[1],ls=dx*dx+dy*dy;
        if (!ls) return Math.sqrt((p[0]-a[0])*(p[0]-a[0])+(p[1]-a[1])*(p[1]-a[1]));
        return Math.abs(dy*p[0]-dx*p[1]+b[0]*a[1]-b[1]*a[0])/Math.sqrt(ls);
    }

    function isConvex(pts) {
        var n=pts.length,s=0;
        for (var i=0;i<n;i++){var a=pts[i],b=pts[(i+1)%n],c=pts[(i+2)%n];
            var cr=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);
            if (cr!==0){if (!s) s=cr>0?1:-1; else if ((cr>0?1:-1)!==s) return false;}}
        return true;
    }

    function polygonArea(pts) {
        var a=0;for(var i=0;i<pts.length;i++){var j=(i+1)%pts.length;a+=pts[i][0]*pts[j][1]-pts[j][0]*pts[i][1];}return Math.abs(a)/2;
    }

    function orderCorners(pts) {
        var ss=pts.map(function(p){return p[0]+p[1];}),ds=pts.map(function(p){return p[1]-p[0];});
        return [pts[ss.indexOf(Math.min.apply(null,ss))],pts[ds.indexOf(Math.min.apply(null,ds))],pts[ss.indexOf(Math.max.apply(null,ss))],pts[ds.indexOf(Math.max.apply(null,ds))]];
    }

    function perspectiveWarp(src, corners, dw, dh) {
        var sctx=src.getContext('2d',{willReadFrequently:true});
        var sd=sctx.getImageData(0,0,src.width,src.height),sp=sd.data,sw=src.width,sh=src.height;
        var oc=document.createElement('canvas');oc.width=dw;oc.height=dh;
        var octx=oc.getContext('2d'),od=octx.createImageData(dw,dh),op=od.data;
        var tl=corners[0],tr=corners[1],br=corners[2],bl=corners[3];
        for (var dy2=0;dy2<dh;dy2++){
            var v=dy2/(dh-1);
            var lx=tl[0]+v*(bl[0]-tl[0]),ly=tl[1]+v*(bl[1]-tl[1]);
            var rx=tr[0]+v*(br[0]-tr[0]),ry=tr[1]+v*(br[1]-tr[1]);
            for (var dx2=0;dx2<dw;dx2++){
                var u=dx2/(dw-1),sx=lx+u*(rx-lx),sy=ly+u*(ry-ly);
                var ix=Math.round(sx),iy=Math.round(sy),oi=(dy2*dw+dx2)*4;
                if (ix>=0&&ix<sw&&iy>=0&&iy<sh){var si=(iy*sw+ix)*4;op[oi]=sp[si];op[oi+1]=sp[si+1];op[oi+2]=sp[si+2];op[oi+3]=255;}
            }
        }
        octx.putImageData(od,0,0);return oc;
    }

    function validateCrop(cc) {
        var w=cc.width,h=cc.height,vc=cc.getContext('2d',{willReadFrequently:true});
        var id=vc.getImageData(0,0,w,h),px=id.data,tp=w*h;
        var bs=0;for(var i=0;i<tp;i++){var o=i*4;bs+=0.299*px[o]+0.587*px[o+1]+0.114*px[o+2];}
        var mb=bs/tp;if(mb<40||mb>240) return false;
        var vs=0;for(var i=0;i<tp;i++){var o=i*4;var l=0.299*px[o]+0.587*px[o+1]+0.114*px[o+2];vs+=(l-mb)*(l-mb);}
        if(vs/tp<100) return false;
        var th=Math.round(h*0.2),tt=w*th,tg=new Uint8Array(tt);
        for(var y=0;y<th;y++)for(var x=0;x<w;x++){var o=(y*w+x)*4;tg[y*w+x]=Math.round(0.299*px[o]+0.587*px[o+1]+0.114*px[o+2]);}
        var te=0;
        for(var y=1;y<th-1;y++)for(var x=1;x<w-1;x++){
            var idx=y*w+x;
            var gx=-tg[idx-w-1]+tg[idx-w+1]-2*tg[idx-1]+2*tg[idx+1]-tg[idx+w-1]+tg[idx+w+1];
            var gy=-tg[idx-w-1]-2*tg[idx-w]-tg[idx-w+1]+tg[idx+w-1]+2*tg[idx+w]+tg[idx+w+1];
            if(Math.sqrt(gx*gx+gy*gy)>40) te++;
        }
        return te/tt >= 0.03;
    }

    function smartCrop(src) {
        var w=src.width,h=src.height;
        var cw=Math.round(w*0.65),ch=Math.round(cw*(3.5/2.5));
        var ah=Math.min(ch,Math.round(h*0.85)),aw=Math.round(ah*(2.5/3.5));
        var x=Math.round((w-aw)/2),y=Math.round((h-ah)/2);
        var cc=document.createElement('canvas');cc.width=aw;cc.height=ah;
        cc.getContext('2d').drawImage(src,x,y,aw,ah,0,0,aw,ah);return cc;
    }

    function canvasToBlob(cvs,q){return new Promise(function(res){cvs.toBlob(function(b){res(b);},'image/jpeg',q||0.92);});}

    // ============================================================
    // Shutter
    // ============================================================
    function onShutter() {
        if (nextSlot>=TOTAL_SLOTS&&captures.every(Boolean)){showPreview();return;}
        var slot=nextSlot<TOTAL_SLOTS?nextSlot:captures.findIndex(function(c){return !c;});
        if (slot<0){showPreview();return;}

        flash.classList.add('active');
        setTimeout(function(){flash.classList.remove('active');},150);
        if (navigator.vibrate) navigator.vibrate(50);

        var fc=captureFrame();
        if (!fc){showToast('Camera not ready');return;}
        shutterBtn.disabled=true;

        var cropped=cropCardFromFrame(fc);
        if (cropped&&!validateCrop(cropped)) cropped=null;
        if (!cropped) cropped=smartCrop(fc);

        canvasToBlob(cropped).then(function(blob){
            var du=URL.createObjectURL(blob);
            reviewSlot=slot; reviewBlob=blob; reviewDataUrl=du;
            showCaptureToast(slot);
            showReviewScreen(slot,du);
            shutterBtn.disabled=false;
        });
    }

    function showCaptureToast(slot){
        var rem=TOTAL_SLOTS-captures.filter(Boolean).length-1;
        toastNextEl.innerHTML=rem>0?'Move to next card &rarr;':'All cards captured!';
        captureToastEl.classList.add('show');
        if(captureToastTimer) clearTimeout(captureToastTimer);
        captureToastTimer=setTimeout(function(){captureToastEl.classList.remove('show');captureToastTimer=null;},800);
    }

    function onSlotTap(idx){if(captures[idx]){nextSlot=idx;updateUI();}}

    function showReviewScreen(slot,du){
        reviewImg.src=du;
        reviewTitleEl.textContent='Card '+(slot+1)+' (R'+(Math.floor(slot/COLS)+1)+'C'+((slot%COLS)+1)+')';
        reviewStatus.textContent='';
        stopDetection();
        screenReview.classList.add('visible');
    }

    function onKeep(){
        if(captures[reviewSlot]&&captures[reviewSlot].dataUrl) URL.revokeObjectURL(captures[reviewSlot].dataUrl);
        captures[reviewSlot]={blob:reviewBlob,dataUrl:reviewDataUrl};
        screenReview.classList.remove('visible');
        nextSlot=computeNextSlot();
        updateUI();
        reviewBlob=null;reviewDataUrl=null;
        startDetection();
        setHint('Move to next card \u2192','ready');
    }

    function onRetake(){
        if(reviewDataUrl) URL.revokeObjectURL(reviewDataUrl);
        reviewBlob=null;reviewDataUrl=null;
        screenReview.classList.remove('visible');
        startDetection();
    }

    function showPreview(){
        var filled=captures.filter(Boolean).length;
        if(!filled){showToast('Capture at least one card first');return;}
        stopDetection();
        previewTitleEl.textContent='Ready to Identify';
        previewSub.textContent=filled+' card'+(filled!==1?'s':'')+' captured';
        previewGrid.innerHTML='';
        previewGo.style.display='';
        previewBack.textContent='Back';
        previewBack.onclick=null;

        for (var i=0;i<TOTAL_SLOTS;i++){
            var cell=document.createElement('div');cell.className='preview-cell';
            var nl=document.createElement('div');nl.className='cell-num';
            nl.textContent='R'+(Math.floor(i/COLS)+1)+'C'+((i%COLS)+1);
            cell.appendChild(nl);
            if(captures[i]){var img=document.createElement('img');img.src=captures[i].dataUrl;cell.appendChild(img);}
            else{var em=document.createElement('div');em.className='cell-empty';em.textContent='Empty';cell.appendChild(em);}
            cell.addEventListener('click',(function(idx){return function(){screenPreview.classList.remove('visible');nextSlot=idx;updateUI();startDetection();};})(i));
            previewGrid.appendChild(cell);
        }
        screenPreview.classList.add('visible');
    }

    function onSubmit(){
        var fi=[];for(var i=0;i<TOTAL_SLOTS;i++) if(captures[i]) fi.push(i);
        if(!fi.length){showToast('No cards');return;}
        screenPreview.classList.remove('visible');
        screenUploading.classList.add('visible');
        var fd=new FormData();
        for(var j=0;j<fi.length;j++){var idx=fi[j];fd.append('card_'+idx,captures[idx].blob,'card_'+idx+'.jpg');}
        fetch('/slide-scan/identify?variants=true',{method:'POST',body:fd})
        .then(function(r){return r.json();})
        .then(function(data){
            screenUploading.classList.remove('visible');
            if(data.error){showToast('Error: '+data.error);return;}
            showResults(data);
        }).catch(function(err){
            screenUploading.classList.remove('visible');
            showToast('Upload failed: '+err.message);
        });
    }

    function showResults(data){
        var cards=data.cards||[],tv=data.total_value||0;
        previewTitleEl.textContent='Results';
        previewSub.textContent=cards.length+' card'+(cards.length!==1?'s':'')+' identified \u2014 Total: $'+tv.toFixed(2);
        previewGrid.innerHTML='';
        for(var i=0;i<cards.length;i++){
            var card=cards[i],cell=document.createElement('div');cell.className='preview-cell';cell.style.cursor='default';
            var iu=card.local_image_url||card.image_url||card.segment_image_url;
            if(iu){var img=document.createElement('img');img.src=iu;cell.appendChild(img);}
            var info=document.createElement('div');
            info.style.cssText='position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(0,0,0,0.85));padding:4px 5px 3px;';
            var nm=document.createElement('div');
            nm.style.cssText='font-size:9px;font-weight:600;color:#fff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
            nm.textContent=card.card_name||'Unknown';info.appendChild(nm);
            var pr=card.variant_price||card.market_price;
            if(pr){var pe=document.createElement('div');pe.style.cssText='font-size:10px;font-weight:700;color:#4ecca3;';
                pe.textContent='$'+pr.toFixed(2);
                if(card.detected_variant&&card.detected_variant!=='normal') pe.textContent+=' ('+card.detected_variant+')';
                info.appendChild(pe);}
            cell.appendChild(info);previewGrid.appendChild(cell);
        }
        previewBack.textContent='Scan Again';
        previewBack.onclick=function(){
            for(var k=0;k<TOTAL_SLOTS;k++){if(captures[k]&&captures[k].dataUrl) URL.revokeObjectURL(captures[k].dataUrl);captures[k]=null;}
            nextSlot=0;updateUI();screenPreview.classList.remove('visible');startDetection();
        };
        previewGo.style.display='none';
        screenPreview.classList.add('visible');
    }

    var toastTimer=null;
    function showToast(msg){toastEl.textContent=msg;toastEl.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(function(){toastEl.classList.remove('show');},3000);}

    // ---- Events ----
    shutterBtn.addEventListener('click', onShutter);
    btnKeep.addEventListener('click', onKeep);
    btnRetake.addEventListener('click', onRetake);
    submitBtn.addEventListener('click', showPreview);
    previewBack.addEventListener('click', function(){screenPreview.classList.remove('visible');startDetection();});
    previewGo.addEventListener('click', onSubmit);

    document.addEventListener('gesturestart', function(e){e.preventDefault();});
    document.addEventListener('touchmove', function(e){if(e.touches.length>1) e.preventDefault();}, {passive:false});

    // ---- Init ----
    buildSlots();
    updateUI();
    startCamera();
})();
</script>
</body>
</html>
"""
