"""Condition grading camera UI with live overlay and auto-capture.

Like ID verification apps: shows a live camera feed with a card-shaped
template overlay. The user positions their card to fill the template,
and the system auto-captures when alignment is detected.

Captures multiple angles for condition assessment:
  1. Front straight-on (0°)
  2. Front tilted left (~45°)
  3. Front tilted right (~45°)

The tilt detection uses the card's perspective shape — a straight-on card
appears rectangular, while a tilted card has converging edges (trapezoid).

Integration into server.py:
    elif self.path == "/condition/camera":
        from cardprice.condition_camera_ui import CAMERA_HTML
        self._send_html(CAMERA_HTML)

    elif self.path.startswith("/condition/camera/"):
        from cardprice.condition_camera_ui import render_camera_html
        card_id = unquote(self.path.split("/condition/camera/", 1)[1])
        self._send_html(render_camera_html(card_id))
"""


def render_camera_html(card_id=None, card_name=None):
    """Return CAMERA_HTML with card context substituted."""
    html = CAMERA_HTML
    html = html.replace("{{CARD_ID}}", card_id or "")
    html = html.replace("{{CARD_NAME}}", card_name or "")
    return html


CAMERA_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Card Condition Scanner</title>
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

/* ---- Top bar: step indicator ---- */
.top-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%);
    padding: 16px 20px 30px;
}

.step-label {
    font-size: 14px;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.step-title {
    font-size: 22px;
    font-weight: 700;
}

.step-instruction {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    margin-top: 4px;
}

/* Step dots */
.step-dots {
    display: flex;
    gap: 8px;
    margin-top: 12px;
}
.step-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: rgba(255,255,255,0.3);
    transition: all 0.3s;
}
.step-dot.active {
    background: #4ecca3;
    box-shadow: 0 0 8px rgba(78, 204, 163, 0.5);
}
.step-dot.done {
    background: #4ecca3;
}

/* ---- Bottom bar: status + manual capture ---- */
.bottom-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%);
    padding: 40px 20px 30px;
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
    transition: color 0.2s;
}
.status-text.ready {
    color: #4ecca3;
    font-weight: 600;
}

/* Manual capture button (fallback) */
.capture-btn {
    width: 72px; height: 72px;
    border-radius: 50%;
    border: 4px solid rgba(255,255,255,0.8);
    background: transparent;
    cursor: pointer;
    position: relative;
    transition: all 0.2s;
}
.capture-btn::after {
    content: '';
    position: absolute;
    top: 4px; left: 4px; right: 4px; bottom: 4px;
    border-radius: 50%;
    background: rgba(255,255,255,0.9);
    transition: all 0.15s;
}
.capture-btn:active::after {
    transform: scale(0.9);
    background: #4ecca3;
}
.capture-btn.disabled {
    opacity: 0.3;
    pointer-events: none;
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

/* ---- Review overlay ---- */
.review-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 15;
    background: rgba(0,0,0,0.85);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
}
.review-overlay.visible {
    display: flex;
}
.review-img {
    max-width: 80%;
    max-height: 50vh;
    border-radius: 8px;
    border: 2px solid #4ecca3;
}
.review-btns {
    display: flex;
    gap: 20px;
    margin-top: 20px;
}
.review-btns button {
    padding: 14px 28px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
}
.btn-retake {
    background: #333;
    color: #fff;
}
.btn-accept {
    background: #4ecca3;
    color: #1a1a2e;
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
.done-overlay.visible {
    display: flex;
}
.done-overlay h2 {
    font-size: 24px;
    color: #4ecca3;
    margin-bottom: 10px;
}
.done-overlay p {
    color: rgba(255,255,255,0.7);
    margin-bottom: 20px;
    text-align: center;
}
.thumbs-row {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}
.thumb {
    width: 90px;
    border-radius: 6px;
    border: 2px solid #333;
}
.done-overlay .submit-btn {
    padding: 16px 40px;
    background: #4ecca3;
    color: #1a1a2e;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
}
.done-overlay .submit-btn:disabled {
    opacity: 0.5;
}
.grade-result {
    margin-top: 20px;
    text-align: center;
}
.grade-badge {
    display: inline-block;
    padding: 12px 30px;
    border-radius: 10px;
    font-size: 28px;
    font-weight: 800;
}
.grade-NM { background: #2ecc71; color: #fff; }
.grade-LP { background: #f1c40f; color: #333; }
.grade-MP { background: #e67e22; color: #fff; }
.grade-HP { background: #e74c3c; color: #fff; }
.grade-DMG { background: #8b0000; color: #fff; }

/* ---- Training data section ---- */
.training-section {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.15);
    text-align: center;
}
.training-label {
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    margin-bottom: 10px;
}
.training-btns {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}
.training-btns button {
    padding: 10px 16px;
    border: 2px solid transparent;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    opacity: 0.85;
    transition: all 0.15s;
}
.training-btns button:hover,
.training-btns button.selected {
    opacity: 1;
    transform: scale(1.05);
}
.training-btns button.selected {
    border-color: #fff;
    box-shadow: 0 0 10px rgba(255,255,255,0.3);
}
.training-btns .tb-NM  { background: #2ecc71; color: #fff; }
.training-btns .tb-LP  { background: #f1c40f; color: #333; }
.training-btns .tb-MP  { background: #e67e22; color: #fff; }
.training-btns .tb-HP  { background: #e74c3c; color: #fff; }
.training-btns .tb-DMG { background: #8b0000; color: #fff; }
.training-save-btn {
    margin-top: 12px;
    padding: 10px 28px;
    background: #555;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
}
.training-save-btn:disabled {
    opacity: 0.4;
    cursor: default;
}
.training-status {
    margin-top: 8px;
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    min-height: 20px;
}

/* ---- Angle indicator arrow ---- */
.angle-arrow {
    position: absolute;
    z-index: 5;
    top: 50%;
    font-size: 48px;
    color: rgba(255,255,255,0.5);
    transform: translateY(-50%);
    animation: pulse 1.5s infinite;
    pointer-events: none;
}
.angle-arrow.left { left: 12px; }
.angle-arrow.right { right: 12px; }

@keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.8; }
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
        <div class="step-label" id="stepLabel">Step 1 of 2</div>
        <div class="step-title" id="stepTitle">Front — Straight On</div>
        <div class="step-instruction" id="stepInstr">Hold the card flat, facing the camera</div>
        <div class="step-dots" id="stepDots"></div>
    </div>

    <div id="angleArrow" class="angle-arrow" style="display:none;"></div>

    <div class="bottom-bar">
        <div class="status-text" id="status">Starting camera...</div>
        <button class="capture-btn" id="captureBtn" onclick="manualCapture()"></button>
    </div>

    <!-- Review captured photo -->
    <div class="review-overlay" id="reviewOverlay">
        <img class="review-img" id="reviewImg" src="">
        <div class="review-btns">
            <button class="btn-retake" onclick="retake()">Retake</button>
            <button class="btn-accept" onclick="acceptPhoto()">Looks Good</button>
        </div>
    </div>

    <!-- All captures done -->
    <div class="done-overlay" id="doneOverlay">
        <h2>All Photos Captured</h2>
        <p>Review your captures below, then submit for grading.</p>
        <div class="thumbs-row" id="thumbsRow"></div>
        <button class="submit-btn" id="submitBtn" onclick="submitForGrading()">
            Submit for Grading
        </button>
        <div class="grade-result" id="gradeResult"></div>
        <div class="training-section" id="trainingSection" style="display:none;">
            <div class="training-label">Save as training data — select correct condition:</div>
            <div class="training-btns" id="trainingBtns">
                <button class="tb-NM" onclick="selectTrainingLabel('NM',this)">NM</button>
                <button class="tb-LP" onclick="selectTrainingLabel('LP',this)">LP</button>
                <button class="tb-MP" onclick="selectTrainingLabel('MP',this)">MP</button>
                <button class="tb-HP" onclick="selectTrainingLabel('HP',this)">HP</button>
                <button class="tb-DMG" onclick="selectTrainingLabel('DMG',this)">DMG</button>
            </div>
            <button class="training-save-btn" id="trainingSaveBtn" disabled onclick="saveTrainingData()">Save Training Sample</button>
            <div class="training-status" id="trainingStatus"></div>
        </div>
    </div>
</div>

<script>
// ===== Configuration =====
const STEPS = [
    {
        id: 'front_straight',
        label: 'Step 1 of 2',
        title: 'Front — Straight On',
        instruction: 'Hold the card flat, facing the camera',
        targetSkew: 0,       // no perspective skew expected
        skewTolerance: 0.08, // how rectangular it must look
        arrow: null,
    },
    {
        id: 'front_tilt_left',
        label: 'Step 2 of 2',
        title: 'Front — Tilt Left 45°',
        instruction: 'Tilt the left edge toward you',
        targetSkew: -0.15,   // left side appears wider
        skewTolerance: 0.12,
        arrow: 'left',
    },
];

const CARD_ASPECT = 63 / 88; // Pokemon card W/H ratio
const TEMPLATE_HEIGHT_FRAC = 0.55; // card template is 55% of viewport height

let currentStep = 0;
let captures = [];       // {stepId, dataUrl, blob}
let detecting = false;
let autoCapturePending = null;
let stream = null;

// Detection state
let frameCount = 0;
let goodFrames = 0;        // consecutive frames where card is aligned
const GOOD_FRAMES_NEEDED = 8; // ~0.5s at 15fps

// ===== Camera setup =====
const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const captureCanvas = document.getElementById('capture');
const ctx = overlay.getContext('2d');
const capCtx = captureCanvas.getContext('2d');

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1920 },
                height: { ideal: 1080 },
            },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();
        // Wait for video dimensions to be available
        video.addEventListener('loadedmetadata', () => {
            resizeOverlay();
            requestAnimationFrame(detectLoop);
        });
        setStatus('Position your card in the frame');
    } catch (e) {
        setStatus('Camera error: ' + e.message);
        console.error('Camera error:', e);
    }
}

function resizeOverlay() {
    overlay.width = overlay.clientWidth * (window.devicePixelRatio || 1);
    overlay.height = overlay.clientHeight * (window.devicePixelRatio || 1);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
}
window.addEventListener('resize', resizeOverlay);

// ===== UI updates =====
function updateStepUI() {
    const step = STEPS[currentStep];
    document.getElementById('stepLabel').textContent = step.label;
    document.getElementById('stepTitle').textContent = step.title;
    document.getElementById('stepInstr').textContent = step.instruction;

    // Dots
    const dotsEl = document.getElementById('stepDots');
    dotsEl.innerHTML = '';
    for (let i = 0; i < STEPS.length; i++) {
        const dot = document.createElement('div');
        dot.className = 'step-dot' + (i < currentStep ? ' done' : i === currentStep ? ' active' : '');
        dotsEl.appendChild(dot);
    }

    // Arrow indicator for tilt steps
    const arrow = document.getElementById('angleArrow');
    if (step.arrow === 'left') {
        arrow.textContent = '◀';
        arrow.className = 'angle-arrow left';
        arrow.style.display = 'block';
    } else if (step.arrow === 'right') {
        arrow.textContent = '▶';
        arrow.className = 'angle-arrow right';
        arrow.style.display = 'block';
    } else {
        arrow.style.display = 'none';
    }

    goodFrames = 0;
}

function setStatus(text, ready) {
    const el = document.getElementById('status');
    el.textContent = text;
    el.classList.toggle('ready', !!ready);
}

// ===== Card detection using contour analysis =====
// We sample the video frame, find the card rectangle, and check:
// 1. Is it roughly filling the template area?
// 2. Does its perspective skew match the required angle?

function detectLoop() {
    if (currentStep >= STEPS.length) return;

    const w = overlay.clientWidth;
    const h = overlay.clientHeight;

    // Draw template overlay
    drawOverlay(w, h);

    // Run detection every 3rd frame (~10fps detection)
    frameCount++;
    if (frameCount % 3 === 0 && !detecting) {
        detecting = true;
        const result = detectCard(w, h);
        detecting = false;

        if (result.aligned) {
            goodFrames++;
            if (goodFrames >= GOOD_FRAMES_NEEDED) {
                setStatus('Hold still...', true);
                if (!autoCapturePending) {
                    autoCapturePending = setTimeout(() => {
                        doCapture();
                        autoCapturePending = null;
                    }, 200);
                }
            } else {
                setStatus(`Aligning... (${Math.round(goodFrames / GOOD_FRAMES_NEEDED * 100)}%)`, false);
            }
        } else {
            goodFrames = Math.max(0, goodFrames - 2); // decay faster than build
            if (autoCapturePending) {
                clearTimeout(autoCapturePending);
                autoCapturePending = null;
            }
            setStatus(result.hint || 'Position your card in the frame');
        }
    }

    requestAnimationFrame(detectLoop);
}

function drawOverlay(w, h) {
    ctx.clearRect(0, 0, w, h);

    // Template rectangle (card-shaped)
    const tplH = h * TEMPLATE_HEIGHT_FRAC;
    const tplW = tplH * CARD_ASPECT;
    const tplX = (w - tplW) / 2;
    const tplY = (h - tplH) / 2 - h * 0.03; // slightly above center

    // Dim everything outside the template
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, w, h);

    // Cut out the template area
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    roundRect(ctx, tplX, tplY, tplW, tplH, 10);
    ctx.fill();
    ctx.restore();

    // Template border
    const progress = goodFrames / GOOD_FRAMES_NEEDED;
    const borderColor = progress > 0.8 ? '#4ecca3' : progress > 0.3 ? '#f1c40f' : 'rgba(255,255,255,0.5)';
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    roundRect(ctx, tplX, tplY, tplW, tplH, 10);
    ctx.stroke();

    // Corner brackets for visual guidance
    const bracketLen = 25;
    const bracketW = 4;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = bracketW;
    ctx.lineCap = 'round';

    // Top-left
    ctx.beginPath();
    ctx.moveTo(tplX, tplY + bracketLen);
    ctx.lineTo(tplX, tplY);
    ctx.lineTo(tplX + bracketLen, tplY);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(tplX + tplW - bracketLen, tplY);
    ctx.lineTo(tplX + tplW, tplY);
    ctx.lineTo(tplX + tplW, tplY + bracketLen);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(tplX, tplY + tplH - bracketLen);
    ctx.lineTo(tplX, tplY + tplH);
    ctx.lineTo(tplX + bracketLen, tplY + tplH);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(tplX + tplW - bracketLen, tplY + tplH);
    ctx.lineTo(tplX + tplW, tplY + tplH);
    ctx.lineTo(tplX + tplW, tplY + tplH - bracketLen);
    ctx.stroke();

    // For tilt steps, draw the expected perspective shape
    const step = STEPS[currentStep];
    if (step.arrow) {
        ctx.setLineDash([6, 6]);
        ctx.strokeStyle = 'rgba(78, 204, 163, 0.4)';
        ctx.lineWidth = 2;
        const skew = step.targetSkew;
        const inset = tplW * Math.abs(skew);
        ctx.beginPath();
        if (skew < 0) {
            // Left tilt: left side wider, right side narrower
            ctx.moveTo(tplX - inset, tplY - inset * 0.5);
            ctx.lineTo(tplX + tplW + inset * 0.3, tplY + inset * 0.5);
            ctx.lineTo(tplX + tplW + inset * 0.3, tplY + tplH - inset * 0.5);
            ctx.lineTo(tplX - inset, tplY + tplH + inset * 0.5);
        } else {
            // Right tilt: right side wider, left side narrower
            ctx.moveTo(tplX - inset * 0.3, tplY + inset * 0.5);
            ctx.lineTo(tplX + tplW + inset, tplY - inset * 0.5);
            ctx.lineTo(tplX + tplW + inset, tplY + tplH + inset * 0.5);
            ctx.lineTo(tplX - inset * 0.3, tplY + tplH - inset * 0.5);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
}

function detectCard(viewW, viewH) {
    // Sample a small version of the video for speed
    const sampleW = 320;
    const sampleH = Math.round(sampleW * (video.videoHeight / video.videoWidth));

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = sampleW;
    tmpCanvas.height = sampleH;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(video, 0, 0, sampleW, sampleH);
    const imgData = tmpCtx.getImageData(0, 0, sampleW, sampleH);

    // Convert to grayscale
    const gray = new Uint8Array(sampleW * sampleH);
    for (let i = 0; i < gray.length; i++) {
        const j = i * 4;
        gray[i] = Math.round(0.299 * imgData.data[j] + 0.587 * imgData.data[j+1] + 0.114 * imgData.data[j+2]);
    }

    // Compute edge magnitude using Sobel-like operator
    const edges = new Uint8Array(sampleW * sampleH);
    for (let y = 1; y < sampleH - 1; y++) {
        for (let x = 1; x < sampleW - 1; x++) {
            const idx = y * sampleW + x;
            const gx = -gray[idx - sampleW - 1] + gray[idx - sampleW + 1]
                       -2*gray[idx - 1] + 2*gray[idx + 1]
                       -gray[idx + sampleW - 1] + gray[idx + sampleW + 1];
            const gy = -gray[idx - sampleW - 1] - 2*gray[idx - sampleW] - gray[idx - sampleW + 1]
                       +gray[idx + sampleW - 1] + 2*gray[idx + sampleW] + gray[idx + sampleW + 1];
            edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
        }
    }

    // Find strong horizontal and vertical edge accumulations
    // to detect the card rectangle boundaries
    const tplH = viewH * TEMPLATE_HEIGHT_FRAC;
    const tplW = tplH * CARD_ASPECT;
    const tplCx = viewW / 2;
    const tplCy = viewH / 2 - viewH * 0.03;

    // Map template to sample coordinates
    const scaleX = sampleW / viewW;
    const scaleY = sampleH / viewH;
    const sTplCx = tplCx * scaleX;
    const sTplCy = tplCy * scaleY;
    const sTplW = tplW * scaleX;
    const sTplH = tplH * scaleY;

    // Check if there's significant edge activity in the template region
    // (indicates a card is present)
    let edgeSum = 0;
    let edgeCount = 0;
    const margin = 0.15; // check within 15% margin of template edges
    const x1 = Math.max(0, Math.round(sTplCx - sTplW/2 - sTplW * margin));
    const x2 = Math.min(sampleW, Math.round(sTplCx + sTplW/2 + sTplW * margin));
    const y1 = Math.max(0, Math.round(sTplCy - sTplH/2 - sTplH * margin));
    const y2 = Math.min(sampleH, Math.round(sTplCy + sTplH/2 + sTplH * margin));

    for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
            const e = edges[y * sampleW + x];
            if (e > 40) {
                edgeSum += e;
                edgeCount++;
            }
        }
    }

    const regionArea = (x2 - x1) * (y2 - y1);
    const edgeDensity = edgeCount / Math.max(regionArea, 1);

    if (edgeDensity < 0.03) {
        return { aligned: false, hint: 'No card detected — move closer' };
    }

    // Simple approach: check brightness contrast between template center
    // and template border region. A well-positioned card will have
    // different brightness inside vs outside the template.
    let innerBright = 0, innerCount = 0;
    let outerBright = 0, outerCount = 0;
    const innerMargin = 0.2;

    for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
            const relX = (x - (sTplCx - sTplW/2)) / sTplW;
            const relY = (y - (sTplCy - sTplH/2)) / sTplH;
            const inside = relX > innerMargin && relX < (1 - innerMargin) &&
                          relY > innerMargin && relY < (1 - innerMargin);
            const pixel = gray[y * sampleW + x];
            if (inside) {
                innerBright += pixel;
                innerCount++;
            } else {
                outerBright += pixel;
                outerCount++;
            }
        }
    }

    const avgInner = innerBright / Math.max(innerCount, 1);
    const avgOuter = outerBright / Math.max(outerCount, 1);
    const contrast = Math.abs(avgInner - avgOuter);

    if (contrast < 15) {
        return { aligned: false, hint: 'Position card to fill the frame' };
    }

    // For tilt detection: check left vs right edge brightness gradient
    // A tilted card will have different illumination on the near vs far side
    const step = STEPS[currentStep];

    if (step.targetSkew === 0) {
        // Straight-on: check that left-right edge density is roughly symmetric
        let leftEdges = 0, rightEdges = 0;
        const edgeBand = Math.round(sTplW * 0.15);
        const leftX = Math.round(sTplCx - sTplW/2);
        const rightX = Math.round(sTplCx + sTplW/2 - edgeBand);
        const yStart = Math.round(sTplCy - sTplH * 0.3);
        const yEnd = Math.round(sTplCy + sTplH * 0.3);

        for (let y = yStart; y < yEnd; y++) {
            for (let dx = 0; dx < edgeBand; dx++) {
                if (leftX + dx >= 0 && leftX + dx < sampleW)
                    leftEdges += edges[y * sampleW + leftX + dx] > 30 ? 1 : 0;
                if (rightX + dx >= 0 && rightX + dx < sampleW)
                    rightEdges += edges[y * sampleW + rightX + dx] > 30 ? 1 : 0;
            }
        }

        const edgeBalance = Math.min(leftEdges, rightEdges) / Math.max(Math.max(leftEdges, rightEdges), 1);
        if (edgeBalance < 0.3) {
            return { aligned: false, hint: 'Hold the card flat and centered' };
        }

        // Good enough for straight-on
        return { aligned: true };

    } else {
        // Tilt detection: for a left-tilted card, the left edge appears
        // brighter (closer to camera/light) and right edge appears darker.
        // We use the brightness gradient across the card as a proxy for tilt.
        let leftBright = 0, rightBright = 0;
        let lCount = 0, rCount = 0;
        const band = Math.round(sTplW * 0.25);
        const lStart = Math.round(sTplCx - sTplW * 0.4);
        const rStart = Math.round(sTplCx + sTplW * 0.15);
        const yMid1 = Math.round(sTplCy - sTplH * 0.2);
        const yMid2 = Math.round(sTplCy + sTplH * 0.2);

        for (let y = yMid1; y < yMid2; y++) {
            for (let dx = 0; dx < band; dx++) {
                const lx = lStart + dx;
                const rx = rStart + dx;
                if (lx >= 0 && lx < sampleW) { leftBright += gray[y * sampleW + lx]; lCount++; }
                if (rx >= 0 && rx < sampleW) { rightBright += gray[y * sampleW + rx]; rCount++; }
            }
        }

        const avgL = leftBright / Math.max(lCount, 1);
        const avgR = rightBright / Math.max(rCount, 1);
        const brightDiff = (avgL - avgR) / 255; // positive = left brighter

        // For left tilt (targetSkew < 0): expect left side brighter OR just accept
        // that the user is following the instruction (loose detection for tilt)
        // Since reliably detecting 45° tilt from brightness alone is fragile,
        // we use a generous tolerance — the main goal is that the user moves
        // the card at all, and we get a different perspective.
        const tiltDetected = Math.abs(brightDiff) > 0.02; // any asymmetry

        if (step.targetSkew < 0) {
            // Left tilt
            if (!tiltDetected) {
                return { aligned: false, hint: 'Tilt the left edge toward you' };
            }
            return { aligned: true };
        } else {
            // Right tilt
            if (!tiltDetected) {
                return { aligned: false, hint: 'Tilt the right edge toward you' };
            }
            return { aligned: true };
        }
    }
}

// ===== Capture =====
function doCapture() {
    // Capture full-resolution frame from video
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    capCtx.drawImage(video, 0, 0);

    // Flash effect
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 150);

    // Get data URL and blob
    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);
    captureCanvas.toBlob(blob => {
        // Show review
        const reviewImg = document.getElementById('reviewImg');
        reviewImg.src = dataUrl;
        document.getElementById('reviewOverlay').classList.add('visible');

        // Store temporarily
        window._pendingCapture = {
            stepId: STEPS[currentStep].id,
            dataUrl: dataUrl,
            blob: blob,
        };
    }, 'image/jpeg', 0.92);
}

function manualCapture() {
    if (currentStep >= STEPS.length) return;
    doCapture();
}

function retake() {
    window._pendingCapture = null;
    document.getElementById('reviewOverlay').classList.remove('visible');
    goodFrames = 0;
}

function acceptPhoto() {
    if (!window._pendingCapture) return;

    captures.push(window._pendingCapture);
    window._pendingCapture = null;
    document.getElementById('reviewOverlay').classList.remove('visible');

    currentStep++;
    goodFrames = 0;

    if (currentStep >= STEPS.length) {
        showDone();
    } else {
        updateStepUI();
        requestAnimationFrame(detectLoop);
    }
}

// ===== Done / Submit =====
function showDone() {
    const doneEl = document.getElementById('doneOverlay');
    const thumbsRow = document.getElementById('thumbsRow');
    thumbsRow.innerHTML = '';

    for (const cap of captures) {
        const img = document.createElement('img');
        img.src = cap.dataUrl;
        img.className = 'thumb';
        thumbsRow.appendChild(img);

        const label = document.createElement('div');
        label.style.cssText = 'color: rgba(255,255,255,0.5); font-size: 11px; text-align: center; margin-top: -6px;';
        label.textContent = cap.stepId.replace(/_/g, ' ');
    }

    doneEl.classList.add('visible');
}

async function submitForGrading() {
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.textContent = 'Grading...';

    const cardId = '{{CARD_ID}}';

    try {
        const formData = new FormData();
        if (cardId) formData.append('card_id', cardId);

        for (const cap of captures) {
            formData.append(cap.stepId, cap.blob, cap.stepId + '.jpg');
        }

        const resp = await fetch('/condition/camera/assess', {
            method: 'POST',
            body: formData,
        });
        const result = await resp.json();

        // Show grade
        const gradeEl = document.getElementById('gradeResult');
        if (result.overall_grade) {
            let html = `
                <div class="grade-badge grade-${result.overall_grade}">
                    ${result.overall_grade}
                </div>`;
            if (result.card_name) {
                html += `<p style="margin-top:10px; font-size:16px;">${result.card_name}</p>`;
            }
            // Sub-grades table
            const sg = result.sub_grades || {};
            html += `<table style="margin:12px auto 0;font-size:13px;color:rgba(255,255,255,0.7);">`;
            for (const [k, v] of Object.entries(sg)) {
                if (v > 0) {
                    const bar = '█'.repeat(Math.round(v));
                    html += `<tr><td style="text-align:right;padding:2px 8px;">${k}</td>
                             <td style="text-align:left;padding:2px 8px;color:${v >= 9 ? '#2ecc71' : v >= 7 ? '#f1c40f' : '#e74c3c'}">${v}/10 ${bar}</td></tr>`;
                }
            }
            html += `</table>`;
            html += `<p style="margin-top:8px; color: rgba(255,255,255,0.5); font-size:12px;">
                Confidence: ${Math.round((result.overall_confidence || 0) * 100)}%
                ${result.price_multiplier ? ' · Price: ' + result.price_multiplier.toFixed(0) + '% of NM' : ''}
            </p>`;
            gradeEl.innerHTML = html;

            // Show training section and pre-select predicted grade
            document.getElementById('trainingSection').style.display = 'block';
            if (result.overall_grade) {
                const preBtn = document.querySelector('#trainingBtns .tb-' + result.overall_grade);
                if (preBtn) selectTrainingLabel(result.overall_grade, preBtn);
            }
        } else {
            gradeEl.innerHTML = '<p style="color:#e74c3c;">Grading failed — try again with better lighting</p>';
            // Still show training section for manual labeling
            document.getElementById('trainingSection').style.display = 'block';
        }
    } catch (e) {
        console.error('Submit error:', e);
        document.getElementById('gradeResult').innerHTML =
            '<p style="color:#e74c3c;">Error: ' + e.message + '</p>';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Submit for Grading';
    }
}

// ===== Training data =====
let selectedTrainingLabel = null;

function selectTrainingLabel(label, btn) {
    selectedTrainingLabel = label;
    // Update button selection state
    document.querySelectorAll('#trainingBtns button').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    document.getElementById('trainingSaveBtn').disabled = false;
}

async function saveTrainingData() {
    if (!selectedTrainingLabel || captures.length === 0) return;

    const btn = document.getElementById('trainingSaveBtn');
    btn.disabled = true;
    btn.textContent = 'Saving...';

    const cardId = '{{CARD_ID}}';

    try {
        const formData = new FormData();
        formData.append('condition', selectedTrainingLabel);
        if (cardId) formData.append('card_id', cardId);

        for (const cap of captures) {
            formData.append(cap.stepId, cap.blob, cap.stepId + '.jpg');
        }

        const resp = await fetch('/condition/training/save', {
            method: 'POST',
            body: formData,
        });
        const result = await resp.json();

        if (result.error) {
            document.getElementById('trainingStatus').innerHTML =
                '<span style="color:#e74c3c;">' + result.error + '</span>';
        } else {
            document.getElementById('trainingStatus').innerHTML =
                '<span style="color:#4ecca3;">Saved! Total samples: ' + result.total_count + '</span>';
            btn.textContent = 'Saved';
        }
    } catch (e) {
        document.getElementById('trainingStatus').innerHTML =
            '<span style="color:#e74c3c;">Error: ' + e.message + '</span>';
        btn.disabled = false;
        btn.textContent = 'Save Training Sample';
    }
}

// ===== Init =====
updateStepUI();
startCamera();
</script>
</body>
</html>
"""
