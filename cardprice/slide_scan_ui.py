"""Slide-scan binder camera UI -- simplified manual capture flow.

User points phone at one card at a time, taps shutter to capture.
System auto-crops the card from the frame, user reviews and accepts or retakes.
Thumbnail strip shows progress (9 slots). Submit when ready.

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
/*  SCREEN: CAMERA (capture mode)                                    */
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
    position: relative;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.15s;
    overflow: hidden;
    flex-shrink: 0;
}
.thumb-slot.next-slot {
    border-color: var(--accent);
    border-style: solid;
}
.thumb-slot .slot-num {
    font-size: 12px;
    color: rgba(255,255,255,0.2);
    font-weight: 600;
}
.thumb-slot img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 4px;
}
.thumb-slot.filled {
    border: 2px solid var(--accent);
    cursor: pointer;
}
.thumb-slot.filled:active {
    transform: scale(0.92);
}
/* redo badge on filled slots */
.thumb-slot .redo-badge {
    position: absolute;
    top: -4px;
    right: -4px;
    width: 18px;
    height: 18px;
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
.thumb-slot.filled .redo-badge {
    display: flex;
}

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
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* Card guide overlay */
.card-guide {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 62%;
    aspect-ratio: 2.5 / 3.5;
    border: 2px solid rgba(78, 204, 163, 0.4);
    border-radius: 10px;
    pointer-events: none;
    z-index: 3;
}
.card-guide-label {
    position: absolute;
    bottom: -28px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 12px;
    color: var(--text-dim);
    white-space: nowrap;
}

/* Flash feedback */
.flash-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(78, 204, 163, 0.3);
    z-index: 20;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
}
.flash-overlay.active {
    opacity: 1;
    transition: none;
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

.shutter-btn {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    border: 4px solid var(--accent);
    background: transparent;
    cursor: pointer;
    position: relative;
    transition: transform 0.1s;
    -webkit-tap-highlight-color: transparent;
}
.shutter-btn:active {
    transform: scale(0.9);
}
.shutter-btn::after {
    content: '';
    position: absolute;
    top: 5px; left: 5px; right: 5px; bottom: 5px;
    border-radius: 50%;
    background: var(--accent);
    transition: background 0.15s;
}
.shutter-btn:active::after {
    background: #3bb88e;
}
.shutter-btn:disabled {
    opacity: 0.3;
    pointer-events: none;
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

/* Hidden processing canvas */
canvas.proc-canvas { display: none; }

/* ================================================================ */
/*  SCREEN: REVIEW (keep / retake)                                   */
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

.review-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    color: var(--text);
}

.review-image {
    max-width: 70%;
    max-height: 55vh;
    border-radius: 10px;
    border: 3px solid var(--accent);
    object-fit: contain;
    margin-bottom: 8px;
}

.review-status {
    font-size: 14px;
    color: var(--text-dim);
    margin-bottom: 20px;
    min-height: 20px;
}

.review-buttons {
    display: flex;
    gap: 16px;
}

.review-btn {
    padding: 14px 32px;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.1s;
}
.review-btn:active { transform: scale(0.93); }
.review-btn.keep {
    background: var(--accent);
    color: var(--bg);
}
.review-btn.retake {
    background: rgba(255,255,255,0.12);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.2);
}

/* ================================================================ */
/*  SCREEN: PREVIEW (grid + submit)                                  */
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

.preview-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 4px;
}
.preview-sub {
    font-size: 13px;
    color: var(--text-dim);
    margin-bottom: 16px;
}

.preview-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    max-width: 320px;
    width: 100%;
    margin-bottom: 20px;
}
.preview-cell {
    aspect-ratio: 2.5 / 3.5;
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
.preview-cell img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.preview-cell .cell-empty {
    font-size: 12px;
    color: rgba(255,255,255,0.2);
}
.preview-cell .cell-num {
    position: absolute;
    top: 4px;
    left: 6px;
    font-size: 11px;
    font-weight: 700;
    color: rgba(255,255,255,0.5);
    background: rgba(0,0,0,0.5);
    padding: 1px 5px;
    border-radius: 4px;
}

.preview-buttons {
    display: flex;
    gap: 12px;
    margin-top: 4px;
}

.preview-btn {
    padding: 14px 28px;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.1s;
}
.preview-btn:active { transform: scale(0.93); }
.preview-btn.go {
    background: var(--accent);
    color: var(--bg);
}
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
    width: 48px;
    height: 48px;
    border: 4px solid rgba(255,255,255,0.1);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.upload-text {
    font-size: 16px;
    color: var(--text-dim);
}

/* ================================================================ */
/*  ERROR TOAST                                                      */
/* ================================================================ */
.toast {
    position: fixed;
    bottom: 120px;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: rgba(233, 69, 96, 0.92);
    color: #fff;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    z-index: 50;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s, transform 0.25s;
    text-align: center;
    max-width: 85%;
}
.toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}
</style>
</head>
<body>

<!-- CAMERA SCREEN -->
<div id="screen-camera">
    <!-- Thumbnail strip -->
    <div class="thumb-strip">
        <div class="thumb-strip-header">
            <div class="thumb-counter"><span id="capture-count">0</span> of 9 captured</div>
        </div>
        <div class="thumb-slots" id="thumb-slots"></div>
    </div>

    <!-- Camera viewport -->
    <div class="camera-area">
        <video id="cam-video" autoplay playsinline muted></video>
        <div class="card-guide">
            <div class="card-guide-label">Center a card in frame</div>
        </div>
        <div class="flash-overlay" id="flash"></div>
    </div>

    <!-- Bottom bar -->
    <div class="bottom-bar">
        <button class="shutter-btn" id="shutter-btn" aria-label="Capture"></button>
        <button class="submit-btn" id="submit-btn">Submit</button>
    </div>

    <canvas class="proc-canvas" id="proc-canvas"></canvas>
</div>

<!-- REVIEW SCREEN -->
<div id="screen-review">
    <div class="review-title" id="review-title">Card Preview</div>
    <img class="review-image" id="review-img" alt="Captured card">
    <div class="review-status" id="review-status"></div>
    <div class="review-buttons">
        <button class="review-btn retake" id="btn-retake">Retake</button>
        <button class="review-btn keep" id="btn-keep">Keep</button>
    </div>
</div>

<!-- PREVIEW SCREEN (grid before submit) -->
<div id="screen-preview">
    <div class="preview-title" id="preview-title">Ready to Identify</div>
    <div class="preview-sub" id="preview-sub">9 cards captured</div>
    <div class="preview-grid" id="preview-grid"></div>
    <div class="preview-buttons">
        <button class="preview-btn back" id="preview-back">Back</button>
        <button class="preview-btn go" id="preview-go">Identify Cards</button>
    </div>
</div>

<!-- UPLOADING SCREEN -->
<div id="screen-uploading">
    <div class="spinner"></div>
    <div class="upload-text">Identifying cards...</div>
</div>

<!-- TOAST -->
<div class="toast" id="toast"></div>

<script>
(function() {
    'use strict';

    var TOTAL_SLOTS = 9;

    // State
    var captures = new Array(TOTAL_SLOTS).fill(null);
    var nextSlot = 0;
    var reviewSlot = -1;
    var reviewBlob = null;
    var reviewDataUrl = null;
    var cameraStream = null;

    // DOM refs
    var video = document.getElementById('cam-video');
    var canvas = document.getElementById('proc-canvas');
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    var shutterBtn = document.getElementById('shutter-btn');
    var submitBtn = document.getElementById('submit-btn');
    var flash = document.getElementById('flash');
    var countEl = document.getElementById('capture-count');
    var thumbSlotsEl = document.getElementById('thumb-slots');

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

    // ---- Build thumbnail slots ----
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

    // ---- Camera ----
    function startCamera() {
        navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            },
            audio: false
        }).then(function(stream) {
            cameraStream = stream;
            video.srcObject = stream;
            video.play();
        }).catch(function(err) {
            showToast('Camera access denied. Please allow camera permissions.');
            console.error('Camera error:', err);
        });
    }

    // ---- Capture ----
    function captureFrame() {
        if (!video.videoWidth) return null;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        return canvas;
    }

    function smartCrop(sourceCanvas) {
        var w = sourceCanvas.width;
        var h = sourceCanvas.height;

        // Crop center region matching card aspect ratio (2.5:3.5)
        var cropW = Math.round(w * 0.65);
        var cropH = Math.round(cropW * (3.5 / 2.5));
        var actualCropH = Math.min(cropH, Math.round(h * 0.85));
        var actualCropW = Math.round(actualCropH * (2.5 / 3.5));

        var x = Math.round((w - actualCropW) / 2);
        var y = Math.round((h - actualCropH) / 2);

        var cropCanvas = document.createElement('canvas');
        cropCanvas.width = actualCropW;
        cropCanvas.height = actualCropH;
        var cropCtx = cropCanvas.getContext('2d');
        cropCtx.drawImage(sourceCanvas, x, y, actualCropW, actualCropH, 0, 0, actualCropW, actualCropH);

        return cropCanvas;
    }

    function canvasToBlob(cvs, quality) {
        return new Promise(function(resolve) {
            cvs.toBlob(function(blob) { resolve(blob); }, 'image/jpeg', quality || 0.92);
        });
    }

    function onShutter() {
        if (nextSlot >= TOTAL_SLOTS && captures.every(Boolean)) {
            showPreview();
            return;
        }

        var slot = nextSlot < TOTAL_SLOTS ? nextSlot : captures.findIndex(function(c) { return !c; });
        if (slot < 0) {
            showPreview();
            return;
        }

        // Flash feedback
        flash.classList.add('active');
        setTimeout(function() { flash.classList.remove('active'); }, 150);

        var frameCanvas = captureFrame();
        if (!frameCanvas) {
            showToast('Camera not ready');
            return;
        }

        shutterBtn.disabled = true;

        var cropped = smartCrop(frameCanvas);
        canvasToBlob(cropped).then(function(blob) {
            var dataUrl = URL.createObjectURL(blob);

            reviewSlot = slot;
            reviewBlob = blob;
            reviewDataUrl = dataUrl;

            showReviewScreen(slot, dataUrl);
            shutterBtn.disabled = false;
        });
    }

    // ---- Slot tap (redo) ----
    function onSlotTap(idx) {
        if (captures[idx]) {
            nextSlot = idx;
            updateUI();
        }
    }

    // ---- Review screen ----
    function showReviewScreen(slot, dataUrl) {
        reviewImg.src = dataUrl;
        reviewTitleEl.textContent = 'Card ' + (slot + 1) + ' of ' + TOTAL_SLOTS;
        reviewStatus.textContent = '';
        screenReview.classList.add('visible');
    }

    function onKeep() {
        if (captures[reviewSlot] && captures[reviewSlot].dataUrl) {
            URL.revokeObjectURL(captures[reviewSlot].dataUrl);
        }
        captures[reviewSlot] = {
            blob: reviewBlob,
            dataUrl: reviewDataUrl
        };
        screenReview.classList.remove('visible');
        nextSlot = computeNextSlot();
        updateUI();
        reviewBlob = null;
        reviewDataUrl = null;
    }

    function onRetake() {
        if (reviewDataUrl) URL.revokeObjectURL(reviewDataUrl);
        reviewBlob = null;
        reviewDataUrl = null;
        screenReview.classList.remove('visible');
    }

    // ---- Preview screen ----
    function showPreview() {
        var filled = captures.filter(Boolean).length;
        if (filled === 0) {
            showToast('Capture at least one card first');
            return;
        }

        previewTitleEl.textContent = 'Ready to Identify';
        previewSub.textContent = filled + ' card' + (filled !== 1 ? 's' : '') + ' captured';
        previewGrid.innerHTML = '';
        previewGo.style.display = '';
        previewBack.textContent = 'Back';
        previewBack.onclick = null;

        for (var i = 0; i < TOTAL_SLOTS; i++) {
            var cell = document.createElement('div');
            cell.className = 'preview-cell';

            var numLabel = document.createElement('div');
            numLabel.className = 'cell-num';
            numLabel.textContent = i + 1;
            cell.appendChild(numLabel);

            if (captures[i]) {
                var img = document.createElement('img');
                img.src = captures[i].dataUrl;
                cell.appendChild(img);
            } else {
                var empty = document.createElement('div');
                empty.className = 'cell-empty';
                empty.textContent = 'Empty';
                cell.appendChild(empty);
            }

            cell.addEventListener('click', (function(idx) {
                return function() {
                    screenPreview.classList.remove('visible');
                    nextSlot = idx;
                    updateUI();
                };
            })(i));

            previewGrid.appendChild(cell);
        }

        screenPreview.classList.add('visible');
    }

    // ---- Submit ----
    function onSubmit() {
        var filledIndices = [];
        for (var i = 0; i < TOTAL_SLOTS; i++) {
            if (captures[i]) filledIndices.push(i);
        }
        if (filledIndices.length === 0) {
            showToast('No cards captured');
            return;
        }

        screenPreview.classList.remove('visible');
        screenUploading.classList.add('visible');

        var formData = new FormData();
        for (var j = 0; j < filledIndices.length; j++) {
            var idx = filledIndices[j];
            formData.append('card_' + idx, captures[idx].blob, 'card_' + idx + '.jpg');
        }

        fetch('/slide-scan/identify?variants=true', {
            method: 'POST',
            body: formData
        }).then(function(resp) {
            return resp.json();
        }).then(function(data) {
            screenUploading.classList.remove('visible');
            if (data.error) {
                showToast('Error: ' + data.error);
                return;
            }
            showResults(data);
        }).catch(function(err) {
            screenUploading.classList.remove('visible');
            showToast('Upload failed: ' + err.message);
            console.error('Submit error:', err);
        });
    }

    // ---- Results display ----
    function showResults(data) {
        var cards = data.cards || [];
        var totalValue = data.total_value || 0;

        previewTitleEl.textContent = 'Results';
        previewSub.textContent = cards.length + ' card' + (cards.length !== 1 ? 's' : '') +
            ' identified \u2014 Total: $' + totalValue.toFixed(2);

        previewGrid.innerHTML = '';

        for (var i = 0; i < cards.length; i++) {
            var card = cards[i];
            var cell = document.createElement('div');
            cell.className = 'preview-cell';
            cell.style.cursor = 'default';

            var imgUrl = card.local_image_url || card.image_url || card.segment_image_url;
            if (imgUrl) {
                var img = document.createElement('img');
                img.src = imgUrl;
                cell.appendChild(img);
            }

            var info = document.createElement('div');
            info.style.cssText = 'position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(0,0,0,0.85));padding:4px 5px 3px;';

            var name = document.createElement('div');
            name.style.cssText = 'font-size:9px;font-weight:600;color:#fff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
            name.textContent = card.card_name || 'Unknown';
            info.appendChild(name);

            var price = card.variant_price || card.market_price;
            if (price) {
                var priceEl = document.createElement('div');
                priceEl.style.cssText = 'font-size:10px;font-weight:700;color:#4ecca3;';
                priceEl.textContent = '$' + price.toFixed(2);
                if (card.detected_variant && card.detected_variant !== 'normal') {
                    priceEl.textContent += ' (' + card.detected_variant + ')';
                }
                info.appendChild(priceEl);
            }

            cell.appendChild(info);
            previewGrid.appendChild(cell);
        }

        previewBack.textContent = 'Scan Again';
        previewBack.onclick = function() {
            for (var k = 0; k < TOTAL_SLOTS; k++) {
                if (captures[k] && captures[k].dataUrl) {
                    URL.revokeObjectURL(captures[k].dataUrl);
                }
                captures[k] = null;
            }
            nextSlot = 0;
            updateUI();
            screenPreview.classList.remove('visible');
        };
        previewGo.style.display = 'none';

        screenPreview.classList.add('visible');
    }

    // ---- Toast ----
    var toastTimer = null;
    function showToast(msg) {
        toastEl.textContent = msg;
        toastEl.classList.add('show');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function() { toastEl.classList.remove('show'); }, 3000);
    }

    // ---- Event listeners ----
    shutterBtn.addEventListener('click', onShutter);
    btnKeep.addEventListener('click', onKeep);
    btnRetake.addEventListener('click', onRetake);
    submitBtn.addEventListener('click', showPreview);
    previewBack.addEventListener('click', function() {
        screenPreview.classList.remove('visible');
    });
    previewGo.addEventListener('click', onSubmit);

    // ---- Init ----
    buildSlots();
    updateUI();
    startCamera();

})();
</script>
</body>
</html>
"""
