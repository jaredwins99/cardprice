"""Slide-scan UI: motion-based real-time card capture for 9-card binder pages.

Flow: For each of 3 rows, user taps "Scan Row" and slides phone across the row.
MotionAnalyzer detects card transitions in real time and auto-captures 3 cards
per row.  After 3 rows (9 cards), auto-submits to /slide-scan/identify.

Capture triggers (from MotionAnalyzer):
  - Pause capture: micro-pause while card is centered (diff drops below avg)
  - Transition capture: card boundary spike (captures previous stable frame)
  - Cadence capture: fallback timer if neither fires within expected window

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
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#000;color:#fff;font-family:-apple-system,system-ui,sans-serif;
  overflow:hidden;height:100dvh;width:100vw;display:flex;flex-direction:column}
#video{width:100%;flex:1;object-fit:cover}
canvas#drawCanvas{display:none}

#topBar{position:absolute;top:0;left:0;right:0;padding:12px 16px;
  display:flex;justify-content:space-between;align-items:center;
  background:linear-gradient(rgba(0,0,0,.6),transparent);z-index:10}
#rowLabel{font-size:20px;font-weight:700}
#status{font-size:14px;opacity:.8}

#thumbStrip{position:absolute;top:56px;left:0;right:0;display:flex;
  justify-content:center;gap:4px;padding:8px;z-index:10}
.thumb{width:36px;height:50px;border:2px solid rgba(255,255,255,.3);
  border-radius:4px;background:rgba(0,0,0,.4);object-fit:cover}
.thumb.filled{border-color:#4f4}
.thumb.active{border-color:#ff0;box-shadow:0 0 8px #ff0}

#diagBar{position:absolute;bottom:88px;left:12px;right:12px;height:40px;
  z-index:10;display:none;flex-direction:column;gap:2px}
#diffMeter{height:6px;background:rgba(255,255,255,.12);border-radius:3px}
#diffFill{height:100%;width:0;border-radius:3px;transition:width 60ms;background:#4f4}
#diagText{font-size:11px;font-family:monospace;opacity:.7;text-align:center}

#bottomBar{position:absolute;bottom:0;left:0;right:0;padding:20px;
  display:flex;justify-content:center;z-index:10;
  background:linear-gradient(transparent,rgba(0,0,0,.7))}
#scanBtn{padding:16px 48px;font-size:20px;font-weight:700;border:none;
  border-radius:50px;background:#4f4;color:#000;cursor:pointer;transition:all .15s}
#scanBtn:active{transform:scale(.95)}
#scanBtn:disabled{background:#555;color:#999}
#scanBtn.scanning{background:#f44;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.7}}

#overlay{position:absolute;inset:0;display:none;z-index:20;
  background:rgba(0,0,0,.85);justify-content:center;align-items:center;
  flex-direction:column;gap:16px}
#overlay.show{display:flex}
#overlay .msg{font-size:22px;font-weight:700}
#overlay .sub{font-size:14px;opacity:.7}
.flash{position:absolute;inset:0;background:#fff;z-index:15;
  animation:flashAnim .15s forwards;pointer-events:none}
@keyframes flashAnim{from{opacity:.6}to{opacity:0}}
</style>
</head>
<body>
<video id="video" autoplay playsinline muted></video>
<canvas id="drawCanvas"></canvas>

<div id="topBar">
  <span id="rowLabel">Row 1 / 3</span>
  <span id="status">Ready</span>
</div>

<div id="thumbStrip"></div>

<div id="diagBar">
  <div id="diffMeter"><div id="diffFill"></div></div>
  <div id="diagText">--</div>
</div>

<div id="bottomBar">
  <button id="scanBtn" onclick="toggleScan()">Scan Row 1</button>
</div>

<div id="overlay">
  <div class="msg" id="overlayMsg">Submitting...</div>
  <div class="sub" id="overlaySub">Identifying 9 cards</div>
</div>

<script>
// ===========================================================================
// MotionAnalyzer -- inlined for single-file deployment
// ===========================================================================

class MotionAnalyzer {
    constructor(opts = {}) {
        this.sampleWidth       = opts.sampleWidth       || 160;
        this.sampleHeight      = opts.sampleHeight      || 90;
        this.historyLen        = opts.historyLen         || 30;
        this.idleThreshold     = opts.idleThreshold      ?? 1.5;
        this.scanStartFrames   = opts.scanStartFrames    || 8;
        this.scanEndFrames     = opts.scanEndFrames      || 15;
        this.pauseRatio        = opts.pauseRatio         ?? 0.4;
        this.transitionRatio   = opts.transitionRatio    ?? 2.0;
        this.minCaptureGapMs   = opts.minCaptureGapMs    || 500;
        this.cadenceTimeoutMs  = opts.cadenceTimeoutMs   || 1500;
        this.stableBufferLen   = opts.stableBufferLen    || 5;

        this._prevGray        = null;
        this._diffHistory     = [];
        this._consecutiveMove = 0;
        this._consecutiveIdle = 0;
        this._state           = 'idle';
        this._lastCaptureTime = 0;
        this._lastCaptureType = null;
        this._stableFrames    = [];
        this._scanStartTime   = 0;
        this._captureCount    = 0;
        this._prevHProj       = null;
        this._sampleCanvas    = null;
        this._sampleCtx       = null;
    }

    processFrame(video, canvas, ctx) {
        const now = performance.now();
        const gray = this._downsampleGray(video);

        if (!this._prevGray) {
            this._prevGray = gray;
            this._prevHProj = this._horizontalProjection(gray);
            return this._result(null, null, 0, 0, 0, 0, false, now);
        }

        const diff = this._meanAbsDiff(gray, this._prevGray);
        const hProj = this._horizontalProjection(gray);
        const speed = this._estimateShift(this._prevHProj, hProj);

        this._prevGray = gray;
        this._prevHProj = hProj;

        this._diffHistory.push({ diff, time: now });
        if (this._diffHistory.length > this.historyLen) this._diffHistory.shift();

        const avgDiff = this._diffHistory.reduce((s, d) => s + d.diff, 0) / this._diffHistory.length;
        const transitionScore = diff / Math.max(avgDiff, 0.01);
        const moving = diff > this.idleThreshold;

        if (moving) { this._consecutiveMove++; this._consecutiveIdle = 0; }
        else        { this._consecutiveIdle++; this._consecutiveMove = 0; }

        let capture = null;
        let captureFrame = null;

        if (this._state === 'idle') {
            if (this._consecutiveMove >= this.scanStartFrames) {
                this._state = 'scanning';
                this._scanStartTime = now;
                this._captureCount = 0;
                this._lastCaptureTime = now;
                this._stableFrames = [];
            }
        }

        if (this._state === 'scanning') {
            this._bufferStableFrame(canvas, diff, now);
            const timeSinceCapture = now - this._lastCaptureTime;
            const canCapture = timeSinceCapture >= this.minCaptureGapMs;

            if (canCapture) {
                // (a) Pause capture
                if (this._diffHistory.length >= 5 && diff < avgDiff * this.pauseRatio && !moving) {
                    captureFrame = this._getBestStableFrame();
                    if (captureFrame) capture = 'pause';
                }
                // (b) Transition capture
                if (!capture && transitionScore > this.transitionRatio && this._diffHistory.length >= 5) {
                    captureFrame = this._getBestStableFrame();
                    if (captureFrame) capture = 'transition';
                }
                // (c) Cadence capture
                if (!capture && timeSinceCapture >= this.cadenceTimeoutMs && moving) {
                    captureFrame = this._getBestStableFrame() || this._snapshotCanvas(canvas);
                    capture = 'cadence';
                }
            }

            if (capture) {
                this._lastCaptureTime = now;
                this._lastCaptureType = capture;
                this._captureCount++;
                this._stableFrames = [];
            }

            if (this._consecutiveIdle >= this.scanEndFrames) {
                this._state = 'ending';
            }
        }

        if (this._state === 'ending') {
            if (this._consecutiveMove >= 3) this._state = 'scanning';
        }

        return this._result(capture, captureFrame, diff, avgDiff, transitionScore, speed, moving, now);
    }

    reset() {
        this._prevGray = null; this._prevHProj = null;
        this._diffHistory = []; this._consecutiveMove = 0; this._consecutiveIdle = 0;
        this._state = 'idle'; this._lastCaptureTime = 0; this._lastCaptureType = null;
        this._stableFrames = []; this._scanStartTime = 0; this._captureCount = 0;
    }

    get state() { return this._state; }
    get captureCount() { return this._captureCount; }

    _downsampleGray(video) {
        const w = this.sampleWidth, h = this.sampleHeight;
        if (!this._sampleCanvas) {
            this._sampleCanvas = document.createElement('canvas');
            this._sampleCanvas.width = w; this._sampleCanvas.height = h;
            this._sampleCtx = this._sampleCanvas.getContext('2d', { willReadFrequently: true });
        }
        this._sampleCtx.drawImage(video, 0, 0, w, h);
        const rgba = this._sampleCtx.getImageData(0, 0, w, h).data;
        const gray = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
            gray[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
        }
        return gray;
    }

    _meanAbsDiff(a, b) {
        let sum = 0; const len = a.length; const len4 = len & ~3;
        for (let i = 0; i < len4; i += 4) {
            sum += Math.abs(a[i] - b[i]) + Math.abs(a[i+1] - b[i+1])
                 + Math.abs(a[i+2] - b[i+2]) + Math.abs(a[i+3] - b[i+3]);
        }
        for (let i = len4; i < len; i++) sum += Math.abs(a[i] - b[i]);
        return sum / len;
    }

    _horizontalProjection(gray) {
        const w = this.sampleWidth, h = this.sampleHeight;
        const proj = new Float32Array(w);
        for (let x = 0; x < w; x++) {
            let sum = 0;
            for (let y = 0; y < h; y++) sum += gray[y * w + x];
            proj[x] = sum / h;
        }
        return proj;
    }

    _estimateShift(prevProj, currProj, maxShift) {
        if (!prevProj || !currProj) return 0;
        maxShift = maxShift || 20;
        const len = prevProj.length;
        let bestCorr = -Infinity, bestShift = 0;
        for (let shift = -maxShift; shift <= maxShift; shift++) {
            let corr = 0, count = 0;
            const start = Math.max(0, -shift), end = Math.min(len, len - shift);
            for (let i = start; i < end; i++) { corr += prevProj[i] * currProj[i + shift]; count++; }
            if (count > 0) { corr /= count; if (corr > bestCorr) { bestCorr = corr; bestShift = shift; } }
        }
        return bestShift;
    }

    _bufferStableFrame(canvas, diff, time) {
        const avgDiff = this._diffHistory.length > 0
            ? this._diffHistory.reduce((s, d) => s + d.diff, 0) / this._diffHistory.length : diff;
        if (diff < avgDiff * 1.5) {
            const snapshot = this._snapshotCanvas(canvas);
            this._stableFrames.push({ canvas: snapshot, diff, time });
            if (this._stableFrames.length > this.stableBufferLen) this._stableFrames.shift();
        }
    }

    _getBestStableFrame() {
        if (this._stableFrames.length === 0) return null;
        let best = this._stableFrames[0];
        for (let i = 1; i < this._stableFrames.length; i++) {
            if (this._stableFrames[i].diff < best.diff) best = this._stableFrames[i];
        }
        return best.canvas;
    }

    _snapshotCanvas(sourceCanvas) {
        const snap = document.createElement('canvas');
        snap.width = sourceCanvas.width; snap.height = sourceCanvas.height;
        snap.getContext('2d').drawImage(sourceCanvas, 0, 0);
        return snap;
    }

    _result(capture, captureFrame, diff, avgDiff, transitionScore, speed, moving, now) {
        return {
            state: this._state, capture, captureFrame,
            diff: Math.round(diff * 100) / 100,
            avgDiff: Math.round(avgDiff * 100) / 100,
            transitionScore: Math.round(transitionScore * 100) / 100,
            speed, moving,
            scanDurationMs: this._state === 'scanning' ? Math.round(now - this._scanStartTime) : 0,
            captureCount: this._captureCount,
        };
    }
}

// ===========================================================================
// Slide-scan UI logic
// ===========================================================================

const V = document.getElementById('video');
const C = document.getElementById('drawCanvas');
const ctx = C.getContext('2d', { willReadFrequently: true });
const thumbStrip = document.getElementById('thumbStrip');
const scanBtn = document.getElementById('scanBtn');
const rowLabel = document.getElementById('rowLabel');
const statusEl = document.getElementById('status');
const diagBar = document.getElementById('diagBar');
const diffFill = document.getElementById('diffFill');
const diagText = document.getElementById('diagText');
const overlay = document.getElementById('overlay');

// State
let currentRow = 0;        // 0, 1, 2
let scanning = false;
let captures = new Array(9).fill(null);    // blob URLs for thumbnails
let captureBlobs = new Array(9).fill(null); // actual blobs for upload
let rowCaptures = 0;       // cards captured in current row
let rafId = null;

const CARDS_PER_ROW = 3;
const TOTAL_ROWS = 3;

// Motion analyzer instance
const motion = new MotionAnalyzer({
    scanStartFrames: 6,    // ~200ms at 30fps to confirm scanning started
    scanEndFrames: 20,     // ~670ms of stillness to end scan
    minCaptureGapMs: 400,  // at least 400ms between captures
    cadenceTimeoutMs: 1800, // force capture if nothing for 1.8s
    idleThreshold: 1.5,    // mean pixel diff below this = idle
    pauseRatio: 0.4,       // diff < 40% of avg = micro-pause
    transitionRatio: 2.0,  // diff > 2x avg = card boundary
});

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
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not available. Need HTTPS — use the tunnel URL from the QR code.');
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false
        });
        V.srcObject = stream;
        await new Promise((resolve, reject) => {
            V.addEventListener('loadeddata', resolve, { once: true });
            V.addEventListener('error', reject, { once: true });
            setTimeout(resolve, 8000);
        });
        await V.play();
        C.width = V.videoWidth;
        C.height = V.videoHeight;
    } catch (e) {
        document.body.innerHTML = '<div style="padding:30px;text-align:center;color:#e74c3c;font-size:18px;">Camera Error<br><br><span style="font-size:14px;color:#aaa;">' + e.message + '</span><br><br><span style="font-size:12px;color:#666;">Make sure you\'re using the HTTPS tunnel URL<br>(scan QR code from main page)</span></div>';
    }
}
initCamera();

function updateActiveThumb() {
    for (let i = 0; i < 9; i++) {
        const t = document.getElementById('thumb_' + i);
        t.classList.toggle('active', i === currentRow * CARDS_PER_ROW + rowCaptures && scanning);
    }
}

// -- Scan control --------------------------------------------------------

function toggleScan() {
    if (scanning) {
        stopScanning();
    } else {
        startScanning();
    }
}

function startScanning() {
    scanning = true;
    rowCaptures = 0;
    motion.reset();

    scanBtn.textContent = 'Cancel';
    scanBtn.classList.add('scanning');
    statusEl.textContent = 'Slide across row ' + (currentRow + 1) + '...';
    diagBar.style.display = 'flex';
    updateActiveThumb();

    rafId = requestAnimationFrame(scanLoop);
}

function stopScanning() {
    scanning = false;
    cancelAnimationFrame(rafId);
    scanBtn.classList.remove('scanning');
    diagBar.style.display = 'none';

    if (currentRow < TOTAL_ROWS) {
        scanBtn.textContent = 'Scan Row ' + (currentRow + 1);
    }
    statusEl.textContent = 'Ready';
    updateActiveThumb();
}

// -- Main scan loop ------------------------------------------------------

function scanLoop() {
    if (!scanning || rowCaptures >= CARDS_PER_ROW) return;

    // Draw current video frame onto canvas (full resolution for capture)
    ctx.drawImage(V, 0, 0, C.width, C.height);

    // Run motion analysis
    const result = motion.processFrame(V, C, ctx);

    // Update diagnostic display
    updateDiagnostics(result);

    // Update status based on motion state
    if (result.state === 'idle') {
        statusEl.textContent = 'Start sliding... (' + rowCaptures + '/' + CARDS_PER_ROW + ')';
    } else if (result.state === 'scanning') {
        statusEl.textContent = 'Scanning... ' + rowCaptures + '/' + CARDS_PER_ROW + ' cards';
    } else if (result.state === 'ending') {
        statusEl.textContent = 'Scan paused. Keep sliding or tap Cancel.';
    }

    // Handle capture if MotionAnalyzer triggered one
    if (result.capture && result.captureFrame && rowCaptures < CARDS_PER_ROW) {
        captureCard(result.captureFrame, result.capture);
    }

    // Auto-end row if we got all cards and scan is ending
    if (rowCaptures >= CARDS_PER_ROW) {
        finishRow();
        return;
    }

    rafId = requestAnimationFrame(scanLoop);
}

function updateDiagnostics(result) {
    // Diff bar: scale 0-20 range to 0-100% width
    const pct = Math.min(100, (result.diff / 20) * 100);
    diffFill.style.width = pct + '%';

    // Color: green=idle, yellow=moving, red=transition spike
    if (result.transitionScore > 2.0) {
        diffFill.style.background = '#f44';
    } else if (result.moving) {
        diffFill.style.background = '#ff0';
    } else {
        diffFill.style.background = '#4f4';
    }

    diagText.textContent = result.state + ' | diff=' + result.diff
        + ' avg=' + result.avgDiff + ' tr=' + result.transitionScore
        + ' spd=' + result.speed;
}

// -- Capture handling ----------------------------------------------------

function captureCard(frameCanvas, triggerType) {
    const pos = currentRow * CARDS_PER_ROW + rowCaptures;

    // Flash effect
    const flash = document.createElement('div');
    flash.className = 'flash';
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 200);

    // Convert captured canvas to blob
    frameCanvas.toBlob(blob => {
        if (!blob) return;
        captureBlobs[pos] = blob;
        const url = URL.createObjectURL(blob);
        captures[pos] = url;

        const thumb = document.getElementById('thumb_' + pos);
        thumb.src = url;
        thumb.classList.add('filled');

        rowCaptures++;
        statusEl.textContent = rowCaptures + '/' + CARDS_PER_ROW
            + ' cards (' + triggerType + ')';
        updateActiveThumb();

        console.log('[slide-scan] Captured card ' + pos + ' via ' + triggerType);

        if (rowCaptures >= CARDS_PER_ROW) {
            finishRow();
        }
    }, 'image/jpeg', 0.92);
}

// Manual capture fallback: tap video during scan to force-capture
V.addEventListener('click', () => {
    if (!scanning || rowCaptures >= CARDS_PER_ROW) return;
    // Grab current live frame
    ctx.drawImage(V, 0, 0, C.width, C.height);
    const snap = document.createElement('canvas');
    snap.width = C.width;
    snap.height = C.height;
    snap.getContext('2d').drawImage(C, 0, 0);
    captureCard(snap, 'manual');
});

// -- Row and page management ---------------------------------------------

function finishRow() {
    scanning = false;
    cancelAnimationFrame(rafId);
    scanBtn.classList.remove('scanning');
    diagBar.style.display = 'none';
    currentRow++;

    if (currentRow >= TOTAL_ROWS) {
        // All 9 cards captured -- submit
        scanBtn.disabled = true;
        scanBtn.textContent = 'Submitting...';
        rowLabel.textContent = 'Done!';
        statusEl.textContent = 'Identifying cards...';
        submitCards();
    } else {
        rowLabel.textContent = 'Row ' + (currentRow + 1) + ' / ' + TOTAL_ROWS;
        scanBtn.textContent = 'Scan Row ' + (currentRow + 1);
        statusEl.textContent = 'Row ' + currentRow + ' done!';
    }
}

async function submitCards() {
    overlay.classList.add('show');
    document.getElementById('overlayMsg').textContent = 'Identifying...';
    document.getElementById('overlaySub').textContent =
        captureBlobs.filter(Boolean).length + ' cards';

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
    motion.reset();

    rowLabel.textContent = 'Row 1 / ' + TOTAL_ROWS;
    scanBtn.textContent = 'Scan Row 1';
    scanBtn.disabled = false;
    scanBtn.classList.remove('scanning');
    statusEl.textContent = 'Ready';
    diagBar.style.display = 'none';

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
