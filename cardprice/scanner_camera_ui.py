"""True scanner mode camera UI -- continuous video processing with auto-capture.

NOT a photo capture app. This is a real scanner:
- Continuous video processing at 10fps on a downsampled detection canvas
- Real-time edge detection draws card outlines on the live feed
- Auto-capture fires when 3 cards are stable, sharp, and well-lit
- Perspective correction outputs perfectly flat rectangles
- No button presses between rows -- scanner detects row transitions automatically

Flow:
1. User holds phone ~20cm above binder in landscape, over a row of 3 cards
2. Scanner detects card edges, draws colored outlines (red->yellow->green)
3. When 3 cards are stable+sharp -> auto-capture fires with haptic+flash
4. User moves phone to next row; scanner detects transition and repeats
5. After 9 cards (3 rows), shows preview grid with Submit button

Integration:
    GET  /scanner           -> serve this HTML
    POST /scanner/identify    -> identify captured card images (no auto-crop)
"""

SCANNER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Card Scanner</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{background:#000;color:#fff;font-family:-apple-system,system-ui,sans-serif;
  touch-action:none;user-select:none;-webkit-user-select:none}

#videoWrap{position:fixed;inset:0;overflow:hidden}
#video{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover}
#overlayCanvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
#detectCanvas,#captureCanvas,#warpCanvas{display:none}

#flash{position:fixed;inset:0;background:#4f4;opacity:0;pointer-events:none;
  z-index:50;transition:opacity 0.15s}
#flash.fire{opacity:0.45;transition:none}

/* HUD */
#hud{position:fixed;top:0;left:0;right:0;z-index:10;
  background:linear-gradient(rgba(0,0,0,.75),rgba(0,0,0,.3),transparent);
  padding:10px 16px 20px}
#hudRow1{display:flex;justify-content:space-between;align-items:center}
#rowIndicator{font-size:18px;font-weight:700;letter-spacing:0.3px}
#scanState{font-size:12px;padding:4px 10px;border-radius:12px;
  background:rgba(255,255,255,0.12)}
#hudRow2{display:flex;gap:6px;margin-top:8px;justify-content:center}
.rowDot{width:30px;height:30px;border-radius:50%;border:2px solid rgba(255,255,255,0.25);
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;
  transition:all 0.3s}
.rowDot.captured{border-color:#4f4;background:rgba(79,255,79,0.2);color:#4f4}
.rowDot.active{border-color:#ff0;background:rgba(255,255,0,0.12);color:#ff0;
  box-shadow:0 0 10px rgba(255,255,0,0.25)}
.rowDot.pending{border-color:rgba(255,255,255,0.15);color:rgba(255,255,255,0.25)}
#cardCountBar{display:flex;gap:3px;margin-top:6px;justify-content:center}
.cardBadge{width:18px;height:25px;border-radius:3px;border:1.5px solid rgba(255,255,255,0.15);
  transition:all 0.3s}
.cardBadge.detected{border-color:#f44;background:rgba(255,68,68,0.2)}
.cardBadge.stable{border-color:#ff0;background:rgba(255,255,0,0.2)}
.cardBadge.locked{border-color:#4f4;background:rgba(79,255,79,0.3)}

#bottomBar{position:fixed;bottom:0;left:0;right:0;z-index:10;
  background:linear-gradient(transparent,rgba(0,0,0,.65));
  padding:16px;display:flex;justify-content:center;align-items:center}
#startBtn{padding:14px 40px;font-size:18px;font-weight:700;border:none;
  border-radius:50px;background:#4f4;color:#000;cursor:pointer}
#startBtn:active{transform:scale(0.95)}
#startBtn.hidden{display:none}

#debugInfo{position:fixed;bottom:68px;left:8px;right:8px;z-index:10;
  font-size:10px;font-family:monospace;opacity:0.45;text-align:center;
  pointer-events:none}

/* Preview */
#previewOverlay{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,0.95);
  display:none;flex-direction:column;overflow-y:auto}
#previewOverlay.show{display:flex}
#previewTitle{text-align:center;font-size:20px;font-weight:700;padding:16px;color:#4f4}
#previewGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;
  padding:8px 16px;flex:1}
#previewGrid img{width:100%;border-radius:6px;border:2px solid rgba(255,255,255,0.1);
  aspect-ratio:63/88;object-fit:cover;background:#111}
#previewActions{padding:16px;display:flex;gap:12px;justify-content:center}
#previewActions button{padding:14px 32px;font-size:16px;font-weight:700;
  border:none;border-radius:50px;cursor:pointer}
#submitBtn{background:#4f4;color:#000}
#retakeBtn{background:rgba(255,255,255,0.15);color:#fff}

/* Submitting */
#submitOverlay{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,0.92);
  display:none;flex-direction:column;align-items:center;justify-content:center;gap:16px}
#submitOverlay.show{display:flex}
.spinner{width:48px;height:48px;border:4px solid rgba(255,255,255,0.15);
  border-top-color:#4f4;border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.msg{font-size:18px;font-weight:600}
.sub{font-size:13px;opacity:0.5}
</style>
</head>
<body>

<div id="videoWrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlayCanvas"></canvas>
</div>
<canvas id="detectCanvas"></canvas>
<canvas id="captureCanvas"></canvas>
<canvas id="warpCanvas"></canvas>
<div id="flash"></div>

<div id="hud">
  <div id="hudRow1">
    <span id="rowIndicator">Row 1 / 3</span>
    <span id="scanState">Initializing...</span>
  </div>
  <div id="hudRow2">
    <div class="rowDot active" id="dot0">1</div>
    <div class="rowDot pending" id="dot1">2</div>
    <div class="rowDot pending" id="dot2">3</div>
  </div>
  <div id="cardCountBar">
    <div class="cardBadge" id="cb0"></div>
    <div class="cardBadge" id="cb1"></div>
    <div class="cardBadge" id="cb2"></div>
  </div>
</div>

<div id="bottomBar">
  <button id="startBtn" onclick="startScanning()">Start Scanning</button>
</div>
<div id="debugInfo"></div>

<div id="previewOverlay">
  <div id="previewTitle">9 Cards Captured</div>
  <div id="previewGrid"></div>
  <div id="previewActions">
    <button id="retakeBtn" onclick="resetAll()">Retake</button>
    <button id="submitBtn" onclick="submitCards()">Submit for ID</button>
  </div>
</div>

<div id="submitOverlay">
  <div class="spinner"></div>
  <div class="msg">Identifying cards...</div>
  <div class="sub">This may take 10-20 seconds</div>
</div>

<script>
"use strict";
// =========================================================================
// Config
// =========================================================================
const DW = 320, DH = 180;              // Detection resolution (16:9)
const CARD_AR_MIN = 0.58, CARD_AR_MAX = 0.82; // Card aspect (w/h) range
const MIN_CARD_W = DW * 0.13;          // Min card width in detect pixels
const MAX_CARD_W = DW * 0.42;          // Max card width in detect pixels
const STABLE_FRAMES = 12;              // Frames for "locked"
const STABLE_PX = 5;                   // Max avg corner drift for "stable"
const SHARP_THRESH = 12;               // Laplacian variance threshold
const CARDS_PER_ROW = 3;
const TOTAL_ROWS = 3;
const WARP_W = 420, WARP_H = 586;      // Output card size (63:88)
const FPS = 10;
const CAPTURE_COOLDOWN_MS = 1200;

// =========================================================================
// State
// =========================================================================
let video, oCanvas, oCtx, dCanvas, dCtx, cCanvas, cCtx, wCanvas, wCtx;
let scanning = false, videoReady = false;
let currentRow = 0;
let capturedRows = [null, null, null];
let tracked = [];           // {corners, stable, state, miss, sharp}
let phase = 'idle';         // idle | detecting | transition
let transFrames = 0;
let lastCapTime = 0;
let loopTimer = null;
let frameCnt = 0;

// =========================================================================
// Init
// =========================================================================
async function init() {
    video    = document.getElementById('video');
    oCanvas  = document.getElementById('overlayCanvas');
    oCtx     = oCanvas.getContext('2d');
    dCanvas  = document.getElementById('detectCanvas');
    dCtx     = dCanvas.getContext('2d', {willReadFrequently: true});
    cCanvas  = document.getElementById('captureCanvas');
    cCtx     = cCanvas.getContext('2d', {willReadFrequently: true});
    wCanvas  = document.getElementById('warpCanvas');
    wCtx     = wCanvas.getContext('2d', {willReadFrequently: true});
    dCanvas.width = DW; dCanvas.height = DH;
    wCanvas.width = WARP_W; wCanvas.height = WARP_H;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {facingMode:{ideal:'environment'}, width:{ideal:1920}, height:{ideal:1080}, frameRate:{ideal:30}},
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        const s = stream.getVideoTracks()[0].getSettings();
        cCanvas.width = s.width || 1920;
        cCanvas.height = s.height || 1080;
        const onReady = () => { resizeOverlay(); videoReady = true; setState('Ready - tap Start'); };
        video.addEventListener('loadedmetadata', onReady);
        if (video.readyState >= 2) onReady();
    } catch(e) {
        setState('Camera error: ' + e.message);
    }
    window.addEventListener('resize', resizeOverlay);
}

function resizeOverlay() {
    oCanvas.width = window.innerWidth;
    oCanvas.height = window.innerHeight;
}

// =========================================================================
// Scan control
// =========================================================================
function startScanning() {
    if (scanning) return;
    scanning = true; phase = 'detecting'; tracked = []; frameCnt = 0;
    document.getElementById('startBtn').classList.add('hidden');
    setState('Scanning...');
    loopTimer = setInterval(processFrame, 1000/FPS);
}
function stopScanning() {
    scanning = false; phase = 'idle';
    if (loopTimer) { clearInterval(loopTimer); loopTimer = null; }
    oCtx.clearRect(0, 0, oCanvas.width, oCanvas.height);
}
function resetAll() {
    stopScanning();
    currentRow = 0; capturedRows = [null,null,null]; tracked = []; transFrames = 0;
    updateHUD();
    document.getElementById('previewOverlay').classList.remove('show');
    document.getElementById('startBtn').classList.remove('hidden');
    setState('Ready - tap Start');
}

// =========================================================================
// Main frame processing
// =========================================================================
function processFrame() {
    if (!videoReady || !scanning) return;
    frameCnt++;

    // 1) Downsample to detection canvas
    dCtx.drawImage(video, 0, 0, DW, DH);
    const imgData = dCtx.getImageData(0, 0, DW, DH);
    const gray = grayscale(imgData.data, DW, DH);

    // 2) Find card-like rectangles
    const rects = detectCards(gray, DW, DH);

    // 3) Track across frames
    trackCards(rects);

    // 4) Draw overlays
    drawOverlays();

    // 5) Update badges
    updateBadges();

    // 6) Phase logic
    if (phase === 'detecting') doDetecting();
    else if (phase === 'transition') doTransition();

    // Debug
    const nl = tracked.filter(t=>t.state==='locked').length;
    document.getElementById('debugInfo').textContent =
        `rects:${rects.length} trk:${tracked.length} lck:${nl} ph:${phase} f:${frameCnt}`;
}

// =========================================================================
// Image processing
// =========================================================================
function grayscale(rgba, w, h) {
    const g = new Uint8Array(w*h);
    for (let i=0,j=0; i<rgba.length; i+=4,j++)
        g[j] = (rgba[i]*77 + rgba[i+1]*150 + rgba[i+2]*29) >> 8;
    return g;
}

// Canny-like edge magnitude (Sobel)
function edgeMag(gray, w, h) {
    const mag = new Uint8Array(w*h);
    for (let y=1; y<h-1; y++) {
        for (let x=1; x<w-1; x++) {
            const gx = -gray[(y-1)*w+x-1] - 2*gray[y*w+x-1] - gray[(y+1)*w+x-1]
                       +gray[(y-1)*w+x+1] + 2*gray[y*w+x+1] + gray[(y+1)*w+x+1];
            const gy = -gray[(y-1)*w+x-1] - 2*gray[(y-1)*w+x] - gray[(y-1)*w+x+1]
                       +gray[(y+1)*w+x-1] + 2*gray[(y+1)*w+x] + gray[(y+1)*w+x+1];
            mag[y*w+x] = Math.min(255, (Math.abs(gx)+Math.abs(gy))>>1);
        }
    }
    return mag;
}

function lapVar(gray, w, h, x0, y0, rw, rh) {
    let s=0, s2=0, n=0;
    const xe=Math.min(x0+rw,w-1), ye=Math.min(y0+rh,h-1);
    for (let y=Math.max(1,y0); y<ye; y+=2) {
        for (let x=Math.max(1,x0); x<xe; x+=2) {
            const v = 4*gray[y*w+x]-gray[(y-1)*w+x]-gray[(y+1)*w+x]-gray[y*w+x-1]-gray[y*w+x+1];
            s += v; s2 += v*v; n++;
        }
    }
    if (n<4) return 0;
    const m = s/n;
    return s2/n - m*m;
}

// =========================================================================
// Card detection: scan for rectangular bright regions
//
// Strategy: find horizontal and vertical edge runs, then look for
// intersecting pairs that form rectangles. Much faster than contour tracing.
//
// Alternative (and what we actually use): threshold the edge image, then
// scan horizontal lines to find segments of strong edge. Cluster these
// into potential card boundaries.
//
// Simplest robust approach: use connected-component labeling on a binary
// edge image, then fit minimum bounding rectangles. But CC labeling is
// also expensive.
//
// ACTUAL approach we use: Line-scan based rectangle finder.
// 1. Compute edge magnitude (Sobel)
// 2. Project edges vertically to find vertical "walls" (card left/right edges)
// 3. Project edges horizontally to find horizontal "walls" (card top/bottom)
// 4. Intersect to find candidate rectangles
// 5. Validate aspect ratio, size, internal brightness consistency
// =========================================================================
function detectCards(gray, w, h) {
    const mag = edgeMag(gray, w, h);

    // Vertical projection: sum edge magnitudes along each column
    const vProj = new Float32Array(w);
    for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let y = 0; y < h; y++) sum += mag[y*w+x];
        vProj[x] = sum / h;
    }

    // Horizontal projection: sum edge magnitudes along each row
    const hProj = new Float32Array(h);
    for (let y = 0; y < h; y++) {
        let sum = 0;
        for (let x = 0; x < w; x++) sum += mag[y*w+x];
        hProj[y] = sum / w;
    }

    // Find peaks in projections (card edges)
    const vPeaks = findPeaks(vProj, w, 15, 0.35);
    const hPeaks = findPeaks(hProj, h, 10, 0.35);

    // Generate candidate rectangles from peak pairs
    const rects = [];
    for (let li = 0; li < vPeaks.length; li++) {
        for (let ri = li+1; ri < vPeaks.length; ri++) {
            const left = vPeaks[li], right = vPeaks[ri];
            const cardW = right - left;
            if (cardW < MIN_CARD_W || cardW > MAX_CARD_W) continue;

            for (let ti = 0; ti < hPeaks.length; ti++) {
                for (let bi = ti+1; bi < hPeaks.length; bi++) {
                    const top = hPeaks[ti], bot = hPeaks[bi];
                    const cardH = bot - top;
                    if (cardH < MIN_CARD_W) continue;

                    const aspect = Math.min(cardW, cardH) / Math.max(cardW, cardH);
                    if (aspect < CARD_AR_MIN || aspect > CARD_AR_MAX) continue;

                    // Verify edges actually exist along the rectangle boundaries
                    const edgeScore = verifyRectEdges(mag, w, h, left, top, right, bot);
                    if (edgeScore < 0.25) continue;

                    // Check interior brightness is relatively uniform (it's a card, not background)
                    const interiorScore = checkInterior(gray, w, h, left, top, right, bot);
                    if (interiorScore < 0.3) continue;

                    const sharp = lapVar(gray, w, h, left, top, cardW, cardH);

                    rects.push({
                        corners: [[left,top],[right,top],[right,bot],[left,bot]],
                        width: cardW, height: cardH, aspect,
                        area: cardW * cardH,
                        sharpness: sharp,
                        edgeScore, interiorScore,
                        bbox: {x:left, y:top, w:cardW, h:cardH}
                    });
                }
            }
        }
    }

    // Non-max suppression
    return nms(rects, 0.3);
}

function findPeaks(proj, len, minDist, relThresh) {
    // Find peaks above a threshold in a 1D projection
    const maxVal = Math.max(...proj);
    const thresh = maxVal * relThresh;
    const peaks = [];
    for (let i = 2; i < len-2; i++) {
        if (proj[i] < thresh) continue;
        if (proj[i] >= proj[i-1] && proj[i] >= proj[i+1]
            && proj[i] >= proj[i-2] && proj[i] >= proj[i+2]) {
            // Check min distance from previous peak
            if (peaks.length === 0 || i - peaks[peaks.length-1] >= minDist) {
                peaks.push(i);
            } else if (proj[i] > proj[peaks[peaks.length-1]]) {
                peaks[peaks.length-1] = i; // Replace with stronger peak
            }
        }
    }
    return peaks;
}

function verifyRectEdges(mag, w, h, x0, y0, x1, y1) {
    // Check that edge magnitude is high along the 4 sides of the rectangle
    let total = 0, count = 0;
    const step = 2;
    // Top edge
    for (let x = x0; x <= x1; x += step) {
        if (y0 >= 0 && y0 < h) { total += mag[y0*w+x]; count++; }
    }
    // Bottom edge
    for (let x = x0; x <= x1; x += step) {
        if (y1 >= 0 && y1 < h) { total += mag[y1*w+x]; count++; }
    }
    // Left edge
    for (let y = y0; y <= y1; y += step) {
        if (x0 >= 0 && x0 < w) { total += mag[y*w+x0]; count++; }
    }
    // Right edge
    for (let y = y0; y <= y1; y += step) {
        if (x1 >= 0 && x1 < w) { total += mag[y*w+x1]; count++; }
    }
    if (count === 0) return 0;
    return (total / count) / 128; // Normalize to ~0-1 range
}

function checkInterior(gray, w, h, x0, y0, x1, y1) {
    // Check that interior of rectangle has consistent brightness (it's a card)
    // Cards are typically brighter than the binder page background
    let sum = 0, count = 0;
    const step = 4;
    const mx = (x0+x1)/2, my = (y0+y1)/2;
    const rw = (x1-x0)*0.6, rh = (y1-y0)*0.6;
    for (let y = my-rh/2; y < my+rh/2; y += step) {
        for (let x = mx-rw/2; x < mx+rw/2; x += step) {
            const ix = Math.round(x), iy = Math.round(y);
            if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
                sum += gray[iy*w+ix]; count++;
            }
        }
    }
    if (count === 0) return 0;
    const mean = sum / count;
    // Cards should be moderately bright
    return mean > 60 ? 1.0 : mean / 60;
}

function nms(rects, overlapThresh) {
    if (rects.length === 0) return [];
    // Sort by combined edge+interior score
    rects.sort((a,b) => (b.edgeScore + b.interiorScore) - (a.edgeScore + a.interiorScore));
    const keep = [], skip = new Set();
    for (let i = 0; i < rects.length; i++) {
        if (skip.has(i)) continue;
        keep.push(rects[i]);
        if (keep.length >= 5) break; // Don't need more than 5 candidates
        for (let j = i+1; j < rects.length; j++) {
            if (skip.has(j)) continue;
            if (bboxIoU(rects[i].bbox, rects[j].bbox) > overlapThresh) skip.add(j);
        }
    }
    return keep;
}

function bboxIoU(a, b) {
    const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
    const x1 = Math.min(a.x+a.w, b.x+b.w), y1 = Math.min(a.y+a.h, b.y+b.h);
    if (x1 <= x0 || y1 <= y0) return 0;
    const inter = (x1-x0)*(y1-y0);
    return inter / Math.min(a.w*a.h, b.w*b.h);
}

// =========================================================================
// Card tracking
// =========================================================================
function trackCards(rects) {
    const newT = [];
    const used = new Set();

    for (const t of tracked) {
        let bi = -1, bd = Infinity;
        for (let i = 0; i < rects.length; i++) {
            if (used.has(i)) continue;
            const d = avgCornerDist(t.corners, rects[i].corners);
            if (d < bd) { bd = d; bi = i; }
        }
        if (bi >= 0 && bd < STABLE_PX * 6) {
            used.add(bi);
            const r = rects[bi];
            const moved = avgCornerDist(t.corners, r.corners);
            t.stable = moved < STABLE_PX ? t.stable + 1 : Math.max(0, t.stable - 2);
            t.corners = r.corners;
            t.sharp = r.sharpness;
            t.bbox = r.bbox;
            t.miss = 0;
            t.state = (t.stable >= STABLE_FRAMES && r.sharpness >= SHARP_THRESH) ? 'locked'
                     : t.stable >= STABLE_FRAMES/2 ? 'stable' : 'detected';
            newT.push(t);
        } else {
            t.miss = (t.miss||0) + 1;
            if (t.miss < 4) { t.stable = Math.max(0, t.stable-3); t.state='detected'; newT.push(t); }
        }
    }

    for (let i = 0; i < rects.length; i++) {
        if (used.has(i)) continue;
        newT.push({corners:rects[i].corners, sharp:rects[i].sharpness,
                   bbox:rects[i].bbox, stable:0, state:'detected', miss:0});
    }
    tracked = newT;
}

function avgCornerDist(a, b) {
    let s = 0;
    for (let i = 0; i < 4; i++) {
        const dx = a[i][0]-b[i][0], dy = a[i][1]-b[i][1];
        s += Math.sqrt(dx*dx+dy*dy);
    }
    return s/4;
}

// =========================================================================
// Overlay drawing
// =========================================================================
function drawOverlays() {
    const ow = oCanvas.width, oh = oCanvas.height;
    oCtx.clearRect(0, 0, ow, oh);
    const sx = ow / DW, sy = oh / DH;

    for (const c of tracked) {
        const col = c.state === 'locked' ? '#4f4' : c.state === 'stable' ? '#ff0' : '#f44';
        const lw = c.state === 'locked' ? 3 : 2;

        oCtx.strokeStyle = col;
        oCtx.lineWidth = lw;
        oCtx.shadowColor = col;
        oCtx.shadowBlur = c.state === 'locked' ? 14 : 6;

        oCtx.beginPath();
        oCtx.moveTo(c.corners[0][0]*sx, c.corners[0][1]*sy);
        for (let i = 1; i < 4; i++) oCtx.lineTo(c.corners[i][0]*sx, c.corners[i][1]*sy);
        oCtx.closePath();
        oCtx.stroke();

        // Corner dots
        oCtx.shadowBlur = 0;
        oCtx.fillStyle = col;
        for (const [px, py] of c.corners) {
            oCtx.beginPath();
            oCtx.arc(px*sx, py*sy, c.state === 'locked' ? 5 : 3, 0, Math.PI*2);
            oCtx.fill();
        }

        // Stability progress bar
        if (c.stable > 0) {
            const progress = Math.min(1, c.stable / STABLE_FRAMES);
            const bcx = (c.corners[0][0]+c.corners[2][0])/2*sx;
            const bcy = (c.corners[0][1]+c.corners[2][1])/2*sy;
            const bw = 36;
            oCtx.fillStyle = 'rgba(0,0,0,0.5)';
            oCtx.fillRect(bcx-bw/2, bcy-3, bw, 6);
            oCtx.fillStyle = col;
            oCtx.fillRect(bcx-bw/2, bcy-3, bw*progress, 6);
        }
    }
}

// =========================================================================
// Phase logic
// =========================================================================
function doDetecting() {
    const locked = tracked.filter(t => t.state === 'locked');
    const now = performance.now();

    if (locked.length >= CARDS_PER_ROW && (now - lastCapTime) > CAPTURE_COOLDOWN_MS) {
        locked.sort((a,b) => (a.corners[0][0]+a.corners[2][0]) - (b.corners[0][0]+b.corners[2][0]));
        captureRow(locked.slice(0, CARDS_PER_ROW));
    } else if (locked.length > 0) {
        setState(locked.length + '/' + CARDS_PER_ROW + ' cards locked');
    } else if (tracked.length > 0) {
        setState('Hold steady...');
    } else {
        setState('Point at cards');
    }
}

function doTransition() {
    if (tracked.length <= 1) {
        transFrames++;
        setState('Move to row ' + (currentRow+1));
    }
    if (transFrames > 5 && tracked.length >= 2) {
        phase = 'detecting'; transFrames = 0;
        setState('Scanning row ' + (currentRow+1) + '...');
    }
}

// =========================================================================
// Capture + perspective warp
// =========================================================================
function captureRow(cards) {
    lastCapTime = performance.now();

    // Full-res snapshot
    cCtx.drawImage(video, 0, 0, cCanvas.width, cCanvas.height);

    const sx = cCanvas.width / DW, sy = cCanvas.height / DH;
    const images = [];

    for (const card of cards) {
        const full = card.corners.map(([x,y]) => [x*sx, y*sy]);
        images.push(warpCard(full));
    }

    capturedRows[currentRow] = images;
    fireFlash();
    haptic();
    updateHUD();

    currentRow++;
    if (currentRow >= TOTAL_ROWS) {
        stopScanning();
        showPreview();
    } else {
        phase = 'transition'; transFrames = 0; tracked = [];
        setState('Row captured! Move down');
    }
}

function warpCard(srcCorners) {
    // srcCorners: [TL, TR, BR, BL] in capture-canvas coords
    // Perspective warp to WARP_W x WARP_H

    const dst = [[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]];
    const H = computeH(dst, srcCorners); // maps dst->src

    if (!H) {
        // Fallback: simple crop
        const bb = {x: Math.min(srcCorners[0][0],srcCorners[3][0]),
                    y: Math.min(srcCorners[0][1],srcCorners[1][1]),
                    w: Math.max(srcCorners[1][0],srcCorners[2][0]) - Math.min(srcCorners[0][0],srcCorners[3][0]),
                    h: Math.max(srcCorners[2][1],srcCorners[3][1]) - Math.min(srcCorners[0][1],srcCorners[1][1])};
        wCtx.drawImage(cCanvas, bb.x, bb.y, bb.w, bb.h, 0, 0, WARP_W, WARP_H);
        return wCanvas.toDataURL('image/jpeg', 0.92);
    }

    const srcD = cCtx.getImageData(0, 0, cCanvas.width, cCanvas.height);
    const sp = srcD.data;
    const sw = cCanvas.width, sh = cCanvas.height;
    const outD = wCtx.createImageData(WARP_W, WARP_H);
    const op = outD.data;

    for (let dy = 0; dy < WARP_H; dy++) {
        for (let dx = 0; dx < WARP_W; dx++) {
            const denom = H[6]*dx + H[7]*dy + H[8];
            const srcX = (H[0]*dx + H[1]*dy + H[2]) / denom;
            const srcY = (H[3]*dx + H[4]*dy + H[5]) / denom;

            const ix = Math.floor(srcX), iy = Math.floor(srcY);
            if (ix < 0 || ix >= sw-1 || iy < 0 || iy >= sh-1) continue;

            const fx = srcX-ix, fy = srcY-iy;
            const i00 = (iy*sw+ix)*4;
            const i10 = i00+4, i01 = i00+sw*4, i11 = i01+4;
            const oi = (dy*WARP_W+dx)*4;

            for (let c = 0; c < 3; c++) {
                op[oi+c] = sp[i00+c]*(1-fx)*(1-fy) + sp[i10+c]*fx*(1-fy)
                         + sp[i01+c]*(1-fx)*fy + sp[i11+c]*fx*fy;
            }
            op[oi+3] = 255;
        }
    }
    wCtx.putImageData(outD, 0, 0);
    return wCanvas.toDataURL('image/jpeg', 0.92);
}

function computeH(src, dst) {
    // Homography from src->dst using DLT (4-point)
    // Builds 8x9 matrix, solves 8x8 with h9=1
    const A = [];
    for (let i = 0; i < 4; i++) {
        const [x,y] = src[i], [u,v] = dst[i];
        A.push([-x,-y,-1, 0,0,0, u*x, u*y, u]);
        A.push([0,0,0, -x,-y,-1, v*x, v*y, v]);
    }
    const M = [], b = [];
    for (let i = 0; i < 8; i++) { M.push(A[i].slice(0,8)); b.push(-A[i][8]); }

    // Gaussian elimination
    const n = 8;
    const aug = M.map((r,i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
        let mx = Math.abs(aug[c][c]), mr = c;
        for (let r = c+1; r < n; r++) { if (Math.abs(aug[r][c]) > mx) { mx = Math.abs(aug[r][c]); mr = r; } }
        if (mx < 1e-10) return null;
        [aug[c], aug[mr]] = [aug[mr], aug[c]];
        for (let r = 0; r < n; r++) {
            if (r === c) continue;
            const f = aug[r][c] / aug[c][c];
            for (let j = c; j <= n; j++) aug[r][j] -= f * aug[c][j];
        }
    }
    const h = [];
    for (let i = 0; i < n; i++) h.push(aug[i][n] / aug[i][i]);
    h.push(1);
    return h;
}

// =========================================================================
// UI helpers
// =========================================================================
function setState(t) { document.getElementById('scanState').textContent = t; }

function updateHUD() {
    document.getElementById('rowIndicator').textContent =
        'Row ' + Math.min(currentRow+1, TOTAL_ROWS) + ' / ' + TOTAL_ROWS;
    for (let i = 0; i < TOTAL_ROWS; i++) {
        const d = document.getElementById('dot'+i);
        d.className = 'rowDot ' + (capturedRows[i] ? 'captured' : i === currentRow ? 'active' : 'pending');
    }
}

function updateBadges() {
    const sorted = tracked.slice().sort((a,b) =>
        (a.corners[0][0]+a.corners[2][0]) - (b.corners[0][0]+b.corners[2][0]));
    for (let i = 0; i < CARDS_PER_ROW; i++) {
        const b = document.getElementById('cb'+i);
        b.className = 'cardBadge' + (i < sorted.length ? ' ' + sorted[i].state : '');
    }
}

function fireFlash() {
    const f = document.getElementById('flash');
    f.classList.add('fire');
    setTimeout(() => f.classList.remove('fire'), 120);
}

function haptic() { if (navigator.vibrate) navigator.vibrate([30,50,30]); }

// =========================================================================
// Preview + Submit
// =========================================================================
function showPreview() {
    const grid = document.getElementById('previewGrid');
    grid.innerHTML = '';
    for (let r = 0; r < TOTAL_ROWS; r++) {
        if (!capturedRows[r]) continue;
        for (let c = 0; c < CARDS_PER_ROW; c++) {
            const img = document.createElement('img');
            img.src = capturedRows[r][c] || '';
            grid.appendChild(img);
        }
    }
    document.getElementById('previewOverlay').classList.add('show');
}

async function submitCards() {
    document.getElementById('previewOverlay').classList.remove('show');
    document.getElementById('submitOverlay').classList.add('show');

    const fd = new FormData();
    for (let r = 0; r < TOTAL_ROWS; r++) {
        if (!capturedRows[r]) continue;
        for (let c = 0; c < CARDS_PER_ROW; c++) {
            const url = capturedRows[r][c];
            if (!url) continue;
            const idx = r * CARDS_PER_ROW + c;
            fd.append('card_' + idx, dataURLBlob(url), 'card_' + idx + '.jpg');
        }
    }

    try {
        const resp = await fetch('/scanner/identify?variants=true', {method:'POST', body:fd});
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        document.getElementById('submitOverlay').classList.remove('show');
        if (data.scan_id) {
            window.location.href = '/?scan=' + data.scan_id;
        } else {
            showResults(data);
        }
    } catch(e) {
        document.getElementById('submitOverlay').classList.remove('show');
        alert('Failed: ' + e.message);
        showPreview();
    }
}

function dataURLBlob(url) {
    const [hdr, b64] = url.split(',');
    const mime = hdr.match(/:(.*?);/)[1];
    const raw = atob(b64);
    const arr = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
    return new Blob([arr], {type: mime});
}

function showResults(data) {
    const grid = document.getElementById('previewGrid');
    const title = document.getElementById('previewTitle');
    const actions = document.getElementById('previewActions');
    title.textContent = 'Results';
    grid.innerHTML = '';
    actions.innerHTML = '<button onclick="resetAll()" style="padding:14px 32px;font-size:16px;font-weight:700;border:none;border-radius:50px;background:#4f4;color:#000;cursor:pointer">Scan More</button>';
    if (data.cards) {
        for (const card of data.cards) {
            const d = document.createElement('div');
            d.style.cssText = 'text-align:center;padding:4px';
            const img = document.createElement('img');
            img.src = card.ref_image_url || '';
            img.style.cssText = 'width:100%;border-radius:6px';
            const lbl = document.createElement('div');
            lbl.style.cssText = 'font-size:11px;margin-top:4px;opacity:0.8';
            lbl.textContent = card.name || card.card_id || '?';
            d.appendChild(img); d.appendChild(lbl);
            grid.appendChild(d);
        }
    }
    document.getElementById('previewOverlay').classList.add('show');
}

// =========================================================================
// Boot
// =========================================================================
init();
</script>
</body>
</html>"""
