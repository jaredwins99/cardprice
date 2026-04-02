"""Page Scanner UI: hands-free full-page binder capture.

Zero-tap auto-capture of 3x3 binder pages using motion detection.
Captures 3-5 frames when stable, picks sharpest (Laplacian variance),
sends to /scan-page for segmentation + identification.

Integration into server.py:
    elif self.path == "/page-scanner":
        from cardprice.page_scanner import PAGE_SCANNER_HTML
        self._send_html(PAGE_SCANNER_HTML)
"""

PAGE_SCANNER_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Page Scanner</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background: #000; color: #fff;
    height: 100vh; height: 100dvh;
    overflow: hidden;
    display: flex; flex-direction: column;
}
#cameraWrap {
    flex: 1; position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    background: #000;
}
video {
    width: 100%; height: 100%; object-fit: cover;
}
/* Grid overlay to help align binder page */
#gridOverlay {
    position: absolute; top: 10%; left: 10%; width: 80%; height: 80%;
    pointer-events: none; z-index: 5;
    border: 2px solid rgba(78, 204, 163, 0.3);
    border-radius: 8px;
}
#statusBar {
    position: absolute; bottom: 0; left: 0; right: 0; z-index: 10;
    background: rgba(0,0,0,0.7); padding: 12px 16px;
    font-size: 15px; font-weight: 600; text-align: center;
    backdrop-filter: blur(4px);
}
#statusBar .sub {
    font-size: 11px; font-weight: 400; color: rgba(255,255,255,0.5);
    margin-top: 4px;
}
#captureFlash {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: #fff; opacity: 0; pointer-events: none; z-index: 15;
    transition: opacity 0.05s;
}

/* Results overlay */
.results-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 25;
    background: #1a1a2e; display: none; flex-direction: column;
    overflow-y: auto; padding: 20px 16px 40px;
}
.results-overlay.visible { display: flex; }
.results-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
}
.results-header h2 { font-size: 22px; color: #4ecca3; }
.results-summary {
    font-size: 13px; color: rgba(255,255,255,0.5);
}
.total-value {
    font-size: 20px; font-weight: 700; color: #4ecca3;
    text-align: center; margin-bottom: 16px;
    padding: 10px; background: rgba(78,204,163,0.1);
    border-radius: 8px;
}
.card-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 8px; margin-bottom: 16px;
}
.result-card {
    background: rgba(255,255,255,0.05);
    border-radius: 8px; padding: 8px;
    display: flex; flex-direction: column; align-items: center;
    text-align: center;
}
.result-card img {
    width: 100%; aspect-ratio: 5/7; object-fit: cover;
    border-radius: 4px; margin-bottom: 6px;
}
.result-card .name {
    font-weight: 600; font-size: 12px; line-height: 1.2;
    max-height: 2.4em; overflow: hidden;
}
.result-card .set {
    font-size: 10px; color: rgba(255,255,255,0.5); margin-top: 2px;
}
.result-card .price {
    font-size: 14px; font-weight: 700; color: #4ecca3; margin-top: 4px;
}
.result-card .variant-badge {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    background: #f0c040; color: #1a1a2e; font-size: 9px;
    font-weight: 700; text-transform: uppercase; margin-top: 3px;
}
.reset-btn {
    display: block; width: 100%; padding: 16px;
    background: #4ecca3; color: #1a1a2e;
    border: none; border-radius: 12px;
    font-size: 17px; font-weight: 700; cursor: pointer;
    margin-top: 12px;
}
.reset-btn:active { background: #3aa88a; }
</style>
</head>
<body>
<div id="cameraWrap">
    <video id="cam" autoplay playsinline muted></video>
    <div id="gridOverlay"></div>
    <div id="captureFlash"></div>
    <div id="statusBar">
        <div id="statusText">Starting camera...</div>
        <div class="sub" id="statusSub"></div>
    </div>
</div>

<div class="results-overlay" id="resultsOverlay">
    <div class="results-header">
        <h2>Page Results</h2>
        <span class="results-summary" id="resultsSummary"></span>
    </div>
    <div class="total-value" id="totalValue"></div>
    <div class="card-grid" id="cardGrid"></div>
    <button class="reset-btn" onclick="resetAll()">Scan Another Page</button>
</div>

<script>
const V = document.getElementById('cam');
const statusText = document.getElementById('statusText');
const statusSub = document.getElementById('statusSub');
const flash = document.getElementById('captureFlash');

let phase = 'starting'; // starting, stabilizing, capturing, uploading, done
let prevGray = null;

// Motion detection config (same as slide_scan_v7)
const STILL_THRESHOLD = 5.0;
const MOVE_THRESHOLD = 8.0;
const STILL_NEEDED = 8;

// Sharpest-frame selection
const BURST_COUNT = 5;
let burstFrames = [];   // [{blob, sharpness}]
let stillFrames = 0;
let burstStarted = false;

function setStatus(main, sub) {
    statusText.textContent = main;
    statusSub.textContent = sub || '';
}

// ===== Camera init =====
async function init() {
    if (!navigator.mediaDevices) {
        setStatus('No camera API. Need HTTPS.');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false
        });
        V.srcObject = stream;
        await V.play();

        // Wait for real video dimensions
        for (let i = 0; i < 50; i++) {
            if (V.videoWidth > 0) break;
            await new Promise(r => setTimeout(r, 100));
        }
        if (!V.videoWidth) {
            setStatus('Camera failed to start');
            return;
        }

        setStatus('Hold phone over binder page...', V.videoWidth + 'x' + V.videoHeight);
        phase = 'stabilizing';
        requestAnimationFrame(tick);
    } catch(e) {
        setStatus('Camera error: ' + e.message);
    }
}

// ===== Grayscale center crop (motion detection) =====
function getGrayCenter() {
    const c = document.createElement('canvas');
    const sw = 80, sh = 60;
    c.width = sw; c.height = sh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    const sx = V.videoWidth * 0.25, sy = V.videoHeight * 0.25;
    const sWidth = V.videoWidth * 0.5, sHeight = V.videoHeight * 0.5;
    ctx.drawImage(V, sx, sy, sWidth, sHeight, 0, 0, sw, sh);
    const d = ctx.getImageData(0, 0, sw, sh).data;
    const gray = new Float32Array(sw * sh);
    for (let i = 0; i < gray.length; i++) {
        gray[i] = 0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2];
    }
    return gray;
}

function frameDiff(a, b) {
    if (!a || !b || a.length !== b.length) return 999;
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
}

// ===== Laplacian variance (sharpness) =====
function computeSharpness(canvas) {
    // Downsample to manageable size for Laplacian
    const sw = 160, sh = 120;
    const c = document.createElement('canvas');
    c.width = sw; c.height = sh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(canvas, 0, 0, sw, sh);
    const d = ctx.getImageData(0, 0, sw, sh).data;

    // Convert to grayscale
    const gray = new Float32Array(sw * sh);
    for (let i = 0; i < gray.length; i++) {
        gray[i] = 0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2];
    }

    // Laplacian: L(x,y) = g(x-1,y) + g(x+1,y) + g(x,y-1) + g(x,y+1) - 4*g(x,y)
    let sumSq = 0;
    let count = 0;
    for (let y = 1; y < sh - 1; y++) {
        for (let x = 1; x < sw - 1; x++) {
            const idx = y * sw + x;
            const lap = gray[idx - 1] + gray[idx + 1] + gray[idx - sw] + gray[idx + sw] - 4 * gray[idx];
            sumSq += lap * lap;
            count++;
        }
    }
    return sumSq / count; // Laplacian variance
}

// ===== Capture a frame into burst buffer =====
function captureFrame() {
    const c = document.createElement('canvas');
    c.width = V.videoWidth;
    c.height = V.videoHeight;
    c.getContext('2d').drawImage(V, 0, 0);

    const sharpness = computeSharpness(c);

    return new Promise(resolve => {
        c.toBlob(blob => {
            resolve({ blob, sharpness });
        }, 'image/jpeg', 0.95);
    });
}

function doFlash() {
    flash.style.opacity = '0.4';
    setTimeout(() => { flash.style.opacity = '0'; }, 80);
    if (navigator.vibrate) navigator.vibrate(30);
}

// ===== Main tick loop =====
async function tick() {
    if (phase !== 'stabilizing' && phase !== 'capturing') return;

    const gray = getGrayCenter();
    const diff = frameDiff(gray, prevGray);
    prevGray = gray;

    if (diff > MOVE_THRESHOLD) {
        stillFrames = 0;
        if (burstStarted) {
            // Movement during burst -- reset
            burstFrames = [];
            burstStarted = false;
            phase = 'stabilizing';
        }
    } else if (diff < STILL_THRESHOLD) {
        stillFrames++;
    }

    // State display
    if (phase === 'stabilizing') {
        if (stillFrames >= STILL_NEEDED) {
            // Start burst capture
            phase = 'capturing';
            burstStarted = true;
            burstFrames = [];
            setStatus('Capturing...', 'Hold steady');
            doFlash();
        } else {
            const progress = Math.min(stillFrames / STILL_NEEDED * 100, 100);
            const stateStr = diff > MOVE_THRESHOLD ? 'MOVING' : 'SETTLING';
            setStatus('Hold phone over binder page...',
                stateStr + ' | stability: ' + Math.round(progress) + '%');
        }
    }

    if (phase === 'capturing') {
        // Capture a frame
        const frame = await captureFrame();
        burstFrames.push(frame);
        setStatus('Capturing...', burstFrames.length + '/' + BURST_COUNT + ' frames');

        if (burstFrames.length >= BURST_COUNT) {
            // Pick sharpest and upload
            phase = 'uploading';
            uploadBest();
            return;
        }

        // Small delay between burst frames to get variety
        await new Promise(r => setTimeout(r, 150));
    }

    requestAnimationFrame(tick);
}

// ===== Upload sharpest frame =====
async function uploadBest() {
    // Find sharpest frame
    let best = burstFrames[0];
    for (const f of burstFrames) {
        if (f.sharpness > best.sharpness) best = f;
    }

    const sharpnessStr = burstFrames.map(f => f.sharpness.toFixed(0)).join(', ');
    setStatus('Uploading best frame...', 'Sharpness: [' + sharpnessStr + '] -> picked ' + best.sharpness.toFixed(0));

    try {
        const fd = new FormData();
        fd.append('image', best.blob, 'page.jpg');

        const resp = await fetch('/scan-page', { method: 'POST', body: fd });
        if (!resp.ok) {
            const errText = await resp.text();
            setStatus('Server error: ' + resp.status, errText.substring(0, 100));
            // Allow retry
            setTimeout(() => {
                phase = 'stabilizing';
                stillFrames = 0;
                burstStarted = false;
                burstFrames = [];
                prevGray = null;
                setStatus('Hold phone over binder page...', 'Retrying...');
                requestAnimationFrame(tick);
            }, 2000);
            return;
        }
        const result = await resp.json();
        showResults(result);
    } catch(e) {
        setStatus('Upload error: ' + e.message);
        setTimeout(() => {
            phase = 'stabilizing';
            stillFrames = 0;
            burstStarted = false;
            burstFrames = [];
            prevGray = null;
            setStatus('Hold phone over binder page...', 'Retrying...');
            requestAnimationFrame(tick);
        }, 2000);
    }
}

// ===== Results =====
function showResults(result) {
    phase = 'done';
    const overlay = document.getElementById('resultsOverlay');
    const grid = document.getElementById('cardGrid');
    const summary = document.getElementById('resultsSummary');
    const totalEl = document.getElementById('totalValue');
    grid.innerHTML = '';

    const cards = result.cards || [];
    summary.textContent = cards.length + ' cards found';

    const total = result.total_value || 0;
    const totalMp = result.total_mp || 0;
    if (total > 0) {
        totalEl.textContent = 'Total: $' + total.toFixed(2) +
            (totalMp > 0 ? '  (MP: $' + totalMp.toFixed(2) + ')' : '');
        totalEl.style.display = '';
    } else {
        totalEl.style.display = 'none';
    }

    if (!cards.length) {
        grid.innerHTML = '<p style="color:#888;grid-column:1/-1;text-align:center;padding:20px;">No cards identified.</p>';
    }

    for (const card of cards) {
        const div = document.createElement('div');
        div.className = 'result-card';

        const name = card.card_name || 'Unknown';
        const setName = card.set_name || '';
        const price = card.variant_price || card.market_price;
        const imgSrc = card.segment_image_url || card.local_image_url || card.image_url || '';
        const variant = card.detected_variant || 'normal';

        let html = '';
        if (imgSrc) html += '<img src="' + imgSrc + '" loading="lazy">';
        html += '<div class="name">' + name + '</div>';
        if (setName) html += '<div class="set">' + setName + '</div>';
        if (price) html += '<div class="price">$' + parseFloat(price).toFixed(2) + '</div>';
        if (variant && variant !== 'normal') {
            html += '<span class="variant-badge">' + variant.replace(/-/g, ' ') + '</span>';
        }

        div.innerHTML = html;
        grid.appendChild(div);
    }

    if (result.error) {
        grid.innerHTML += '<p style="color:#e74c3c;grid-column:1/-1;text-align:center;">' + result.error + '</p>';
    }

    overlay.classList.add('visible');
}

function resetAll() {
    phase = 'stabilizing';
    prevGray = null;
    stillFrames = 0;
    burstStarted = false;
    burstFrames = [];
    document.getElementById('resultsOverlay').classList.remove('visible');
    setStatus('Hold phone over binder page...', '');
    requestAnimationFrame(tick);
}

init();
</script>
</body>
</html>
"""
