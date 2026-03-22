"""Slide-scan binder camera UI with row guidance system.

User holds phone ~15cm from binder and slowly slides across each row.
The camera detects individual cards as they pass through a central detection
zone, auto-capturing at peak sharpness/alignment. All detection runs
client-side in JS -- no server round-trips during scanning.

Row guidance features:
    - Row indicator with progress dots
    - Animated direction arrows (zigzag: L->R odd rows, R->L even rows)
    - Per-row card counter ("Card 2/3 captured")
    - Green flash + thumbnail slide-in animation on capture
    - Row completion banner with downward arrow animation
    - Completion screen: 3x3 grid of thumbnails + "Submit for identification"
    - Tap any thumbnail to re-capture that card
    - Speed warning when user moves too fast (motion blur / rapid captures)

Integration into server.py:
    GET  /slide-scan       -> serve this HTML
    POST /slide-scan       -> receive 9 card images, identify, return JSON
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

/* Detection zone overlay canvas */
canvas#overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    pointer-events: none;
}

/* ================================================================ */
/*  TOP HUD: row progress + direction + card counter                 */
/* ================================================================ */
.top-hud {
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to bottom, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.55) 80%, transparent 100%);
    padding: 10px 14px 24px;
    pointer-events: none;
}

/* Row label: "Row 1 of 3" */
.row-label {
    font-size: 16px;
    font-weight: 700;
    text-align: center;
    color: rgba(255,255,255,0.95);
    margin-bottom: 6px;
}

/* Row progress dots */
.row-dots {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 8px;
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

/* Direction arrow + label */
.direction-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-bottom: 4px;
}
.dir-arrow {
    font-size: 28px;
    color: #4ecca3;
    animation: slideRight 1.2s ease-in-out infinite;
}
.dir-arrow.left {
    animation-name: slideLeft;
}
.dir-label {
    font-size: 13px;
    color: rgba(255,255,255,0.7);
}

@keyframes slideRight {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(12px); }
}
@keyframes slideLeft {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(-12px); }
}

/* Card counter: "Card 2/3 captured" */
.card-counter {
    text-align: center;
    font-size: 13px;
    color: rgba(255,255,255,0.65);
}

/* ================================================================ */
/*  LEFT: row progress vertical dots (secondary indicator)           */
/* ================================================================ */
.row-progress-side {
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

/* ================================================================ */
/*  FILMSTRIP: bottom thumbnail bar                                  */
/* ================================================================ */
.filmstrip {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    z-index: 10;
    background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.5) 70%, transparent 100%);
    padding: 30px 10px 10px;
    pointer-events: none;
}

.strip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    padding: 0 4px;
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
    position: relative;
}
.strip-slot.active-slot {
    border-color: #e94560;
    border-style: solid;
}
.strip-slot .slot-pos {
    font-size: 9px;
    color: rgba(255,255,255,0.3);
}

/* Filled thumbnail */
.strip-thumb {
    flex-shrink: 0;
    width: 52px;
    height: 72px;
    border-radius: 4px;
    border: 2px solid #4ecca3;
    object-fit: cover;
    opacity: 0;
    transform: translateY(20px) scale(0.8);
    transition: all 0.35s ease;
    position: relative;
    cursor: pointer;
    pointer-events: auto;
}
.strip-thumb.visible {
    opacity: 1;
    transform: translateY(0) scale(1);
}
.strip-thumb.latest {
    border-color: #4ecca3;
    box-shadow: 0 0 10px rgba(78, 204, 163, 0.6);
}

/* ================================================================ */
/*  BOTTOM CONTROLS                                                  */
/* ================================================================ */
.bottom-controls {
    position: absolute;
    bottom: 95px; left: 0; right: 0;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    pointer-events: none;
}

.status-text {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    text-align: center;
    min-height: 18px;
    transition: color 0.2s;
}
.status-text.ready {
    color: #4ecca3;
    font-weight: 600;
}

.detection-meter {
    width: 180px;
    height: 3px;
    background: rgba(255,255,255,0.12);
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

.btn-row {
    display: flex;
    gap: 12px;
    pointer-events: auto;
}

.btn {
    padding: 11px 22px;
    border: none;
    border-radius: 10px;
    font-size: 14px;
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
/*  SPEED WARNING                                                    */
/* ================================================================ */
.speed-warning {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(0.8);
    background: rgba(255, 68, 68, 0.9);
    color: #fff;
    font-weight: 700;
    font-size: 20px;
    padding: 14px 32px;
    border-radius: 16px;
    z-index: 25;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s, transform 0.25s;
}
.speed-warning.show {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
}

/* ================================================================ */
/*  ROW COMPLETE OVERLAY                                             */
/* ================================================================ */
.row-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 15;
    background: rgba(0,0,0,0.88);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
}
.row-overlay.visible { display: flex; }
.row-overlay .row-check {
    font-size: 48px;
    color: #4ecca3;
}
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
.row-overlay .down-bounce {
    font-size: 40px;
    color: #4ecca3;
    animation: bounceDown 0.8s ease-in-out infinite;
}
@keyframes bounceDown {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(12px); }
}
.row-overlay .btn {
    margin-top: 8px;
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
    bottom: 110px;
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
    <canvas id="detect" class="hidden-canvas"></canvas>
    <canvas id="capture" class="hidden-canvas"></canvas>

    <!-- ======== TOP HUD ======== -->
    <div class="top-hud" id="topHud">
        <div class="row-label" id="rowLabel">Row 1 of 3</div>
        <div class="row-dots" id="rowDots"></div>
        <div class="direction-row" id="directionRow">
            <span class="dir-arrow" id="dirArrow">&#8594;</span>
            <span class="dir-label" id="dirLabel">Slide right</span>
        </div>
        <div class="card-counter" id="cardCounter">Card 0/3 captured</div>
    </div>

    <!-- ======== SIDE ROW DOTS ======== -->
    <div class="row-progress-side" id="rowProgressSide"></div>

    <!-- ======== BOTTOM CONTROLS ======== -->
    <div class="bottom-controls">
        <div class="status-text" id="statusText">Starting camera...</div>
        <div class="detection-meter">
            <div class="detection-fill" id="detectionFill"></div>
        </div>
        <div class="btn-row">
            <button class="btn btn-manual" id="btnManual" onclick="manualCapture()">Capture</button>
            <button class="btn btn-next-row" id="btnNextRow" onclick="nextRow()">Next Row</button>
            <button class="btn btn-done" id="btnDone" onclick="showDone()">Done</button>
        </div>
    </div>

    <!-- ======== FILMSTRIP ======== -->
    <div class="filmstrip" id="filmstrip">
        <div class="strip-header">
            <span class="strip-title" id="stripTitle">Row 1</span>
            <span class="strip-counter" id="stripCounter">0 / 9</span>
        </div>
        <div class="strip-thumbs" id="stripThumbs"></div>
    </div>

    <!-- ======== GREEN FLASH ======== -->
    <div class="flash" id="flash"></div>

    <!-- ======== SPEED WARNING ======== -->
    <div class="speed-warning" id="speedWarning">Slow down!</div>

    <!-- ======== ROW COMPLETE OVERLAY ======== -->
    <div class="row-overlay" id="rowOverlay">
        <div class="row-check">&#10003;</div>
        <h2 id="rowOverlayTitle">Row 1 Complete</h2>
        <p id="rowOverlayText">Move your phone down to row 2</p>
        <div class="down-bounce">&#8595;</div>
        <button class="btn btn-next-row" style="display:inline-block" onclick="continueNextRow()">Continue Scanning</button>
    </div>

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
    zoneW: 0.50,
    zoneH: 0.70,

    // Thresholds
    edgeDensityMin: 0.12,
    sharpnessMin: 15.0,
    contrastMin: 10,
    readyFramesNeeded: 3,
    frameSampleInterval: 3,

    // Detection canvas width (for speed)
    detectWidth: 320,

    debug: false,
};

// ============================================================
// State
// ============================================================
let video, overlay, detectCanvas, captureCanvas;
let overlayCtx, detectCtx, captureCtx;
let captures = [];           // array of {dataUrl, row, col} indexed by grid position
let currentRow = 0;          // 0-indexed
let currentColInRow = 0;     // cards captured in current row
let readyCount = 0;
let cooldown = false;
let cooldownFrames = 0;
let frameCount = 0;
let scanning = true;
let animFrameId = null;
let captureTimestamps = [];  // for speed warning
let speedWarningTimer = null;

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

        // Wait for video frames to be available before playing.
        // On iOS Safari, loadedmetadata fires too early.
        await new Promise((resolve, reject) => {
            video.addEventListener('loadeddata', resolve, { once: true });
            video.addEventListener('error', reject, { once: true });
            // Safety timeout
            setTimeout(() => resolve(), 10000);
        });

        await video.play();
        setStatus('Slide across cards slowly...');

        // Delay resize slightly so layout settles on mobile
        setTimeout(() => {
            resizeOverlay();
            window.addEventListener('resize', resizeOverlay);
            animFrameId = requestAnimationFrame(detectionLoop);
        }, 200);
    } catch (err) {
        setStatus('Camera error: ' + err.message);
        console.error('Camera init failed:', err);
        // Show error visibly on screen
        document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#e74c3c;font-size:18px;">Camera Error<br><br><span style="font-size:14px;color:#888;">' + err.message + '<br><br>Make sure you allow camera access.<br>On some devices, HTTPS is required.</span></div>';
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
    buildHud();
    buildThumbs();
    buildSideDots();
}

function buildHud() {
    // Row label
    document.getElementById('rowLabel').textContent =
        'Row ' + (currentRow + 1) + ' of ' + CFG.rows;

    // Row dots
    const dotsEl = document.getElementById('rowDots');
    dotsEl.innerHTML = '';
    for (let r = 0; r < CFG.rows; r++) {
        const dot = document.createElement('div');
        dot.className = 'row-dot';
        if (r < currentRow) dot.classList.add('done');
        if (r === currentRow) dot.classList.add('active');
        dotsEl.appendChild(dot);
    }

    // Direction arrow: zigzag (even rows L->R, odd rows R->L)
    const goRight = (currentRow % 2 === 0);
    const dirArrow = document.getElementById('dirArrow');
    const dirLabel = document.getElementById('dirLabel');
    dirArrow.innerHTML = goRight ? '&#8594;' : '&#8592;';
    dirArrow.className = 'dir-arrow' + (goRight ? '' : ' left');
    dirLabel.textContent = goRight ? 'Slide right' : 'Slide left';

    // Card counter
    document.getElementById('cardCounter').textContent =
        'Card ' + currentColInRow + '/' + CFG.cols + ' captured';
}

function buildThumbs() {
    const container = document.getElementById('stripThumbs');
    container.innerHTML = '';

    for (let i = 0; i < CFG.total; i++) {
        const cap = captures[i];
        if (cap) {
            const img = document.createElement('img');
            img.className = 'strip-thumb visible';
            if (i === captures.length - 1 ||
                (captures.filter(Boolean).length > 0 &&
                 i === captures.reduce((last, c, idx) => c ? idx : last, -1))) {
                // Mark latest captured
            }
            img.src = cap.dataUrl;
            img.onclick = () => rescanSlot(i);
            container.appendChild(img);
        } else {
            const slot = document.createElement('div');
            slot.className = 'strip-slot';
            // Highlight active slot
            const activeCol = getActiveCol();
            const activeIdx = currentRow * CFG.cols + activeCol;
            if (i === activeIdx && scanning) {
                slot.classList.add('active-slot');
            }
            const r = Math.floor(i / CFG.cols) + 1;
            const c = (i % CFG.cols) + 1;
            const posLabel = document.createElement('span');
            posLabel.className = 'slot-pos';
            posLabel.textContent = r + '.' + c;
            slot.appendChild(posLabel);
            container.appendChild(slot);
        }
    }

    document.getElementById('stripTitle').textContent = 'Row ' + (currentRow + 1);
    document.getElementById('stripCounter').textContent =
        captures.filter(Boolean).length + ' / ' + CFG.total;
}

function buildSideDots() {
    const container = document.getElementById('rowProgressSide');
    container.innerHTML = '';
    for (let r = 0; r < CFG.rows; r++) {
        const dot = document.createElement('div');
        dot.className = 'row-dot';
        if (r < currentRow) dot.classList.add('done');
        if (r === currentRow) dot.classList.add('active');
        container.appendChild(dot);
    }
}

function getActiveCol() {
    // Zigzag: even rows L->R, odd rows R->L
    if (currentRow % 2 === 0) {
        return currentColInRow;
    } else {
        return CFG.cols - 1 - currentColInRow;
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
    el.textContent = Object.entries(info).map(([k,v]) =>
        k + ': ' + (typeof v === 'number' ? v.toFixed(2) : v)).join('\n');
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
// Draw overlay (detection zone rectangle + direction arrow)
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
    overlayCtx.fillRect(0, 0, W, zy);
    overlayCtx.fillRect(0, zy + zh, W, H - zy - zh);
    overlayCtx.fillRect(0, zy, zx, zh);
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

    // Corner brackets
    const bLen = 20;
    overlayCtx.strokeStyle = borderColor;
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

    // Animated direction arrow on canvas (below detection zone)
    if (!cooldown && readyCount === 0 && captures.filter(Boolean).length < CFG.total) {
        const goRight = (currentRow % 2 === 0);
        overlayCtx.save();
        overlayCtx.globalAlpha = 0.35 + 0.15 * Math.sin(Date.now() / 400);
        overlayCtx.fillStyle = '#4ecca3';
        overlayCtx.font = '26px sans-serif';
        overlayCtx.textAlign = 'center';
        // Pulse position
        const pulse = Math.sin(Date.now() / 400) * 8;
        const arrowX = goRight ? W / 2 + pulse : W / 2 - pulse;
        overlayCtx.fillText(goRight ? '\u25B6' : '\u25C0', arrowX, zy + zh + 28);
        overlayCtx.restore();
    }
}

// ============================================================
// Frame processing
// ============================================================
function processFrame() {
    if (!video.videoWidth || cooldown) {
        if (cooldown) {
            cooldownFrames++;
            const info = analyzeZone();
            if (info && info.edgeDensity < CFG.edgeDensityMin * 0.5) {
                cooldown = false;
                cooldownFrames = 0;
                setStatus('Slide to next card...', false);
                setMeter(0);
            } else if (cooldownFrames > 30) {
                cooldown = false;
                cooldownFrames = 0;
                setStatus('Slide to next card...', false);
                setMeter(0);
            }
            updateDebug({ state: 'cooldown', frames: cooldownFrames,
                          edgeDensity: info ? info.edgeDensity : 0 });
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
        row: currentRow + 1,
        col: getActiveCol() + 1,
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
            if (captures.filter(Boolean).length < CFG.total) {
                setStatus('Slide across cards slowly...', false);
            }
        }
    }
}

// ============================================================
// Analyze the detection zone
// ============================================================
function analyzeZone() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    const scale = CFG.detectWidth / vw;
    const dw = CFG.detectWidth;
    const dh = Math.round(vh * scale);
    detectCanvas.width = dw;
    detectCanvas.height = dh;
    detectCtx.drawImage(video, 0, 0, dw, dh);

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

    // Edge density via Sobel
    let edgeCount = 0;
    const edgeThreshold = 60;
    for (let y = 1; y < zh - 1; y++) {
        for (let x = 1; x < zw - 1; x++) {
            const idx = y * zw + x;
            const gx = -gray[idx - zw - 1] + gray[idx - zw + 1]
                       -2 * gray[idx - 1] + 2 * gray[idx + 1]
                       -gray[idx + zw - 1] + gray[idx + zw + 1];
            const gy = -gray[idx - zw - 1] - 2 * gray[idx - zw] - gray[idx - zw + 1]
                       +gray[idx + zw - 1] + 2 * gray[idx + zw] + gray[idx + zw + 1];
            if (Math.sqrt(gx * gx + gy * gy) > edgeThreshold) edgeCount++;
        }
    }
    const edgeDensity = edgeCount / total;

    // Sharpness via Laplacian variance
    let lapSum = 0, lapSumSq = 0, lapCount = 0;
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

    // Contrast: center vs border brightness
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
                centerSum += v; centerCount++;
            } else {
                borderSum += v; borderCount++;
            }
        }
    }
    const contrast = Math.abs(
        (centerCount ? centerSum / centerCount : 128) -
        (borderCount ? borderSum / borderCount : 128)
    );

    return { edgeDensity, sharpness, contrast };
}

// ============================================================
// Capture card at full resolution
// ============================================================
function captureCard() {
    if (captures.filter(Boolean).length >= CFG.total) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    const zx = Math.round(vw * (1 - CFG.zoneW) / 2);
    const zy = Math.round(vh * (1 - CFG.zoneH) / 2);
    const zw = Math.round(vw * CFG.zoneW);
    const zh = Math.round(vh * CFG.zoneH);

    captureCanvas.width = zw;
    captureCanvas.height = zh;
    captureCtx.drawImage(video, zx, zy, zw, zh, 0, 0, zw, zh);

    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);

    // Compute grid position using zigzag
    const col = getActiveCol();
    const slotIdx = currentRow * CFG.cols + col;

    captures[slotIdx] = { dataUrl, row: currentRow, col };
    currentColInRow++;

    // Speed warning: check timing between captures
    const now = Date.now();
    captureTimestamps.push(now);
    if (captureTimestamps.length >= 2) {
        const dt = now - captureTimestamps[captureTimestamps.length - 2];
        if (dt < 1200) {
            showSpeedWarning();
        }
    }

    // Green flash
    triggerFlash();

    // Haptic
    if (navigator.vibrate) navigator.vibrate(50);

    // Update UI
    buildUI();

    // Scroll filmstrip to latest
    const thumbsEl = document.getElementById('stripThumbs');
    thumbsEl.scrollLeft = thumbsEl.scrollWidth;

    // Check row completion
    if (currentColInRow >= CFG.cols) {
        if (currentRow >= CFG.rows - 1) {
            // All rows done
            setStatus('All cards captured!', true);
            document.getElementById('btnDone').style.display = 'inline-block';
            document.getElementById('btnManual').style.display = 'none';
            scanning = false;
            // Auto-show completion after brief delay
            setTimeout(() => showDone(), 600);
        } else {
            showRowComplete();
        }
    }

    // Cooldown
    readyCount = 0;
    cooldown = true;
    cooldownFrames = 0;
    setMeter(0);
}

function triggerFlash() {
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 150);
}

function showSpeedWarning() {
    const el = document.getElementById('speedWarning');
    el.classList.add('show');
    if (speedWarningTimer) clearTimeout(speedWarningTimer);
    speedWarningTimer = setTimeout(() => {
        el.classList.remove('show');
        speedWarningTimer = null;
    }, 2000);
}

// ============================================================
// Manual capture
// ============================================================
function manualCapture() {
    if (captures.filter(Boolean).length >= CFG.total) return;
    readyCount = CFG.readyFramesNeeded;
    captureCard();
}

// ============================================================
// Row transitions
// ============================================================
function showRowComplete() {
    scanning = false;
    const ovl = document.getElementById('rowOverlay');
    document.getElementById('rowOverlayTitle').textContent =
        'Row ' + (currentRow + 1) + ' Complete!';
    document.getElementById('rowOverlayText').textContent =
        'Move your phone down to row ' + (currentRow + 2);
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

    const goRight = (currentRow % 2 === 0);
    setStatus('Slide ' + (goRight ? 'right' : 'left') + ' across row ' + (currentRow + 1) + '...', false);
    setMeter(0);
    animFrameId = requestAnimationFrame(detectionLoop);
}

function nextRow() {
    showRowComplete();
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
            // Empty slot placeholder
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

        // Tap cell to re-scan
        ((idx) => {
            cell.onclick = () => rescanSlot(idx);
        })(i);

        grid.appendChild(cell);
    }

    grid.style.gridTemplateColumns = 'repeat(' + CFG.cols + ', 1fr)';

    // Enable/disable submit
    document.getElementById('btnSubmit').disabled = (count < CFG.total);

    ovl.classList.add('visible');
}

// ============================================================
// Re-scan a specific slot
// ============================================================
function rescanSlot(idx) {
    // Hide done overlay, go back to camera for one capture
    document.getElementById('doneOverlay').classList.remove('visible');

    // Set position to capture into this slot
    const targetRow = Math.floor(idx / CFG.cols);
    const targetCol = idx % CFG.cols;

    // Temporarily override: capture next card into this slot
    const origRow = currentRow;
    const origCol = currentColInRow;

    // Manual single-capture mode
    scanning = true;
    currentRow = targetRow;
    // For zigzag, figure out what currentColInRow should be
    if (targetRow % 2 === 0) {
        currentColInRow = targetCol;
    } else {
        currentColInRow = CFG.cols - 1 - targetCol;
    }

    buildUI();
    setStatus('Capture card for R' + (targetRow + 1) + ' C' + (targetCol + 1), false);

    // After this capture, go back to done screen
    const origCaptureCard = captureCard;
    const self = this;
    const patchedCapture = () => {
        // Do normal capture
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const zx = Math.round(vw * (1 - CFG.zoneW) / 2);
        const zy = Math.round(vh * (1 - CFG.zoneH) / 2);
        const zw = Math.round(vw * CFG.zoneW);
        const zh = Math.round(vh * CFG.zoneH);
        captureCanvas.width = zw;
        captureCanvas.height = zh;
        captureCtx.drawImage(video, zx, zy, zw, zh, 0, 0, zw, zh);
        const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);

        captures[idx] = { dataUrl, row: targetRow, col: targetCol };
        triggerFlash();
        if (navigator.vibrate) navigator.vibrate(50);

        // Restore state
        scanning = false;
        currentRow = origRow;
        currentColInRow = origCol;

        // Remove patch
        window._rescanCapture = null;

        // Show done overlay again
        setTimeout(() => showDone(), 300);
    };
    window._rescanCapture = patchedCapture;

    // Override manual capture to use patched version
    document.getElementById('btnManual').onclick = () => {
        if (window._rescanCapture) {
            window._rescanCapture();
        } else {
            manualCapture();
        }
    };

    setMeter(0);
    animFrameId = requestAnimationFrame(rescanDetectionLoop);
}

function rescanDetectionLoop() {
    if (!scanning) return;

    frameCount++;
    drawOverlay();

    if (frameCount % CFG.frameSampleInterval === 0) {
        if (!video.videoWidth || cooldown) {
            if (cooldown) {
                cooldownFrames++;
                if (cooldownFrames > 30) {
                    cooldown = false;
                    cooldownFrames = 0;
                }
            }
        } else {
            const info = analyzeZone();
            if (info) {
                const isReady = info.edgeDensity >= CFG.edgeDensityMin
                             && info.sharpness >= CFG.sharpnessMin
                             && info.contrast >= CFG.contrastMin;

                if (isReady) {
                    readyCount++;
                    setMeter(Math.min(readyCount / CFG.readyFramesNeeded, 1));
                    if (readyCount >= CFG.readyFramesNeeded && window._rescanCapture) {
                        window._rescanCapture();
                        return;
                    }
                } else {
                    if (readyCount > 0) readyCount = Math.max(0, readyCount - 1);
                    setMeter(Math.min(readyCount / CFG.readyFramesNeeded, 1));
                }
            }
        }
    }

    animFrameId = requestAnimationFrame(rescanDetectionLoop);
}

// ============================================================
// Submit cards for identification
// ============================================================
async function submitCards() {
    const count = captures.filter(Boolean).length;
    if (count === 0) return;

    const btn = document.getElementById('btnSubmit');
    btn.disabled = true;

    // Hide done overlay, show ID overlay
    document.getElementById('doneOverlay').classList.remove('visible');
    const idOvl = document.getElementById('idOverlay');
    idOvl.classList.add('visible');

    // Start timer
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
            // Convert dataUrl to blob
            const resp0 = await fetch(captures[i].dataUrl);
            const blob = await resp0.blob();
            // Field name: card_0 through card_8 (position index)
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

        // Images
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

        // Info
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

        // Price
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
    currentRow = 0;
    currentColInRow = 0;
    readyCount = 0;
    cooldown = false;
    cooldownFrames = 0;
    captureTimestamps = [];
    scanning = true;

    document.getElementById('btnDone').style.display = 'none';
    document.getElementById('btnManual').style.display = 'inline-block';
    document.getElementById('btnManual').onclick = manualCapture;

    buildUI();
    setStatus('Slide across cards slowly...', false);
    setMeter(0);
    animFrameId = requestAnimationFrame(detectionLoop);
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
