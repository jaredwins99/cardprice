"""Condition assessment UI template for multi-angle card capture.

This module provides the HTML/JS template for the /condition endpoint.
It implements a 4-step guided capture wizard (front, back, oblique, edge)
and displays grading results with sub-grades and pricing.

Integration into server.py:
    # In do_GET:
    elif self.path == "/condition":
        from cardprice.condition_ui import CONDITION_HTML
        self._send_html(CONDITION_HTML)

    # Per-card capture (linked from scan results):
    elif self.path.startswith("/condition/capture/"):
        from cardprice.condition_ui import render_capture_html
        card_id = unquote(self.path.split("/condition/capture/", 1)[1])
        self._send_html(render_capture_html(card_id, card_name, set_name, image_url))

    # In do_POST:
    elif self.path == "/condition/assess":
        self._handle_condition_assess()
    elif self.path.startswith("/condition/photo/"):
        self._handle_condition_photo()  # per-step upload with quality feedback
"""


def render_capture_html(card_id, card_name=None, set_name=None, image_url=None):
    """Return CONDITION_CAPTURE_HTML with card context substituted."""
    html = CONDITION_CAPTURE_HTML
    html = html.replace("{{CARD_ID}}", card_id or "")
    html = html.replace("{{CARD_NAME}}", card_name or "Unknown Card")
    html = html.replace("{{SET_NAME}}", set_name or "")
    html = html.replace("{{IMAGE_URL}}", image_url or "")
    return html


# ---------------------------------------------------------------------------
# Per-card capture HTML (linked from scan results via "Grade Condition")
# ---------------------------------------------------------------------------

CONDITION_CAPTURE_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Grade: {{CARD_NAME}}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
    min-height: 100dvh;
    overflow-x: hidden;
}

/* ---- Header ---- */
.header {
    text-align: center;
    padding: 12px 15px 8px;
    border-bottom: 1px solid #16213e;
}
.header h1 {
    font-size: 20px;
    color: #e94560;
}
.header .subtitle {
    font-size: 13px;
    color: #888;
    margin-top: 2px;
}
.back-link {
    position: absolute;
    top: 14px;
    left: 15px;
    color: #4ecca3;
    text-decoration: none;
    font-size: 14px;
}

/* ---- Card Identity Banner ---- */
.card-banner {
    display: flex;
    gap: 12px;
    align-items: center;
    background: #16213e;
    padding: 12px 15px;
    margin: 0 0 4px;
}
.card-banner img {
    width: 50px;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.card-banner .cb-info h3 {
    font-size: 15px;
    color: #fff;
    margin-bottom: 1px;
}
.card-banner .cb-info .cb-set {
    font-size: 12px;
    color: #888;
}
.card-banner .cb-info .cb-id {
    font-size: 10px;
    color: #555;
    font-family: monospace;
}

/* ---- Progress Bar ---- */
.progress-bar {
    display: flex;
    justify-content: center;
    gap: 8px;
    padding: 16px 20px 12px;
}
.progress-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    flex: 1;
    max-width: 80px;
}
.progress-dot {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid #333;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #555;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.progress-dot.active {
    border-color: #e94560;
    color: #e94560;
    box-shadow: 0 0 12px rgba(233, 69, 96, 0.3);
}
.progress-dot.done {
    border-color: #4ecca3;
    background: #4ecca3;
    color: #1a1a2e;
}
.progress-dot img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
}
.progress-label {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: color 0.3s ease;
}
.progress-step.active .progress-label { color: #e94560; }
.progress-step.done .progress-label { color: #4ecca3; }
.progress-connector {
    width: 20px;
    height: 2px;
    background: #333;
    align-self: center;
    margin-bottom: 16px;
    transition: background 0.3s ease;
}
.progress-connector.done { background: #4ecca3; }

/* ---- Wizard Steps ---- */
.wizard-container {
    max-width: 500px;
    margin: 0 auto;
    padding: 0 15px;
}
.wizard-step {
    display: none;
    flex-direction: column;
    align-items: center;
    animation: fadeIn 0.3s ease;
}
.wizard-step.active { display: flex; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Viewfinder ---- */
.viewfinder-container {
    width: 100%;
    max-width: 400px;
    aspect-ratio: 3/4;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    margin: 8px 0;
}
.viewfinder-container video {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.viewfinder-container .captured-preview {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: none;
}
.viewfinder-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
}
/* Card outline guide */
.card-guide {
    position: absolute;
    top: 8%;
    left: 12%;
    right: 12%;
    bottom: 8%;
    border: 2px dashed rgba(78, 204, 163, 0.5);
    border-radius: 12px;
}
.card-guide.oblique {
    transform: perspective(400px) rotateX(25deg);
    top: 12%;
    bottom: 12%;
}
.card-guide.edge {
    top: 30%;
    bottom: 30%;
    left: 20%;
    right: 20%;
    border-radius: 4px;
}

/* ---- Instructions ---- */
.step-instruction {
    text-align: center;
    padding: 12px 10px;
}
.step-instruction h2 {
    font-size: 18px;
    color: #fff;
    margin-bottom: 4px;
}
.step-instruction p {
    font-size: 14px;
    color: #888;
    line-height: 1.4;
}

/* ---- Photo feedback ---- */
.photo-feedback {
    text-align: center;
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px 0;
    display: none;
    animation: fadeIn 0.3s ease;
}
.photo-feedback.good {
    display: block;
    color: #4ecca3;
    background: rgba(78, 204, 163, 0.15);
}
.photo-feedback.warn {
    display: block;
    color: #f0a500;
    background: rgba(240, 165, 0, 0.15);
}
.photo-feedback.bad {
    display: block;
    color: #e94560;
    background: rgba(233, 69, 96, 0.15);
}
.photo-feedback.checking {
    display: block;
    color: #888;
    background: rgba(136, 136, 136, 0.15);
}

/* ---- Gyro feedback ---- */
.gyro-indicator {
    display: none;
    text-align: center;
    margin: 4px 0;
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 20px;
    background: #16213e;
}
.gyro-indicator.good { color: #4ecca3; background: rgba(78, 204, 163, 0.15); }
.gyro-indicator.adjust { color: #f0a500; background: rgba(240, 165, 0, 0.15); }

/* ---- Capture Button ---- */
.capture-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 12px 0 20px;
    width: 100%;
}
.capture-btn {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    border: 3px solid #e94560;
    background: transparent;
    cursor: pointer;
    position: relative;
    transition: all 0.15s ease;
}
.capture-btn:active {
    transform: scale(0.92);
}
.capture-btn::after {
    content: '';
    position: absolute;
    top: 4px; left: 4px; right: 4px; bottom: 4px;
    border-radius: 50%;
    background: #e94560;
}
.capture-btn:disabled {
    border-color: #333;
    opacity: 0.5;
}
.capture-btn:disabled::after {
    background: #333;
}
.retake-btn {
    padding: 8px 16px;
    background: #16213e;
    border: 1px solid #333;
    color: #888;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
    display: none;
}
.retake-btn:active { background: #0f3460; }
.gallery-fallback-btn {
    padding: 8px 16px;
    background: #16213e;
    border: 1px solid #333;
    color: #888;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
}
.gallery-fallback-btn:active { background: #0f3460; }

/* ---- Upload / Spinner ---- */
.upload-section {
    display: none;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    animation: fadeIn 0.3s ease;
}
.upload-section.active { display: flex; }
.thumb-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    width: 100%;
    max-width: 400px;
    margin-bottom: 16px;
}
.thumb-grid .thumb {
    aspect-ratio: 3/4;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    background: #16213e;
}
.thumb-grid .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.thumb-grid .thumb .thumb-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.7);
    font-size: 10px;
    color: #ccc;
    text-align: center;
    padding: 2px;
    text-transform: uppercase;
}
.submit-btn {
    display: block;
    width: 100%;
    max-width: 400px;
    padding: 16px;
    font-size: 17px;
    font-weight: bold;
    background: #4ecca3;
    color: #1a1a2e;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    margin: 8px 0;
}
.submit-btn:active { background: #3dbb91; }
.submit-btn:disabled {
    background: #333;
    color: #666;
    cursor: not-allowed;
}
.spinner-section {
    display: none;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
    animation: fadeIn 0.3s ease;
}
.spinner-section.active { display: flex; }
.spin-ring {
    width: 48px;
    height: 48px;
    border: 4px solid #16213e;
    border-top-color: #e94560;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-section p {
    margin-top: 16px;
    color: #888;
    font-size: 14px;
}

/* ---- Results ---- */
.results-section {
    display: none;
    max-width: 500px;
    margin: 0 auto;
    padding: 0 15px 30px;
    animation: fadeIn 0.4s ease;
}
.results-section.active { display: block; }

.card-identity {
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    display: flex;
    gap: 14px;
    align-items: center;
}
.card-identity img {
    width: 80px;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.card-identity .card-info h3 {
    color: #fff;
    font-size: 17px;
    margin-bottom: 2px;
}
.card-identity .card-info .set-name {
    color: #888;
    font-size: 13px;
}
.card-identity .card-info .card-id-text {
    color: #555;
    font-size: 11px;
    font-family: monospace;
    margin-top: 2px;
}

/* Overall grade */
.overall-grade {
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}
.grade-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    margin: 0 auto 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: bold;
}
.grade-nm { border: 3px solid #4ecca3; color: #4ecca3; }
.grade-lp { border: 3px solid #7bc47f; color: #7bc47f; }
.grade-mp { border: 3px solid #f0a500; color: #f0a500; }
.grade-hp { border: 3px solid #e94560; color: #e94560; }
.grade-dmg { border: 3px solid #ff2e63; color: #ff2e63; }
.overall-label {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 2px;
}
.overall-sublabel {
    font-size: 13px;
    color: #888;
}

/* Sub-grades */
.subgrades {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
}
.subgrade-card {
    background: #16213e;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.subgrade-card .sg-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.subgrade-card .sg-score {
    font-size: 24px;
    font-weight: bold;
}
.subgrade-card .sg-bar {
    height: 4px;
    border-radius: 2px;
    background: #333;
    margin-top: 8px;
    overflow: hidden;
}
.subgrade-card .sg-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

/* Defect annotations */
.defects-section {
    background: #16213e;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.defects-section h4 {
    font-size: 14px;
    color: #888;
    margin-bottom: 10px;
}
.defect-thumbs {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 4px;
}
.defect-thumb {
    flex-shrink: 0;
    width: 100px;
    border-radius: 8px;
    overflow: hidden;
    background: #0f3460;
}
.defect-thumb img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
}
.defect-thumb .defect-label {
    font-size: 11px;
    color: #ccc;
    padding: 4px 6px;
    text-align: center;
}
.no-defects {
    color: #4ecca3;
    font-size: 13px;
    text-align: center;
    padding: 8px;
}

/* Pricing */
.pricing-section {
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.pricing-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
}
.pricing-row .pr-label {
    color: #888;
    font-size: 14px;
}
.pricing-row .pr-value {
    font-size: 18px;
    font-weight: bold;
}
.pricing-row .pr-value.assessed {
    color: #4ecca3;
}
.pricing-row .pr-value.nm {
    color: #888;
}
.pricing-divider {
    border: none;
    border-top: 1px solid #333;
    margin: 6px 0;
}
.pricing-row .pr-value.diff-up { color: #4ecca3; }
.pricing-row .pr-value.diff-down { color: #e94560; }

/* Action buttons in results */
.result-actions {
    display: flex;
    gap: 8px;
    margin-top: 16px;
}
.result-actions button {
    flex: 1;
    padding: 14px;
    font-size: 15px;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    cursor: pointer;
}
.btn-add-inventory {
    background: #4ecca3;
    color: #1a1a2e;
}
.btn-add-inventory:active { background: #3dbb91; }
.btn-scan-another {
    background: #16213e;
    border: 2px solid #e94560 !important;
    color: #e94560;
}
.btn-scan-another:active { background: #0f3460; }

/* Utility */
.hidden { display: none !important; }
</style>
</head>
<body>

<div class="header" style="position:relative;">
    <a href="/" class="back-link">&larr; Scanner</a>
    <h1>Condition Grader</h1>
    <div class="subtitle">4-angle capture for accurate grading</div>
</div>

<!-- Card Identity Banner -->
<div class="card-banner" id="cardBanner">
    <img id="bannerImg" src="{{IMAGE_URL}}" alt="" onerror="this.style.display='none'">
    <div class="cb-info">
        <h3 id="bannerName">{{CARD_NAME}}</h3>
        <div class="cb-set" id="bannerSet">{{SET_NAME}}</div>
        <div class="cb-id" id="bannerId">{{CARD_ID}}</div>
    </div>
</div>

<!-- Progress Bar -->
<div class="progress-bar" id="progressBar">
    <div class="progress-step active" data-step="0">
        <div class="progress-dot active" id="dot0">1</div>
        <div class="progress-label">Front</div>
    </div>
    <div class="progress-connector" id="conn0"></div>
    <div class="progress-step" data-step="1">
        <div class="progress-dot" id="dot1">2</div>
        <div class="progress-label">Back</div>
    </div>
    <div class="progress-connector" id="conn1"></div>
    <div class="progress-step" data-step="2">
        <div class="progress-dot" id="dot2">3</div>
        <div class="progress-label">Oblique</div>
    </div>
    <div class="progress-connector" id="conn2"></div>
    <div class="progress-step" data-step="3">
        <div class="progress-dot" id="dot3">4</div>
        <div class="progress-label">Edge</div>
    </div>
</div>

<div class="wizard-container">

    <!-- Step 0: Front -->
    <div class="wizard-step active" id="step0">
        <div class="step-instruction">
            <h2>Step 1: Front Face</h2>
            <p>Hold the card flat, straight on. Fill the frame with the front of the card. Used for centering and surface analysis.</p>
        </div>
        <div class="viewfinder-container" id="vf0">
            <video id="video0" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview0">
            <div class="viewfinder-overlay">
                <div class="card-guide"></div>
            </div>
        </div>
        <div class="photo-feedback" id="feedback0"></div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput0">Gallery</label>
            <input type="file" id="galleryInput0" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn0" onclick="capturePhoto(0)"></button>
            <button class="retake-btn" id="retakeBtn0" onclick="retakePhoto(0)">Retake</button>
        </div>
    </div>

    <!-- Step 1: Back -->
    <div class="wizard-step" id="step1">
        <div class="step-instruction">
            <h2>Step 2: Back Face</h2>
            <p>Flip the card over. Hold flat, centered in the frame. Reveals back centering and edge whitening.</p>
        </div>
        <div class="viewfinder-container" id="vf1">
            <video id="video1" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview1">
            <div class="viewfinder-overlay">
                <div class="card-guide"></div>
            </div>
        </div>
        <div class="photo-feedback" id="feedback1"></div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput1">Gallery</label>
            <input type="file" id="galleryInput1" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn1" onclick="capturePhoto(1)"></button>
            <button class="retake-btn" id="retakeBtn1" onclick="retakePhoto(1)">Retake</button>
        </div>
    </div>

    <!-- Step 2: Oblique -->
    <div class="wizard-step" id="step2">
        <div class="step-instruction">
            <h2>Step 3: Oblique Angle</h2>
            <p>Tilt the card ~30 degrees toward a light source. Reveals holo scratches and surface imperfections.</p>
        </div>
        <div class="gyro-indicator" id="gyroIndicator">
            Tilt: <span id="gyroAngle">--</span>
        </div>
        <div class="viewfinder-container" id="vf2">
            <video id="video2" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview2">
            <div class="viewfinder-overlay">
                <div class="card-guide oblique"></div>
            </div>
        </div>
        <div class="photo-feedback" id="feedback2"></div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput2">Gallery</label>
            <input type="file" id="galleryInput2" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn2" onclick="capturePhoto(2)"></button>
            <button class="retake-btn" id="retakeBtn2" onclick="retakePhoto(2)">Retake</button>
        </div>
    </div>

    <!-- Step 3: Edge -->
    <div class="wizard-step" id="step3">
        <div class="step-instruction">
            <h2>Step 4: Edge Close-up</h2>
            <p>Hold the card nearly edge-on, showing the top corners. Reveals corner wear and edge whitening.</p>
        </div>
        <div class="viewfinder-container" id="vf3">
            <video id="video3" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview3">
            <div class="viewfinder-overlay">
                <div class="card-guide edge"></div>
            </div>
        </div>
        <div class="photo-feedback" id="feedback3"></div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput3">Gallery</label>
            <input type="file" id="galleryInput3" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn3" onclick="capturePhoto(3)"></button>
            <button class="retake-btn" id="retakeBtn3" onclick="retakePhoto(3)">Retake</button>
        </div>
    </div>

    <!-- Upload / Review -->
    <div class="upload-section" id="uploadSection">
        <h2 style="color:#fff;margin-bottom:12px;">Review Captures</h2>
        <div class="thumb-grid" id="thumbGrid">
            <div class="thumb"><img id="thumb0"><div class="thumb-label">Front</div></div>
            <div class="thumb"><img id="thumb1"><div class="thumb-label">Back</div></div>
            <div class="thumb"><img id="thumb2"><div class="thumb-label">Oblique</div></div>
            <div class="thumb"><img id="thumb3"><div class="thumb-label">Edge</div></div>
        </div>
        <button class="submit-btn" id="submitBtn" onclick="submitForGrading()">Grade This Card</button>
        <button class="retake-btn" style="display:block;margin:8px auto;" onclick="startOver()">Start Over</button>
    </div>

    <!-- Spinner -->
    <div class="spinner-section" id="spinnerSection">
        <div class="spin-ring"></div>
        <p>Analyzing card condition...</p>
        <p style="font-size:12px;color:#555;margin-top:4px;">Checking centering, surface, edges, corners</p>
    </div>
</div>

<!-- Results -->
<div class="results-section" id="resultsSection">

    <!-- Card identity -->
    <div class="card-identity" id="cardIdentity">
        <img id="resultRefImage" src="" alt="">
        <div class="card-info">
            <h3 id="resultCardName">--</h3>
            <div class="set-name" id="resultSetName">--</div>
            <div class="card-id-text" id="resultCardId"></div>
        </div>
    </div>

    <!-- Overall Grade -->
    <div class="overall-grade">
        <div class="grade-circle" id="gradeCircle">
            <span id="gradeNumber">--</span>
        </div>
        <div class="overall-label" id="overallLabel">--</div>
        <div class="overall-sublabel" id="overallSublabel">--</div>
    </div>

    <!-- Sub-grades -->
    <div class="subgrades" id="subgrades">
        <div class="subgrade-card">
            <div class="sg-label">Centering</div>
            <div class="sg-score" id="sgCentering">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barCentering"></div></div>
        </div>
        <div class="subgrade-card">
            <div class="sg-label">Surface</div>
            <div class="sg-score" id="sgSurface">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barSurface"></div></div>
        </div>
        <div class="subgrade-card">
            <div class="sg-label">Edges</div>
            <div class="sg-score" id="sgEdges">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barEdges"></div></div>
        </div>
        <div class="subgrade-card">
            <div class="sg-label">Corners</div>
            <div class="sg-score" id="sgCorners">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barCorners"></div></div>
        </div>
    </div>

    <!-- Defect annotations -->
    <div class="defects-section" id="defectsSection">
        <h4>Defects Found</h4>
        <div id="defectsContent"></div>
    </div>

    <!-- Pricing -->
    <div class="pricing-section" id="pricingSection">
        <div class="pricing-row">
            <span class="pr-label">NM Market Price</span>
            <span class="pr-value nm" id="priceNM">--</span>
        </div>
        <hr class="pricing-divider">
        <div class="pricing-row">
            <span class="pr-label">Assessed Condition Price</span>
            <span class="pr-value assessed" id="priceAssessed">--</span>
        </div>
        <div class="pricing-row">
            <span class="pr-label">Difference</span>
            <span class="pr-value" id="priceDiff">--</span>
        </div>
    </div>

    <!-- Actions -->
    <div class="result-actions">
        <button class="btn-add-inventory" id="resultAddBtn" onclick="addGradedToInventory()">Add to Inventory</button>
        <button class="btn-scan-another" onclick="window.location='/'">Back to Scanner</button>
    </div>
    <div id="resultInventoryMsg" style="display:none;font-size:13px;text-align:center;margin-top:8px;"></div>
</div>

<script>
(function() {
"use strict";

// ---- Card context (injected by server) ----
var CARD_ID = "{{CARD_ID}}";
var STEPS = ['front', 'back', 'oblique', 'edge'];
var currentStep = 0;
var captures = [null, null, null, null];  // Blob for each step
var captureURLs = [null, null, null, null]; // Object URLs for preview
var stream = null;
var resultData = null;

// ---- Camera ----
function startCamera(stepIdx) {
    var video = document.getElementById('video' + stepIdx);
    if (!video) return;

    // Stop any existing stream
    stopCamera();

    // Prefer rear camera
    var constraints = {
        video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1920 },
            height: { ideal: 2560 }
        },
        audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(s) {
            stream = s;
            video.srcObject = s;
            video.style.display = 'block';
            document.getElementById('preview' + stepIdx).style.display = 'none';
            document.getElementById('captureBtn' + stepIdx).disabled = false;
        })
        .catch(function(err) {
            console.warn('Camera access failed:', err);
            // Camera unavailable - gallery-only mode
            video.style.display = 'none';
            document.getElementById('captureBtn' + stepIdx).disabled = true;
        });
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(function(t) { t.stop(); });
        stream = null;
    }
}

// ---- Capture ----
window.capturePhoto = function(stepIdx) {
    var video = document.getElementById('video' + stepIdx);
    if (!video || !video.videoWidth) return;

    // Draw frame to canvas and extract as blob
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(function(blob) {
        if (!blob) return;
        onPhotoCaptured(stepIdx, blob, URL.createObjectURL(blob));
    }, 'image/jpeg', 0.92);
};

function handleGalleryFile(stepIdx, file) {
    if (!file) return;
    var url = URL.createObjectURL(file);
    onPhotoCaptured(stepIdx, file, url);
}

function onPhotoCaptured(stepIdx, blob, objectURL) {
    // Store
    if (captureURLs[stepIdx]) URL.revokeObjectURL(captureURLs[stepIdx]);
    captures[stepIdx] = blob;
    captureURLs[stepIdx] = objectURL;

    // Show preview in viewfinder
    var video = document.getElementById('video' + stepIdx);
    var preview = document.getElementById('preview' + stepIdx);
    if (video) video.style.display = 'none';
    preview.src = objectURL;
    preview.style.display = 'block';
    stopCamera();

    // Show retake, hide capture
    document.getElementById('captureBtn' + stepIdx).style.display = 'none';
    document.getElementById('retakeBtn' + stepIdx).style.display = 'inline-block';

    // Update progress dot with thumbnail
    var dot = document.getElementById('dot' + stepIdx);
    dot.innerHTML = '<img src="' + objectURL + '">';
    dot.classList.remove('active');
    dot.classList.add('done');
    var stepEl = dot.closest('.progress-step');
    stepEl.classList.remove('active');
    stepEl.classList.add('done');

    // Mark connector done
    if (stepIdx > 0) {
        document.getElementById('conn' + (stepIdx - 1)).classList.add('done');
    }

    // Upload photo for immediate quality feedback
    uploadPhotoForFeedback(stepIdx, blob);
}

// ---- Per-photo quality check ----
function uploadPhotoForFeedback(stepIdx, blob) {
    var feedbackEl = document.getElementById('feedback' + stepIdx);
    feedbackEl.className = 'photo-feedback checking';
    feedbackEl.textContent = 'Checking image quality...';

    var fd = new FormData();
    fd.append('photo', blob, STEPS[stepIdx] + '.jpg');

    var encodedCardId = encodeURIComponent(CARD_ID);
    fetch('/condition/photo/' + encodedCardId + '/' + stepIdx, {
        method: 'POST',
        body: fd
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.quality === 'good') {
            feedbackEl.className = 'photo-feedback good';
            feedbackEl.textContent = data.message || 'Good quality - sharp and well-lit';
        } else if (data.quality === 'acceptable') {
            feedbackEl.className = 'photo-feedback warn';
            feedbackEl.textContent = data.message || 'Acceptable - consider retaking for better results';
        } else {
            feedbackEl.className = 'photo-feedback bad';
            feedbackEl.textContent = data.message || 'Poor quality - please retake';
        }

        // Auto-advance after feedback (short delay)
        setTimeout(function() {
            if (stepIdx < 3) {
                goToStep(stepIdx + 1);
            } else {
                showUploadReview();
            }
        }, 800);
    })
    .catch(function(err) {
        // On network error, still advance (feedback is best-effort)
        feedbackEl.className = 'photo-feedback warn';
        feedbackEl.textContent = 'Could not check quality (offline?)';
        setTimeout(function() {
            if (stepIdx < 3) {
                goToStep(stepIdx + 1);
            } else {
                showUploadReview();
            }
        }, 600);
    });
}

window.retakePhoto = function(stepIdx) {
    // Clear capture
    if (captureURLs[stepIdx]) URL.revokeObjectURL(captureURLs[stepIdx]);
    captures[stepIdx] = null;
    captureURLs[stepIdx] = null;

    // Reset UI
    var preview = document.getElementById('preview' + stepIdx);
    preview.style.display = 'none';
    preview.src = '';
    document.getElementById('captureBtn' + stepIdx).style.display = '';
    document.getElementById('captureBtn' + stepIdx).disabled = false;
    document.getElementById('retakeBtn' + stepIdx).style.display = 'none';

    // Reset feedback
    var feedbackEl = document.getElementById('feedback' + stepIdx);
    feedbackEl.className = 'photo-feedback';
    feedbackEl.textContent = '';

    // Reset progress dot
    var dot = document.getElementById('dot' + stepIdx);
    dot.innerHTML = '' + (stepIdx + 1);
    dot.classList.remove('done');
    dot.classList.add('active');
    var stepEl = dot.closest('.progress-step');
    stepEl.classList.remove('done');
    stepEl.classList.add('active');

    // Restart camera
    startCamera(stepIdx);
};

// ---- Step Navigation ----
function goToStep(stepIdx) {
    // Hide all steps
    for (var i = 0; i < 4; i++) {
        document.getElementById('step' + i).classList.remove('active');
    }
    document.getElementById('uploadSection').classList.remove('active');

    currentStep = stepIdx;
    document.getElementById('step' + stepIdx).classList.add('active');

    // Update progress dots
    for (var i = 0; i < 4; i++) {
        var dot = document.getElementById('dot' + i);
        var stepEl = dot.closest('.progress-step');
        if (captures[i]) {
            // Already captured
        } else if (i === stepIdx) {
            dot.classList.add('active');
            dot.classList.remove('done');
            stepEl.classList.add('active');
            stepEl.classList.remove('done');
        } else {
            dot.classList.remove('active', 'done');
            stepEl.classList.remove('active', 'done');
        }
    }

    // Start camera for this step if not already captured
    if (!captures[stepIdx]) {
        startCamera(stepIdx);
    }
}

function showUploadReview() {
    // Hide wizard steps
    for (var i = 0; i < 4; i++) {
        document.getElementById('step' + i).classList.remove('active');
    }

    // Populate thumbnails
    for (var i = 0; i < 4; i++) {
        var thumbImg = document.getElementById('thumb' + i);
        if (captureURLs[i]) {
            thumbImg.src = captureURLs[i];
        }
    }

    document.getElementById('uploadSection').classList.add('active');
    document.getElementById('progressBar').style.display = 'none';
    stopCamera();
}

// ---- Upload ----
window.submitForGrading = function() {
    // Verify all 4 captures exist
    for (var i = 0; i < 4; i++) {
        if (!captures[i]) {
            alert('Missing ' + STEPS[i] + ' photo. Please capture all 4 angles.');
            return;
        }
    }

    document.getElementById('uploadSection').classList.remove('active');
    document.getElementById('spinnerSection').classList.add('active');

    var fd = new FormData();
    fd.append('front', captures[0], 'front.jpg');
    fd.append('back', captures[1], 'back.jpg');
    fd.append('oblique', captures[2], 'oblique.jpg');
    fd.append('edge', captures[3], 'edge.jpg');
    fd.append('card_id', CARD_ID);

    fetch('/condition/assess', { method: 'POST', body: fd })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            document.getElementById('spinnerSection').classList.remove('active');
            if (data.error) {
                alert('Error: ' + data.error);
                document.getElementById('uploadSection').classList.add('active');
                return;
            }
            resultData = data;
            showResults(data);
        })
        .catch(function(err) {
            document.getElementById('spinnerSection').classList.remove('active');
            alert('Upload failed: ' + err);
            document.getElementById('uploadSection').classList.add('active');
        });
};

// ---- Results Display ----
function showResults(data) {
    var section = document.getElementById('resultsSection');
    section.classList.add('active');

    // Hide wizard
    document.getElementById('progressBar').style.display = 'none';

    // Card identity
    var refImg = document.getElementById('resultRefImage');
    if (data.image_url || data.local_image_url) {
        refImg.src = data.local_image_url || data.image_url;
        refImg.style.display = 'block';
    } else {
        refImg.style.display = 'none';
    }
    document.getElementById('resultCardName').textContent = data.card_name || '{{CARD_NAME}}';
    document.getElementById('resultSetName').textContent = data.set_name || '{{SET_NAME}}';
    document.getElementById('resultCardId').textContent = data.card_id || CARD_ID;

    // Overall grade
    var overall = data.overall_grade || 0;
    var condition = data.condition || 'NM';
    document.getElementById('gradeNumber').textContent = overall.toFixed(1);
    document.getElementById('overallLabel').textContent = condition;

    var conditionDescriptions = {
        'NM': 'Near Mint',
        'LP': 'Lightly Played',
        'MP': 'Moderately Played',
        'HP': 'Heavily Played',
        'DMG': 'Damaged'
    };
    document.getElementById('overallSublabel').textContent =
        conditionDescriptions[condition] || condition;

    // Grade circle color
    var circle = document.getElementById('gradeCircle');
    circle.className = 'grade-circle';
    if (condition === 'NM') circle.classList.add('grade-nm');
    else if (condition === 'LP') circle.classList.add('grade-lp');
    else if (condition === 'MP') circle.classList.add('grade-mp');
    else if (condition === 'HP') circle.classList.add('grade-hp');
    else circle.classList.add('grade-dmg');

    // Sub-grades
    var grades = data.sub_grades || {};
    setSubgrade('Centering', grades.centering);
    setSubgrade('Surface', grades.surface);
    setSubgrade('Edges', grades.edges);
    setSubgrade('Corners', grades.corners);

    // Defects
    var defects = data.defects || [];
    var defectsContent = document.getElementById('defectsContent');
    if (defects.length === 0) {
        defectsContent.innerHTML = '<div class="no-defects">No significant defects detected</div>';
    } else {
        var html = '<div class="defect-thumbs">';
        for (var i = 0; i < defects.length; i++) {
            var d = defects[i];
            html += '<div class="defect-thumb">';
            if (d.image_url) {
                html += '<img src="' + d.image_url + '" alt="' + (d.label || '') + '">';
            } else {
                html += '<div style="width:100%;aspect-ratio:1;background:#0a1628;display:flex;align-items:center;justify-content:center;color:#555;font-size:24px;">';
                html += defectIcon(d.type);
                html += '</div>';
            }
            html += '<div class="defect-label">' + (d.label || d.type || 'Defect') + '</div>';
            html += '</div>';
        }
        html += '</div>';
        defectsContent.innerHTML = html;
    }

    // Pricing
    var nmPrice = data.nm_price;
    var assessedPrice = data.assessed_price;
    document.getElementById('priceNM').textContent = nmPrice ? '$' + nmPrice.toFixed(2) : '--';
    document.getElementById('priceAssessed').textContent = assessedPrice ? '$' + assessedPrice.toFixed(2) : '--';

    var diffEl = document.getElementById('priceDiff');
    if (nmPrice && assessedPrice) {
        var diff = assessedPrice - nmPrice;
        var pct = ((diff / nmPrice) * 100).toFixed(0);
        var sign = diff >= 0 ? '+' : '';
        diffEl.textContent = sign + '$' + diff.toFixed(2) + ' (' + sign + pct + '%)';
        diffEl.className = 'pr-value ' + (diff >= 0 ? 'diff-up' : 'diff-down');
    } else {
        diffEl.textContent = '--';
        diffEl.className = 'pr-value';
    }

    // Store card_id for inventory
    document.getElementById('resultAddBtn').dataset.cardId = data.card_id || CARD_ID;
    document.getElementById('resultAddBtn').dataset.condition = condition;
}

function setSubgrade(name, score) {
    var id = name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
    if (score === undefined || score === null) score = 0;

    var scoreEl = document.getElementById('sg' + id);
    var barEl = document.getElementById('bar' + id);
    if (!scoreEl || !barEl) return;

    scoreEl.textContent = score.toFixed(1);
    var pct = (score / 10) * 100;
    barEl.style.width = pct + '%';

    // Color based on score
    if (score >= 8) barEl.style.background = '#4ecca3';
    else if (score >= 6) barEl.style.background = '#7bc47f';
    else if (score >= 4) barEl.style.background = '#f0a500';
    else barEl.style.background = '#e94560';
}

function defectIcon(type) {
    var icons = {
        'scratch': '/',
        'whitening': 'W',
        'dent': 'D',
        'crease': '~',
        'miscut': 'M',
        'print_line': '|'
    };
    return icons[type] || '!';
}

// ---- Inventory ----
window.addGradedToInventory = function() {
    var btn = document.getElementById('resultAddBtn');
    var msg = document.getElementById('resultInventoryMsg');
    var cardId = btn.dataset.cardId;
    if (!cardId) return;

    btn.disabled = true;
    btn.textContent = 'Adding...';

    fetch('/inventory/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            card_id: cardId,
            quantity: 1,
            condition: btn.dataset.condition || 'NM'
        })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false;
        btn.textContent = 'Add to Inventory';
        msg.style.display = 'block';
        if (data.error) {
            msg.style.color = '#e94560';
            msg.textContent = data.error;
        } else {
            msg.style.color = '#4ecca3';
            msg.textContent = 'Added! Total in inventory: ' + data.quantity;
        }
    })
    .catch(function(e) {
        btn.disabled = false;
        btn.textContent = 'Add to Inventory';
        msg.style.display = 'block';
        msg.style.color = '#e94560';
        msg.textContent = 'Error: ' + e;
    });
};

// ---- Start Over ----
window.startOver = function() {
    stopCamera();

    // Reset captures
    for (var i = 0; i < 4; i++) {
        if (captureURLs[i]) URL.revokeObjectURL(captureURLs[i]);
        captures[i] = null;
        captureURLs[i] = null;

        // Reset viewfinder
        var preview = document.getElementById('preview' + i);
        preview.style.display = 'none';
        preview.src = '';
        var video = document.getElementById('video' + i);
        if (video) video.style.display = 'block';
        document.getElementById('captureBtn' + i).style.display = '';
        document.getElementById('captureBtn' + i).disabled = false;
        document.getElementById('retakeBtn' + i).style.display = 'none';

        // Reset progress dots
        var dot = document.getElementById('dot' + i);
        dot.innerHTML = '' + (i + 1);
        dot.className = 'progress-dot';
        var stepEl = dot.closest('.progress-step');
        stepEl.className = 'progress-step';

        // Reset feedback
        var feedbackEl = document.getElementById('feedback' + i);
        feedbackEl.className = 'photo-feedback';
        feedbackEl.textContent = '';
    }
    // Reset connectors
    for (var i = 0; i < 3; i++) {
        document.getElementById('conn' + i).classList.remove('done');
    }

    // Hide sections
    document.getElementById('uploadSection').classList.remove('active');
    document.getElementById('spinnerSection').classList.remove('active');
    document.getElementById('resultsSection').classList.remove('active');
    document.getElementById('resultInventoryMsg').style.display = 'none';

    // Show progress bar and first step
    document.getElementById('progressBar').style.display = 'flex';
    currentStep = 0;
    resultData = null;
    goToStep(0);
};

// ---- Gyroscope Feedback (Step 2 - Oblique) ----
function initGyroscope() {
    var indicator = document.getElementById('gyroIndicator');
    var angleSpan = document.getElementById('gyroAngle');

    if (!window.DeviceOrientationEvent) return;

    // Request permission on iOS 13+
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        document.addEventListener('click', function requestGyro() {
            DeviceOrientationEvent.requestPermission().then(function(state) {
                if (state === 'granted') listenGyro();
            }).catch(function() {});
            document.removeEventListener('click', requestGyro);
        }, { once: true });
    } else {
        listenGyro();
    }

    function listenGyro() {
        window.addEventListener('deviceorientation', function(e) {
            // Only show during oblique step
            if (currentStep !== 2) {
                indicator.style.display = 'none';
                return;
            }
            indicator.style.display = 'block';

            var beta = e.beta; // front-back tilt (-180 to 180)
            if (beta === null) {
                indicator.style.display = 'none';
                return;
            }

            var absBeta = Math.abs(beta);
            angleSpan.textContent = absBeta.toFixed(0) + ' deg';

            // Ideal oblique angle: 20-45 degrees
            if (absBeta >= 20 && absBeta <= 45) {
                indicator.className = 'gyro-indicator good';
                angleSpan.textContent += ' - Good angle!';
            } else if (absBeta < 20) {
                indicator.className = 'gyro-indicator adjust';
                angleSpan.textContent += ' - Tilt more';
            } else {
                indicator.className = 'gyro-indicator adjust';
                angleSpan.textContent += ' - Too steep';
            }
        });
    }
}

// ---- Gallery file input handlers ----
for (var i = 0; i < 4; i++) {
    (function(idx) {
        document.getElementById('galleryInput' + idx).addEventListener('change', function() {
            if (this.files && this.files[0]) {
                handleGalleryFile(idx, this.files[0]);
            }
            this.value = '';
        });
    })(i);
}

// ---- Init ----
initGyroscope();
startCamera(0);

})();
</script>
</body>
</html>
"""

CONDITION_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Card Condition Grader</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
    min-height: 100dvh;
    overflow-x: hidden;
}

/* ---- Header ---- */
.header {
    text-align: center;
    padding: 12px 15px 8px;
    border-bottom: 1px solid #16213e;
}
.header h1 {
    font-size: 20px;
    color: #e94560;
}
.header .subtitle {
    font-size: 13px;
    color: #888;
    margin-top: 2px;
}
.back-link {
    position: absolute;
    top: 14px;
    left: 15px;
    color: #4ecca3;
    text-decoration: none;
    font-size: 14px;
}

/* ---- Progress Bar ---- */
.progress-bar {
    display: flex;
    justify-content: center;
    gap: 8px;
    padding: 16px 20px 12px;
}
.progress-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    flex: 1;
    max-width: 80px;
}
.progress-dot {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid #333;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #555;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.progress-dot.active {
    border-color: #e94560;
    color: #e94560;
    box-shadow: 0 0 12px rgba(233, 69, 96, 0.3);
}
.progress-dot.done {
    border-color: #4ecca3;
    background: #4ecca3;
    color: #1a1a2e;
}
.progress-dot img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
}
.progress-label {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: color 0.3s ease;
}
.progress-step.active .progress-label { color: #e94560; }
.progress-step.done .progress-label { color: #4ecca3; }
.progress-connector {
    width: 20px;
    height: 2px;
    background: #333;
    align-self: center;
    margin-bottom: 16px;
    transition: background 0.3s ease;
}
.progress-connector.done { background: #4ecca3; }

/* ---- Wizard Steps ---- */
.wizard-container {
    max-width: 500px;
    margin: 0 auto;
    padding: 0 15px;
}
.wizard-step {
    display: none;
    flex-direction: column;
    align-items: center;
    animation: fadeIn 0.3s ease;
}
.wizard-step.active { display: flex; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Viewfinder ---- */
.viewfinder-container {
    width: 100%;
    max-width: 400px;
    aspect-ratio: 3/4;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    margin: 8px 0;
}
.viewfinder-container video {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.viewfinder-container .captured-preview {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: none;
}
.viewfinder-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
}
/* Card outline guide */
.card-guide {
    position: absolute;
    top: 8%;
    left: 12%;
    right: 12%;
    bottom: 8%;
    border: 2px dashed rgba(78, 204, 163, 0.5);
    border-radius: 12px;
}
.card-guide.oblique {
    transform: perspective(400px) rotateX(25deg);
    top: 12%;
    bottom: 12%;
}
.card-guide.edge {
    top: 30%;
    bottom: 30%;
    left: 20%;
    right: 20%;
    border-radius: 4px;
}

/* ---- Instructions ---- */
.step-instruction {
    text-align: center;
    padding: 12px 10px;
}
.step-instruction h2 {
    font-size: 18px;
    color: #fff;
    margin-bottom: 4px;
}
.step-instruction p {
    font-size: 14px;
    color: #888;
    line-height: 1.4;
}

/* ---- Gyro feedback ---- */
.gyro-indicator {
    display: none;
    text-align: center;
    margin: 4px 0;
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 20px;
    background: #16213e;
}
.gyro-indicator.good { color: #4ecca3; background: rgba(78, 204, 163, 0.15); }
.gyro-indicator.adjust { color: #f0a500; background: rgba(240, 165, 0, 0.15); }

/* ---- Capture Button ---- */
.capture-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 12px 0 20px;
    width: 100%;
}
.capture-btn {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    border: 3px solid #e94560;
    background: transparent;
    cursor: pointer;
    position: relative;
    transition: all 0.15s ease;
}
.capture-btn:active {
    transform: scale(0.92);
}
.capture-btn::after {
    content: '';
    position: absolute;
    top: 4px; left: 4px; right: 4px; bottom: 4px;
    border-radius: 50%;
    background: #e94560;
}
.capture-btn:disabled {
    border-color: #333;
    opacity: 0.5;
}
.capture-btn:disabled::after {
    background: #333;
}
.retake-btn {
    padding: 8px 16px;
    background: #16213e;
    border: 1px solid #333;
    color: #888;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
    display: none;
}
.retake-btn:active { background: #0f3460; }
.gallery-fallback-btn {
    padding: 8px 16px;
    background: #16213e;
    border: 1px solid #333;
    color: #888;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
}
.gallery-fallback-btn:active { background: #0f3460; }

/* ---- Upload / Spinner ---- */
.upload-section {
    display: none;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    animation: fadeIn 0.3s ease;
}
.upload-section.active { display: flex; }
.thumb-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    width: 100%;
    max-width: 400px;
    margin-bottom: 16px;
}
.thumb-grid .thumb {
    aspect-ratio: 3/4;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    background: #16213e;
}
.thumb-grid .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.thumb-grid .thumb .thumb-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.7);
    font-size: 10px;
    color: #ccc;
    text-align: center;
    padding: 2px;
    text-transform: uppercase;
}
.submit-btn {
    display: block;
    width: 100%;
    max-width: 400px;
    padding: 16px;
    font-size: 17px;
    font-weight: bold;
    background: #4ecca3;
    color: #1a1a2e;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    margin: 8px 0;
}
.submit-btn:active { background: #3dbb91; }
.submit-btn:disabled {
    background: #333;
    color: #666;
    cursor: not-allowed;
}
.spinner-section {
    display: none;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
    animation: fadeIn 0.3s ease;
}
.spinner-section.active { display: flex; }
.spin-ring {
    width: 48px;
    height: 48px;
    border: 4px solid #16213e;
    border-top-color: #e94560;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-section p {
    margin-top: 16px;
    color: #888;
    font-size: 14px;
}

/* ---- Results ---- */
.results-section {
    display: none;
    max-width: 500px;
    margin: 0 auto;
    padding: 0 15px 30px;
    animation: fadeIn 0.4s ease;
}
.results-section.active { display: block; }

.card-identity {
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    display: flex;
    gap: 14px;
    align-items: center;
}
.card-identity img {
    width: 80px;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.card-identity .card-info h3 {
    color: #fff;
    font-size: 17px;
    margin-bottom: 2px;
}
.card-identity .card-info .set-name {
    color: #888;
    font-size: 13px;
}
.card-identity .card-info .card-id-text {
    color: #555;
    font-size: 11px;
    font-family: monospace;
    margin-top: 2px;
}

/* Overall grade */
.overall-grade {
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}
.grade-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    margin: 0 auto 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: bold;
}
.grade-nm { border: 3px solid #4ecca3; color: #4ecca3; }
.grade-lp { border: 3px solid #7bc47f; color: #7bc47f; }
.grade-mp { border: 3px solid #f0a500; color: #f0a500; }
.grade-hp { border: 3px solid #e94560; color: #e94560; }
.grade-dmg { border: 3px solid #ff2e63; color: #ff2e63; }
.overall-label {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 2px;
}
.overall-sublabel {
    font-size: 13px;
    color: #888;
}

/* Sub-grades */
.subgrades {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
}
.subgrade-card {
    background: #16213e;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.subgrade-card .sg-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.subgrade-card .sg-score {
    font-size: 24px;
    font-weight: bold;
}
.subgrade-card .sg-bar {
    height: 4px;
    border-radius: 2px;
    background: #333;
    margin-top: 8px;
    overflow: hidden;
}
.subgrade-card .sg-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

/* Defect annotations */
.defects-section {
    background: #16213e;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.defects-section h4 {
    font-size: 14px;
    color: #888;
    margin-bottom: 10px;
}
.defect-thumbs {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 4px;
}
.defect-thumb {
    flex-shrink: 0;
    width: 100px;
    border-radius: 8px;
    overflow: hidden;
    background: #0f3460;
}
.defect-thumb img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
}
.defect-thumb .defect-label {
    font-size: 11px;
    color: #ccc;
    padding: 4px 6px;
    text-align: center;
}
.no-defects {
    color: #4ecca3;
    font-size: 13px;
    text-align: center;
    padding: 8px;
}

/* Pricing */
.pricing-section {
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.pricing-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
}
.pricing-row .pr-label {
    color: #888;
    font-size: 14px;
}
.pricing-row .pr-value {
    font-size: 18px;
    font-weight: bold;
}
.pricing-row .pr-value.assessed {
    color: #4ecca3;
}
.pricing-row .pr-value.nm {
    color: #888;
}
.pricing-divider {
    border: none;
    border-top: 1px solid #333;
    margin: 6px 0;
}
.pricing-row .pr-value.diff-up { color: #4ecca3; }
.pricing-row .pr-value.diff-down { color: #e94560; }

/* Action buttons in results */
.result-actions {
    display: flex;
    gap: 8px;
    margin-top: 16px;
}
.result-actions button {
    flex: 1;
    padding: 14px;
    font-size: 15px;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    cursor: pointer;
}
.btn-add-inventory {
    background: #4ecca3;
    color: #1a1a2e;
}
.btn-add-inventory:active { background: #3dbb91; }
.btn-scan-another {
    background: #16213e;
    border: 2px solid #e94560 !important;
    color: #e94560;
}
.btn-scan-another:active { background: #0f3460; }

/* Utility */
.hidden { display: none !important; }
</style>
</head>
<body>

<div class="header" style="position:relative;">
    <a href="/" class="back-link">&larr; Scanner</a>
    <h1>Condition Grader</h1>
    <div class="subtitle">4-angle capture for accurate grading</div>
</div>

<!-- Progress Bar -->
<div class="progress-bar" id="progressBar">
    <div class="progress-step active" data-step="0">
        <div class="progress-dot active" id="dot0">1</div>
        <div class="progress-label">Front</div>
    </div>
    <div class="progress-connector" id="conn0"></div>
    <div class="progress-step" data-step="1">
        <div class="progress-dot" id="dot1">2</div>
        <div class="progress-label">Back</div>
    </div>
    <div class="progress-connector" id="conn1"></div>
    <div class="progress-step" data-step="2">
        <div class="progress-dot" id="dot2">3</div>
        <div class="progress-label">Oblique</div>
    </div>
    <div class="progress-connector" id="conn2"></div>
    <div class="progress-step" data-step="3">
        <div class="progress-dot" id="dot3">4</div>
        <div class="progress-label">Edge</div>
    </div>
</div>

<div class="wizard-container">

    <!-- Step 0: Front -->
    <div class="wizard-step active" id="step0">
        <div class="step-instruction">
            <h2>Front Face</h2>
            <p>Hold the card flat, fill the frame with the front of the card.</p>
        </div>
        <div class="viewfinder-container" id="vf0">
            <video id="video0" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview0">
            <div class="viewfinder-overlay">
                <div class="card-guide"></div>
            </div>
        </div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput0">Gallery</label>
            <input type="file" id="galleryInput0" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn0" onclick="capturePhoto(0)"></button>
            <button class="retake-btn" id="retakeBtn0" onclick="retakePhoto(0)">Retake</button>
        </div>
    </div>

    <!-- Step 1: Back -->
    <div class="wizard-step" id="step1">
        <div class="step-instruction">
            <h2>Back Face</h2>
            <p>Flip the card over. Hold flat, centered in the frame.</p>
        </div>
        <div class="viewfinder-container" id="vf1">
            <video id="video1" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview1">
            <div class="viewfinder-overlay">
                <div class="card-guide"></div>
            </div>
        </div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput1">Gallery</label>
            <input type="file" id="galleryInput1" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn1" onclick="capturePhoto(1)"></button>
            <button class="retake-btn" id="retakeBtn1" onclick="retakePhoto(1)">Retake</button>
        </div>
    </div>

    <!-- Step 2: Oblique -->
    <div class="wizard-step" id="step2">
        <div class="step-instruction">
            <h2>Oblique / Glare</h2>
            <p>Tilt the card toward a light source until you see reflections across the surface.</p>
        </div>
        <div class="gyro-indicator" id="gyroIndicator">
            Tilt: <span id="gyroAngle">--</span>
        </div>
        <div class="viewfinder-container" id="vf2">
            <video id="video2" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview2">
            <div class="viewfinder-overlay">
                <div class="card-guide oblique"></div>
            </div>
        </div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput2">Gallery</label>
            <input type="file" id="galleryInput2" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn2" onclick="capturePhoto(2)"></button>
            <button class="retake-btn" id="retakeBtn2" onclick="retakePhoto(2)">Retake</button>
        </div>
    </div>

    <!-- Step 3: Edge -->
    <div class="wizard-step" id="step3">
        <div class="step-instruction">
            <h2>Edge View</h2>
            <p>Hold the card nearly edge-on, showing the top corners. Reveals edge wear and whitening.</p>
        </div>
        <div class="viewfinder-container" id="vf3">
            <video id="video3" autoplay playsinline muted></video>
            <img class="captured-preview" id="preview3">
            <div class="viewfinder-overlay">
                <div class="card-guide edge"></div>
            </div>
        </div>
        <div class="capture-row">
            <label class="gallery-fallback-btn" for="galleryInput3">Gallery</label>
            <input type="file" id="galleryInput3" accept="image/*" style="display:none;">
            <button class="capture-btn" id="captureBtn3" onclick="capturePhoto(3)"></button>
            <button class="retake-btn" id="retakeBtn3" onclick="retakePhoto(3)">Retake</button>
        </div>
    </div>

    <!-- Upload / Review -->
    <div class="upload-section" id="uploadSection">
        <h2 style="color:#fff;margin-bottom:12px;">Review Captures</h2>
        <div class="thumb-grid" id="thumbGrid">
            <div class="thumb"><img id="thumb0"><div class="thumb-label">Front</div></div>
            <div class="thumb"><img id="thumb1"><div class="thumb-label">Back</div></div>
            <div class="thumb"><img id="thumb2"><div class="thumb-label">Oblique</div></div>
            <div class="thumb"><img id="thumb3"><div class="thumb-label">Edge</div></div>
        </div>
        <button class="submit-btn" id="submitBtn" onclick="submitForGrading()">Grade This Card</button>
        <button class="retake-btn" style="display:block;margin:8px auto;" onclick="startOver()">Start Over</button>
    </div>

    <!-- Spinner -->
    <div class="spinner-section" id="spinnerSection">
        <div class="spin-ring"></div>
        <p>Analyzing card condition...</p>
        <p style="font-size:12px;color:#555;margin-top:4px;">Checking centering, surface, edges, corners</p>
    </div>
</div>

<!-- Results -->
<div class="results-section" id="resultsSection">

    <!-- Card identity -->
    <div class="card-identity" id="cardIdentity">
        <img id="resultRefImage" src="" alt="">
        <div class="card-info">
            <h3 id="resultCardName">--</h3>
            <div class="set-name" id="resultSetName">--</div>
            <div class="card-id-text" id="resultCardId"></div>
        </div>
    </div>

    <!-- Overall Grade -->
    <div class="overall-grade">
        <div class="grade-circle" id="gradeCircle">
            <span id="gradeNumber">--</span>
        </div>
        <div class="overall-label" id="overallLabel">--</div>
        <div class="overall-sublabel" id="overallSublabel">--</div>
    </div>

    <!-- Sub-grades -->
    <div class="subgrades" id="subgrades">
        <div class="subgrade-card">
            <div class="sg-label">Centering</div>
            <div class="sg-score" id="sgCentering">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barCentering"></div></div>
        </div>
        <div class="subgrade-card">
            <div class="sg-label">Surface</div>
            <div class="sg-score" id="sgSurface">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barSurface"></div></div>
        </div>
        <div class="subgrade-card">
            <div class="sg-label">Edges</div>
            <div class="sg-score" id="sgEdges">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barEdges"></div></div>
        </div>
        <div class="subgrade-card">
            <div class="sg-label">Corners</div>
            <div class="sg-score" id="sgCorners">--</div>
            <div class="sg-bar"><div class="sg-bar-fill" id="barCorners"></div></div>
        </div>
    </div>

    <!-- Defect annotations -->
    <div class="defects-section" id="defectsSection">
        <h4>Defects Found</h4>
        <div id="defectsContent"></div>
    </div>

    <!-- Pricing -->
    <div class="pricing-section" id="pricingSection">
        <div class="pricing-row">
            <span class="pr-label">NM Market Price</span>
            <span class="pr-value nm" id="priceNM">--</span>
        </div>
        <hr class="pricing-divider">
        <div class="pricing-row">
            <span class="pr-label">Assessed Condition Price</span>
            <span class="pr-value assessed" id="priceAssessed">--</span>
        </div>
        <div class="pricing-row">
            <span class="pr-label">Difference</span>
            <span class="pr-value" id="priceDiff">--</span>
        </div>
    </div>

    <!-- Actions -->
    <div class="result-actions">
        <button class="btn-add-inventory" id="resultAddBtn" onclick="addGradedToInventory()">Add to Inventory</button>
        <button class="btn-scan-another" onclick="startOver()">Scan Another</button>
    </div>
    <div id="resultInventoryMsg" style="display:none;font-size:13px;text-align:center;margin-top:8px;"></div>
</div>

<script>
(function() {
"use strict";

// ---- State ----
var STEPS = ['front', 'back', 'oblique', 'edge'];
var currentStep = 0;
var captures = [null, null, null, null];  // Blob for each step
var captureURLs = [null, null, null, null]; // Object URLs for preview
var stream = null;
var resultData = null;

// ---- Camera ----
function startCamera(stepIdx) {
    var video = document.getElementById('video' + stepIdx);
    if (!video) return;

    // Stop any existing stream
    stopCamera();

    // Prefer rear camera
    var constraints = {
        video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1920 },
            height: { ideal: 2560 }
        },
        audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(s) {
            stream = s;
            video.srcObject = s;
            video.style.display = 'block';
            document.getElementById('preview' + stepIdx).style.display = 'none';
            document.getElementById('captureBtn' + stepIdx).disabled = false;
        })
        .catch(function(err) {
            console.warn('Camera access failed:', err);
            // Camera unavailable - gallery-only mode
            video.style.display = 'none';
            document.getElementById('captureBtn' + stepIdx).disabled = true;
        });
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(function(t) { t.stop(); });
        stream = null;
    }
}

// ---- Capture ----
window.capturePhoto = function(stepIdx) {
    var video = document.getElementById('video' + stepIdx);
    if (!video || !video.videoWidth) return;

    // Draw frame to canvas and extract as blob
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(function(blob) {
        if (!blob) return;
        onPhotoCaptured(stepIdx, blob, URL.createObjectURL(blob));
    }, 'image/jpeg', 0.92);
};

function handleGalleryFile(stepIdx, file) {
    if (!file) return;
    var url = URL.createObjectURL(file);
    onPhotoCaptured(stepIdx, file, url);
}

function onPhotoCaptured(stepIdx, blob, objectURL) {
    // Store
    if (captureURLs[stepIdx]) URL.revokeObjectURL(captureURLs[stepIdx]);
    captures[stepIdx] = blob;
    captureURLs[stepIdx] = objectURL;

    // Show preview in viewfinder
    var video = document.getElementById('video' + stepIdx);
    var preview = document.getElementById('preview' + stepIdx);
    if (video) video.style.display = 'none';
    preview.src = objectURL;
    preview.style.display = 'block';
    stopCamera();

    // Show retake, hide capture
    document.getElementById('captureBtn' + stepIdx).style.display = 'none';
    document.getElementById('retakeBtn' + stepIdx).style.display = 'inline-block';

    // Update progress dot with thumbnail
    var dot = document.getElementById('dot' + stepIdx);
    dot.innerHTML = '<img src="' + objectURL + '">';
    dot.classList.remove('active');
    dot.classList.add('done');
    var stepEl = dot.closest('.progress-step');
    stepEl.classList.remove('active');
    stepEl.classList.add('done');

    // Mark connector done
    if (stepIdx > 0) {
        document.getElementById('conn' + (stepIdx - 1)).classList.add('done');
    }

    // Auto-advance after short delay
    setTimeout(function() {
        if (stepIdx < 3) {
            goToStep(stepIdx + 1);
        } else {
            showUploadReview();
        }
    }, 500);
}

window.retakePhoto = function(stepIdx) {
    // Clear capture
    if (captureURLs[stepIdx]) URL.revokeObjectURL(captureURLs[stepIdx]);
    captures[stepIdx] = null;
    captureURLs[stepIdx] = null;

    // Reset UI
    var preview = document.getElementById('preview' + stepIdx);
    preview.style.display = 'none';
    preview.src = '';
    document.getElementById('captureBtn' + stepIdx).style.display = '';
    document.getElementById('captureBtn' + stepIdx).disabled = false;
    document.getElementById('retakeBtn' + stepIdx).style.display = 'none';

    // Reset progress dot
    var dot = document.getElementById('dot' + stepIdx);
    dot.innerHTML = '' + (stepIdx + 1);
    dot.classList.remove('done');
    dot.classList.add('active');
    var stepEl = dot.closest('.progress-step');
    stepEl.classList.remove('done');
    stepEl.classList.add('active');

    // Restart camera
    startCamera(stepIdx);
};

// ---- Step Navigation ----
function goToStep(stepIdx) {
    // Hide all steps
    for (var i = 0; i < 4; i++) {
        document.getElementById('step' + i).classList.remove('active');
    }
    document.getElementById('uploadSection').classList.remove('active');

    currentStep = stepIdx;
    document.getElementById('step' + stepIdx).classList.add('active');

    // Update progress dots
    for (var i = 0; i < 4; i++) {
        var dot = document.getElementById('dot' + i);
        var stepEl = dot.closest('.progress-step');
        if (captures[i]) {
            // Already captured
        } else if (i === stepIdx) {
            dot.classList.add('active');
            dot.classList.remove('done');
            stepEl.classList.add('active');
            stepEl.classList.remove('done');
        } else {
            dot.classList.remove('active', 'done');
            stepEl.classList.remove('active', 'done');
        }
    }

    // Start camera for this step if not already captured
    if (!captures[stepIdx]) {
        startCamera(stepIdx);
    }
}

function showUploadReview() {
    // Hide wizard steps
    for (var i = 0; i < 4; i++) {
        document.getElementById('step' + i).classList.remove('active');
    }

    // Populate thumbnails
    for (var i = 0; i < 4; i++) {
        var thumbImg = document.getElementById('thumb' + i);
        if (captureURLs[i]) {
            thumbImg.src = captureURLs[i];
        }
    }

    document.getElementById('uploadSection').classList.add('active');
    document.getElementById('progressBar').style.display = 'none';
    stopCamera();
}

// ---- Upload ----
window.submitForGrading = function() {
    // Verify all 4 captures exist
    for (var i = 0; i < 4; i++) {
        if (!captures[i]) {
            alert('Missing ' + STEPS[i] + ' photo. Please capture all 4 angles.');
            return;
        }
    }

    document.getElementById('uploadSection').classList.remove('active');
    document.getElementById('spinnerSection').classList.add('active');

    var fd = new FormData();
    fd.append('front', captures[0], 'front.jpg');
    fd.append('back', captures[1], 'back.jpg');
    fd.append('oblique', captures[2], 'oblique.jpg');
    fd.append('edge', captures[3], 'edge.jpg');

    fetch('/condition/assess', { method: 'POST', body: fd })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            document.getElementById('spinnerSection').classList.remove('active');
            if (data.error) {
                alert('Error: ' + data.error);
                document.getElementById('uploadSection').classList.add('active');
                return;
            }
            resultData = data;
            showResults(data);
        })
        .catch(function(err) {
            document.getElementById('spinnerSection').classList.remove('active');
            alert('Upload failed: ' + err);
            document.getElementById('uploadSection').classList.add('active');
        });
};

// ---- Results Display ----
function showResults(data) {
    var section = document.getElementById('resultsSection');
    section.classList.add('active');

    // Card identity
    var refImg = document.getElementById('resultRefImage');
    if (data.image_url || data.local_image_url) {
        refImg.src = data.local_image_url || data.image_url;
        refImg.style.display = 'block';
    } else {
        refImg.style.display = 'none';
    }
    document.getElementById('resultCardName').textContent = data.card_name || 'Unknown Card';
    document.getElementById('resultSetName').textContent = data.set_name || '';
    document.getElementById('resultCardId').textContent = data.card_id || '';

    // Overall grade
    var overall = data.overall_grade || 0;
    var condition = data.condition || 'NM';
    document.getElementById('gradeNumber').textContent = overall.toFixed(1);
    document.getElementById('overallLabel').textContent = condition;

    var conditionDescriptions = {
        'NM': 'Near Mint',
        'LP': 'Lightly Played',
        'MP': 'Moderately Played',
        'HP': 'Heavily Played',
        'DMG': 'Damaged'
    };
    document.getElementById('overallSublabel').textContent =
        conditionDescriptions[condition] || condition;

    // Grade circle color
    var circle = document.getElementById('gradeCircle');
    circle.className = 'grade-circle';
    if (condition === 'NM') circle.classList.add('grade-nm');
    else if (condition === 'LP') circle.classList.add('grade-lp');
    else if (condition === 'MP') circle.classList.add('grade-mp');
    else if (condition === 'HP') circle.classList.add('grade-hp');
    else circle.classList.add('grade-dmg');

    // Sub-grades
    var grades = data.sub_grades || {};
    setSubgrade('Centering', grades.centering);
    setSubgrade('Surface', grades.surface);
    setSubgrade('Edges', grades.edges);
    setSubgrade('Corners', grades.corners);

    // Defects
    var defects = data.defects || [];
    var defectsContent = document.getElementById('defectsContent');
    if (defects.length === 0) {
        defectsContent.innerHTML = '<div class="no-defects">No significant defects detected</div>';
    } else {
        var html = '<div class="defect-thumbs">';
        for (var i = 0; i < defects.length; i++) {
            var d = defects[i];
            html += '<div class="defect-thumb">';
            if (d.image_url) {
                html += '<img src="' + d.image_url + '" alt="' + (d.label || '') + '">';
            } else {
                html += '<div style="width:100%;aspect-ratio:1;background:#0a1628;display:flex;align-items:center;justify-content:center;color:#555;font-size:24px;">';
                html += defectIcon(d.type);
                html += '</div>';
            }
            html += '<div class="defect-label">' + (d.label || d.type || 'Defect') + '</div>';
            html += '</div>';
        }
        html += '</div>';
        defectsContent.innerHTML = html;
    }

    // Pricing
    var nmPrice = data.nm_price;
    var assessedPrice = data.assessed_price;
    document.getElementById('priceNM').textContent = nmPrice ? '$' + nmPrice.toFixed(2) : '--';
    document.getElementById('priceAssessed').textContent = assessedPrice ? '$' + assessedPrice.toFixed(2) : '--';

    var diffEl = document.getElementById('priceDiff');
    if (nmPrice && assessedPrice) {
        var diff = assessedPrice - nmPrice;
        var pct = ((diff / nmPrice) * 100).toFixed(0);
        var sign = diff >= 0 ? '+' : '';
        diffEl.textContent = sign + '$' + diff.toFixed(2) + ' (' + sign + pct + '%)';
        diffEl.className = 'pr-value ' + (diff >= 0 ? 'diff-up' : 'diff-down');
    } else {
        diffEl.textContent = '--';
        diffEl.className = 'pr-value';
    }

    // Store card_id for inventory
    document.getElementById('resultAddBtn').dataset.cardId = data.card_id || '';
    document.getElementById('resultAddBtn').dataset.condition = condition;
}

function setSubgrade(name, score) {
    var id = name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
    // Score might come as e.g. "centering" -> need lowercase match
    if (score === undefined || score === null) score = 0;

    var scoreEl = document.getElementById('sg' + id);
    var barEl = document.getElementById('bar' + id);
    if (!scoreEl || !barEl) return;

    scoreEl.textContent = score.toFixed(1);
    var pct = (score / 10) * 100;
    barEl.style.width = pct + '%';

    // Color based on score
    if (score >= 8) barEl.style.background = '#4ecca3';
    else if (score >= 6) barEl.style.background = '#7bc47f';
    else if (score >= 4) barEl.style.background = '#f0a500';
    else barEl.style.background = '#e94560';
}

function defectIcon(type) {
    var icons = {
        'scratch': '/',
        'whitening': 'W',
        'dent': 'D',
        'crease': '~',
        'miscut': 'M',
        'print_line': '|'
    };
    return icons[type] || '!';
}

// ---- Inventory ----
window.addGradedToInventory = function() {
    var btn = document.getElementById('resultAddBtn');
    var msg = document.getElementById('resultInventoryMsg');
    var cardId = btn.dataset.cardId;
    if (!cardId) return;

    btn.disabled = true;
    btn.textContent = 'Adding...';

    fetch('/inventory/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            card_id: cardId,
            quantity: 1,
            condition: btn.dataset.condition || 'NM'
        })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false;
        btn.textContent = 'Add to Inventory';
        msg.style.display = 'block';
        if (data.error) {
            msg.style.color = '#e94560';
            msg.textContent = data.error;
        } else {
            msg.style.color = '#4ecca3';
            msg.textContent = 'Added! Total in inventory: ' + data.quantity;
        }
    })
    .catch(function(e) {
        btn.disabled = false;
        btn.textContent = 'Add to Inventory';
        msg.style.display = 'block';
        msg.style.color = '#e94560';
        msg.textContent = 'Error: ' + e;
    });
};

// ---- Start Over ----
window.startOver = function() {
    stopCamera();

    // Reset captures
    for (var i = 0; i < 4; i++) {
        if (captureURLs[i]) URL.revokeObjectURL(captureURLs[i]);
        captures[i] = null;
        captureURLs[i] = null;

        // Reset viewfinder
        var preview = document.getElementById('preview' + i);
        preview.style.display = 'none';
        preview.src = '';
        var video = document.getElementById('video' + i);
        if (video) video.style.display = 'block';
        document.getElementById('captureBtn' + i).style.display = '';
        document.getElementById('captureBtn' + i).disabled = false;
        document.getElementById('retakeBtn' + i).style.display = 'none';

        // Reset progress dots
        var dot = document.getElementById('dot' + i);
        dot.innerHTML = '' + (i + 1);
        dot.className = 'progress-dot';
        var stepEl = dot.closest('.progress-step');
        stepEl.className = 'progress-step';
    }
    // Reset connectors
    for (var i = 0; i < 3; i++) {
        document.getElementById('conn' + i).classList.remove('done');
    }

    // Hide sections
    document.getElementById('uploadSection').classList.remove('active');
    document.getElementById('spinnerSection').classList.remove('active');
    document.getElementById('resultsSection').classList.remove('active');
    document.getElementById('resultInventoryMsg').style.display = 'none';

    // Show progress bar and first step
    document.getElementById('progressBar').style.display = 'flex';
    currentStep = 0;
    resultData = null;
    goToStep(0);
};

// ---- Gyroscope Feedback (Step 2 - Oblique) ----
function initGyroscope() {
    var indicator = document.getElementById('gyroIndicator');
    var angleSpan = document.getElementById('gyroAngle');

    if (!window.DeviceOrientationEvent) return;

    // Request permission on iOS 13+
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        // Will be triggered by first user interaction
        document.addEventListener('click', function requestGyro() {
            DeviceOrientationEvent.requestPermission().then(function(state) {
                if (state === 'granted') listenGyro();
            }).catch(function() {});
            document.removeEventListener('click', requestGyro);
        }, { once: true });
    } else {
        listenGyro();
    }

    function listenGyro() {
        window.addEventListener('deviceorientation', function(e) {
            // Only show during oblique step
            if (currentStep !== 2) {
                indicator.style.display = 'none';
                return;
            }
            indicator.style.display = 'block';

            var beta = e.beta; // front-back tilt (-180 to 180)
            if (beta === null) {
                indicator.style.display = 'none';
                return;
            }

            var absBeta = Math.abs(beta);
            angleSpan.textContent = absBeta.toFixed(0) + ' deg';

            // Ideal oblique angle: 20-45 degrees
            if (absBeta >= 20 && absBeta <= 45) {
                indicator.className = 'gyro-indicator good';
                angleSpan.textContent += ' - Good angle!';
            } else if (absBeta < 20) {
                indicator.className = 'gyro-indicator adjust';
                angleSpan.textContent += ' - Tilt more';
            } else {
                indicator.className = 'gyro-indicator adjust';
                angleSpan.textContent += ' - Too steep';
            }
        });
    }
}

// ---- Gallery file input handlers ----
for (var i = 0; i < 4; i++) {
    (function(idx) {
        document.getElementById('galleryInput' + idx).addEventListener('change', function() {
            if (this.files && this.files[0]) {
                handleGalleryFile(idx, this.files[0]);
            }
            this.value = '';
        });
    })(i);
}

// ---- Init ----
initGyroscope();
startCamera(0);

})();
</script>
</body>
</html>
"""
