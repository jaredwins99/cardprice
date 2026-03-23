/**
 * ScanState — State machine for slide-scan card capture with exit detection.
 *
 * Prevents double-captures by requiring a card to fully EXIT the frame
 * (pass through the binder gutter) before the next card can be captured.
 *
 * State machine:
 *   IDLE → start() → DETECTING
 *   DETECTING → quality met → CAPTURED
 *   CAPTURED → gutter detected for N frames → IN_GUTTER
 *   IN_GUTTER → card-like frame returns → DETECTING
 *   DETECTING → all cards captured → DONE
 *
 * Uses two signals for gutter detection:
 *   1. Center-strip brightness (luminance) — gutters are darker than cards
 *   2. Center-strip saturation — binder material is saturated (blue/orange)
 *      or desaturated+dark (black), distinct from card artwork
 *
 * Also integrates sharpness checking: a capture only fires when the frame
 * is both centered (brightness peak) and sharp (not motion-blurred).
 *
 * Usage:
 *   const scan = new ScanState({ targetCount: 3, binderColor: 'blue' });
 *   scan.start();
 *
 *   // In rAF loop:
 *   const event = scan.processFrame(videoElement, captureCanvas);
 *   if (event && event.type === 'capture') {
 *       // card image is on captureCanvas
 *       sendToServer(captureCanvas.toDataURL());
 *   }
 *   if (scan.state === 'done') {
 *       showComplete();
 *   }
 *
 *   // Debug overlay:
 *   const diag = scan.getDiagnostics();
 */

class ScanState {
    /**
     * @param {Object} opts
     * @param {number}  opts.targetCount        Cards to capture before done (default 3)
     * @param {string}  opts.binderColor        'auto'|'blue'|'orange'|'black'|'white' (default 'auto')
     * @param {number}  opts.gutterFramesNeeded Consecutive gutter frames to confirm exit (default 5)
     * @param {number}  opts.cardFramesNeeded   Consecutive card frames to confirm entry (default 3)
     * @param {number}  opts.stripWidthPct      Center strip width as fraction of frame (default 0.20)
     * @param {number}  opts.brightnessCardMin  Min brightness to consider "card-like" (default 100)
     * @param {number}  opts.saturationGutterMin Min saturation for colored gutter detection (default 60)
     * @param {number}  opts.sharpnessThreshold Laplacian variance threshold for sharp (default 80)
     * @param {number}  opts.peakHoldFrames     Frames past brightness peak before capture (default 2)
     * @param {number}  opts.cooldownFrames     Minimum frames after capture before checking gutter (default 5)
     * @param {number}  opts.adaptiveCalFrames  Frames for adaptive threshold calibration (default 20)
     */
    constructor(opts = {}) {
        this.targetCount        = opts.targetCount        ?? 3;
        this.binderColor        = opts.binderColor        ?? 'auto';
        this.gutterFramesNeeded = opts.gutterFramesNeeded ?? 5;
        this.cardFramesNeeded   = opts.cardFramesNeeded   ?? 3;
        this.stripWidthPct      = opts.stripWidthPct      ?? 0.20;
        this.brightnessCardMin  = opts.brightnessCardMin  ?? 100;
        this.saturationGutterMin = opts.saturationGutterMin ?? 60;
        this.sharpnessThreshold = opts.sharpnessThreshold ?? 80;
        this.peakHoldFrames     = opts.peakHoldFrames     ?? 2;
        this.cooldownFrames     = opts.cooldownFrames     ?? 5;
        this.adaptiveCalFrames  = opts.adaptiveCalFrames  ?? 20;

        // State
        this.state        = 'idle'; // idle | calibrating | detecting | captured | in_gutter | done
        this.captureCount = 0;
        this.captures     = [];     // { cardIndex, frame, brightness, sharpness }

        // Frame counter
        this._frameCount = 0;

        // Gutter/card transition counters
        this._gutterFrames    = 0;  // consecutive gutter-like frames
        this._cardFrames      = 0;  // consecutive card-like frames
        this._cooldownCounter = 0;  // frames since last capture

        // Peak tracking (for capturing at the card center, not the edge)
        this._peakBrightness  = -1;
        this._peakFrame       = -1;
        this._risingFrames    = 0;

        // Adaptive thresholds (computed during calibration)
        this._calBrightness   = [];
        this._calSaturation   = [];
        this._calBlueRatio    = [];
        this._brightnessMean  = null;
        this._brightnessStd   = null;
        this._gutterBrightThresh = null; // below this = gutter
        this._cardBrightThresh   = null; // above this = card

        // Binder color matcher (from GutterDetector-style HSV matching)
        this._gutterMatcher = null;
        this._initGutterMatcher();

        // Signal history for diagnostics
        this._brightnessHistory = [];
        this._saturationHistory = [];
        this._gutterFracHistory = [];
        this._sharpnessHistory  = [];

        // Reusable canvases
        this._stripCanvas = null;
        this._stripCtx    = null;

        // Callbacks
        this.onStateChange  = null; // function(newState, oldState)
        this.onDiagnostics  = null; // function(diagObject)
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /** Begin scanning. Resets all state and enters calibration. */
    start() {
        this._setState('calibrating');
        this.captureCount     = 0;
        this.captures         = [];
        this._frameCount      = 0;
        this._gutterFrames    = 0;
        this._cardFrames      = 0;
        this._cooldownCounter = 0;
        this._peakBrightness  = -1;
        this._peakFrame       = -1;
        this._risingFrames    = 0;
        this._calBrightness   = [];
        this._calSaturation   = [];
        this._calBlueRatio    = [];
        this._brightnessMean  = null;
        this._brightnessStd   = null;
        this._gutterBrightThresh = null;
        this._cardBrightThresh   = null;
        this._brightnessHistory = [];
        this._saturationHistory = [];
        this._gutterFracHistory = [];
        this._sharpnessHistory  = [];
    }

    /** Reset to idle. */
    reset() {
        this._setState('idle');
        this.captureCount     = 0;
        this.captures         = [];
        this._frameCount      = 0;
        this._gutterFrames    = 0;
        this._cardFrames      = 0;
        this._cooldownCounter = 0;
        this._peakBrightness  = -1;
        this._peakFrame       = -1;
        this._risingFrames    = 0;
        this._calBrightness   = [];
        this._calSaturation   = [];
        this._calBlueRatio    = [];
        this._brightnessMean  = null;
        this._brightnessStd   = null;
        this._gutterBrightThresh = null;
        this._cardBrightThresh   = null;
        this._brightnessHistory = [];
        this._saturationHistory = [];
        this._gutterFracHistory = [];
        this._sharpnessHistory  = [];
    }

    /**
     * Process one video frame. Call from requestAnimationFrame.
     *
     * @param {HTMLVideoElement} video         The live camera feed
     * @param {HTMLCanvasElement} captureCanvas Canvas to draw the captured frame onto
     * @returns {Object|null}  { type:'capture', cardIndex, brightness, sharpness, frame }
     *                         or { type:'gutter_entered' } or { type:'card_entered' }
     *                         or null if no event this frame.
     */
    processFrame(video, captureCanvas) {
        if (this.state === 'idle' || this.state === 'done') return null;

        this._frameCount++;

        // --- Sample center strip ---
        const strip = this._sampleCenterStrip(video);
        const { brightness, saturation, hue, gutterFraction } = strip;

        this._brightnessHistory.push(brightness);
        this._saturationHistory.push(saturation);
        this._gutterFracHistory.push(gutterFraction);

        // Trim histories to last 120 frames
        if (this._brightnessHistory.length > 120) {
            this._brightnessHistory.shift();
            this._saturationHistory.shift();
            this._gutterFracHistory.shift();
        }

        // --- Sharpness (lightweight: laplacian variance on center strip) ---
        const sharpness = this._estimateSharpness(video);
        this._sharpnessHistory.push(sharpness);
        if (this._sharpnessHistory.length > 120) this._sharpnessHistory.shift();

        // --- Calibration phase ---
        if (this.state === 'calibrating') {
            this._calBrightness.push(brightness);
            this._calSaturation.push(saturation);
            this._calBlueRatio.push(gutterFraction);

            if (this._calBrightness.length >= this.adaptiveCalFrames) {
                this._computeAdaptiveThresholds();
                this._setState('detecting');
            }

            this._emitDiag(strip, sharpness);
            return null;
        }

        // --- State machine ---
        let event = null;

        switch (this.state) {
            case 'detecting':
                event = this._handleDetecting(brightness, gutterFraction, sharpness, video, captureCanvas);
                break;

            case 'captured':
                event = this._handleCaptured(brightness, gutterFraction);
                break;

            case 'in_gutter':
                event = this._handleInGutter(brightness, gutterFraction);
                break;
        }

        this._emitDiag(strip, sharpness);
        return event;
    }

    /**
     * Get diagnostics snapshot for debug overlay.
     */
    getDiagnostics() {
        const bLen = this._brightnessHistory.length;
        return {
            state:              this.state,
            frameCount:         this._frameCount,
            captureCount:       this.captureCount,
            targetCount:        this.targetCount,
            gutterFrames:       this._gutterFrames,
            cardFrames:         this._cardFrames,
            cooldownCounter:    this._cooldownCounter,
            peakBrightness:     this._peakBrightness,
            gutterBrightThresh: this._gutterBrightThresh,
            cardBrightThresh:   this._cardBrightThresh,
            lastBrightness:     bLen > 0 ? this._brightnessHistory[bLen - 1] : null,
            lastGutterFrac:     this._gutterFracHistory.length > 0
                ? this._gutterFracHistory[this._gutterFracHistory.length - 1] : null,
            lastSharpness:      this._sharpnessHistory.length > 0
                ? this._sharpnessHistory[this._sharpnessHistory.length - 1] : null,
            recentBrightness:   this._brightnessHistory.slice(-60),
            recentGutterFrac:   this._gutterFracHistory.slice(-60),
            recentSharpness:    this._sharpnessHistory.slice(-60),
            binderColor:        this.binderColor,
        };
    }

    // -----------------------------------------------------------------------
    // State handlers
    // -----------------------------------------------------------------------

    /**
     * DETECTING state: looking for a card to capture.
     * Track brightness to find the peak (card center), then capture.
     */
    _handleDetecting(brightness, gutterFraction, sharpness, video, captureCanvas) {
        const isCardLike = this._isCardLike(brightness, gutterFraction);

        if (!isCardLike) {
            // Not seeing a card yet — reset peak tracking
            this._peakBrightness = -1;
            this._peakFrame      = -1;
            this._risingFrames   = 0;
            return null;
        }

        // Card-like frame. Track the brightness peak.
        if (brightness >= this._peakBrightness) {
            // Still rising or at peak
            this._peakBrightness = brightness;
            this._peakFrame      = this._frameCount;
            this._risingFrames++;
        }

        // Check capture conditions:
        // 1. Brightness has started dropping (past the peak)
        // 2. We've been on a card long enough (not a noise spike)
        // 3. Frame is sharp enough for OCR/matching
        const framesPastPeak = this._frameCount - this._peakFrame;
        const isSharp = sharpness >= this.sharpnessThreshold;

        if (framesPastPeak >= this.peakHoldFrames
            && this._risingFrames >= 2
            && isSharp) {
            // Capture!
            return this._captureFrame(video, captureCanvas, brightness, sharpness);
        }

        // If we've gone way past the peak but never had a sharp frame,
        // still capture — a slightly blurry card is better than missing it
        if (framesPastPeak >= this.peakHoldFrames * 3 && this._risingFrames >= 3) {
            return this._captureFrame(video, captureCanvas, brightness, sharpness);
        }

        return null;
    }

    /**
     * CAPTURED state: just captured a card, waiting for it to exit.
     * First wait out the cooldown, then look for gutter.
     */
    _handleCaptured(brightness, gutterFraction) {
        this._cooldownCounter++;

        // Mandatory cooldown: ignore everything for a few frames after capture
        // to avoid re-triggering on the same card
        if (this._cooldownCounter < this.cooldownFrames) {
            return null;
        }

        // Now look for gutter
        const isGutter = this._isGutterLike(brightness, gutterFraction);

        if (isGutter) {
            this._gutterFrames++;
        } else {
            // Reset gutter counter — must be consecutive
            this._gutterFrames = 0;
        }

        // Confirmed gutter: card has exited the frame
        if (this._gutterFrames >= this.gutterFramesNeeded) {
            this._setState('in_gutter');
            this._cardFrames = 0;
            return { type: 'gutter_entered', frame: this._frameCount };
        }

        return null;
    }

    /**
     * IN_GUTTER state: card has exited, waiting for the NEXT card to enter.
     * This prevents capturing during the gutter-to-card transition edge.
     */
    _handleInGutter(brightness, gutterFraction) {
        const isCardLike = this._isCardLike(brightness, gutterFraction);

        if (isCardLike) {
            this._cardFrames++;
        } else {
            // Reset — must be consecutive card frames
            this._cardFrames = 0;
        }

        // New card has entered
        if (this._cardFrames >= this.cardFramesNeeded) {
            this._setState('detecting');
            this._peakBrightness = -1;
            this._peakFrame      = -1;
            this._risingFrames   = 0;
            return { type: 'card_entered', frame: this._frameCount };
        }

        return null;
    }

    // -----------------------------------------------------------------------
    // Classification: is this frame gutter or card?
    // -----------------------------------------------------------------------

    /**
     * Determine if the current center strip looks like a card.
     * Uses adaptive thresholds when available, falls back to absolute.
     */
    _isCardLike(brightness, gutterFraction) {
        // Adaptive: brightness above card threshold AND low gutter color fraction
        if (this._cardBrightThresh !== null) {
            return brightness >= this._cardBrightThresh && gutterFraction < 0.20;
        }
        // Absolute fallback
        return brightness >= this.brightnessCardMin && gutterFraction < 0.20;
    }

    /**
     * Determine if the current center strip looks like a gutter.
     * Either low brightness (any binder) or high gutter color match (colored binder).
     */
    _isGutterLike(brightness, gutterFraction) {
        // High gutter color fraction is a strong signal regardless of brightness
        if (gutterFraction >= 0.35) return true;

        // Adaptive: brightness below gutter threshold
        if (this._gutterBrightThresh !== null) {
            return brightness < this._gutterBrightThresh;
        }

        // Absolute fallback: dark = gutter
        return brightness < this.brightnessCardMin * 0.7;
    }

    // -----------------------------------------------------------------------
    // Adaptive threshold calibration
    // -----------------------------------------------------------------------

    /**
     * Compute adaptive thresholds from the calibration window.
     * The user should be positioned with a mix of card and gutter visible
     * (or sliding slowly) during calibration.
     */
    _computeAdaptiveThresholds() {
        const bMean = _arrMean(this._calBrightness);
        const bStd  = _arrStd(this._calBrightness, bMean);

        this._brightnessMean = bMean;
        this._brightnessStd  = bStd;

        // Card threshold: mean + 0.3 std (cards are brighter)
        // Gutter threshold: mean - 0.3 std (gutters are darker)
        // If std is very low (uniform scene), use absolute thresholds
        if (bStd > 5) {
            this._cardBrightThresh   = bMean + 0.3 * bStd;
            this._gutterBrightThresh = bMean - 0.3 * bStd;
        } else {
            this._cardBrightThresh   = this.brightnessCardMin;
            this._gutterBrightThresh = this.brightnessCardMin * 0.7;
        }
    }

    // -----------------------------------------------------------------------
    // Center strip sampling
    // -----------------------------------------------------------------------

    /**
     * Sample the center vertical strip of the video frame.
     * Returns brightness (0-255), saturation (0-255), dominant hue (0-359),
     * and gutterFraction (0-1).
     */
    _sampleCenterStrip(video) {
        const vw = video.videoWidth  || 640;
        const vh = video.videoHeight || 480;
        if (vw === 0 || vh === 0) {
            return { brightness: 128, saturation: 0, hue: 0, gutterFraction: 0 };
        }

        // Center strip bounds in source coordinates
        const stripW = Math.max(4, Math.round(vw * this.stripWidthPct));
        const stripX = Math.round((vw - stripW) / 2);

        // Downsample to small canvas
        const targetH = 60;
        const targetW = Math.max(2, Math.round(stripW * (targetH / vh)));

        if (!this._stripCanvas) {
            this._stripCanvas = document.createElement('canvas');
            this._stripCtx = this._stripCanvas.getContext('2d', { willReadFrequently: true });
        }
        this._stripCanvas.width  = targetW;
        this._stripCanvas.height = targetH;

        this._stripCtx.drawImage(
            video,
            stripX, 0, stripW, vh,
            0, 0, targetW, targetH
        );

        const imgData = this._stripCtx.getImageData(0, 0, targetW, targetH);
        const px = imgData.data;
        const numPixels = targetW * targetH;

        let sumBright = 0;
        let sumSat    = 0;
        let sumHue    = 0;
        let gutterCount = 0;

        for (let i = 0; i < px.length; i += 4) {
            const r = px[i];
            const g = px[i + 1];
            const b = px[i + 2];

            // Brightness (luminance)
            const bright = 0.299 * r + 0.587 * g + 0.114 * b;
            sumBright += bright;

            // HSV for saturation and gutter matching
            const max = Math.max(r, g, b);
            const min = Math.min(r, g, b);
            const delta = max - min;

            const sat = max === 0 ? 0 : (delta / max) * 255;
            sumSat += sat;

            let hue = 0;
            if (delta > 0) {
                if (max === r)      hue = 60 * (((g - b) / delta) % 6);
                else if (max === g) hue = 60 * (((b - r) / delta) + 2);
                else                hue = 60 * (((r - g) / delta) + 4);
                if (hue < 0) hue += 360;
            }
            sumHue += hue;

            // Check if this pixel matches binder gutter color
            if (this._gutterMatcher && this._gutterMatcher(Math.round(hue), Math.round(sat), max)) {
                gutterCount++;
            }
        }

        return {
            brightness:    sumBright / numPixels,
            saturation:    sumSat / numPixels,
            hue:           sumHue / numPixels,
            gutterFraction: gutterCount / numPixels,
        };
    }

    // -----------------------------------------------------------------------
    // Sharpness estimation (lightweight)
    // -----------------------------------------------------------------------

    /**
     * Quick sharpness estimate via Laplacian variance on the center strip.
     * Reuses the strip canvas from the last _sampleCenterStrip call.
     */
    _estimateSharpness(video) {
        if (!this._stripCanvas || !this._stripCtx) return 0;

        const w = this._stripCanvas.width;
        const h = this._stripCanvas.height;
        if (w < 3 || h < 3) return 0;

        const imgData = this._stripCtx.getImageData(0, 0, w, h);
        const px = imgData.data;

        // Convert to grayscale
        const gray = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < px.length; i += 4, j++) {
            gray[j] = (px[i] * 77 + px[i + 1] * 150 + px[i + 2] * 29) >> 8;
        }

        // Laplacian variance
        let sum = 0;
        let sumSq = 0;
        let n = 0;

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                const lap = -4 * gray[idx]
                           + gray[idx - 1] + gray[idx + 1]
                           + gray[idx - w] + gray[idx + w];
                sum   += lap;
                sumSq += lap * lap;
                n++;
            }
        }

        if (n === 0) return 0;
        const mean = sum / n;
        return (sumSq / n) - (mean * mean);
    }

    // -----------------------------------------------------------------------
    // Capture
    // -----------------------------------------------------------------------

    _captureFrame(video, captureCanvas, brightness, sharpness) {
        if (video && captureCanvas) {
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            captureCanvas.width  = vw;
            captureCanvas.height = vh;
            const ctx = captureCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0, vw, vh);
        }

        const cardIndex = this.captureCount;
        const event = {
            type:       'capture',
            cardIndex:  cardIndex,
            brightness: Math.round(brightness * 10) / 10,
            sharpness:  Math.round(sharpness * 10) / 10,
            frame:      this._frameCount,
        };

        this.captures.push({
            cardIndex:  cardIndex,
            frame:      this._frameCount,
            brightness: event.brightness,
            sharpness:  event.sharpness,
        });

        this.captureCount++;

        if (this.captureCount >= this.targetCount) {
            this._setState('done');
        } else {
            this._setState('captured');
            this._cooldownCounter = 0;
            this._gutterFrames    = 0;
        }

        return event;
    }

    // -----------------------------------------------------------------------
    // Gutter color matching
    // -----------------------------------------------------------------------

    _initGutterMatcher() {
        const color = this.binderColor;

        if (color === 'auto') {
            // Auto: accept any strong binder color OR very dark
            this._gutterMatcher = (h, s, v) => {
                // Black binder
                if (v < 50) return true;
                // Blue binder
                if (h >= 190 && h <= 250 && s > 50 && v > 50) return true;
                // Orange binder
                if (h >= 15 && h <= 45 && s > 80 && v > 80) return true;
                // White binder (low saturation, high value — but card art can also be white)
                // Skip white for auto since it's ambiguous
                return false;
            };
        } else if (color === 'blue') {
            this._gutterMatcher = (h, s, v) => h >= 190 && h <= 250 && s > 50 && v > 50;
        } else if (color === 'orange') {
            this._gutterMatcher = (h, s, v) => h >= 15 && h <= 45 && s > 80 && v > 80;
        } else if (color === 'black') {
            this._gutterMatcher = (h, s, v) => v < 50;
        } else if (color === 'white') {
            this._gutterMatcher = (h, s, v) => s < 25 && v > 200;
        } else {
            // Unknown color — match nothing (rely on brightness alone)
            this._gutterMatcher = () => false;
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    _setState(newState) {
        const oldState = this.state;
        if (oldState === newState) return;
        this.state = newState;
        if (this.onStateChange) {
            this.onStateChange(newState, oldState);
        }
    }

    _emitDiag(strip, sharpness) {
        if (this.onDiagnostics) {
            this.onDiagnostics({
                state:            this.state,
                frame:            this._frameCount,
                brightness:       Math.round(strip.brightness * 10) / 10,
                saturation:       Math.round(strip.saturation * 10) / 10,
                hue:              Math.round(strip.hue),
                gutterFraction:   Math.round(strip.gutterFraction * 1000) / 1000,
                sharpness:        Math.round(sharpness * 10) / 10,
                gutterFrames:     this._gutterFrames,
                cardFrames:       this._cardFrames,
                cooldownCounter:  this._cooldownCounter,
                peakBrightness:   Math.round(this._peakBrightness * 10) / 10,
                captureCount:     this.captureCount,
                targetCount:      this.targetCount,
                gutterBrightThresh: this._gutterBrightThresh,
                cardBrightThresh:   this._cardBrightThresh,
            });
        }
    }
}


// ---------------------------------------------------------------------------
// Utility functions (module-private)
// ---------------------------------------------------------------------------

function _arrMean(arr) {
    if (arr.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    return sum / arr.length;
}

function _arrStd(arr, mean) {
    if (arr.length === 0) return 0;
    let sumSq = 0;
    for (let i = 0; i < arr.length; i++) {
        const d = arr[i] - mean;
        sumSq += d * d;
    }
    return Math.sqrt(sumSq / arr.length);
}


// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------

ScanState.selfTest = function () {
    let passed = 0;
    let failed = 0;

    function assert(condition, name) {
        if (condition) {
            console.log(`  PASS: ${name}`);
            passed++;
        } else {
            console.error(`  FAIL: ${name}`);
            failed++;
        }
    }

    console.log('=== ScanState Self-Test ===');

    // --- Test _arrMean and _arrStd ---
    {
        assert(_arrMean([10, 20, 30]) === 20, 'mean([10,20,30]) = 20');
        assert(Math.abs(_arrStd([10, 20, 30], 20) - 8.165) < 0.01, 'std([10,20,30]) ~ 8.165');
        assert(_arrMean([]) === 0, 'mean([]) = 0');
        assert(_arrStd([], 0) === 0, 'std([]) = 0');
    }

    // --- Test constructor defaults ---
    {
        const s = new ScanState();
        assert(s.targetCount === 3, 'default targetCount = 3');
        assert(s.state === 'idle', 'initial state = idle');
        assert(s.captureCount === 0, 'initial captureCount = 0');
        assert(s.captures.length === 0, 'initial captures empty');
        assert(s.binderColor === 'auto', 'default binderColor = auto');
    }

    // --- Test start/reset ---
    {
        const s = new ScanState();
        s.start();
        assert(s.state === 'calibrating', 'start -> calibrating');
        assert(s.captureCount === 0, 'start resets captureCount');
        s.reset();
        assert(s.state === 'idle', 'reset -> idle');
    }

    // --- Test state change callback ---
    {
        const s = new ScanState();
        const transitions = [];
        s.onStateChange = (newS, oldS) => transitions.push(`${oldS}->${newS}`);
        s.start();
        assert(transitions.length === 1, 'one transition on start');
        assert(transitions[0] === 'idle->calibrating', 'idle->calibrating transition');
        s.reset();
        assert(transitions[1] === 'calibrating->idle', 'calibrating->idle transition');
    }

    // --- Test adaptive threshold computation ---
    {
        const s = new ScanState({ adaptiveCalFrames: 5 });
        s.start();
        // Simulate calibration data: mix of gutter (80) and card (200) brightness
        s._calBrightness = [80, 200, 80, 200, 80];
        s._calSaturation = [150, 30, 150, 30, 150];
        s._calBlueRatio  = [0.6, 0.1, 0.6, 0.1, 0.6];
        s._computeAdaptiveThresholds();

        assert(s._brightnessMean !== null, 'brightness mean computed');
        assert(s._cardBrightThresh > s._brightnessMean,
            `card threshold (${s._cardBrightThresh.toFixed(1)}) > mean (${s._brightnessMean.toFixed(1)})`);
        assert(s._gutterBrightThresh < s._brightnessMean,
            `gutter threshold (${s._gutterBrightThresh.toFixed(1)}) < mean (${s._brightnessMean.toFixed(1)})`);
    }

    // --- Test _isCardLike and _isGutterLike with adaptive thresholds ---
    {
        const s = new ScanState();
        s._cardBrightThresh   = 140;
        s._gutterBrightThresh = 100;

        assert(s._isCardLike(200, 0.05) === true, 'bright + low gutter frac = card');
        assert(s._isCardLike(80, 0.05) === false, 'dark + low gutter frac != card');
        assert(s._isCardLike(200, 0.40) === false, 'bright + high gutter frac != card');

        assert(s._isGutterLike(60, 0.50) === true, 'dark + high gutter frac = gutter');
        assert(s._isGutterLike(200, 0.50) === true, 'bright but high gutter frac = gutter');
        assert(s._isGutterLike(60, 0.10) === true, 'dark + low gutter frac = gutter (brightness alone)');
        assert(s._isGutterLike(200, 0.10) === false, 'bright + low gutter frac != gutter');
    }

    // --- Test gutter color matchers ---
    {
        const sBlue = new ScanState({ binderColor: 'blue' });
        assert(sBlue._gutterMatcher(220, 150, 150) === true, 'blue matcher accepts blue pixel');
        assert(sBlue._gutterMatcher(0, 150, 150) === false, 'blue matcher rejects red pixel');

        const sBlack = new ScanState({ binderColor: 'black' });
        assert(sBlack._gutterMatcher(0, 0, 30) === true, 'black matcher accepts dark pixel');
        assert(sBlack._gutterMatcher(0, 0, 100) === false, 'black matcher rejects bright pixel');

        const sOrange = new ScanState({ binderColor: 'orange' });
        assert(sOrange._gutterMatcher(30, 200, 200) === true, 'orange matcher accepts orange pixel');

        const sAuto = new ScanState({ binderColor: 'auto' });
        assert(sAuto._gutterMatcher(220, 150, 150) === true, 'auto matcher accepts blue');
        assert(sAuto._gutterMatcher(0, 0, 30) === true, 'auto matcher accepts black');
        assert(sAuto._gutterMatcher(30, 200, 200) === true, 'auto matcher accepts orange');
        assert(sAuto._gutterMatcher(0, 0, 200) === false, 'auto matcher rejects bright neutral');
    }

    // --- Test state machine: DETECTING ---
    {
        const s = new ScanState({ targetCount: 1, peakHoldFrames: 1, sharpnessThreshold: 0 });
        s.state = 'detecting';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 60;

        // Simulate rising brightness (card entering)
        let evt;
        s._frameCount = 1;
        evt = s._handleDetecting(120, 0.05, 50, null, null);
        assert(evt === null, 'first card frame -> no capture yet (rising)');
        assert(s._peakBrightness === 120, 'peak tracks brightness');

        s._frameCount = 2;
        evt = s._handleDetecting(150, 0.05, 50, null, null);
        assert(evt === null, 'still rising -> no capture');

        s._frameCount = 3;
        evt = s._handleDetecting(130, 0.05, 50, null, null);
        // Now brightness dropped: 130 < 150 peak, framesPastPeak = 3-2 = 1 >= peakHoldFrames(1)
        // risingFrames = 2 >= 2
        // sharpness 50 >= 0 threshold
        assert(evt !== null && evt.type === 'capture', 'capture fires after peak + hold');
        assert(evt.cardIndex === 0, 'first capture index = 0');
    }

    // --- Test state machine: CAPTURED -> IN_GUTTER ---
    {
        const s = new ScanState({
            cooldownFrames: 2,
            gutterFramesNeeded: 3,
        });
        s.state = 'captured';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 80;
        s._cooldownCounter    = 0;
        s._gutterFrames       = 0;

        // _handleCaptured increments _cooldownCounter internally each call.
        // cooldownFrames=2: first call counter=1 (<2, cooldown), second call counter=2 (>=2, gutter check starts)
        let evt;
        evt = s._handleCaptured(60, 0.50);
        assert(evt === null, 'cooldown frame 1: no event (counter=1 < 2)');

        // Counter=2, past cooldown. Gutter check starts. gutterFrames=1.
        evt = s._handleCaptured(60, 0.50);
        assert(evt === null, 'gutter frame 1: not enough yet');

        // gutterFrames=2
        evt = s._handleCaptured(60, 0.50);
        assert(evt === null, 'gutter frame 2: not enough yet');

        // gutterFrames=3 >= gutterFramesNeeded(3)
        evt = s._handleCaptured(60, 0.50);
        assert(evt !== null && evt.type === 'gutter_entered', 'gutter frame 3: transition to in_gutter');
        assert(s.state === 'in_gutter', 'state is now in_gutter');
    }

    // --- Test state machine: IN_GUTTER -> DETECTING ---
    {
        const s = new ScanState({ cardFramesNeeded: 2 });
        s.state = 'in_gutter';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 80;
        s._cardFrames = 0;

        let evt;
        evt = s._handleInGutter(150, 0.05);
        assert(evt === null, 'card frame 1: not enough yet');

        evt = s._handleInGutter(150, 0.05);
        assert(evt !== null && evt.type === 'card_entered', 'card frame 2: transition to detecting');
        assert(s.state === 'detecting', 'state is now detecting');
    }

    // --- Test IN_GUTTER resets on non-card frame ---
    {
        const s = new ScanState({ cardFramesNeeded: 3 });
        s.state = 'in_gutter';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 80;
        s._cardFrames = 0;

        s._handleInGutter(150, 0.05); // card frame 1
        s._handleInGutter(150, 0.05); // card frame 2
        assert(s._cardFrames === 2, 'two consecutive card frames');

        s._handleInGutter(60, 0.50);  // gutter frame — resets counter
        assert(s._cardFrames === 0, 'gutter frame resets card counter');
        assert(s.state === 'in_gutter', 'still in gutter after reset');
    }

    // --- Test CAPTURED: gutter counter resets on non-gutter frame ---
    {
        const s = new ScanState({ cooldownFrames: 0, gutterFramesNeeded: 3 });
        s.state = 'captured';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 80;
        s._cooldownCounter = 10; // past cooldown

        s._handleCaptured(60, 0.50); // gutter
        s._handleCaptured(60, 0.50); // gutter (2)
        assert(s._gutterFrames === 2, 'two consecutive gutter frames');

        s._handleCaptured(180, 0.05); // card interruption
        assert(s._gutterFrames === 0, 'card frame resets gutter counter');
    }

    // --- Test full cycle: DETECTING -> CAPTURED -> IN_GUTTER -> DETECTING -> DONE ---
    {
        const s = new ScanState({
            targetCount: 2,
            peakHoldFrames: 1,
            cooldownFrames: 1,
            gutterFramesNeeded: 2,
            cardFramesNeeded: 2,
            sharpnessThreshold: 0,
        });
        s.state = 'detecting';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 80;

        // Card 1: rise -> peak -> drop -> capture
        s._frameCount = 1;
        s._handleDetecting(120, 0.05, 10, null, null);
        s._frameCount = 2;
        s._handleDetecting(180, 0.05, 10, null, null);
        s._frameCount = 3;
        const cap1 = s._handleDetecting(160, 0.05, 10, null, null);
        assert(cap1 !== null && cap1.type === 'capture', 'card 1 captured');
        assert(s.state === 'captured', 'state -> captured after card 1');

        // Cooldown
        s._cooldownCounter++;

        // Gutter (card exit)
        s._handleCaptured(60, 0.50);
        s._handleCaptured(60, 0.50);
        assert(s.state === 'in_gutter', 'state -> in_gutter after card 1 exits');

        // New card enters
        s._handleInGutter(170, 0.05);
        s._handleInGutter(170, 0.05);
        assert(s.state === 'detecting', 'state -> detecting for card 2');

        // Card 2: rise -> peak -> drop -> capture
        s._frameCount = 10;
        s._handleDetecting(140, 0.05, 10, null, null);
        s._frameCount = 11;
        s._handleDetecting(200, 0.05, 10, null, null);
        s._frameCount = 12;
        const cap2 = s._handleDetecting(180, 0.05, 10, null, null);
        assert(cap2 !== null && cap2.type === 'capture', 'card 2 captured');
        assert(s.state === 'done', 'state -> done after all cards captured');
        assert(s.captureCount === 2, 'captureCount = 2');
    }

    // --- Test getDiagnostics ---
    {
        const s = new ScanState();
        s.start();
        s._brightnessHistory = [100, 150, 200];
        s._gutterFracHistory = [0.1, 0.5, 0.2];
        s._sharpnessHistory  = [50, 80, 120];
        s._frameCount = 3;

        const diag = s.getDiagnostics();
        assert(diag.state === 'calibrating', 'diagnostics includes state');
        assert(diag.frameCount === 3, 'diagnostics includes frameCount');
        assert(diag.recentBrightness.length === 3, 'diagnostics includes recent brightness');
        assert(diag.lastBrightness === 200, 'diagnostics includes last brightness');
    }

    // --- Test DETECTING doesn't capture when not sharp (and waits) ---
    {
        const s = new ScanState({
            targetCount: 1,
            peakHoldFrames: 1,
            sharpnessThreshold: 100, // high threshold
        });
        s.state = 'detecting';
        s._cardBrightThresh   = 100;
        s._gutterBrightThresh = 80;

        // Card enters and peaks, but sharpness too low
        s._frameCount = 1;
        s._handleDetecting(150, 0.05, 20, null, null); // rising
        s._frameCount = 2;
        s._handleDetecting(200, 0.05, 20, null, null); // peak
        s._frameCount = 3;
        const evt = s._handleDetecting(180, 0.05, 20, null, null); // dropping but not sharp
        assert(evt === null, 'no capture when frame is not sharp enough');

        // Much later: force capture even if blurry (fallback)
        s._frameCount = 8;
        const evt2 = s._handleDetecting(170, 0.05, 20, null, null);
        // framesPastPeak = 8-2 = 6 >= peakHoldFrames*3 = 3, risingFrames = 2+1 = 3 >= 3
        // Wait, risingFrames was set at frames 1 and 2, then frame 3 was dropping
        // risingFrames stays at 2 since frame 3 brightness < peak
        // Actually at frame 3, brightness 180 < 200 so peak stays at 200, risingFrames stays 2
        // But frame 8 brightness 170 < 200 so still dropping, risingFrames still 2
        // peakHoldFrames*3 = 3, need risingFrames >= 3 for fallback, but it's 2
        // So no fallback either
        // Let's add another rising frame first
        s._peakBrightness = -1;
        s._risingFrames = 0;
        s._frameCount = 10;
        s._handleDetecting(140, 0.05, 20, null, null);
        s._frameCount = 11;
        s._handleDetecting(180, 0.05, 20, null, null);
        s._frameCount = 12;
        s._handleDetecting(200, 0.05, 20, null, null);
        s._frameCount = 15;
        const evt3 = s._handleDetecting(190, 0.05, 20, null, null);
        assert(evt3 !== null && evt3.type === 'capture', 'fallback capture fires after extended wait');
    }

    console.log(`=== Results: ${passed} passed, ${failed} failed ===`);
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScanState;
}
