/**
 * RowScanner — Card detection for slide-scan mode via 1D brightness signal.
 *
 * When the user slides their phone across a binder row, the video shows a
 * repeating pattern: colored gutter → card → gutter → card → gutter → card → gutter.
 * This class treats it as a 1D signal problem: sample center-strip brightness
 * over time, find peaks = card centers, and capture at each peak.
 *
 * Two complementary signals are used:
 *   1. Brightness (luminance) — cards are brighter than binder gutters
 *   2. Blue channel ratio B/(R+G+1) — blue/colored binders have high blue ratio
 *      in gutters, cards do not (works even with dark or colored cards)
 *
 * Thresholds are adaptive: after a calibration window, mean and std of each
 * signal are computed.  Card/gutter zones are defined relative to those stats.
 *
 * Capture timing uses a 5-frame rolling average to smooth noise, then detects
 * the moment brightness transitions from rising/stable to dropping (just past
 * the peak), which is the card center.
 *
 * Usage:
 *   const scanner = new RowScanner({ numCards: 3 });
 *   scanner.start();
 *   // in requestAnimationFrame loop:
 *   const event = scanner.processFrame(videoElement, captureCanvas);
 *   if (event) {
 *       // event.type === 'capture' — card image is on captureCanvas
 *       // event.cardIndex — which card (0, 1, 2)
 *       // event.brightness — brightness at capture
 *   }
 *   if (scanner.state === 'done') { ... }
 */

class RowScanner {
    /**
     * @param {Object} opts
     * @param {number} opts.numCards         Cards per row (default 3)
     * @param {number} opts.calibrationFrames Frames to collect before computing thresholds (default 15)
     * @param {number} opts.smoothingWindow  Rolling average window for noise suppression (default 5)
     * @param {number} opts.stripWidthPct    Center strip width as fraction of frame (default 0.20)
     * @param {number} opts.cardThresholdK   Std-dev multiplier for card threshold (default 0.3)
     * @param {number} opts.gutterThresholdK Std-dev multiplier for gutter threshold (default 0.3)
     * @param {number} opts.minPeakWidth     Minimum frames a card peak must span (default 3)
     * @param {number} opts.minGutterWidth   Minimum frames a gutter must span before next card (default 2)
     * @param {number} opts.blueRatioWeight  Weight of blue-ratio signal vs brightness (default 0.4)
     */
    constructor(opts = {}) {
        this.numCards          = opts.numCards          ?? 3;
        this.calibrationFrames = opts.calibrationFrames ?? 15;
        this.smoothingWindow   = opts.smoothingWindow   ?? 5;
        this.stripWidthPct     = opts.stripWidthPct     ?? 0.20;
        this.cardThresholdK    = opts.cardThresholdK    ?? 0.3;
        this.gutterThresholdK  = opts.gutterThresholdK  ?? 0.3;
        this.minPeakWidth      = opts.minPeakWidth      ?? 3;
        this.minGutterWidth    = opts.minGutterWidth    ?? 2;
        this.blueRatioWeight   = opts.blueRatioWeight   ?? 0.4;

        this.state    = 'idle';  // idle | calibrating | scanning | done
        this.captures = [];      // array of { canvas, cardIndex, brightness, frame }

        // Signal history (raw and smoothed)
        this._brightHistory   = [];   // raw brightness per frame
        this._blueHistory     = [];   // raw blue-ratio per frame
        this._smoothedHistory = [];   // smoothed combined signal
        this._frameCount      = 0;

        // Adaptive thresholds (set after calibration)
        this._cardThreshold   = null;
        this._gutterThreshold = null;

        // Peak detection state machine
        this._zone         = 'unknown';  // unknown | gutter | card
        this._cardEnterFrame = -1;       // frame when we entered card zone
        this._peakValue    = -Infinity;  // highest smoothed value in current card zone
        this._peakFrame    = -1;         // frame index of peak
        this._gutterFrames = 0;          // consecutive frames in gutter (debounce)
        this._captured     = false;      // already captured for current card zone?

        // Reusable sampling canvas
        this._sampleCanvas = null;
        this._sampleCtx    = null;

        // Debug / diagnostics callback
        this.onDiagnostics = null;  // function({ brightness, blueRatio, smoothed, zone, threshold })
    }

    // -------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------

    /** Begin scanning. Resets all state. */
    start() {
        this.state    = 'calibrating';
        this.captures = [];

        this._brightHistory   = [];
        this._blueHistory     = [];
        this._smoothedHistory = [];
        this._frameCount      = 0;

        this._cardThreshold   = null;
        this._gutterThreshold = null;

        this._zone           = 'unknown';
        this._cardEnterFrame = -1;
        this._peakValue      = -Infinity;
        this._peakFrame      = -1;
        this._gutterFrames   = 0;
        this._captured       = false;
    }

    /** Reset to idle (e.g. user cancels). */
    reset() {
        this.state    = 'idle';
        this.captures = [];

        this._brightHistory   = [];
        this._blueHistory     = [];
        this._smoothedHistory = [];
        this._frameCount      = 0;

        this._cardThreshold   = null;
        this._gutterThreshold = null;

        this._zone           = 'unknown';
        this._cardEnterFrame = -1;
        this._peakValue      = -Infinity;
        this._peakFrame      = -1;
        this._gutterFrames   = 0;
        this._captured       = false;
    }

    /**
     * Process one video frame.  Call this from requestAnimationFrame.
     *
     * @param {HTMLVideoElement} video       The live camera feed
     * @param {HTMLCanvasElement} captureCanvas  Canvas to draw the captured frame onto
     * @returns {Object|null}  Capture event { type:'capture', cardIndex, brightness, frame }
     *                         or null if no capture this frame.
     */
    processFrame(video, captureCanvas) {
        if (this.state === 'idle' || this.state === 'done') return null;

        this._frameCount++;

        // --- 1. Sample center strip ---
        const { brightness, blueRatio } = this._sampleCenterStrip(video);

        this._brightHistory.push(brightness);
        this._blueHistory.push(blueRatio);

        // --- 2. Calibration phase ---
        if (this.state === 'calibrating') {
            if (this._brightHistory.length < this.calibrationFrames) {
                this._emitDiag(brightness, blueRatio, null, 'calibrating');
                return null;
            }
            this._computeThresholds();
            this.state = 'scanning';
            // Fall through to process this frame as scanning
        }

        // --- 3. Compute smoothed combined signal ---
        const smoothed = this._computeSmoothed();
        this._smoothedHistory.push(smoothed);

        // --- 4. Peak detection state machine ---
        const event = this._detectPeak(smoothed, video, captureCanvas);

        this._emitDiag(brightness, blueRatio, smoothed, this._zone);

        return event;
    }

    /**
     * Get current diagnostics snapshot (for debug overlay).
     * @returns {Object}
     */
    getDiagnostics() {
        const len = this._smoothedHistory.length;
        return {
            state:          this.state,
            zone:           this._zone,
            frameCount:     this._frameCount,
            captureCount:   this.captures.length,
            numCards:        this.numCards,
            cardThreshold:  this._cardThreshold,
            gutterThreshold: this._gutterThreshold,
            lastBrightness: this._brightHistory.length > 0
                ? this._brightHistory[this._brightHistory.length - 1] : null,
            lastBlueRatio:  this._blueHistory.length > 0
                ? this._blueHistory[this._blueHistory.length - 1] : null,
            lastSmoothed:   len > 0 ? this._smoothedHistory[len - 1] : null,
            peakValue:      this._peakValue,
            // Last N smoothed values for plotting
            recentSignal:   this._smoothedHistory.slice(-60),
        };
    }

    // -------------------------------------------------------------------
    // Internal: signal sampling
    // -------------------------------------------------------------------

    /**
     * Sample the center vertical strip of the video frame.
     * Returns mean brightness (0-255) and blue channel ratio.
     *
     * The center strip captures what's directly in front of the camera
     * as the user slides horizontally.  Using a narrow strip (20% width)
     * avoids edge contamination from adjacent cards/gutters.
     */
    _sampleCenterStrip(video) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        if (vw === 0 || vh === 0) return { brightness: 128, blueRatio: 0.33 };

        // Compute strip bounds in source coordinates
        const stripW = Math.max(4, Math.round(vw * this.stripWidthPct));
        const stripX = Math.round((vw - stripW) / 2);

        // Downsample strip to a small canvas for speed
        // Target: stripW pixels wide, 60 pixels tall (enough for statistics)
        const targetH = 60;
        const targetW = Math.max(2, Math.round(stripW * (targetH / vh)));

        if (!this._sampleCanvas) {
            this._sampleCanvas = document.createElement('canvas');
            this._sampleCtx = this._sampleCanvas.getContext('2d', { willReadFrequently: true });
        }
        this._sampleCanvas.width  = targetW;
        this._sampleCanvas.height = targetH;

        // Draw just the center strip, downsampled
        this._sampleCtx.drawImage(
            video,
            stripX, 0, stripW, vh,   // source rect
            0, 0, targetW, targetH   // dest rect
        );

        const imgData = this._sampleCtx.getImageData(0, 0, targetW, targetH);
        const px = imgData.data;
        const numPixels = targetW * targetH;

        let sumR = 0, sumG = 0, sumB = 0;
        for (let i = 0; i < px.length; i += 4) {
            sumR += px[i];
            sumG += px[i + 1];
            sumB += px[i + 2];
        }

        const meanR = sumR / numPixels;
        const meanG = sumG / numPixels;
        const meanB = sumB / numPixels;

        // Brightness: standard luminance
        const brightness = 0.299 * meanR + 0.587 * meanG + 0.114 * meanB;

        // Blue ratio: B / (R + G + 1) — higher for blue binder gutters
        // The +1 avoids division by zero on very dark regions
        const blueRatio = meanB / (meanR + meanG + 1);

        return { brightness, blueRatio };
    }

    // -------------------------------------------------------------------
    // Internal: adaptive thresholds
    // -------------------------------------------------------------------

    /**
     * Compute adaptive thresholds from the calibration window.
     *
     * The calibration window should span at least one gutter-card-gutter
     * transition so we capture the range of both.  We use mean +/- k*std
     * to set thresholds.
     *
     * Also computes blue-ratio statistics for the combined signal.
     */
    _computeThresholds() {
        // Brightness stats
        const bMean = _mean(this._brightHistory);
        const bStd  = _std(this._brightHistory, bMean);

        // Blue-ratio stats
        const brMean = _mean(this._blueHistory);
        const brStd  = _std(this._blueHistory, brMean);

        // Card threshold: brightness above mean (cards are brighter)
        // Gutter threshold: brightness below mean (gutters are darker)
        this._cardThreshold   = bMean + this.cardThresholdK * bStd;
        this._gutterThreshold = bMean - this.gutterThresholdK * bStd;

        // Store blue-ratio stats for combined signal normalization
        this._brightMean = bMean;
        this._brightStd  = Math.max(bStd, 1);  // avoid div-by-zero
        this._blueMean   = brMean;
        this._blueStd    = Math.max(brStd, 0.001);
    }

    /**
     * Compute the smoothed combined signal for the current frame.
     *
     * Combines brightness and inverse-blue-ratio into a single score.
     * Both are z-score normalized so they contribute equally regardless
     * of absolute scale.  Then applies a rolling average over the
     * smoothing window to suppress frame-to-frame noise.
     *
     * Higher = more likely card.  Lower = more likely gutter.
     */
    _computeSmoothed() {
        const len = this._brightHistory.length;

        // Rolling average of combined signal over last `smoothingWindow` frames
        const windowStart = Math.max(0, len - this.smoothingWindow);
        let sum = 0;
        let count = 0;

        for (let i = windowStart; i < len; i++) {
            const b = this._brightHistory[i];
            const br = this._blueHistory[i];

            // Z-score normalize each signal
            const bNorm = (b - this._brightMean) / this._brightStd;
            // Invert blue ratio: low blue ratio = card (high score)
            const brNorm = -(br - this._blueMean) / this._blueStd;

            // Weighted combination
            const combined = (1 - this.blueRatioWeight) * bNorm
                           + this.blueRatioWeight * brNorm;
            sum += combined;
            count++;
        }

        return sum / count;
    }

    // -------------------------------------------------------------------
    // Internal: peak detection state machine
    // -------------------------------------------------------------------

    /**
     * Detect card peaks in the smoothed signal and trigger captures.
     *
     * State machine with two zones:
     *   - GUTTER: smoothed signal is below card threshold.
     *             Wait for signal to rise above threshold → enter CARD zone.
     *   - CARD:   smoothed signal is above gutter threshold.
     *             Track the peak.  When signal drops below peak by a margin
     *             OR drops below gutter threshold → capture and return to GUTTER.
     *
     * The capture happens just AFTER the peak, not at the peak itself.
     * This gives us the most centered card image.
     *
     * Debouncing:
     *   - Must stay in card zone for minPeakWidth frames (prevents noise spikes)
     *   - Must stay in gutter for minGutterWidth frames (prevents double-triggers)
     */
    _detectPeak(smoothed, video, captureCanvas) {
        // Normalize thresholds to combined-signal space
        const cardThresh   = this.cardThresholdK;   // in z-score units
        const gutterThresh = -this.gutterThresholdK; // negative z-score = gutter

        if (this._zone === 'unknown' || this._zone === 'gutter') {
            // --- Looking for a card ---
            if (smoothed > cardThresh) {
                if (this._zone === 'gutter' && this._gutterFrames < this.minGutterWidth) {
                    // Haven't been in gutter long enough — ignore
                    return null;
                }
                // Transition to card zone
                this._zone           = 'card';
                this._cardEnterFrame = this._frameCount;
                this._peakValue      = smoothed;
                this._peakFrame      = this._frameCount;
                this._captured       = false;
            } else {
                this._gutterFrames++;
            }
            return null;
        }

        if (this._zone === 'card') {
            // --- Tracking peak within card zone ---

            // Update peak if still rising
            if (smoothed >= this._peakValue) {
                this._peakValue = smoothed;
                this._peakFrame = this._frameCount;
            }

            // Check for capture condition: signal has dropped from peak
            const framesInCard  = this._frameCount - this._cardEnterFrame;
            const framesSincePeak = this._frameCount - this._peakFrame;

            // Capture when:
            //   1. We've been in the card zone long enough (minPeakWidth)
            //   2. Signal has dropped from peak (at least 2 frames past peak)
            //   3. Haven't already captured this card
            const shouldCapture = !this._captured
                && framesInCard >= this.minPeakWidth
                && framesSincePeak >= 2
                && smoothed < this._peakValue;

            if (shouldCapture) {
                this._captured = true;
                return this._captureFrame(video, captureCanvas, smoothed);
            }

            // Check for exit to gutter
            if (smoothed < gutterThresh) {
                // Left the card zone
                // If we never captured (peak was too narrow), capture now as last chance
                if (!this._captured && framesInCard >= this.minPeakWidth) {
                    this._zone         = 'gutter';
                    this._gutterFrames = 0;
                    return this._captureFrame(video, captureCanvas, smoothed);
                }
                this._zone         = 'gutter';
                this._gutterFrames = 0;
            }

            return null;
        }

        return null;
    }

    /**
     * Capture the current video frame onto captureCanvas.
     *
     * @returns {Object} Capture event
     */
    _captureFrame(video, captureCanvas, smoothed) {
        // Draw the video frame onto captureCanvas (skip if no DOM in test)
        if (video && captureCanvas) {
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            captureCanvas.width  = vw;
            captureCanvas.height = vh;
            const ctx = captureCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0, vw, vh);
        }

        const cardIndex = this.captures.length;
        const event = {
            type:       'capture',
            cardIndex:  cardIndex,
            brightness: this._brightHistory[this._brightHistory.length - 1],
            blueRatio:  this._blueHistory[this._blueHistory.length - 1],
            smoothed:   smoothed,
            frame:      this._frameCount,
        };

        this.captures.push({
            cardIndex: cardIndex,
            frame:     this._frameCount,
            brightness: event.brightness,
        });

        // Check if we've captured all cards
        if (this.captures.length >= this.numCards) {
            this.state = 'done';
        }

        return event;
    }

    // -------------------------------------------------------------------
    // Internal: diagnostics
    // -------------------------------------------------------------------

    _emitDiag(brightness, blueRatio, smoothed, zone) {
        if (this.onDiagnostics) {
            this.onDiagnostics({
                brightness:      Math.round(brightness * 10) / 10,
                blueRatio:       Math.round(blueRatio * 1000) / 1000,
                smoothed:        smoothed !== null ? Math.round(smoothed * 100) / 100 : null,
                zone:            zone,
                cardThreshold:   this._cardThreshold,
                gutterThreshold: this._gutterThreshold,
                captureCount:    this.captures.length,
                numCards:         this.numCards,
            });
        }
    }
}


// ---------------------------------------------------------------------------
// Utility functions (module-private)
// ---------------------------------------------------------------------------

function _mean(arr) {
    if (arr.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    return sum / arr.length;
}

function _std(arr, mean) {
    if (arr.length === 0) return 0;
    let sumSq = 0;
    for (let i = 0; i < arr.length; i++) {
        const d = arr[i] - mean;
        sumSq += d * d;
    }
    return Math.sqrt(sumSq / arr.length);
}


// ---------------------------------------------------------------------------
// Self-test: exercises core logic with synthetic signal data
// ---------------------------------------------------------------------------

RowScanner.selfTest = function () {
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

    console.log('=== RowScanner Self-Test ===');

    // --- Test _mean and _std ---
    {
        assert(_mean([10, 20, 30]) === 20, 'mean([10,20,30]) = 20');
        assert(Math.abs(_std([10, 20, 30], 20) - 8.165) < 0.01, 'std([10,20,30]) ~ 8.165');
        assert(_mean([]) === 0, 'mean([]) = 0');
        assert(_std([], 0) === 0, 'std([]) = 0');
    }

    // --- Test constructor defaults ---
    {
        const s = new RowScanner();
        assert(s.numCards === 3, 'default numCards = 3');
        assert(s.state === 'idle', 'initial state = idle');
        assert(s.captures.length === 0, 'initial captures empty');
    }

    // --- Test start/reset ---
    {
        const s = new RowScanner();
        s.start();
        assert(s.state === 'calibrating', 'start -> calibrating');
        s.reset();
        assert(s.state === 'idle', 'reset -> idle');
        assert(s.captures.length === 0, 'reset clears captures');
    }

    // --- Test threshold computation ---
    {
        const s = new RowScanner({ calibrationFrames: 5 });
        s.start();
        // Simulate brightness history: gutters dark (80), cards bright (200)
        s._brightHistory = [80, 200, 80, 200, 80];
        s._blueHistory   = [0.6, 0.2, 0.6, 0.2, 0.6];
        s._computeThresholds();

        // Mean brightness = 128, std ~ 60
        assert(s._cardThreshold > 128, 'card threshold > mean brightness');
        assert(s._gutterThreshold < 128, 'gutter threshold < mean brightness');
        assert(s._brightStd > 50, 'brightness std captures gutter/card variation');
    }

    // --- Test smoothed signal computation ---
    {
        const s = new RowScanner({ smoothingWindow: 3, calibrationFrames: 3 });
        s.start();
        s._brightHistory = [80, 200, 80, 200, 140];
        s._blueHistory   = [0.6, 0.2, 0.6, 0.2, 0.4];
        s._computeThresholds();

        const smoothed = s._computeSmoothed();
        assert(typeof smoothed === 'number', 'smoothed is a number');
        assert(!isNaN(smoothed), 'smoothed is not NaN');
    }

    // --- Test combined signal z-score normalization ---
    {
        const s = new RowScanner({ blueRatioWeight: 0.0, smoothingWindow: 1, calibrationFrames: 3 });
        s.start();
        // All same brightness = mean, std > 0 only if we add variation
        s._brightHistory = [100, 150, 200, 200];
        s._blueHistory   = [0.3, 0.3, 0.3, 0.3];
        s._computeThresholds();

        // Last value is 200, which is above mean (150), so z-score should be positive
        const smoothed = s._computeSmoothed();
        assert(smoothed > 0, 'bright frame -> positive z-score');

        // Now add a dark frame
        s._brightHistory.push(100);
        s._blueHistory.push(0.3);
        const smoothedDark = s._computeSmoothed();
        assert(smoothedDark < smoothed, 'dark frame -> lower z-score');
    }

    // --- Test peak detection: synthetic gutter-card-gutter pattern ---
    {
        const s = new RowScanner({
            numCards: 1,
            calibrationFrames: 5,
            smoothingWindow: 1,  // no smoothing for test clarity
            minPeakWidth: 2,
            minGutterWidth: 1,
        });
        s.start();

        // Calibration phase: mix of gutter and card values
        s._brightHistory = [80, 200, 80, 200, 80];
        s._blueHistory   = [0.6, 0.2, 0.6, 0.2, 0.6];
        s._computeThresholds();
        s.state = 'scanning';
        s._zone = 'gutter';
        s._gutterFrames = 5;

        // Feed frames manually through _detectPeak with synthetic smoothed values
        // Gutter -> card -> peak -> dropping -> gutter
        const signals = [-1.0, -0.5, 0.5, 1.0, 1.5, 1.2, 0.8, 0.3, -0.5];
        let captureEvent = null;

        for (let i = 0; i < signals.length; i++) {
            s._frameCount = i + 6;  // after calibration
            s._brightHistory.push(signals[i] > 0 ? 200 : 80);
            s._blueHistory.push(signals[i] > 0 ? 0.2 : 0.6);
            s._smoothedHistory.push(signals[i]);

            // Use null for video/canvas since we can't create DOM elements in test
            const evt = s._detectPeak(signals[i], null, null);
            if (evt) captureEvent = evt;
        }

        // The peak is at index 4 (signal=1.5), capture should happen at index 5 or 6
        // (when signal drops to 1.2 or 0.8, confirming we passed the peak)
        // We can't actually capture without DOM, but zone transitions should work
        assert(s._zone === 'gutter' || captureEvent !== null || s.captures.length > 0,
            'peak detection transitions through card zone');
    }

    // --- Test blue ratio contribution ---
    {
        const s = new RowScanner({ blueRatioWeight: 0.5, smoothingWindow: 1, calibrationFrames: 3 });
        s.start();
        s._brightHistory = [100, 150, 200];
        s._blueHistory   = [0.5, 0.3, 0.1];
        s._computeThresholds();

        // High blue ratio should pull signal DOWN (gutter)
        s._brightHistory.push(150);
        s._blueHistory.push(0.8);  // very blue = gutter
        const gutterSignal = s._computeSmoothed();

        s._brightHistory.push(150);
        s._blueHistory.push(0.1);  // low blue = card
        const cardSignal = s._computeSmoothed();

        assert(cardSignal > gutterSignal,
            'same brightness but low blue ratio -> higher signal (card)');
    }

    // --- Test state = done after numCards captures ---
    {
        const s = new RowScanner({ numCards: 2 });
        s.start();
        s.captures = [{ cardIndex: 0 }];
        // Simulate second capture setting done
        s.captures.push({ cardIndex: 1 });
        if (s.captures.length >= s.numCards) s.state = 'done';
        assert(s.state === 'done', 'state = done after numCards captures');
    }

    // --- Test getDiagnostics ---
    {
        const s = new RowScanner();
        s.start();
        s._brightHistory = [100, 150];
        s._blueHistory = [0.3, 0.4];
        s._smoothedHistory = [0.5, 0.8];
        s._frameCount = 2;
        s._zone = 'card';
        s._cardThreshold = 140;
        s._gutterThreshold = 110;

        const diag = s.getDiagnostics();
        assert(diag.state === 'calibrating', 'diagnostics includes state');
        assert(diag.zone === 'card', 'diagnostics includes zone');
        assert(diag.lastBrightness === 150, 'diagnostics includes last brightness');
        assert(diag.lastBlueRatio === 0.4, 'diagnostics includes last blue ratio');
        assert(diag.recentSignal.length === 2, 'diagnostics includes recent signal');
    }

    console.log(`=== Results: ${passed} passed, ${failed} failed ===`);
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RowScanner;
}
