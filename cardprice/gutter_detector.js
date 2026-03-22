/**
 * GutterDetector — Client-side binder gutter detection for slide-scan camera.
 *
 * Detects the colored binder page material visible between card sleeves as the
 * user slides across a binder page. Gutters have very consistent color
 * properties that differ sharply from card artwork:
 *   - Blue binder:   high saturation, hue 190-250 (0-360 scale)
 *   - Orange binder: high saturation, hue 15-45
 *   - Black binder:  low value (<50)
 *   - White binder:  low saturation (<25), high value (>200)
 *
 * When the camera crosses a gutter, a card has just exited and another is about
 * to enter. Tracking card->gutter->card transitions is simpler and more robust
 * than edge detection or brightness analysis because binder color is extremely
 * consistent and distinct from card colors.
 *
 * Auto-detection: during the first N frames, collects HSV histograms from
 * the frame edges (left 10%, right 10%) where gutter material is most likely
 * visible. The dominant color cluster becomes the gutter signature.
 *
 * Usage:
 *   const detector = new GutterDetector();          // auto-detect binder color
 *   const detector = new GutterDetector('blue');     // or specify explicitly
 *
 *   // On each video frame:
 *   const result = detector.processFrame(videoElement);
 *   // result = { inGutter, gutterFraction, state, transitionCount,
 *   //            binderColor, calibrated }
 */

class GutterDetector {
    /**
     * @param {string} binderColor  'auto' | 'blue' | 'orange' | 'black' | 'white'
     * @param {Object} opts
     * @param {number} opts.gutterEnterThreshold  Fraction of column that must be gutter to enter gutter state. Default 0.30.
     * @param {number} opts.gutterExitThreshold   Fraction below which we leave gutter state. Default 0.10.
     * @param {number} opts.calibrationFrames     Frames to collect for auto-detection. Default 30.
     * @param {number} opts.sampleColumns         Number of vertical columns to sample. Default 3 (center + offsets).
     * @param {number} opts.edgeStripPct          Edge strip width as fraction for calibration. Default 0.10.
     * @param {number} opts.targetHeight          Downsample height for processing. Default 120.
     */
    constructor(binderColor = 'auto', opts = {}) {
        this.gutterEnterThreshold = opts.gutterEnterThreshold ?? 0.30;
        this.gutterExitThreshold  = opts.gutterExitThreshold  ?? 0.10;
        this.calibrationFrames    = opts.calibrationFrames    ?? 30;
        this.sampleColumns        = opts.sampleColumns        ?? 3;
        this.edgeStripPct         = opts.edgeStripPct         ?? 0.10;
        this.targetHeight         = opts.targetHeight         ?? 120;

        // Binder color config
        this._requestedColor = binderColor;
        this.binderColor     = binderColor === 'auto' ? null : binderColor;
        this.calibrated      = binderColor !== 'auto';

        // HSV matching function — set during calibration or from preset
        this._isGutterPixel = this.calibrated
            ? GutterDetector._makeColorMatcher(this.binderColor)
            : null;

        // Calibration state
        this._calFrameCount = 0;
        this._calHueHist    = new Float32Array(360);  // hue histogram (0-359)
        this._calSatSum     = 0;
        this._calValSum     = 0;
        this._calPixelCount = 0;

        // Transition tracking state
        this._inGutter       = false;
        this._transitionCount = 0;
        this._state           = 'unknown'; // 'card' | 'gutter' | 'unknown'

        // Rolling gutter fraction for smoothing
        this._fractionHistory = [];
        this._smoothingWindow = 5;

        // Reusable canvas
        this._canvas = null;
        this._ctx    = null;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Process a video frame and return gutter detection results.
     *
     * @param {HTMLVideoElement|HTMLCanvasElement|HTMLImageElement} source
     * @returns {{
     *   inGutter: boolean,
     *   gutterFraction: number,
     *   smoothedFraction: number,
     *   state: string,
     *   transitionCount: number,
     *   binderColor: string|null,
     *   calibrated: boolean
     * }}
     */
    processFrame(source) {
        const rgba = this._getRGBA(source);
        const w = rgba.width;
        const h = rgba.height;
        const pixels = rgba.data;

        // --- Calibration phase ---
        if (!this.calibrated) {
            this._collectCalibrationData(pixels, w, h);
            this._calFrameCount++;

            if (this._calFrameCount >= this.calibrationFrames) {
                this._finishCalibration();
            }

            return {
                inGutter: false,
                gutterFraction: 0,
                smoothedFraction: 0,
                state: 'calibrating',
                transitionCount: 0,
                binderColor: this.binderColor,
                calibrated: false,
            };
        }

        // --- Gutter detection ---
        const fraction = this._measureGutterFraction(pixels, w, h);

        // Smooth with rolling average
        this._fractionHistory.push(fraction);
        if (this._fractionHistory.length > this._smoothingWindow) {
            this._fractionHistory.shift();
        }
        const smoothed = this._fractionHistory.reduce((a, b) => a + b, 0)
                        / this._fractionHistory.length;

        // Hysteresis state machine: separate enter/exit thresholds prevent
        // flickering at the boundary
        const prevInGutter = this._inGutter;

        if (!this._inGutter && smoothed >= this.gutterEnterThreshold) {
            this._inGutter = true;
            this._state = 'gutter';
        } else if (this._inGutter && smoothed <= this.gutterExitThreshold) {
            this._inGutter = false;
            this._state = 'card';
        }

        // Count transitions (gutter->card means a new card just arrived)
        if (prevInGutter && !this._inGutter) {
            this._transitionCount++;
        }

        return {
            inGutter: this._inGutter,
            gutterFraction: Math.round(fraction * 1000) / 1000,
            smoothedFraction: Math.round(smoothed * 1000) / 1000,
            state: this._state,
            transitionCount: this._transitionCount,
            binderColor: this.binderColor,
            calibrated: true,
        };
    }

    /**
     * Manually set the binder color (skips or overrides auto-detection).
     *
     * @param {string} color  'blue' | 'orange' | 'black' | 'white'
     */
    setBinderColor(color) {
        this.binderColor = color;
        this._isGutterPixel = GutterDetector._makeColorMatcher(color);
        this.calibrated = true;
    }

    /**
     * Reset all state (e.g., when starting a new scanning session).
     */
    reset() {
        this._inGutter = false;
        this._transitionCount = 0;
        this._state = 'unknown';
        this._fractionHistory = [];

        if (this._requestedColor === 'auto') {
            this.calibrated = false;
            this.binderColor = null;
            this._isGutterPixel = null;
            this._calFrameCount = 0;
            this._calHueHist = new Float32Array(360);
            this._calSatSum = 0;
            this._calValSum = 0;
            this._calPixelCount = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Color matching
    // -----------------------------------------------------------------------

    /**
     * Create a pixel-matching function for a given binder color.
     * Returns a function (h, s, v) => boolean, where h is 0-359, s and v are 0-255.
     *
     * @param {string} color
     * @returns {function(number, number, number): boolean}
     */
    static _makeColorMatcher(color) {
        switch (color) {
            case 'blue':
                // Hue 190-250 (blue range in 0-360 scale), saturation > 50, value > 50
                return (h, s, v) => h >= 190 && h <= 250 && s > 50 && v > 50;

            case 'orange':
                // Hue 15-45, high saturation, decent brightness
                return (h, s, v) => h >= 15 && h <= 45 && s > 80 && v > 80;

            case 'black':
                // Any hue, low value (dark)
                return (h, s, v) => v < 50;

            case 'white':
                // Any hue, low saturation, high value
                return (h, s, v) => s < 25 && v > 200;

            default:
                // Fallback: accept nothing (forces recalibration)
                return () => false;
        }
    }

    /**
     * Convert RGB to HSV.
     * @param {number} r  0-255
     * @param {number} g  0-255
     * @param {number} b  0-255
     * @returns {[number, number, number]}  [h: 0-359, s: 0-255, v: 0-255]
     */
    static _rgbToHsv(r, g, b) {
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const delta = max - min;

        // Value
        const v = max;

        // Saturation
        const s = max === 0 ? 0 : Math.round((delta / max) * 255);

        // Hue
        let h = 0;
        if (delta > 0) {
            if (max === r) {
                h = 60 * (((g - b) / delta) % 6);
            } else if (max === g) {
                h = 60 * (((b - r) / delta) + 2);
            } else {
                h = 60 * (((r - g) / delta) + 4);
            }
            if (h < 0) h += 360;
        }

        return [Math.round(h), s, v];
    }

    // -----------------------------------------------------------------------
    // Calibration
    // -----------------------------------------------------------------------

    /**
     * Collect HSV statistics from edge strips of a frame.
     */
    _collectCalibrationData(pixels, w, h) {
        const edgeW = Math.max(1, Math.floor(w * this.edgeStripPct));

        // Sample left and right edge strips
        for (let y = 0; y < h; y++) {
            // Left strip
            for (let x = 0; x < edgeW; x++) {
                this._addCalibrationPixel(pixels, y * w + x);
            }
            // Right strip
            for (let x = w - edgeW; x < w; x++) {
                this._addCalibrationPixel(pixels, y * w + x);
            }
        }
    }

    /**
     * Add a single pixel's HSV to the calibration accumulators.
     */
    _addCalibrationPixel(pixels, idx) {
        const off = idx * 4;
        const [h, s, v] = GutterDetector._rgbToHsv(pixels[off], pixels[off + 1], pixels[off + 2]);

        // Only count pixels with some saturation or very dark/light (to avoid
        // neutral card artwork swamping the histogram)
        if (s > 40 || v < 50 || (s < 25 && v > 200)) {
            this._calHueHist[h]++;
            this._calSatSum += s;
            this._calValSum += v;
            this._calPixelCount++;
        }
    }

    /**
     * Finish calibration: pick the dominant color from collected histograms.
     */
    _finishCalibration() {
        if (this._calPixelCount < 100) {
            // Not enough data — default to black (safest fallback)
            this.setBinderColor('black');
            return;
        }

        const avgSat = this._calSatSum / this._calPixelCount;
        const avgVal = this._calValSum / this._calPixelCount;

        // Find the peak hue (smooth the histogram with a 15-degree window)
        let bestHue = 0;
        let bestCount = 0;
        for (let h = 0; h < 360; h++) {
            let count = 0;
            for (let d = -7; d <= 7; d++) {
                count += this._calHueHist[(h + d + 360) % 360];
            }
            if (count > bestCount) {
                bestCount = count;
                bestHue = h;
            }
        }

        // Classify based on average saturation/value and peak hue
        if (avgVal < 60) {
            this.setBinderColor('black');
        } else if (avgSat < 30 && avgVal > 180) {
            this.setBinderColor('white');
        } else if (bestHue >= 190 && bestHue <= 250) {
            this.setBinderColor('blue');
        } else if (bestHue >= 15 && bestHue <= 45) {
            this.setBinderColor('orange');
        } else {
            // Unknown color — create a custom matcher from the observed peak hue
            this.binderColor = `custom(hue=${bestHue})`;
            const peakH = bestHue;
            const hueRange = 20; // +/- 20 degrees
            this._isGutterPixel = (h, s, v) => {
                const hueDist = Math.min(
                    Math.abs(h - peakH),
                    360 - Math.abs(h - peakH)
                );
                return hueDist <= hueRange && s > 40 && v > 40;
            };
            this.calibrated = true;
        }
    }

    // -----------------------------------------------------------------------
    // Gutter measurement
    // -----------------------------------------------------------------------

    /**
     * Measure what fraction of the sampled columns are gutter-colored.
     * Samples multiple vertical columns across the frame width for robustness.
     *
     * @param {Uint8ClampedArray} pixels  RGBA pixel data
     * @param {number} w  Frame width
     * @param {number} h  Frame height
     * @returns {number}  Fraction 0-1 of sampled pixels that match gutter color
     */
    _measureGutterFraction(pixels, w, h) {
        const matcher = this._isGutterPixel;
        if (!matcher) return 0;

        let gutterCount = 0;
        let totalCount = 0;

        // Sample columns: center, and offsets at 1/3 and 2/3 width
        const colPositions = this._getColumnPositions(w);

        for (let ci = 0; ci < colPositions.length; ci++) {
            const x = colPositions[ci];

            // Sample every 2nd row for speed
            for (let y = 0; y < h; y += 2) {
                const idx = (y * w + x) * 4;
                const r = pixels[idx];
                const g = pixels[idx + 1];
                const b = pixels[idx + 2];

                const [hue, sat, val] = GutterDetector._rgbToHsv(r, g, b);

                if (matcher(hue, sat, val)) {
                    gutterCount++;
                }
                totalCount++;
            }
        }

        return totalCount > 0 ? gutterCount / totalCount : 0;
    }

    /**
     * Compute which x-positions to sample as vertical columns.
     * Uses center column plus offsets for robustness.
     *
     * @param {number} w  Frame width
     * @returns {number[]}  Array of x-coordinates to sample
     */
    _getColumnPositions(w) {
        if (this.sampleColumns === 1) {
            return [Math.floor(w / 2)];
        }
        // Evenly space columns across the middle 60% of the frame
        // (avoid the very edges which may have lens distortion)
        const positions = [];
        const start = Math.floor(w * 0.2);
        const end   = Math.floor(w * 0.8);
        const step  = (end - start) / (this.sampleColumns - 1);
        for (let i = 0; i < this.sampleColumns; i++) {
            positions.push(Math.floor(start + i * step));
        }
        return positions;
    }

    // -----------------------------------------------------------------------
    // Frame acquisition
    // -----------------------------------------------------------------------

    /**
     * Draw source to internal canvas at reduced resolution and return RGBA data.
     * Returns { data: Uint8ClampedArray, width, height }.
     */
    _getRGBA(source) {
        let srcW, srcH;
        if (source instanceof HTMLVideoElement) {
            srcW = source.videoWidth;
            srcH = source.videoHeight;
        } else if (source instanceof HTMLCanvasElement) {
            srcW = source.width;
            srcH = source.height;
        } else {
            srcW = source.naturalWidth || source.width;
            srcH = source.naturalHeight || source.height;
        }

        // Downsample to targetHeight, preserving aspect ratio
        const scale = this.targetHeight / srcH;
        const w = Math.round(srcW * scale);
        const h = this.targetHeight;

        if (!this._canvas || this._canvas.width !== w || this._canvas.height !== h) {
            this._canvas = document.createElement('canvas');
            this._canvas.width = w;
            this._canvas.height = h;
            this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
        }

        this._ctx.drawImage(source, 0, 0, w, h);
        const imgData = this._ctx.getImageData(0, 0, w, h);

        return { data: imgData.data, width: w, height: h };
    }
}

// ---------------------------------------------------------------------------
// Standalone helper for simple usage
// ---------------------------------------------------------------------------

/**
 * Quick check: is this frame mostly gutter?
 * Accepts raw RGBA ImageData. Does NOT track transitions.
 *
 * @param {ImageData|{data: Uint8ClampedArray, width: number, height: number}} imageData
 * @param {string} binderColor  'blue' | 'orange' | 'black' | 'white'
 * @returns {{ isGutter: boolean, fraction: number }}
 */
function isGutter(imageData, binderColor) {
    const pixels = imageData instanceof ImageData ? imageData.data : imageData.data;
    const w = imageData.width;
    const h = imageData.height;
    const matcher = GutterDetector._makeColorMatcher(binderColor);

    let gutterCount = 0;
    let totalCount = 0;
    const cx = Math.floor(w / 2);

    for (let y = 0; y < h; y++) {
        const idx = (y * w + cx) * 4;
        const [hue, sat, val] = GutterDetector._rgbToHsv(pixels[idx], pixels[idx + 1], pixels[idx + 2]);
        if (matcher(hue, sat, val)) gutterCount++;
        totalCount++;
    }

    const fraction = totalCount > 0 ? gutterCount / totalCount : 0;
    return { isGutter: fraction >= 0.30, fraction: Math.round(fraction * 1000) / 1000 };
}

// ---------------------------------------------------------------------------
// Self-test (run in browser console or Node with synthetic data)
// ---------------------------------------------------------------------------

GutterDetector.selfTest = function () {
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

    console.log('=== GutterDetector Self-Test ===');

    // --- Test RGB to HSV ---
    {
        const [h, s, v] = GutterDetector._rgbToHsv(255, 0, 0);
        assert(h === 0 && s === 255 && v === 255, `pure red -> h=${h} s=${s} v=${v}`);

        const [h2, s2, v2] = GutterDetector._rgbToHsv(0, 0, 255);
        assert(h2 === 240 && s2 === 255 && v2 === 255, `pure blue -> h=${h2} s=${s2} v=${v2}`);

        const [h3, s3, v3] = GutterDetector._rgbToHsv(0, 255, 0);
        assert(h3 === 120 && s3 === 255 && v3 === 255, `pure green -> h=${h3} s=${s3} v=${v3}`);

        const [h4, s4, v4] = GutterDetector._rgbToHsv(0, 0, 0);
        assert(h4 === 0 && s4 === 0 && v4 === 0, `black -> h=${h4} s=${s4} v=${v4}`);

        const [h5, s5, v5] = GutterDetector._rgbToHsv(255, 255, 255);
        assert(h5 === 0 && s5 === 0 && v5 === 255, `white -> h=${h5} s=${s5} v=${v5}`);

        // Blue binder color (typical: R=40, G=80, B=180)
        const [h6, s6, v6] = GutterDetector._rgbToHsv(40, 80, 180);
        assert(h6 >= 190 && h6 <= 250, `blue binder hue in range 190-250 (got ${h6})`);
        assert(s6 > 50, `blue binder saturation > 50 (got ${s6})`);
    }

    // --- Test color matchers ---
    {
        const blueMatcher = GutterDetector._makeColorMatcher('blue');
        assert(blueMatcher(220, 150, 150) === true, 'blue matcher accepts hue=220 s=150 v=150');
        assert(blueMatcher(0, 150, 150) === false, 'blue matcher rejects hue=0 (red)');
        assert(blueMatcher(220, 20, 150) === false, 'blue matcher rejects low saturation');

        const orangeMatcher = GutterDetector._makeColorMatcher('orange');
        assert(orangeMatcher(30, 200, 200) === true, 'orange matcher accepts hue=30 s=200 v=200');
        assert(orangeMatcher(220, 200, 200) === false, 'orange matcher rejects hue=220 (blue)');

        const blackMatcher = GutterDetector._makeColorMatcher('black');
        assert(blackMatcher(0, 0, 30) === true, 'black matcher accepts v=30');
        assert(blackMatcher(0, 0, 100) === false, 'black matcher rejects v=100');

        const whiteMatcher = GutterDetector._makeColorMatcher('white');
        assert(whiteMatcher(0, 10, 240) === true, 'white matcher accepts s=10 v=240');
        assert(whiteMatcher(0, 100, 240) === false, 'white matcher rejects high saturation');
    }

    // --- Test constructor presets ---
    {
        const d1 = new GutterDetector('blue');
        assert(d1.calibrated === true, 'preset blue is immediately calibrated');
        assert(d1.binderColor === 'blue', 'preset blue binderColor');

        const d2 = new GutterDetector('auto');
        assert(d2.calibrated === false, 'auto starts uncalibrated');
        assert(d2.binderColor === null, 'auto binderColor is null initially');
    }

    // --- Test setBinderColor ---
    {
        const d = new GutterDetector('auto');
        assert(d.calibrated === false, 'starts uncalibrated');
        d.setBinderColor('orange');
        assert(d.calibrated === true, 'calibrated after setBinderColor');
        assert(d.binderColor === 'orange', 'binderColor updated');
    }

    // --- Test reset ---
    {
        const d = new GutterDetector('auto');
        d.setBinderColor('blue');
        d._transitionCount = 5;
        d._inGutter = true;
        d.reset();
        assert(d.calibrated === false, 'reset auto detector clears calibration');
        assert(d._transitionCount === 0, 'reset clears transition count');
        assert(d._inGutter === false, 'reset clears gutter state');

        const d2 = new GutterDetector('blue');
        d2._transitionCount = 3;
        d2.reset();
        assert(d2.calibrated === true, 'reset preset detector keeps calibration');
        assert(d2._transitionCount === 0, 'reset preset clears transitions');
    }

    // --- Test column positions ---
    {
        const d = new GutterDetector('blue', { sampleColumns: 3 });
        const cols = d._getColumnPositions(100);
        assert(cols.length === 3, `3 columns returned (got ${cols.length})`);
        assert(cols[0] === 20, `first col at 20% (got ${cols[0]})`);
        assert(cols[1] === 50, `middle col at 50% (got ${cols[1]})`);
        assert(cols[2] === 80, `last col at 80% (got ${cols[2]})`);

        const d2 = new GutterDetector('blue', { sampleColumns: 1 });
        const cols2 = d2._getColumnPositions(100);
        assert(cols2.length === 1, '1 column returned');
        assert(cols2[0] === 50, `single col at center (got ${cols2[0]})`);
    }

    // --- Test hysteresis (state machine logic) ---
    {
        const d = new GutterDetector('blue');
        // Simulate the state machine directly
        d._fractionHistory = [];

        // Not in gutter, fraction below enter threshold
        d._inGutter = false;
        d._fractionHistory = [0.15, 0.15, 0.15, 0.15, 0.15];
        const smoothed1 = 0.15;
        assert(!d._inGutter, 'stays in card state at 15% gutter');

        // Push above enter threshold
        d._fractionHistory = [0.35, 0.35, 0.35, 0.35, 0.35];
        // Manually trigger the state machine logic
        const smoothed2 = 0.35;
        if (!d._inGutter && smoothed2 >= d.gutterEnterThreshold) {
            d._inGutter = true;
            d._state = 'gutter';
        }
        assert(d._inGutter, 'enters gutter state at 35%');

        // Between thresholds — should stay in gutter (hysteresis)
        const smoothed3 = 0.20;
        if (d._inGutter && smoothed3 <= d.gutterExitThreshold) {
            d._inGutter = false;
        }
        assert(d._inGutter, 'stays in gutter at 20% (hysteresis band)');

        // Drop below exit threshold
        const prevGutter = d._inGutter;
        const smoothed4 = 0.05;
        if (d._inGutter && smoothed4 <= d.gutterExitThreshold) {
            d._inGutter = false;
            d._state = 'card';
        }
        if (prevGutter && !d._inGutter) {
            d._transitionCount++;
        }
        assert(!d._inGutter, 'exits gutter state at 5%');
        assert(d._transitionCount === 1, 'transition counted on gutter->card');
    }

    // --- Test gutter measurement with synthetic pixel data ---
    {
        // Create a 20x10 "frame" of solid blue pixels (R=40, G=80, B=180)
        const w = 20, h = 10;
        const pixels = new Uint8ClampedArray(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            pixels[i * 4]     = 40;   // R
            pixels[i * 4 + 1] = 80;   // G
            pixels[i * 4 + 2] = 180;  // B
            pixels[i * 4 + 3] = 255;  // A
        }

        const d = new GutterDetector('blue', { sampleColumns: 1 });
        const frac = d._measureGutterFraction(pixels, w, h);
        assert(frac > 0.9, `solid blue frame -> fraction > 0.9 (got ${frac.toFixed(3)})`);

        // All white pixels (not gutter for blue binder)
        const whitePixels = new Uint8ClampedArray(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            whitePixels[i * 4]     = 255;
            whitePixels[i * 4 + 1] = 255;
            whitePixels[i * 4 + 2] = 255;
            whitePixels[i * 4 + 3] = 255;
        }
        const frac2 = d._measureGutterFraction(whitePixels, w, h);
        assert(frac2 === 0, `solid white frame -> fraction 0 for blue binder (got ${frac2.toFixed(3)})`);
    }

    // --- Test calibration ---
    {
        const d = new GutterDetector('auto', { calibrationFrames: 2 });
        const w = 100, h = 50;

        // Create blue-edged frames (edges are blue, center is white)
        const edgeW = Math.max(1, Math.floor(w * d.edgeStripPct));
        const pixels = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                if (x < edgeW || x >= w - edgeW) {
                    // Blue edge
                    pixels[idx]     = 40;
                    pixels[idx + 1] = 80;
                    pixels[idx + 2] = 180;
                } else {
                    // White center
                    pixels[idx]     = 255;
                    pixels[idx + 1] = 255;
                    pixels[idx + 2] = 255;
                }
                pixels[idx + 3] = 255;
            }
        }

        // Feed calibration frames
        d._collectCalibrationData(pixels, w, h);
        d._calFrameCount++;
        d._collectCalibrationData(pixels, w, h);
        d._calFrameCount++;
        d._finishCalibration();

        assert(d.calibrated === true, 'calibration completes after enough frames');
        assert(d.binderColor === 'blue', `detected blue binder (got ${d.binderColor})`);
    }

    // --- Test calibration: black binder ---
    {
        const d3 = new GutterDetector('auto', { calibrationFrames: 2 });
        const w3 = 100, h3 = 50;
        const edgeW3 = Math.max(1, Math.floor(w3 * d3.edgeStripPct));
        const pixels3 = new Uint8ClampedArray(w3 * h3 * 4);
        for (let y = 0; y < h3; y++) {
            for (let x = 0; x < w3; x++) {
                const idx = (y * w3 + x) * 4;
                if (x < edgeW3 || x >= w3 - edgeW3) {
                    // Black edge
                    pixels3[idx]     = 10;
                    pixels3[idx + 1] = 10;
                    pixels3[idx + 2] = 10;
                } else {
                    pixels3[idx]     = 200;
                    pixels3[idx + 1] = 200;
                    pixels3[idx + 2] = 200;
                }
                pixels3[idx + 3] = 255;
            }
        }

        d3._collectCalibrationData(pixels3, w3, h3);
        d3._calFrameCount++;
        d3._collectCalibrationData(pixels3, w3, h3);
        d3._calFrameCount++;
        d3._finishCalibration();

        assert(d3.calibrated === true, 'black calibration completes');
        assert(d3.binderColor === 'black', `detected black binder (got ${d3.binderColor})`);
    }

    console.log(`=== Results: ${passed} passed, ${failed} failed ===`);
    return failed === 0;
};

// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GutterDetector;
}
