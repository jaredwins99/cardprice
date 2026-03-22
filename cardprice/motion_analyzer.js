/**
 * MotionAnalyzer -- motion-based card transition detection for slide-scan.
 *
 * When the user slides their phone across a binder row, there are two
 * complementary signals:
 *
 *   1. **Frame differencing** -- inter-frame pixel change rate.  During
 *      sliding the diff is sustained; at card boundaries (card art -> gutter
 *      -> new card art) the diff spikes; during micro-pauses the diff drops.
 *
 *   2. **Center-strip optical flow proxy** -- horizontal displacement of the
 *      center strip between frames, estimated via phase correlation on a 1-D
 *      horizontal projection.  Gives a signed velocity (pixels/frame) that is
 *      more robust than raw diff for speed estimation.
 *
 * Capture strategy (combined):
 *   - Phase 1: detect scan start (sustained diff > idle threshold for N frames).
 *   - Phase 2: during scanning, use two complementary triggers:
 *       (a) **Pause capture** -- diff drops below running average (micro-pause
 *           while card is centered).  Capture the sharpest recent frame.
 *       (b) **Transition capture** -- diff spikes above 2x running average
 *           (card boundary passing through frame).  Capture the *previous*
 *           stable frame (the card that just left center).
 *       (c) **Cadence capture** -- fallback: if neither (a) nor (b) fires
 *           within a time window predicted from scan speed, force a capture
 *           at the expected moment.
 *   - Phase 3: detect scan end (sustained low diff).
 *
 * Usage:
 *   const motion = new MotionAnalyzer();
 *
 *   // In your rAF loop:
 *   ctx.drawImage(video, 0, 0);
 *   const result = motion.processFrame(video, canvas, ctx);
 *   // result.state       -- 'idle' | 'scanning' | 'ending'
 *   // result.capture      -- null | 'pause' | 'transition' | 'cadence'
 *   // result.captureFrame -- HTMLCanvasElement snapshot to capture (or null)
 *   // result.diff, .speed, .transitionScore, etc -- diagnostics
 */

class MotionAnalyzer {
    /**
     * @param {Object} opts
     * @param {number} opts.sampleWidth        Downsampled width for diff computation (default 160)
     * @param {number} opts.sampleHeight       Downsampled height for diff computation (default 90)
     * @param {number} opts.historyLen         Rolling history length in frames (default 30, ~1s at 30fps)
     * @param {number} opts.idleThreshold      Mean abs diff below this = no movement (default 1.5)
     * @param {number} opts.scanStartFrames    Consecutive moving frames to declare scan start (default 8)
     * @param {number} opts.scanEndFrames      Consecutive idle frames to declare scan end (default 15)
     * @param {number} opts.pauseRatio         Diff/avgDiff below this = micro-pause (default 0.4)
     * @param {number} opts.transitionRatio    Diff/avgDiff above this = card boundary (default 2.0)
     * @param {number} opts.minCaptureGapMs    Minimum ms between any two captures (default 500)
     * @param {number} opts.cadenceTimeoutMs   If no capture in this window, force one (default 1500)
     * @param {number} opts.stableBufferLen    Number of recent frames to keep for "best stable" (default 5)
     */
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

        // Internal state
        this._prevGray        = null;      // Uint8Array, sampleWidth * sampleHeight
        this._diffHistory     = [];        // {diff, time} rolling window
        this._consecutiveMove = 0;         // frames above idle threshold
        this._consecutiveIdle = 0;         // frames below idle threshold
        this._state           = 'idle';    // 'idle' | 'scanning' | 'ending'
        this._lastCaptureTime = 0;
        this._lastCaptureType = null;
        this._stableFrames    = [];        // circular buffer of {canvas, diff, time}
        this._scanStartTime   = 0;
        this._captureCount    = 0;         // captures since scan started

        // Center-strip horizontal velocity estimation
        this._prevHProj       = null;      // Float32Array, horizontal projection of center strip

        // Reusable scratch canvas for downsampling
        this._sampleCanvas    = null;
        this._sampleCtx       = null;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Process a single video frame. Call once per rAF tick.
     *
     * @param {HTMLVideoElement} video  - Live video element
     * @param {HTMLCanvasElement} canvas - Full-res canvas (already has frame drawn)
     * @param {CanvasRenderingContext2D} ctx - Context of canvas
     * @returns {{
     *   state: string,
     *   capture: string|null,
     *   captureFrame: HTMLCanvasElement|null,
     *   diff: number,
     *   avgDiff: number,
     *   transitionScore: number,
     *   speed: number,
     *   moving: boolean,
     *   scanDurationMs: number,
     *   captureCount: number
     * }}
     */
    processFrame(video, canvas, ctx) {
        const now = performance.now();
        const gray = this._downsampleGray(video);

        // First frame -- no diff possible
        if (!this._prevGray) {
            this._prevGray = gray;
            this._prevHProj = this._horizontalProjection(gray);
            return this._result(null, null, 0, 0, 0, 0, false, now);
        }

        // Compute frame difference (mean absolute diff, 0-255 scale)
        const diff = this._meanAbsDiff(gray, this._prevGray);

        // Estimate horizontal speed via 1-D phase correlation
        const hProj = this._horizontalProjection(gray);
        const speed = this._estimateShift(this._prevHProj, hProj);

        this._prevGray = gray;
        this._prevHProj = hProj;

        // Update rolling history
        this._diffHistory.push({ diff, time: now });
        if (this._diffHistory.length > this.historyLen) {
            this._diffHistory.shift();
        }

        // Running average of diffs
        const avgDiff = this._diffHistory.reduce((s, d) => s + d.diff, 0) / this._diffHistory.length;
        const transitionScore = diff / Math.max(avgDiff, 0.01);
        const moving = diff > this.idleThreshold;

        // Update consecutive counters
        if (moving) {
            this._consecutiveMove++;
            this._consecutiveIdle = 0;
        } else {
            this._consecutiveIdle++;
            this._consecutiveMove = 0;
        }

        // State machine
        let capture = null;
        let captureFrame = null;

        if (this._state === 'idle') {
            if (this._consecutiveMove >= this.scanStartFrames) {
                this._state = 'scanning';
                this._scanStartTime = now;
                this._captureCount = 0;
                this._lastCaptureTime = now; // reset cadence timer
                this._stableFrames = [];
            }
        }

        if (this._state === 'scanning') {
            // Buffer stable frames (low diff = sharp & centered)
            this._bufferStableFrame(canvas, diff, now);

            const timeSinceCapture = now - this._lastCaptureTime;
            const canCapture = timeSinceCapture >= this.minCaptureGapMs;

            if (canCapture) {
                // (a) Pause capture: diff dropped well below average
                if (this._diffHistory.length >= 5 && diff < avgDiff * this.pauseRatio && !moving) {
                    captureFrame = this._getBestStableFrame();
                    if (captureFrame) {
                        capture = 'pause';
                    }
                }

                // (b) Transition capture: diff spiked above average (card boundary)
                if (!capture && transitionScore > this.transitionRatio && this._diffHistory.length >= 5) {
                    // Grab the *previous* stable frame (the card that was just centered)
                    captureFrame = this._getBestStableFrame();
                    if (captureFrame) {
                        capture = 'transition';
                    }
                }

                // (c) Cadence capture: too long without any capture
                if (!capture && timeSinceCapture >= this.cadenceTimeoutMs && moving) {
                    captureFrame = this._getBestStableFrame() || this._snapshotCanvas(canvas);
                    capture = 'cadence';
                }
            }

            if (capture) {
                this._lastCaptureTime = now;
                this._lastCaptureType = capture;
                this._captureCount++;
                this._stableFrames = []; // clear buffer after capture
            }

            // Check for scan end
            if (this._consecutiveIdle >= this.scanEndFrames) {
                this._state = 'ending';
            }
        }

        if (this._state === 'ending') {
            // If motion resumes, go back to scanning
            if (this._consecutiveMove >= 3) {
                this._state = 'scanning';
            }
        }

        return this._result(capture, captureFrame, diff, avgDiff, transitionScore, speed, moving, now);
    }

    /**
     * Reset all state (e.g., between row scans).
     */
    reset() {
        this._prevGray        = null;
        this._prevHProj       = null;
        this._diffHistory     = [];
        this._consecutiveMove = 0;
        this._consecutiveIdle = 0;
        this._state           = 'idle';
        this._lastCaptureTime = 0;
        this._lastCaptureType = null;
        this._stableFrames    = [];
        this._scanStartTime   = 0;
        this._captureCount    = 0;
    }

    /** Current state: 'idle' | 'scanning' | 'ending'. */
    get state() {
        return this._state;
    }

    /** Number of captures since scan started. */
    get captureCount() {
        return this._captureCount;
    }

    // -----------------------------------------------------------------------
    // Internal: image processing
    // -----------------------------------------------------------------------

    /**
     * Downsample the video frame to a small grayscale Uint8Array.
     * Uses a scratch canvas for speed.
     */
    _downsampleGray(video) {
        const w = this.sampleWidth;
        const h = this.sampleHeight;

        if (!this._sampleCanvas) {
            this._sampleCanvas = document.createElement('canvas');
            this._sampleCanvas.width = w;
            this._sampleCanvas.height = h;
            this._sampleCtx = this._sampleCanvas.getContext('2d', { willReadFrequently: true });
        }

        this._sampleCtx.drawImage(video, 0, 0, w, h);
        const rgba = this._sampleCtx.getImageData(0, 0, w, h).data;

        const gray = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
            // Fast luminance: (77R + 150G + 29B) >> 8
            gray[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
        }
        return gray;
    }

    /**
     * Mean absolute difference between two grayscale frames.
     * Returns value in 0-255 range.
     */
    _meanAbsDiff(a, b) {
        let sum = 0;
        const len = a.length;
        // Unrolled: process 4 at a time for speed
        const len4 = len & ~3;
        for (let i = 0; i < len4; i += 4) {
            sum += Math.abs(a[i] - b[i])
                 + Math.abs(a[i + 1] - b[i + 1])
                 + Math.abs(a[i + 2] - b[i + 2])
                 + Math.abs(a[i + 3] - b[i + 3]);
        }
        for (let i = len4; i < len; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return sum / len;
    }

    /**
     * Compute horizontal projection: for each column x, average all rows.
     * Returns Float32Array of length sampleWidth.
     * This 1-D signal shifts horizontally as the phone slides.
     */
    _horizontalProjection(gray) {
        const w = this.sampleWidth;
        const h = this.sampleHeight;
        const proj = new Float32Array(w);

        for (let x = 0; x < w; x++) {
            let sum = 0;
            for (let y = 0; y < h; y++) {
                sum += gray[y * w + x];
            }
            proj[x] = sum / h;
        }
        return proj;
    }

    /**
     * Estimate horizontal pixel shift between two 1-D projections via
     * cross-correlation on a small search window.
     *
     * Returns signed shift in pixels (positive = rightward motion).
     * Searches +/- maxShift pixels (default 20, ~12% of 160px width).
     */
    _estimateShift(prevProj, currProj, maxShift) {
        if (!prevProj || !currProj) return 0;
        maxShift = maxShift || 20;
        const len = prevProj.length;

        let bestCorr = -Infinity;
        let bestShift = 0;

        for (let shift = -maxShift; shift <= maxShift; shift++) {
            let corr = 0;
            let count = 0;
            const start = Math.max(0, -shift);
            const end = Math.min(len, len - shift);
            for (let i = start; i < end; i++) {
                corr += prevProj[i] * currProj[i + shift];
                count++;
            }
            if (count > 0) {
                corr /= count;
                if (corr > bestCorr) {
                    bestCorr = corr;
                    bestShift = shift;
                }
            }
        }

        return bestShift;
    }

    // -----------------------------------------------------------------------
    // Internal: frame buffering and capture
    // -----------------------------------------------------------------------

    /**
     * Keep a small ring buffer of recent frames with their diff values.
     * The "best stable frame" is the one with the lowest diff (sharpest).
     */
    _bufferStableFrame(canvas, diff, time) {
        // Only buffer frames that are somewhat stable (not a transition spike)
        const avgDiff = this._diffHistory.length > 0
            ? this._diffHistory.reduce((s, d) => s + d.diff, 0) / this._diffHistory.length
            : diff;

        if (diff < avgDiff * 1.5) {
            const snapshot = this._snapshotCanvas(canvas);
            this._stableFrames.push({ canvas: snapshot, diff, time });
            if (this._stableFrames.length > this.stableBufferLen) {
                this._stableFrames.shift();
            }
        }
    }

    /**
     * Return the best (lowest diff) frame from the stable buffer.
     * Returns the canvas element, or null if buffer is empty.
     */
    _getBestStableFrame() {
        if (this._stableFrames.length === 0) return null;

        let best = this._stableFrames[0];
        for (let i = 1; i < this._stableFrames.length; i++) {
            if (this._stableFrames[i].diff < best.diff) {
                best = this._stableFrames[i];
            }
        }
        return best.canvas;
    }

    /**
     * Create a snapshot copy of the current canvas.
     */
    _snapshotCanvas(sourceCanvas) {
        const snap = document.createElement('canvas');
        snap.width = sourceCanvas.width;
        snap.height = sourceCanvas.height;
        snap.getContext('2d').drawImage(sourceCanvas, 0, 0);
        return snap;
    }

    // -----------------------------------------------------------------------
    // Internal: result assembly
    // -----------------------------------------------------------------------

    _result(capture, captureFrame, diff, avgDiff, transitionScore, speed, moving, now) {
        return {
            state:           this._state,
            capture:         capture,         // null | 'pause' | 'transition' | 'cadence'
            captureFrame:    captureFrame,     // HTMLCanvasElement or null
            diff:            Math.round(diff * 100) / 100,
            avgDiff:         Math.round(avgDiff * 100) / 100,
            transitionScore: Math.round(transitionScore * 100) / 100,
            speed:           speed,           // pixels/frame horizontal shift
            moving:          moving,
            scanDurationMs:  this._state === 'scanning' ? Math.round(now - this._scanStartTime) : 0,
            captureCount:    this._captureCount,
        };
    }
}


// ---------------------------------------------------------------------------
// Self-test (run in browser console: MotionAnalyzer.selfTest())
// ---------------------------------------------------------------------------

MotionAnalyzer.selfTest = function () {
    let passed = 0;
    let failed = 0;

    function assert(cond, name) {
        if (cond) { console.log('  PASS: ' + name); passed++; }
        else      { console.error('  FAIL: ' + name); failed++; }
    }

    console.log('=== MotionAnalyzer Self-Test ===');

    const ma = new MotionAnalyzer();

    // --- _meanAbsDiff ---
    {
        const a = new Uint8Array([100, 100, 100, 100]);
        const b = new Uint8Array([100, 100, 100, 100]);
        assert(ma._meanAbsDiff(a, b) === 0, 'identical frames -> diff 0');

        const c = new Uint8Array([0, 0, 0, 0]);
        const d = new Uint8Array([10, 20, 30, 40]);
        assert(ma._meanAbsDiff(c, d) === 25, 'known diff -> 25');

        const e = new Uint8Array(160 * 90).fill(128);
        const f = new Uint8Array(160 * 90).fill(130);
        assert(ma._meanAbsDiff(e, f) === 2, 'uniform 2-step diff -> 2');
    }

    // --- _horizontalProjection ---
    {
        // 4x2 image: each column has same value
        ma.sampleWidth = 4;
        ma.sampleHeight = 2;
        const gray = new Uint8Array([10, 20, 30, 40, 10, 20, 30, 40]);
        const proj = ma._horizontalProjection(gray);
        assert(proj.length === 4, 'projection length matches width');
        assert(proj[0] === 10 && proj[1] === 20 && proj[2] === 30 && proj[3] === 40,
               'projection values correct');
        ma.sampleWidth = 160;
        ma.sampleHeight = 90;
    }

    // --- _estimateShift ---
    {
        const a = new Float32Array([0, 0, 100, 100, 0, 0, 0, 0, 0, 0]);
        const b = new Float32Array([0, 0, 0, 0, 100, 100, 0, 0, 0, 0]);
        const shift = ma._estimateShift(a, b, 5);
        assert(shift === 2 || shift === -2, 'detects 2-pixel shift (got ' + shift + ')');
    }

    // --- State machine: idle -> scanning ---
    {
        const m = new MotionAnalyzer({ scanStartFrames: 3 });
        // Simulate: feed enough "moving" frames
        m._prevGray = new Uint8Array(160 * 90).fill(128);
        m._prevHProj = new Float32Array(160).fill(128);

        // Helper to make a frame with known diff from prev
        function makeDiffFrame(base, offset) {
            const f = new Uint8Array(base.length);
            for (let i = 0; i < f.length; i++) f[i] = Math.min(255, base[i] + offset);
            return f;
        }

        // Feed frames that cause diff > idleThreshold (1.5)
        for (let i = 0; i < 10; i++) {
            const gray = makeDiffFrame(m._prevGray, (i % 2 === 0) ? 5 : -5);
            const diff = m._meanAbsDiff(gray, m._prevGray);
            m._diffHistory.push({ diff, time: i * 33 });
            if (m._diffHistory.length > m.historyLen) m._diffHistory.shift();
            if (diff > m.idleThreshold) {
                m._consecutiveMove++;
                m._consecutiveIdle = 0;
            }
            if (m._state === 'idle' && m._consecutiveMove >= m.scanStartFrames) {
                m._state = 'scanning';
            }
            m._prevGray = gray;
        }
        assert(m._state === 'scanning', 'transitions to scanning after enough moving frames');
    }

    // --- Reset ---
    {
        const m = new MotionAnalyzer();
        m._state = 'scanning';
        m._captureCount = 5;
        m._diffHistory = [{ diff: 1, time: 0 }];
        m.reset();
        assert(m._state === 'idle', 'reset -> idle');
        assert(m._captureCount === 0, 'reset -> captureCount 0');
        assert(m._diffHistory.length === 0, 'reset -> empty history');
        assert(m._prevGray === null, 'reset -> null prevGray');
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MotionAnalyzer;
}
