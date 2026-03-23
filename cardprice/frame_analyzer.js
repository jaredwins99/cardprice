/**
 * Frame Analyzer -- static card presence/quality detection for slide-scan v7.
 *
 * Adapted from condition_camera_ui.py's detectCard() function which works
 * reliably on mobile phones. Simplified for the scanner use case:
 *   - No template alignment or tilt detection (scanner has fixed geometry)
 *   - Added sharpness (Laplacian variance) for blur rejection
 *   - Added fill ratio estimation
 *   - Downsamples to 240px width for speed (~1ms per call)
 *
 * The Sobel edge detection, grayscale conversion, and brightness contrast
 * logic are copied directly from detectCard() with the same coefficients.
 *
 * Usage:
 *   // Create once (reuses internal canvas):
 *   const analyzer = new FrameAnalyzer();
 *
 *   // In your rAF loop:
 *   const result = analyzer.analyzeFrame(video);
 *   // result.cardPresent  -- edges + contrast say there's a card
 *   // result.sharp         -- not motion blurred
 *   // result.fillRatio     -- how much of center region is card-like (0-1)
 *   // result.sharpness     -- raw Laplacian variance value
 *   // result.edgeDensity   -- fraction of center pixels with strong edges
 *   // result.contrast      -- brightness diff between center and border
 *   // result.hint          -- "Move closer" / "Hold steady" / "Ready"
 */

class FrameAnalyzer {
    /**
     * @param {Object} opts
     * @param {number} opts.sampleWidth       Downsample width (default 240)
     * @param {number} opts.edgeThreshold     Sobel magnitude to count as edge (default 40, same as detectCard)
     * @param {number} opts.minEdgeDensity    Min fraction of edge pixels for card present (default 0.03, same as detectCard)
     * @param {number} opts.minContrast       Min brightness diff center vs border (default 15, same as detectCard)
     * @param {number} opts.minSharpness      Min Laplacian variance for "sharp" (default 50)
     * @param {number} opts.minFillRatio      Min fill ratio to consider card filling frame (default 0.3)
     */
    constructor(opts = {}) {
        this.sampleWidth    = opts.sampleWidth    || 240;
        this.edgeThreshold  = opts.edgeThreshold  ?? 40;
        this.minEdgeDensity = opts.minEdgeDensity ?? 0.03;
        this.minContrast    = opts.minContrast    ?? 15;
        this.minSharpness   = opts.minSharpness   ?? 50;
        this.minFillRatio   = opts.minFillRatio   ?? 0.3;

        // Reusable scratch canvas
        this._sampleCanvas = null;
        this._sampleCtx    = null;
    }

    /**
     * Analyze a single video frame for card presence and quality.
     *
     * @param {HTMLVideoElement} video - Live video element
     * @returns {{
     *   cardPresent: boolean,
     *   sharp: boolean,
     *   fillRatio: number,
     *   sharpness: number,
     *   edgeDensity: number,
     *   contrast: number,
     *   hint: string
     * }}
     */
    analyzeFrame(video) {
        if (!video.videoWidth || !video.videoHeight) {
            return this._empty('Waiting for camera...');
        }

        // --- Downsample ---
        const sw = this.sampleWidth;
        const sh = Math.round(sw * (video.videoHeight / video.videoWidth));

        if (!this._sampleCanvas) {
            this._sampleCanvas = document.createElement('canvas');
            this._sampleCtx = this._sampleCanvas.getContext('2d', { willReadFrequently: true });
        }
        this._sampleCanvas.width = sw;
        this._sampleCanvas.height = sh;
        this._sampleCtx.drawImage(video, 0, 0, sw, sh);
        const imgData = this._sampleCtx.getImageData(0, 0, sw, sh);

        // --- Convert to grayscale ---
        // Same coefficients as detectCard: 0.299R + 0.587G + 0.114B
        const gray = new Uint8Array(sw * sh);
        for (let i = 0; i < gray.length; i++) {
            const j = i * 4;
            gray[i] = Math.round(
                0.299 * imgData.data[j] +
                0.587 * imgData.data[j + 1] +
                0.114 * imgData.data[j + 2]
            );
        }

        // --- Center region: 60% of frame (20%-80% X, 15%-85% Y) ---
        const cx1 = Math.round(sw * 0.2);
        const cx2 = Math.round(sw * 0.8);
        const cy1 = Math.round(sh * 0.15);
        const cy2 = Math.round(sh * 0.85);

        // --- 1. Sobel edge detection ---
        // Exact same Sobel operator as detectCard()
        const edges = new Uint8Array(sw * sh);
        for (let y = 1; y < sh - 1; y++) {
            for (let x = 1; x < sw - 1; x++) {
                const idx = y * sw + x;
                const gx = -gray[idx - sw - 1] + gray[idx - sw + 1]
                           - 2 * gray[idx - 1] + 2 * gray[idx + 1]
                           - gray[idx + sw - 1] + gray[idx + sw + 1];
                const gy = -gray[idx - sw - 1] - 2 * gray[idx - sw] - gray[idx - sw + 1]
                           + gray[idx + sw - 1] + 2 * gray[idx + sw] + gray[idx + sw + 1];
                edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
            }
        }

        // Edge density in center region (same threshold=40 as detectCard)
        let edgeCount = 0;
        let centerArea = 0;
        for (let y = cy1; y < cy2; y++) {
            for (let x = cx1; x < cx2; x++) {
                centerArea++;
                if (edges[y * sw + x] > this.edgeThreshold) {
                    edgeCount++;
                }
            }
        }
        const edgeDensity = edgeCount / Math.max(centerArea, 1);

        // --- 2. Sharpness via Laplacian variance ---
        // Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0]
        // Compute variance of Laplacian response in center region
        let lapSum = 0;
        let lapSumSq = 0;
        let lapCount = 0;
        for (let y = cy1 + 1; y < cy2 - 1; y++) {
            for (let x = cx1 + 1; x < cx2 - 1; x++) {
                const idx = y * sw + x;
                const lap = gray[idx - sw] + gray[idx + sw] +
                            gray[idx - 1] + gray[idx + 1] -
                            4 * gray[idx];
                lapSum += lap;
                lapSumSq += lap * lap;
                lapCount++;
            }
        }
        const lapMean = lapSum / Math.max(lapCount, 1);
        const sharpness = (lapSumSq / Math.max(lapCount, 1)) - (lapMean * lapMean);

        // --- 3. Brightness contrast: center vs border ---
        // Same approach as detectCard: inner 60% vs outer ring
        const innerMargin = 0.2; // same as detectCard
        let innerBright = 0, innerCount = 0;
        let outerBright = 0, outerCount = 0;

        for (let y = cy1; y < cy2; y++) {
            for (let x = cx1; x < cx2; x++) {
                const relX = (x - cx1) / (cx2 - cx1);
                const relY = (y - cy1) / (cy2 - cy1);
                const inside = relX > innerMargin && relX < (1 - innerMargin) &&
                               relY > innerMargin && relY < (1 - innerMargin);
                const pixel = gray[y * sw + x];
                if (inside) {
                    innerBright += pixel;
                    innerCount++;
                } else {
                    outerBright += pixel;
                    outerCount++;
                }
            }
        }

        const avgInner = innerBright / Math.max(innerCount, 1);
        const avgOuter = outerBright / Math.max(outerCount, 1);
        const contrast = Math.abs(avgInner - avgOuter);

        // --- 4. Fill ratio: fraction of center region with edges ---
        // Use a lower threshold to catch card artwork edges (not just borders)
        let fillEdges = 0;
        const fillThreshold = 20; // lower than border detection
        for (let y = cy1; y < cy2; y++) {
            for (let x = cx1; x < cx2; x++) {
                if (edges[y * sw + x] > fillThreshold) {
                    fillEdges++;
                }
            }
        }
        const fillRatio = fillEdges / Math.max(centerArea, 1);

        // --- Decision logic ---
        const hasEdges = edgeDensity >= this.minEdgeDensity;
        const hasContrast = contrast >= this.minContrast;
        const isSharp = sharpness >= this.minSharpness;
        const hasFill = fillRatio >= this.minFillRatio;

        const cardPresent = hasEdges && hasContrast;

        // Build hint
        let hint;
        if (!hasEdges) {
            hint = 'No card detected';
        } else if (!hasContrast) {
            hint = 'Move closer';
        } else if (!isSharp) {
            hint = 'Hold steady';
        } else if (!hasFill) {
            hint = 'Move closer';
        } else {
            hint = 'Ready';
        }

        return {
            cardPresent,
            sharp:       isSharp,
            fillRatio:   Math.round(fillRatio * 1000) / 1000,
            sharpness:   Math.round(sharpness * 10) / 10,
            edgeDensity: Math.round(edgeDensity * 1000) / 1000,
            contrast:    Math.round(contrast * 10) / 10,
            hint
        };
    }

    /**
     * Empty result for when video isn't ready.
     */
    _empty(hint) {
        return {
            cardPresent: false,
            sharp:       false,
            fillRatio:   0,
            sharpness:   0,
            edgeDensity: 0,
            contrast:    0,
            hint
        };
    }
}


// ---------------------------------------------------------------------------
// Self-test (run in browser console: FrameAnalyzer.selfTest())
// ---------------------------------------------------------------------------

FrameAnalyzer.selfTest = function () {
    let passed = 0;
    let failed = 0;

    function assert(cond, name) {
        if (cond) { console.log('  PASS: ' + name); passed++; }
        else      { console.error('  FAIL: ' + name); failed++; }
    }

    console.log('=== FrameAnalyzer Self-Test ===');

    const fa = new FrameAnalyzer();

    // --- Empty result when no video dimensions ---
    {
        const fakeVideo = { videoWidth: 0, videoHeight: 0 };
        const r = fa.analyzeFrame(fakeVideo);
        assert(r.cardPresent === false, 'no video -> not present');
        assert(r.hint === 'Waiting for camera...', 'no video -> waiting hint');
    }

    // --- Sobel on uniform gray -> no edges ---
    {
        // Test the core logic by creating a synthetic gray array
        const sw = 60, sh = 40;
        const gray = new Uint8Array(sw * sh).fill(128);

        // Compute Sobel (same as analyzeFrame)
        const edges = new Uint8Array(sw * sh);
        for (let y = 1; y < sh - 1; y++) {
            for (let x = 1; x < sw - 1; x++) {
                const idx = y * sw + x;
                const gx = -gray[idx - sw - 1] + gray[idx - sw + 1]
                           - 2 * gray[idx - 1] + 2 * gray[idx + 1]
                           - gray[idx + sw - 1] + gray[idx + sw + 1];
                const gy = -gray[idx - sw - 1] - 2 * gray[idx - sw] - gray[idx - sw + 1]
                           + gray[idx + sw - 1] + 2 * gray[idx + sw] + gray[idx + sw + 1];
                edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
            }
        }

        let anyEdge = false;
        for (let i = 0; i < edges.length; i++) {
            if (edges[i] > 0) { anyEdge = true; break; }
        }
        assert(!anyEdge, 'uniform gray -> zero Sobel edges');
    }

    // --- Sobel on sharp edge -> detects edge ---
    {
        const sw = 60, sh = 40;
        const gray = new Uint8Array(sw * sh);
        // Left half = 50, right half = 200 -> sharp vertical edge at x=30
        for (let y = 0; y < sh; y++) {
            for (let x = 0; x < sw; x++) {
                gray[y * sw + x] = x < 30 ? 50 : 200;
            }
        }

        const edges = new Uint8Array(sw * sh);
        for (let y = 1; y < sh - 1; y++) {
            for (let x = 1; x < sw - 1; x++) {
                const idx = y * sw + x;
                const gx = -gray[idx - sw - 1] + gray[idx - sw + 1]
                           - 2 * gray[idx - 1] + 2 * gray[idx + 1]
                           - gray[idx + sw - 1] + gray[idx + sw + 1];
                const gy = -gray[idx - sw - 1] - 2 * gray[idx - sw] - gray[idx - sw + 1]
                           + gray[idx + sw - 1] + 2 * gray[idx + sw] + gray[idx + sw + 1];
                edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
            }
        }

        // Check that the edge at x=29,30 has high magnitude
        const edgeVal = edges[20 * sw + 30]; // row 20, at the boundary
        assert(edgeVal > 100, 'sharp edge -> high Sobel magnitude (got ' + edgeVal + ')');
    }

    // --- Laplacian variance: uniform -> 0, textured -> high ---
    {
        const sw = 20, sh = 20;
        const uniform = new Uint8Array(sw * sh).fill(128);
        let lapSum = 0, lapSumSq = 0, lapCount = 0;
        for (let y = 1; y < sh - 1; y++) {
            for (let x = 1; x < sw - 1; x++) {
                const idx = y * sw + x;
                const lap = uniform[idx - sw] + uniform[idx + sw] +
                            uniform[idx - 1] + uniform[idx + 1] -
                            4 * uniform[idx];
                lapSum += lap;
                lapSumSq += lap * lap;
                lapCount++;
            }
        }
        const uniformSharpness = (lapSumSq / lapCount) - (lapSum / lapCount) ** 2;
        assert(uniformSharpness === 0, 'uniform gray -> Laplacian variance 0');

        // Checkerboard pattern -> high variance
        const checker = new Uint8Array(sw * sh);
        for (let y = 0; y < sh; y++) {
            for (let x = 0; x < sw; x++) {
                checker[y * sw + x] = ((x + y) % 2 === 0) ? 50 : 200;
            }
        }
        lapSum = 0; lapSumSq = 0; lapCount = 0;
        for (let y = 1; y < sh - 1; y++) {
            for (let x = 1; x < sw - 1; x++) {
                const idx = y * sw + x;
                const lap = checker[idx - sw] + checker[idx + sw] +
                            checker[idx - 1] + checker[idx + 1] -
                            4 * checker[idx];
                lapSum += lap;
                lapSumSq += lap * lap;
                lapCount++;
            }
        }
        const checkerSharpness = (lapSumSq / lapCount) - (lapSum / lapCount) ** 2;
        assert(checkerSharpness > 1000, 'checkerboard -> high Laplacian variance (got ' + checkerSharpness + ')');
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FrameAnalyzer;
}
