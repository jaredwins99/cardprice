/**
 * SharpnessDetector — client-side motion blur detection for slide-scan camera.
 *
 * Analyzes video frames to determine if they are sharp enough for OCR and
 * DINOv2 matching.  Combines three signals:
 *   1. Laplacian variance (edge density)
 *   2. Gradient magnitude via Sobel (edge strength)
 *   3. High-frequency energy (blur kills high frequencies)
 *
 * Also detects motion blur direction by comparing horizontal vs vertical
 * gradient energy, and tracks a rolling sharpness average so callers can
 * distinguish "user is sliding" from "user is stationary."
 *
 * Usage:
 *   const detector = new SharpnessDetector();
 *   // on each candidate frame (every ~3rd frame):
 *   const result = detector.analyze(videoElement);
 *   // result = { sharp, score, laplacian, gradient, hfEnergy,
 *   //            blurDirection, blurAnisotropy, motionState, rollingAvg }
 */
class SharpnessDetector {
    /**
     * @param {object} opts
     * @param {number} opts.targetWidth      Downsample long edge to this (default 320)
     * @param {number} opts.laplacianWeight  Weight for laplacian variance (default 0.45)
     * @param {number} opts.gradientWeight   Weight for gradient magnitude  (default 0.30)
     * @param {number} opts.hfWeight         Weight for high-freq energy    (default 0.25)
     * @param {number} opts.sharpThreshold   Combined score above this = sharp (default 100)
     * @param {number} opts.rollingSize      Rolling window size (default 10)
     * @param {number} opts.anisotropyThresh Ratio threshold for directional blur (default 2.0)
     */
    constructor(opts = {}) {
        this.targetWidth      = opts.targetWidth      || 320;
        this.laplacianWeight  = opts.laplacianWeight  || 0.45;
        this.gradientWeight   = opts.gradientWeight   || 0.30;
        this.hfWeight         = opts.hfWeight         || 0.25;
        this.sharpThreshold   = opts.sharpThreshold   || 100;
        this.rollingSize      = opts.rollingSize      || 10;
        this.anisotropyThresh = opts.anisotropyThresh || 2.0;

        // Rolling window of recent scores
        this._history = [];

        // Reusable canvas for downsampling
        this._canvas = null;
        this._ctx    = null;
    }

    // ------------------------------------------------------------------ public

    /**
     * Analyze a video element (or canvas/image) for sharpness.
     *
     * @param {HTMLVideoElement|HTMLCanvasElement|HTMLImageElement} source
     * @returns {{ sharp: boolean, score: number, threshold: number,
     *             laplacian: number, gradient: number, hfEnergy: number,
     *             blurDirection: string, blurAnisotropy: number,
     *             motionState: string, rollingAvg: number }}
     */
    analyze(source) {
        const gray = this._getGrayscale(source);
        const w = gray.width;
        const h = gray.height;
        const px = gray.data;  // Uint8Array, one byte per pixel

        // 1. Laplacian variance
        const laplacian = this._laplacianVariance(px, w, h);

        // 2. Sobel gradient magnitude + directional components
        const { total: gradient, horizontal: gH, vertical: gV } =
            this._sobelGradient(px, w, h);

        // 3. High-frequency energy
        const hfEnergy = this._highFreqEnergy(px, w, h);

        // Weighted combination (each metric is in roughly the same 0-500 range
        // for typical card images, but we normalize by dividing by the threshold
        // contribution so the combined score has threshold = sharpThreshold)
        const score = laplacian  * this.laplacianWeight
                    + gradient   * this.gradientWeight
                    + hfEnergy   * this.hfWeight;

        const sharp = score >= this.sharpThreshold;

        // Directional blur analysis
        const anisotropy = (gV > 0.001) ? gH / gV : 999;
        const invAniso   = (gH > 0.001) ? gV / gH : 999;
        let blurDirection = 'none';
        if (!sharp) {
            if (anisotropy < 1 / this.anisotropyThresh) {
                blurDirection = 'horizontal';  // horizontal edges lost → horizontal motion
            } else if (invAniso < 1 / this.anisotropyThresh) {
                blurDirection = 'vertical';
            } else {
                blurDirection = 'uniform';     // general defocus or omnidirectional blur
            }
        }

        // Rolling average
        this._history.push(score);
        if (this._history.length > this.rollingSize) {
            this._history.shift();
        }
        const rollingAvg = this._history.reduce((a, b) => a + b, 0) / this._history.length;

        // Motion state inference
        let motionState = 'stationary';
        if (this._history.length >= 3) {
            const recentSharp = this._history.slice(-3).filter(s => s >= this.sharpThreshold).length;
            if (recentSharp === 0) {
                motionState = 'sliding';
            } else if (recentSharp < 3) {
                motionState = 'settling';   // transitioning from motion to still
            }
        }

        return {
            sharp,
            score:     Math.round(score * 10) / 10,
            threshold: this.sharpThreshold,
            laplacian: Math.round(laplacian * 10) / 10,
            gradient:  Math.round(gradient  * 10) / 10,
            hfEnergy:  Math.round(hfEnergy  * 10) / 10,
            blurDirection,
            blurAnisotropy: Math.round(Math.min(anisotropy, invAniso) * 100) / 100,
            motionState,
            rollingAvg: Math.round(rollingAvg * 10) / 10,
        };
    }

    /** Reset rolling history (e.g. when starting a new scan session). */
    reset() {
        this._history = [];
    }

    // ----------------------------------------------------------------- private

    /**
     * Downsample source to targetWidth and return single-channel grayscale.
     * Returns { data: Uint8Array, width, height }.
     */
    _getGrayscale(source) {
        let srcW, srcH;
        if (source instanceof HTMLVideoElement) {
            srcW = source.videoWidth;
            srcH = source.videoHeight;
        } else {
            srcW = source.width;
            srcH = source.height;
        }

        // Compute downsampled dimensions
        const scale = this.targetWidth / Math.max(srcW, srcH);
        const w = Math.round(srcW * scale);
        const h = Math.round(srcH * scale);

        // Lazily create / resize canvas
        if (!this._canvas || this._canvas.width !== w || this._canvas.height !== h) {
            this._canvas = document.createElement('canvas');
            this._canvas.width  = w;
            this._canvas.height = h;
            this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
        }

        this._ctx.drawImage(source, 0, 0, w, h);
        const rgba = this._ctx.getImageData(0, 0, w, h).data;

        // Convert to grayscale (luminosity formula)
        const gray = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
            gray[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
        }

        return { data: gray, width: w, height: h };
    }

    /**
     * Laplacian variance: convolve with 3x3 Laplacian kernel, return variance.
     * Kernel: [0 1 0; 1 -4 1; 0 1 0]
     * Sharp images: 200-500+.  Motion-blurred: 20-80.
     */
    _laplacianVariance(px, w, h) {
        let sum = 0;
        let sumSq = 0;
        let n = 0;

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                const lap = -4 * px[idx]
                           + px[idx - 1] + px[idx + 1]
                           + px[idx - w] + px[idx + w];
                sum   += lap;
                sumSq += lap * lap;
                n++;
            }
        }

        const mean = sum / n;
        return (sumSq / n) - (mean * mean);  // variance
    }

    /**
     * Sobel gradient magnitude.  Returns total, horizontal, and vertical
     * average absolute gradient.
     *
     * Sobel-X: [-1 0 1; -2 0 2; -1 0 1]  → detects vertical edges
     * Sobel-Y: [-1 -2 -1; 0 0 0; 1 2 1]  → detects horizontal edges
     */
    _sobelGradient(px, w, h) {
        let sumH = 0;   // Sobel-X magnitude (vertical edge detector)
        let sumV = 0;   // Sobel-Y magnitude (horizontal edge detector)
        let n = 0;

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const tl = px[(y - 1) * w + (x - 1)];
                const tc = px[(y - 1) * w + x];
                const tr = px[(y - 1) * w + (x + 1)];
                const ml = px[y * w + (x - 1)];
                const mr = px[y * w + (x + 1)];
                const bl = px[(y + 1) * w + (x - 1)];
                const bc = px[(y + 1) * w + x];
                const br = px[(y + 1) * w + (x + 1)];

                // Sobel-X (detects vertical edges — survives horizontal motion)
                const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
                // Sobel-Y (detects horizontal edges — survives vertical motion)
                const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;

                sumH += Math.abs(gx);
                sumV += Math.abs(gy);
                n++;
            }
        }

        // Normalize to per-pixel average (scale to roughly 0-500 range)
        const normH = (sumH / n);
        const normV = (sumV / n);
        const total = (sumH + sumV) / n;

        return { total, horizontal: normH, vertical: normV };
    }

    /**
     * High-frequency energy: subtract a box-blurred version from the original,
     * measure variance of the difference.  Motion blur removes high frequencies
     * so the difference will be small.
     *
     * Uses a fast 5x5 box blur (separable: two 1D passes).
     */
    _highFreqEnergy(px, w, h) {
        const size = w * h;

        // --- Horizontal pass (radius 2, kernel width 5) ---
        const tmp = new Float32Array(size);
        for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
                let sum = 0;
                let cnt = 0;
                const x0 = Math.max(0, x - 2);
                const x1 = Math.min(w - 1, x + 2);
                for (let xx = x0; xx <= x1; xx++) {
                    sum += px[row + xx];
                    cnt++;
                }
                tmp[row + x] = sum / cnt;
            }
        }

        // --- Vertical pass ---
        const blurred = new Float32Array(size);
        for (let x = 0; x < w; x++) {
            for (let y = 0; y < h; y++) {
                let sum = 0;
                let cnt = 0;
                const y0 = Math.max(0, y - 2);
                const y1 = Math.min(h - 1, y + 2);
                for (let yy = y0; yy <= y1; yy++) {
                    sum += tmp[yy * w + x];
                    cnt++;
                }
                blurred[y * w + x] = sum / cnt;
            }
        }

        // --- High-pass = original - blurred, compute variance ---
        let sum = 0;
        let sumSq = 0;
        for (let i = 0; i < size; i++) {
            const diff = px[i] - blurred[i];
            sum   += diff;
            sumSq += diff * diff;
        }
        const mean = sum / size;
        return (sumSq / size) - (mean * mean);
    }
}


// --------------------------------------------------------------------------
// Standalone helper matching the requested API
// --------------------------------------------------------------------------

/**
 * Simple function API: returns { sharp, score, threshold }.
 * For richer data, use SharpnessDetector.analyze() directly.
 */
function isSharp(imageData, width, height) {
    // imageData can be an ImageData object or a Uint8ClampedArray of RGBA pixels
    const rgba = imageData instanceof ImageData ? imageData.data : imageData;

    // Convert to grayscale
    const gray = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
        gray[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
    }

    // Laplacian variance
    let sum = 0, sumSq = 0, n = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            const lap = -4 * gray[idx]
                       + gray[idx - 1] + gray[idx + 1]
                       + gray[idx - width] + gray[idx + width];
            sum += lap;
            sumSq += lap * lap;
            n++;
        }
    }
    const lapMean = sum / n;
    const laplacian = (sumSq / n) - (lapMean * lapMean);

    // Sobel gradient magnitude
    let gSum = 0;
    n = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const tl = gray[(y-1)*width+(x-1)], tc = gray[(y-1)*width+x], tr = gray[(y-1)*width+(x+1)];
            const ml = gray[y*width+(x-1)], mr = gray[y*width+(x+1)];
            const bl = gray[(y+1)*width+(x-1)], bc = gray[(y+1)*width+x], br = gray[(y+1)*width+(x+1)];
            const gx = -tl + tr - 2*ml + 2*mr - bl + br;
            const gy = -tl - 2*tc - tr + bl + 2*bc + br;
            gSum += Math.abs(gx) + Math.abs(gy);
            n++;
        }
    }
    const gradient = gSum / n;

    // Weighted score (simplified — no HF energy for the lightweight version)
    const threshold = 100;
    const score = laplacian * 0.55 + gradient * 0.45;
    return { sharp: score >= threshold, score: Math.round(score * 10) / 10, threshold };
}
