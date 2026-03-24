/**
 * BinderDetector — Detects binder page background color and produces binary
 * card/background masks to help the contour detector separate cards from the
 * binder page.
 *
 * When scanning a binder page close-up, the camera sees:
 *   - Cards (varied colors, artwork)
 *   - Binder page background (solid orange, blue, black, or white)
 *   - Sleeve edges (clear/reflective — high V, low S)
 *   - Binder ring shadow (dark band in the middle)
 *
 * This class samples the frame edges (where background is most visible),
 * identifies the dominant background color via a lightweight k=2 clustering,
 * classifies it as orange/blue/black/white, and produces a binary mask where
 * 1 = likely card, 0 = likely background.
 *
 * Designed to run every few frames, not every frame — results are cached.
 *
 * Usage:
 *   const detector = new BinderDetector();
 *
 *   // Every ~5 frames:
 *   const bg = detector.detectBackground(imageData, width, height);
 *   // bg.color   — {r, g, b} dominant background color
 *   // bg.type    — 'orange' | 'blue' | 'black' | 'white' | 'unknown'
 *   // bg.mask    — Uint8Array, 1 = background, 0 = card
 *   // bg.confidence — 0-1
 *
 *   // Use cached background to generate a card mask:
 *   const cardMask = detector.createCardMask(imageData, width, height);
 *   // Uint8Array where 1 = likely card, 0 = likely background
 */

class BinderDetector {
    /**
     * @param {Object} opts
     * @param {number} opts.edgePct          Fraction of frame edges to sample (default 0.10)
     * @param {number} opts.colorTolerance   Max Euclidean RGB distance for "similar to background" (default 55)
     * @param {number} opts.morphRadius      Radius for morphological cleanup (default 2)
     * @param {number} opts.kmeansIterations Max k-means iterations (default 8)
     * @param {number} opts.sampleStride     Pixel stride for k-means sampling (default 4)
     */
    constructor(opts = {}) {
        this.edgePct          = opts.edgePct          ?? 0.10;
        this.colorTolerance   = opts.colorTolerance   ?? 55;
        this.morphRadius      = opts.morphRadius      ?? 2;
        this.kmeansIterations = opts.kmeansIterations ?? 8;
        this.sampleStride     = opts.sampleStride     ?? 4;

        // Cached detection result
        this.binderColor = null;  // {r, g, b}
        this.binderType  = null;  // 'orange' | 'blue' | 'black' | 'white' | 'unknown'
        this._lastConfidence = 0;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Analyze a frame to detect the binder background color and produce a mask.
     *
     * @param {Uint8ClampedArray} imageData  RGBA pixel data (from getImageData().data)
     * @param {number} width
     * @param {number} height
     * @returns {{
     *   color: {r: number, g: number, b: number},
     *   type: string,
     *   mask: Uint8Array,
     *   confidence: number
     * }}
     */
    detectBackground(imageData, width, height) {
        // Step 1: Collect edge-region pixels and all-region pixels
        const edgePixels = this._sampleEdgePixels(imageData, width, height);
        const allPixels  = this._sampleAllPixels(imageData, width, height);

        // Step 2: K-means with k=2 on all sampled pixels
        const clusters = this._kMeans2(allPixels);

        // Step 3: The cluster appearing more in edge regions = background
        const edgeCounts = [0, 0];
        for (let i = 0; i < edgePixels.length; i += 3) {
            const r = edgePixels[i], g = edgePixels[i + 1], b = edgePixels[i + 2];
            const d0 = this._colorDistSq(r, g, b, clusters[0].r, clusters[0].g, clusters[0].b);
            const d1 = this._colorDistSq(r, g, b, clusters[1].r, clusters[1].g, clusters[1].b);
            edgeCounts[d0 <= d1 ? 0 : 1]++;
        }

        const bgIdx = edgeCounts[0] >= edgeCounts[1] ? 0 : 1;
        const bgColor = clusters[bgIdx];

        // Step 4: Classify the background color
        const bgType = this._classifyColor(bgColor.r, bgColor.g, bgColor.b);

        // Confidence: how dominant is the background cluster in edge regions?
        const totalEdge = edgeCounts[0] + edgeCounts[1];
        const confidence = totalEdge > 0
            ? edgeCounts[bgIdx] / totalEdge
            : 0;

        // Cache for createCardMask
        this.binderColor = { r: bgColor.r, g: bgColor.g, b: bgColor.b };
        this.binderType  = bgType;
        this._lastConfidence = confidence;

        // Step 5: Create background mask (1 = background, 0 = card)
        const mask = this._buildMask(imageData, width, height, bgColor);

        // Step 6: Morphological cleanup
        this._morphClose(mask, width, height, this.morphRadius);
        this._morphOpen(mask, width, height, this.morphRadius);

        return {
            color: { r: bgColor.r, g: bgColor.g, b: bgColor.b },
            type: bgType,
            mask,
            confidence: Math.round(confidence * 1000) / 1000,
        };
    }

    /**
     * Create a card mask using the previously detected background color.
     * Call detectBackground() first (at least once) to calibrate.
     *
     * @param {Uint8ClampedArray} imageData  RGBA pixel data
     * @param {number} width
     * @param {number} height
     * @returns {Uint8Array}  1 = likely card, 0 = likely background
     */
    createCardMask(imageData, width, height) {
        if (!this.binderColor) {
            // Not calibrated — return all-card mask
            return new Uint8Array(width * height).fill(1);
        }

        const bgMask = this._buildMask(imageData, width, height, this.binderColor);
        this._morphClose(bgMask, width, height, this.morphRadius);
        this._morphOpen(bgMask, width, height, this.morphRadius);

        // Invert: background mask -> card mask
        const cardMask = new Uint8Array(width * height);
        for (let i = 0; i < cardMask.length; i++) {
            cardMask[i] = bgMask[i] === 0 ? 1 : 0;
        }
        return cardMask;
    }

    /**
     * Get the last detected background info without recomputing.
     * @returns {{ color: {r,g,b}|null, type: string|null, confidence: number }}
     */
    getLastResult() {
        return {
            color: this.binderColor,
            type: this.binderType,
            confidence: this._lastConfidence,
        };
    }

    /**
     * Reset cached detection (e.g., when switching binder pages).
     */
    reset() {
        this.binderColor = null;
        this.binderType = null;
        this._lastConfidence = 0;
    }

    // -----------------------------------------------------------------------
    // Edge and pixel sampling
    // -----------------------------------------------------------------------

    /**
     * Sample RGB values from the frame edge strips (top/bottom/left/right).
     * These regions are most likely to show binder background material.
     *
     * @returns {Uint8Array}  Packed [r,g,b, r,g,b, ...] triplets
     */
    _sampleEdgePixels(imageData, width, height) {
        const edgeX = Math.max(1, Math.floor(width * this.edgePct));
        const edgeY = Math.max(1, Math.floor(height * this.edgePct));
        const stride = this.sampleStride;

        // Estimate max pixels (oversize is fine, we track actual count)
        const maxPixels = Math.ceil(width * edgeY * 2 / (stride * stride))
                        + Math.ceil(edgeX * height * 2 / (stride * stride));
        const result = new Uint8Array(maxPixels * 3);
        let count = 0;

        // Top strip
        for (let y = 0; y < edgeY; y += stride) {
            for (let x = 0; x < width; x += stride) {
                const idx = (y * width + x) * 4;
                result[count++] = imageData[idx];
                result[count++] = imageData[idx + 1];
                result[count++] = imageData[idx + 2];
            }
        }

        // Bottom strip
        for (let y = height - edgeY; y < height; y += stride) {
            for (let x = 0; x < width; x += stride) {
                const idx = (y * width + x) * 4;
                result[count++] = imageData[idx];
                result[count++] = imageData[idx + 1];
                result[count++] = imageData[idx + 2];
            }
        }

        // Left strip (excluding corners already covered)
        for (let y = edgeY; y < height - edgeY; y += stride) {
            for (let x = 0; x < edgeX; x += stride) {
                const idx = (y * width + x) * 4;
                result[count++] = imageData[idx];
                result[count++] = imageData[idx + 1];
                result[count++] = imageData[idx + 2];
            }
        }

        // Right strip (excluding corners already covered)
        for (let y = edgeY; y < height - edgeY; y += stride) {
            for (let x = width - edgeX; x < width; x += stride) {
                const idx = (y * width + x) * 4;
                result[count++] = imageData[idx];
                result[count++] = imageData[idx + 1];
                result[count++] = imageData[idx + 2];
            }
        }

        return result.subarray(0, count);
    }

    /**
     * Sample RGB values from the entire frame at stride intervals.
     * Used as input to k-means clustering.
     *
     * @returns {Uint8Array}  Packed [r,g,b, r,g,b, ...] triplets
     */
    _sampleAllPixels(imageData, width, height) {
        const stride = this.sampleStride;
        const maxPixels = Math.ceil(width / stride) * Math.ceil(height / stride);
        const result = new Uint8Array(maxPixels * 3);
        let count = 0;

        for (let y = 0; y < height; y += stride) {
            for (let x = 0; x < width; x += stride) {
                const idx = (y * width + x) * 4;
                result[count++] = imageData[idx];
                result[count++] = imageData[idx + 1];
                result[count++] = imageData[idx + 2];
            }
        }

        return result.subarray(0, count);
    }

    // -----------------------------------------------------------------------
    // K-means clustering (k=2)
    // -----------------------------------------------------------------------

    /**
     * Lightweight k-means with k=2 on packed RGB triplets.
     * Returns two cluster centers sorted by total membership count.
     *
     * @param {Uint8Array} pixels  Packed [r,g,b, ...] triplets
     * @returns {Array<{r: number, g: number, b: number, count: number}>}
     */
    _kMeans2(pixels) {
        const n = pixels.length / 3;
        if (n < 2) {
            return [
                { r: 0, g: 0, b: 0, count: 0 },
                { r: 255, g: 255, b: 255, count: 0 },
            ];
        }

        // Initialize centroids: pick the first pixel and the most-distant pixel
        let c0r = pixels[0], c0g = pixels[1], c0b = pixels[2];
        let c1r = c0r, c1g = c0g, c1b = c0b;
        let maxDist = 0;

        for (let i = 0; i < n; i++) {
            const off = i * 3;
            const d = this._colorDistSq(pixels[off], pixels[off + 1], pixels[off + 2], c0r, c0g, c0b);
            if (d > maxDist) {
                maxDist = d;
                c1r = pixels[off];
                c1g = pixels[off + 1];
                c1b = pixels[off + 2];
            }
        }

        // Iterate
        for (let iter = 0; iter < this.kmeansIterations; iter++) {
            let s0r = 0, s0g = 0, s0b = 0, cnt0 = 0;
            let s1r = 0, s1g = 0, s1b = 0, cnt1 = 0;

            for (let i = 0; i < n; i++) {
                const off = i * 3;
                const r = pixels[off], g = pixels[off + 1], b = pixels[off + 2];
                const d0 = this._colorDistSq(r, g, b, c0r, c0g, c0b);
                const d1 = this._colorDistSq(r, g, b, c1r, c1g, c1b);

                if (d0 <= d1) {
                    s0r += r; s0g += g; s0b += b; cnt0++;
                } else {
                    s1r += r; s1g += g; s1b += b; cnt1++;
                }
            }

            if (cnt0 > 0) {
                c0r = Math.round(s0r / cnt0);
                c0g = Math.round(s0g / cnt0);
                c0b = Math.round(s0b / cnt0);
            }
            if (cnt1 > 0) {
                c1r = Math.round(s1r / cnt1);
                c1g = Math.round(s1g / cnt1);
                c1b = Math.round(s1b / cnt1);
            }
        }

        // Final assignment counts
        let cnt0 = 0, cnt1 = 0;
        for (let i = 0; i < n; i++) {
            const off = i * 3;
            const d0 = this._colorDistSq(pixels[off], pixels[off + 1], pixels[off + 2], c0r, c0g, c0b);
            const d1 = this._colorDistSq(pixels[off], pixels[off + 1], pixels[off + 2], c1r, c1g, c1b);
            if (d0 <= d1) cnt0++; else cnt1++;
        }

        return [
            { r: c0r, g: c0g, b: c0b, count: cnt0 },
            { r: c1r, g: c1g, b: c1b, count: cnt1 },
        ];
    }

    // -----------------------------------------------------------------------
    // Color classification
    // -----------------------------------------------------------------------

    /**
     * Classify an RGB color as a binder type using HSV analysis.
     *
     * @param {number} r  0-255
     * @param {number} g  0-255
     * @param {number} b  0-255
     * @returns {string}  'orange' | 'blue' | 'black' | 'white' | 'unknown'
     */
    _classifyColor(r, g, b) {
        const [h, s, v] = BinderDetector._rgbToHsv(r, g, b);

        // Black: low brightness regardless of hue/saturation
        if (v < 60) return 'black';

        // White: low saturation + high brightness
        if (s < 30 && v > 200) return 'white';

        // Orange: hue 10-25 (on 0-180 OpenCV scale) = 20-50 on 0-360 scale,
        // with decent saturation
        // Using 0-360 scale: hue 10-50, saturation > 100 (out of 255)
        if (h >= 10 && h <= 50 && s > 100) return 'orange';

        // Blue: hue 200-260 on 0-360 scale, saturation > 60
        if (h >= 200 && h <= 260 && s > 60) return 'blue';

        return 'unknown';
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

        const v = max;
        const s = max === 0 ? 0 : Math.round((delta / max) * 255);

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
    // Mask building
    // -----------------------------------------------------------------------

    /**
     * Build a binary mask: 1 = background (similar to bgColor), 0 = not background.
     *
     * Uses Euclidean distance in RGB space with colorTolerance threshold.
     * Also marks sleeve edges (high V, low S) as background since they are
     * transparent binder material, not card content.
     *
     * @param {Uint8ClampedArray} imageData  RGBA pixels
     * @param {number} width
     * @param {number} height
     * @param {{r: number, g: number, b: number}} bgColor
     * @returns {Uint8Array}  1 = background, 0 = not background
     */
    _buildMask(imageData, width, height, bgColor) {
        const mask = new Uint8Array(width * height);
        const tolSq = this.colorTolerance * this.colorTolerance;
        const bgR = bgColor.r, bgG = bgColor.g, bgB = bgColor.b;

        for (let i = 0; i < width * height; i++) {
            const idx = i * 4;
            const r = imageData[idx];
            const g = imageData[idx + 1];
            const b = imageData[idx + 2];

            // Check if pixel is similar to background color
            const dr = r - bgR;
            const dg = g - bgG;
            const db = b - bgB;
            if (dr * dr + dg * dg + db * db <= tolSq) {
                mask[i] = 1;
                continue;
            }

            // Check for sleeve edges: high brightness, low saturation, reflective
            // These are transparent sleeve material catching light
            const max = Math.max(r, g, b);
            const min = Math.min(r, g, b);
            const delta = max - min;
            const sat = max === 0 ? 0 : (delta / max) * 255;

            if (max > 220 && sat < 20) {
                // Bright near-white highlight — likely sleeve reflection
                mask[i] = 1;
            }
        }

        return mask;
    }

    // -----------------------------------------------------------------------
    // Morphological operations (binary mask cleanup)
    // -----------------------------------------------------------------------

    /**
     * Morphological close (dilate then erode) to fill small holes.
     * Operates in-place on the mask.
     */
    _morphClose(mask, width, height, radius) {
        const temp = this._dilate(mask, width, height, radius);
        const result = this._erode(temp, width, height, radius);
        for (let i = 0; i < mask.length; i++) mask[i] = result[i];
    }

    /**
     * Morphological open (erode then dilate) to remove small noise.
     * Operates in-place on the mask.
     */
    _morphOpen(mask, width, height, radius) {
        const temp = this._erode(mask, width, height, radius);
        const result = this._dilate(temp, width, height, radius);
        for (let i = 0; i < mask.length; i++) mask[i] = result[i];
    }

    /**
     * Binary dilation with a square structuring element.
     * A pixel is 1 if any pixel in its neighborhood is 1.
     */
    _dilate(mask, width, height, radius) {
        const out = new Uint8Array(width * height);

        for (let y = 0; y < height; y++) {
            const yMin = Math.max(0, y - radius);
            const yMax = Math.min(height - 1, y + radius);

            for (let x = 0; x < width; x++) {
                // Fast check: if center is already 1, no need to search
                if (mask[y * width + x] === 1) {
                    out[y * width + x] = 1;
                    continue;
                }

                const xMin = Math.max(0, x - radius);
                const xMax = Math.min(width - 1, x + radius);
                let found = false;

                for (let ny = yMin; ny <= yMax && !found; ny++) {
                    for (let nx = xMin; nx <= xMax && !found; nx++) {
                        if (mask[ny * width + nx] === 1) {
                            found = true;
                        }
                    }
                }

                out[y * width + x] = found ? 1 : 0;
            }
        }

        return out;
    }

    /**
     * Binary erosion with a square structuring element.
     * A pixel is 1 only if all pixels in its neighborhood are 1.
     */
    _erode(mask, width, height, radius) {
        const out = new Uint8Array(width * height);

        for (let y = 0; y < height; y++) {
            const yMin = Math.max(0, y - radius);
            const yMax = Math.min(height - 1, y + radius);

            for (let x = 0; x < width; x++) {
                // Fast check: if center is 0, result is 0
                if (mask[y * width + x] === 0) {
                    continue;
                }

                const xMin = Math.max(0, x - radius);
                const xMax = Math.min(width - 1, x + radius);
                let allSet = true;

                for (let ny = yMin; ny <= yMax && allSet; ny++) {
                    for (let nx = xMin; nx <= xMax && allSet; nx++) {
                        if (mask[ny * width + nx] === 0) {
                            allSet = false;
                        }
                    }
                }

                out[y * width + x] = allSet ? 1 : 0;
            }
        }

        return out;
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /**
     * Squared Euclidean distance between two RGB colors.
     */
    _colorDistSq(r1, g1, b1, r2, g2, b2) {
        const dr = r1 - r2;
        const dg = g1 - g2;
        const db = b1 - b2;
        return dr * dr + dg * dg + db * db;
    }
}


// ---------------------------------------------------------------------------
// Self-test (run in browser console: BinderDetector.selfTest())
// ---------------------------------------------------------------------------

BinderDetector.selfTest = function () {
    let passed = 0;
    let failed = 0;

    function assert(condition, name) {
        if (condition) {
            console.log('  PASS: ' + name);
            passed++;
        } else {
            console.error('  FAIL: ' + name);
            failed++;
        }
    }

    console.log('=== BinderDetector Self-Test ===');

    // --- Test RGB to HSV ---
    {
        const [h, s, v] = BinderDetector._rgbToHsv(255, 0, 0);
        assert(h === 0 && s === 255 && v === 255, 'pure red HSV: h=' + h + ' s=' + s + ' v=' + v);

        const [h2, s2, v2] = BinderDetector._rgbToHsv(0, 0, 255);
        assert(h2 === 240 && s2 === 255 && v2 === 255, 'pure blue HSV: h=' + h2);

        const [h3, s3, v3] = BinderDetector._rgbToHsv(0, 0, 0);
        assert(h3 === 0 && s3 === 0 && v3 === 0, 'black HSV');

        const [h4, s4, v4] = BinderDetector._rgbToHsv(255, 255, 255);
        assert(h4 === 0 && s4 === 0 && v4 === 255, 'white HSV');
    }

    // --- Test color classification ---
    {
        const bd = new BinderDetector();
        assert(bd._classifyColor(20, 20, 20) === 'black', 'dark -> black');
        assert(bd._classifyColor(240, 240, 240) === 'white', 'bright unsaturated -> white');
        assert(bd._classifyColor(40, 80, 200) === 'blue', 'blue binder color -> blue');
        assert(bd._classifyColor(220, 120, 30) === 'orange', 'orange binder color -> orange');
        assert(bd._classifyColor(0, 200, 0) === 'unknown', 'green -> unknown');
    }

    // --- Test k-means with two obvious clusters ---
    {
        const bd = new BinderDetector();
        // 50 dark blue pixels + 50 bright red pixels
        const pixels = new Uint8Array(100 * 3);
        for (let i = 0; i < 50; i++) {
            pixels[i * 3]     = 30;   // R
            pixels[i * 3 + 1] = 50;   // G
            pixels[i * 3 + 2] = 180;  // B
        }
        for (let i = 50; i < 100; i++) {
            pixels[i * 3]     = 220;  // R
            pixels[i * 3 + 1] = 60;   // G
            pixels[i * 3 + 2] = 40;   // B
        }

        const clusters = bd._kMeans2(pixels);
        assert(clusters.length === 2, 'returns 2 clusters');
        assert(clusters[0].count + clusters[1].count === 100, 'all pixels assigned');

        // One cluster should be near blue, the other near red
        const blueCluster = clusters[0].b > clusters[1].b ? clusters[0] : clusters[1];
        const redCluster  = clusters[0].r > clusters[1].r ? clusters[0] : clusters[1];
        assert(blueCluster.b > 150 && blueCluster.r < 60, 'blue cluster detected: r=' + blueCluster.r + ' b=' + blueCluster.b);
        assert(redCluster.r > 180 && redCluster.b < 80, 'red cluster detected: r=' + redCluster.r + ' b=' + redCluster.b);
    }

    // --- Test mask building with solid color frame ---
    {
        const bd = new BinderDetector({ colorTolerance: 50 });
        const w = 20, h = 10;
        const pixels = new Uint8ClampedArray(w * h * 4);

        // Fill with blue (background)
        for (let i = 0; i < w * h; i++) {
            pixels[i * 4]     = 40;
            pixels[i * 4 + 1] = 60;
            pixels[i * 4 + 2] = 180;
            pixels[i * 4 + 3] = 255;
        }

        const bgColor = { r: 40, g: 60, b: 180 };
        const mask = bd._buildMask(pixels, w, h, bgColor);

        let bgCount = 0;
        for (let i = 0; i < mask.length; i++) {
            if (mask[i] === 1) bgCount++;
        }
        assert(bgCount === w * h, 'solid blue frame -> all background (' + bgCount + '/' + (w * h) + ')');
    }

    // --- Test mask with mixed content ---
    {
        const bd = new BinderDetector({ colorTolerance: 50 });
        const w = 20, h = 10;
        const pixels = new Uint8ClampedArray(w * h * 4);

        // Left half: blue background
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = (y * w + x) * 4;
                if (x < 10) {
                    pixels[i] = 40; pixels[i + 1] = 60; pixels[i + 2] = 180;
                } else {
                    pixels[i] = 200; pixels[i + 1] = 50; pixels[i + 2] = 30;
                }
                pixels[i + 3] = 255;
            }
        }

        const bgColor = { r: 40, g: 60, b: 180 };
        const mask = bd._buildMask(pixels, w, h, bgColor);

        let leftBg = 0, rightBg = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (mask[y * w + x] === 1) {
                    if (x < 10) leftBg++; else rightBg++;
                }
            }
        }
        assert(leftBg === 100, 'left half (blue) -> all background (' + leftBg + ')');
        assert(rightBg === 0, 'right half (red card) -> no background (' + rightBg + ')');
    }

    // --- Test morphological operations ---
    {
        const bd = new BinderDetector();
        const w = 10, h = 10;

        // Single isolated pixel (noise) — open should remove it
        const mask = new Uint8Array(w * h);
        mask[5 * w + 5] = 1;

        bd._morphOpen(mask, w, h, 1);
        let sum = 0;
        for (let i = 0; i < mask.length; i++) sum += mask[i];
        assert(sum === 0, 'morph open removes isolated pixel');

        // Small hole in solid region — close should fill it
        const mask2 = new Uint8Array(w * h).fill(1);
        mask2[5 * w + 5] = 0;  // single hole

        bd._morphClose(mask2, w, h, 1);
        assert(mask2[5 * w + 5] === 1, 'morph close fills single-pixel hole');
    }

    // --- Test detectBackground with synthetic binder frame ---
    {
        const bd = new BinderDetector({ sampleStride: 1, morphRadius: 0 });
        const w = 40, h = 30;
        const pixels = new Uint8ClampedArray(w * h * 4);

        // Edges: blue binder background
        // Center: red card artwork
        const edgeX = Math.floor(w * 0.10);
        const edgeY = Math.floor(h * 0.10);

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = (y * w + x) * 4;
                const isEdge = y < edgeY || y >= h - edgeY || x < edgeX || x >= w - edgeX;
                if (isEdge) {
                    // Blue binder
                    pixels[i] = 30; pixels[i + 1] = 60; pixels[i + 2] = 200;
                } else {
                    // Red card
                    pixels[i] = 220; pixels[i + 1] = 40; pixels[i + 2] = 30;
                }
                pixels[i + 3] = 255;
            }
        }

        const result = bd.detectBackground(pixels, w, h);
        assert(result.type === 'blue', 'detected blue background (got ' + result.type + ')');
        assert(result.confidence > 0.5, 'confidence > 0.5 (got ' + result.confidence + ')');
        assert(result.mask.length === w * h, 'mask has correct size');

        // Edge pixels should be background (1), center should be card (0)
        assert(result.mask[0] === 1, 'top-left corner is background');
        const centerIdx = Math.floor(h / 2) * w + Math.floor(w / 2);
        assert(result.mask[centerIdx] === 0, 'center pixel is not background');
    }

    // --- Test createCardMask uses cached color ---
    {
        const bd = new BinderDetector({ colorTolerance: 50, morphRadius: 0 });
        // Manually set cached binder color
        bd.binderColor = { r: 40, g: 60, b: 180 };

        const w = 10, h = 10;
        const pixels = new Uint8ClampedArray(w * h * 4);

        // All blue -> should be all background -> card mask all 0
        for (let i = 0; i < w * h; i++) {
            pixels[i * 4] = 40; pixels[i * 4 + 1] = 60;
            pixels[i * 4 + 2] = 180; pixels[i * 4 + 3] = 255;
        }

        const cardMask = bd.createCardMask(pixels, w, h);
        let cardCount = 0;
        for (let i = 0; i < cardMask.length; i++) cardCount += cardMask[i];
        assert(cardCount === 0, 'all-blue frame -> card mask is all 0');

        // Not calibrated -> all card
        bd.reset();
        const cardMask2 = bd.createCardMask(pixels, w, h);
        let cardCount2 = 0;
        for (let i = 0; i < cardMask2.length; i++) cardCount2 += cardMask2[i];
        assert(cardCount2 === w * h, 'uncalibrated -> card mask is all 1');
    }

    // --- Test sleeve edge detection (high V, low S -> background) ---
    {
        const bd = new BinderDetector({ colorTolerance: 50, morphRadius: 0 });
        bd.binderColor = { r: 40, g: 60, b: 180 };

        const w = 10, h = 1;
        const pixels = new Uint8ClampedArray(w * h * 4);

        // Bright near-white pixel (sleeve reflection)
        pixels[0] = 245; pixels[1] = 248; pixels[2] = 250; pixels[3] = 255;
        // Colored card pixel
        pixels[4] = 200; pixels[5] = 50; pixels[6] = 30; pixels[7] = 255;

        const mask = bd._buildMask(pixels, w, h, bd.binderColor);
        assert(mask[0] === 1, 'sleeve reflection (245,248,250) -> background');
        assert(mask[1] === 0, 'red card pixel (200,50,30) -> not background');
    }

    // --- Test reset ---
    {
        const bd = new BinderDetector();
        bd.binderColor = { r: 40, g: 60, b: 180 };
        bd.binderType = 'blue';
        bd._lastConfidence = 0.9;

        bd.reset();
        assert(bd.binderColor === null, 'reset clears binderColor');
        assert(bd.binderType === null, 'reset clears binderType');
        assert(bd._lastConfidence === 0, 'reset clears confidence');
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BinderDetector;
}
