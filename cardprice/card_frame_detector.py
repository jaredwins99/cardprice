"""Card frame detector for close-range video capture.

Detects a single Pokemon card in a phone camera frame when the phone is
held ~15cm from the binder page. At this distance a single card fills
60-80% of the frame.

Exports CARD_FRAME_DETECTOR_JS: a self-contained JavaScript class that can
be embedded into any HTML page. No external dependencies.

Usage from another UI module:
    from cardprice.card_frame_detector import CARD_FRAME_DETECTOR_JS
    # Then include it in a <script> tag in your HTML.

The class provides:
    CardFrameDetector - stateful detector with transition tracking
    detect(imageData, width, height) -> detection result
    getTransition() -> 'none' | 'entering' | 'centered' | 'exiting'
"""

CARD_FRAME_DETECTOR_JS = r"""
// =============================================================================
// CardFrameDetector: detect a Pokemon card in close-range video frames.
//
// Designed for phone held ~15cm from binder page where a single card fills
// 60-80% of the frame. Handles:
//   - Binder sleeve edges visible around the card
//   - Adjacent cards partially visible at frame edges
//   - Orange/blue binder background around card
//   - Phone movement causing card to shift across frame
// =============================================================================

class CardFrameDetector {
    constructor(options = {}) {
        // Pokemon card aspect ratio: 63mm x 88mm
        this.CARD_ASPECT = options.cardAspect || (63 / 88);  // W/H = 0.716

        // Detection thresholds
        this.MIN_FILL = options.minFill || 0.25;     // minimum card area / frame area
        this.MAX_FILL = options.maxFill || 0.90;     // maximum (too close)
        this.TARGET_FILL_LO = options.targetFillLo || 0.40;
        this.TARGET_FILL_HI = options.targetFillHi || 0.80;
        this.CENTER_TOLERANCE = options.centerTolerance || 0.15; // 15% of frame dim
        this.ASPECT_TOLERANCE = options.aspectTolerance || 0.18; // allow ~18% aspect deviation
        this.EDGE_MARGIN = options.edgeMargin || 0.02; // 2% of frame = card edge too close to boundary

        // Downscale resolution for processing speed
        this.SAMPLE_W = options.sampleWidth || 240;

        // Saturation thresholds for card vs binder segmentation
        // Cards are generally low-saturation (white/gray borders, artwork varies)
        // Binder backgrounds are high-saturation (orange, blue, etc.)
        this.SAT_THRESHOLD = options.satThreshold || 60;  // below = likely card
        this.BRIGHT_LO = options.brightLo || 30;          // ignore very dark pixels
        this.BRIGHT_HI = options.brightHi || 245;         // ignore blown-out pixels

        // Edge detection threshold (Sobel magnitude)
        this.EDGE_THRESHOLD = options.edgeThreshold || 35;

        // Transition tracking state
        this._history = [];          // last N detection results
        this._historyMax = 10;
        this._transition = 'none';   // 'none' | 'entering' | 'centered' | 'exiting'
        this._centeredFrames = 0;
        this._centeredRequired = 5;  // consecutive centered frames to confirm

        // Reusable buffers (allocated on first use)
        this._gray = null;
        this._sat = null;
        this._val = null;
        this._mask = null;
        this._edges = null;
        this._tmpCanvas = null;
        this._tmpCtx = null;
    }

    // =========================================================================
    // Main detection entry point
    // =========================================================================

    /**
     * Detect card in a video frame.
     *
     * @param {HTMLVideoElement|HTMLCanvasElement|ImageData} source
     *   Video element, canvas, or raw ImageData to analyze.
     * @param {number} [width]  - Frame width (required if source is ImageData)
     * @param {number} [height] - Frame height (required if source is ImageData)
     * @returns {{
     *   detected: boolean,
     *   centered: boolean,
     *   fillRatio: number,
     *   cardRect: {x: number, y: number, w: number, h: number} | null,
     *   confidence: number,
     *   hint: string,
     *   transition: string
     * }}
     */
    detect(source, width, height) {
        // Get ImageData from whatever source we received
        const imgData = this._getImageData(source, width, height);
        if (!imgData) {
            return this._noDetection('Cannot read frame');
        }

        const w = imgData.width;
        const h = imgData.height;
        const pixels = imgData.data;

        // Allocate or resize buffers
        this._ensureBuffers(w, h);

        // Step 1: Convert to grayscale + HSV-like channels
        this._computeChannels(pixels, w, h);

        // Step 2: Build binary mask - card pixels vs background
        this._buildMask(w, h);

        // Step 3: Clean up mask with morphological operations
        this._cleanMask(w, h);

        // Step 4: Find bounding box of the largest connected region
        const bbox = this._findLargestBlob(w, h);
        if (!bbox) {
            const result = this._noDetection('No card detected');
            this._updateTransition(result);
            return result;
        }

        // Step 5: Refine bounding box using edge detection
        const refined = this._refineWithEdges(bbox, w, h);

        // Step 6: Validate the detection
        const result = this._validate(refined, w, h);
        this._updateTransition(result);
        return result;
    }

    /**
     * Get the current transition state.
     * @returns {'none'|'entering'|'centered'|'exiting'}
     */
    getTransition() {
        return this._transition;
    }

    /**
     * Reset transition tracking state.
     */
    reset() {
        this._history = [];
        this._transition = 'none';
        this._centeredFrames = 0;
    }

    // =========================================================================
    // Internal: image data extraction
    // =========================================================================

    _getImageData(source, width, height) {
        // Raw ImageData passed directly
        if (source instanceof ImageData) {
            return source;
        }

        // Video or Canvas element - draw to temp canvas at reduced size
        let srcW, srcH;
        if (source instanceof HTMLVideoElement) {
            srcW = source.videoWidth;
            srcH = source.videoHeight;
        } else if (source instanceof HTMLCanvasElement) {
            srcW = source.width;
            srcH = source.height;
        } else if (source.width && source.height && source.data) {
            // ImageData-like object
            return source;
        } else {
            return null;
        }

        if (!srcW || !srcH) return null;

        const sampleW = this.SAMPLE_W;
        const sampleH = Math.round(sampleW * (srcH / srcW));

        if (!this._tmpCanvas) {
            this._tmpCanvas = document.createElement('canvas');
            this._tmpCtx = this._tmpCanvas.getContext('2d', { willReadFrequently: true });
        }
        this._tmpCanvas.width = sampleW;
        this._tmpCanvas.height = sampleH;
        this._tmpCtx.drawImage(source, 0, 0, sampleW, sampleH);
        return this._tmpCtx.getImageData(0, 0, sampleW, sampleH);
    }

    // =========================================================================
    // Internal: buffer management
    // =========================================================================

    _ensureBuffers(w, h) {
        const n = w * h;
        if (!this._gray || this._gray.length !== n) {
            this._gray = new Uint8Array(n);
            this._sat = new Uint8Array(n);
            this._val = new Uint8Array(n);
            this._mask = new Uint8Array(n);
            this._edges = new Uint8Array(n);
        }
    }

    // =========================================================================
    // Internal: color conversion
    // =========================================================================

    _computeChannels(pixels, w, h) {
        const gray = this._gray;
        const sat = this._sat;
        const val = this._val;
        const n = w * h;

        for (let i = 0; i < n; i++) {
            const j = i * 4;
            const r = pixels[j];
            const g = pixels[j + 1];
            const b = pixels[j + 2];

            // Grayscale (luminance)
            gray[i] = (r * 77 + g * 150 + b * 29) >> 8;

            // Value (max of RGB) and Saturation
            const mx = Math.max(r, g, b);
            const mn = Math.min(r, g, b);
            val[i] = mx;
            sat[i] = mx === 0 ? 0 : Math.round(((mx - mn) / mx) * 255);
        }
    }

    // =========================================================================
    // Internal: mask building
    //
    // Strategy: segment card from binder background using a combination of:
    // 1. Saturation: binder backgrounds are high-saturation (orange/blue),
    //    card borders and most card content are lower saturation.
    // 2. Edge information: strong edges often appear at card boundaries.
    //
    // We also use adaptive thresholding on brightness to handle cards with
    // colorful artwork (which have high saturation too).
    // =========================================================================

    _buildMask(w, h) {
        const mask = this._mask;
        const sat = this._sat;
        const val = this._val;
        const gray = this._gray;

        // Compute global brightness stats for adaptive thresholding
        let brightSum = 0;
        let brightCount = 0;
        const n = w * h;

        for (let i = 0; i < n; i++) {
            if (val[i] > this.BRIGHT_LO && val[i] < this.BRIGHT_HI) {
                brightSum += gray[i];
                brightCount++;
            }
        }
        const avgBright = brightCount > 0 ? brightSum / brightCount : 128;

        // Compute saturation histogram to find the split between card and binder
        const satHist = new Uint32Array(256);
        for (let i = 0; i < n; i++) {
            if (val[i] > this.BRIGHT_LO) {
                satHist[sat[i]]++;
            }
        }

        // Find Otsu-like threshold on saturation
        let satThresh = this._otsuThreshold(satHist, 256);
        // Clamp to reasonable range
        satThresh = Math.max(35, Math.min(120, satThresh));

        // Build the mask: card = 1, background = 0
        for (let i = 0; i < n; i++) {
            // Very dark or very bright pixels are ambiguous - exclude
            if (val[i] <= this.BRIGHT_LO || val[i] >= this.BRIGHT_HI) {
                mask[i] = 0;
                continue;
            }

            // Low saturation = likely card (white border, gray text areas)
            // But also include mid-saturation pixels that are bright enough
            // (card artwork can be colorful)
            if (sat[i] < satThresh) {
                mask[i] = 1;
            } else if (gray[i] > avgBright * 0.7 && sat[i] < satThresh + 30) {
                // Moderately saturated but bright - could be card artwork
                mask[i] = 1;
            } else {
                mask[i] = 0;
            }
        }
    }

    /**
     * Otsu's method to find optimal threshold for bimodal histogram.
     */
    _otsuThreshold(hist, size) {
        let total = 0;
        let sum = 0;
        for (let i = 0; i < size; i++) {
            total += hist[i];
            sum += i * hist[i];
        }
        if (total === 0) return 128;

        let sumB = 0;
        let wB = 0;
        let maxVariance = 0;
        let threshold = 0;

        for (let i = 0; i < size; i++) {
            wB += hist[i];
            if (wB === 0) continue;
            const wF = total - wB;
            if (wF === 0) break;

            sumB += i * hist[i];
            const mB = sumB / wB;
            const mF = (sum - sumB) / wF;
            const variance = wB * wF * (mB - mF) * (mB - mF);

            if (variance > maxVariance) {
                maxVariance = variance;
                threshold = i;
            }
        }
        return threshold;
    }

    // =========================================================================
    // Internal: morphological cleanup
    // =========================================================================

    _cleanMask(w, h) {
        const mask = this._mask;

        // Erode then dilate (opening) to remove noise
        // Use a 3x3 kernel, applied twice for stronger effect
        this._erode(mask, w, h);
        this._erode(mask, w, h);
        this._dilate(mask, w, h);
        this._dilate(mask, w, h);
        this._dilate(mask, w, h);

        // Fill holes: flood fill from edges to find background,
        // then anything not reached and not already mask=1 becomes mask=1
        this._fillHoles(mask, w, h);
    }

    _erode(mask, w, h) {
        // A pixel survives only if all 4-connected neighbors are also set
        const tmp = new Uint8Array(w * h);
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const i = y * w + x;
                if (mask[i] &&
                    mask[i - 1] && mask[i + 1] &&
                    mask[i - w] && mask[i + w]) {
                    tmp[i] = 1;
                }
            }
        }
        tmp.forEach((v, i) => mask[i] = v);
    }

    _dilate(mask, w, h) {
        const tmp = new Uint8Array(mask);
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const i = y * w + x;
                if (tmp[i]) {
                    mask[i - 1] = 1; mask[i + 1] = 1;
                    mask[i - w] = 1; mask[i + w] = 1;
                }
            }
        }
    }

    _fillHoles(mask, w, h) {
        // Flood fill from all border pixels that are 0 -> mark as "exterior"
        const exterior = new Uint8Array(w * h);
        const stack = [];

        // Seed from all 4 borders
        for (let x = 0; x < w; x++) {
            if (!mask[x]) stack.push(x);                      // top row
            if (!mask[(h - 1) * w + x]) stack.push((h - 1) * w + x);  // bottom row
        }
        for (let y = 0; y < h; y++) {
            if (!mask[y * w]) stack.push(y * w);              // left col
            if (!mask[y * w + w - 1]) stack.push(y * w + w - 1);  // right col
        }

        while (stack.length > 0) {
            const idx = stack.pop();
            if (exterior[idx]) continue;
            if (mask[idx]) continue;
            exterior[idx] = 1;

            const x = idx % w;
            const y = (idx - x) / w;
            if (x > 0) stack.push(idx - 1);
            if (x < w - 1) stack.push(idx + 1);
            if (y > 0) stack.push(idx - w);
            if (y < h - 1) stack.push(idx + w);
        }

        // Anything not exterior and not already mask -> it's a hole inside the card
        for (let i = 0; i < w * h; i++) {
            if (!exterior[i]) mask[i] = 1;
        }
    }

    // =========================================================================
    // Internal: find largest connected blob
    // =========================================================================

    _findLargestBlob(w, h) {
        const mask = this._mask;
        const labels = new Int32Array(w * h);
        let nextLabel = 1;
        let bestLabel = 0;
        let bestSize = 0;
        const sizes = {};

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = y * w + x;
                if (!mask[i] || labels[i]) continue;

                // BFS flood fill
                const label = nextLabel++;
                const queue = [i];
                let qHead = 0;
                let size = 0;

                while (qHead < queue.length) {
                    const idx = queue[qHead++];
                    if (labels[idx]) continue;
                    if (!mask[idx]) continue;
                    labels[idx] = label;
                    size++;

                    const px = idx % w;
                    const py = (idx - px) / w;
                    if (px > 0 && mask[idx - 1] && !labels[idx - 1]) queue.push(idx - 1);
                    if (px < w - 1 && mask[idx + 1] && !labels[idx + 1]) queue.push(idx + 1);
                    if (py > 0 && mask[idx - w] && !labels[idx - w]) queue.push(idx - w);
                    if (py < h - 1 && mask[idx + w] && !labels[idx + w]) queue.push(idx + w);
                }

                sizes[label] = size;
                if (size > bestSize) {
                    bestSize = size;
                    bestLabel = label;
                }
            }
        }

        // Need at least 5% of frame to be considered a real blob
        if (bestSize < w * h * 0.05) return null;

        // Find bounding box of the largest blob
        let minX = w, minY = h, maxX = 0, maxY = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (labels[y * w + x] === bestLabel) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // Also compute fill ratio within the bounding box (rectangularity)
        const bboxArea = (maxX - minX + 1) * (maxY - minY + 1);
        const rectangularity = bestSize / bboxArea;

        return {
            x: minX, y: minY,
            w: maxX - minX + 1,
            h: maxY - minY + 1,
            area: bestSize,
            rectangularity: rectangularity,
            label: bestLabel,
            labels: labels,
        };
    }

    // =========================================================================
    // Internal: refine bbox using edge detection
    //
    // The saturation-based mask gives a rough blob. Edges give us precise
    // card boundaries. We look for strong horizontal/vertical edge lines
    // near the blob boundaries.
    // =========================================================================

    _refineWithEdges(bbox, w, h) {
        const gray = this._gray;
        const edges = this._edges;

        // Compute Sobel edges
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const i = y * w + x;
                const gx = -gray[i - w - 1] + gray[i - w + 1]
                           - 2 * gray[i - 1] + 2 * gray[i + 1]
                           - gray[i + w - 1] + gray[i + w + 1];
                const gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1]
                           + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
                edges[i] = Math.min(255, Math.round(Math.sqrt(gx * gx + gy * gy)));
            }
        }

        // Search for the strongest horizontal edge lines near top/bottom of blob
        const searchMargin = Math.round(bbox.h * 0.15);

        const topEdge = this._findEdgeLine(
            edges, w, h, 'horizontal',
            Math.max(0, bbox.y - searchMargin),
            Math.min(h - 1, bbox.y + searchMargin),
            bbox.x, bbox.x + bbox.w
        );

        const bottomEdge = this._findEdgeLine(
            edges, w, h, 'horizontal',
            Math.max(0, bbox.y + bbox.h - searchMargin),
            Math.min(h - 1, bbox.y + bbox.h + searchMargin),
            bbox.x, bbox.x + bbox.w
        );

        const leftEdge = this._findEdgeLine(
            edges, w, h, 'vertical',
            Math.max(0, bbox.x - searchMargin),
            Math.min(w - 1, bbox.x + searchMargin),
            bbox.y, bbox.y + bbox.h
        );

        const rightEdge = this._findEdgeLine(
            edges, w, h, 'vertical',
            Math.max(0, bbox.x + bbox.w - searchMargin),
            Math.min(w - 1, bbox.x + bbox.w + searchMargin),
            bbox.y, bbox.y + bbox.h
        );

        // Use edge-refined boundaries where we found strong edges,
        // fall back to blob boundaries otherwise
        const refinedX = leftEdge !== null ? leftEdge : bbox.x;
        const refinedY = topEdge !== null ? topEdge : bbox.y;
        const refinedR = rightEdge !== null ? rightEdge : bbox.x + bbox.w;
        const refinedB = bottomEdge !== null ? bottomEdge : bbox.y + bbox.h;

        return {
            x: refinedX,
            y: refinedY,
            w: refinedR - refinedX,
            h: refinedB - refinedY,
            area: bbox.area,
            rectangularity: bbox.rectangularity,
            edgeRefined: (topEdge !== null || bottomEdge !== null ||
                          leftEdge !== null || rightEdge !== null),
        };
    }

    /**
     * Find the row (horizontal) or column (vertical) with the strongest
     * edge response in a search region.
     */
    _findEdgeLine(edges, w, h, direction, searchStart, searchEnd, perpStart, perpEnd) {
        let bestPos = null;
        let bestScore = 0;
        const threshold = this.EDGE_THRESHOLD;

        perpStart = Math.max(0, perpStart);

        if (direction === 'horizontal') {
            perpEnd = Math.min(w, perpEnd);
            for (let y = searchStart; y <= searchEnd; y++) {
                let score = 0;
                for (let x = perpStart; x < perpEnd; x++) {
                    const e = edges[y * w + x];
                    if (e > threshold) score += e;
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestPos = y;
                }
            }
        } else {
            perpEnd = Math.min(h, perpEnd);
            for (let x = searchStart; x <= searchEnd; x++) {
                let score = 0;
                for (let y = perpStart; y < perpEnd; y++) {
                    const e = edges[y * w + x];
                    if (e > threshold) score += e;
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestPos = x;
                }
            }
        }

        // Only accept if the edge line has a reasonable density
        const lineLen = (direction === 'horizontal')
            ? (perpEnd - perpStart)
            : (perpEnd - perpStart);
        const minScore = lineLen * threshold * 0.15;  // at least 15% of line has edges

        return bestScore > minScore ? bestPos : null;
    }

    // =========================================================================
    // Internal: validation
    // =========================================================================

    _validate(rect, frameW, frameH) {
        const fillRatio = (rect.w * rect.h) / (frameW * frameH);
        const cardCx = rect.x + rect.w / 2;
        const cardCy = rect.y + rect.h / 2;
        const frameCx = frameW / 2;
        const frameCy = frameH / 2;

        // Centering: distance from frame center as fraction of frame dimensions
        const offX = Math.abs(cardCx - frameCx) / frameW;
        const offY = Math.abs(cardCy - frameCy) / frameH;
        const centered = offX < this.CENTER_TOLERANCE && offY < this.CENTER_TOLERANCE;

        // Aspect ratio check
        const detectedAspect = rect.w / rect.h;
        const aspectError = Math.abs(detectedAspect - this.CARD_ASPECT) / this.CARD_ASPECT;
        const aspectOk = aspectError < this.ASPECT_TOLERANCE;

        // Edge visibility: all 4 card edges should be away from frame boundary
        const edgeMarginPx = Math.round(Math.min(frameW, frameH) * this.EDGE_MARGIN);
        const topVisible = rect.y > edgeMarginPx;
        const bottomVisible = (rect.y + rect.h) < (frameH - edgeMarginPx);
        const leftVisible = rect.x > edgeMarginPx;
        const rightVisible = (rect.x + rect.w) < (frameW - edgeMarginPx);
        const allEdgesVisible = topVisible && bottomVisible && leftVisible && rightVisible;
        const visibleEdgeCount = [topVisible, bottomVisible, leftVisible, rightVisible]
            .filter(Boolean).length;

        // Rectangularity check (how rectangular is the blob)
        const rectOk = rect.rectangularity > 0.65;

        // Fill ratio check
        const fillOk = fillRatio >= this.MIN_FILL && fillRatio <= this.MAX_FILL;
        const fillIdeal = fillRatio >= this.TARGET_FILL_LO && fillRatio <= this.TARGET_FILL_HI;

        // Build confidence score (0-1)
        let confidence = 0;

        // Aspect ratio contributes up to 0.25
        if (aspectOk) {
            confidence += 0.25 * (1 - aspectError / this.ASPECT_TOLERANCE);
        }

        // Rectangularity contributes up to 0.20
        if (rectOk) {
            confidence += 0.20 * Math.min(1, (rect.rectangularity - 0.65) / 0.30);
        }

        // Fill ratio contributes up to 0.20
        if (fillOk) {
            if (fillIdeal) {
                confidence += 0.20;
            } else {
                confidence += 0.10;
            }
        }

        // Centering contributes up to 0.20
        if (centered) {
            const centerScore = 1 - (offX + offY) / (2 * this.CENTER_TOLERANCE);
            confidence += 0.20 * centerScore;
        }

        // Edge visibility contributes up to 0.15
        confidence += 0.15 * (visibleEdgeCount / 4);

        // Determine if detected
        const detected = fillOk && aspectOk && rectOk;

        // Generate user-facing hint
        let hint = '';
        if (!detected) {
            if (!fillOk && fillRatio < this.MIN_FILL) {
                hint = 'Move closer to the card';
            } else if (!fillOk && fillRatio > this.MAX_FILL) {
                hint = 'Move further from the card';
            } else if (!aspectOk) {
                hint = 'Rotate phone to match card orientation';
            } else if (!rectOk) {
                hint = 'Card not clearly visible';
            }
        } else if (!allEdgesVisible) {
            const clipped = [];
            if (!topVisible) clipped.push('top');
            if (!bottomVisible) clipped.push('bottom');
            if (!leftVisible) clipped.push('left');
            if (!rightVisible) clipped.push('right');
            hint = 'Card ' + clipped.join('+') + ' edge clipped — move away';
        } else if (!centered) {
            // Tell user which direction to move
            const dirs = [];
            if (offX > this.CENTER_TOLERANCE) {
                dirs.push(cardCx < frameCx ? 'right' : 'left');
            }
            if (offY > this.CENTER_TOLERANCE) {
                dirs.push(cardCy < frameCy ? 'down' : 'up');
            }
            hint = 'Move card ' + dirs.join(' and ');
        } else if (!fillIdeal) {
            hint = fillRatio < this.TARGET_FILL_LO
                ? 'Move a bit closer'
                : 'Move a bit further';
        } else {
            hint = 'Card aligned';
        }

        // Scale rect back to original frame coordinates if we downsampled
        // (the caller gets coordinates in the sample space; they need to
        // scale if they want original-frame coords. We return normalized.)
        const result = {
            detected: detected,
            centered: centered && allEdgesVisible,
            fillRatio: fillRatio,
            cardRect: {
                x: rect.x / frameW,
                y: rect.y / frameH,
                w: rect.w / frameW,
                h: rect.h / frameH,
            },
            confidence: Math.round(confidence * 1000) / 1000,
            allEdgesVisible: allEdgesVisible,
            visibleEdges: { top: topVisible, bottom: bottomVisible, left: leftVisible, right: rightVisible },
            aspectRatio: detectedAspect,
            rectangularity: rect.rectangularity,
            hint: hint,
            transition: this._transition,
        };

        return result;
    }

    _noDetection(hint) {
        return {
            detected: false,
            centered: false,
            fillRatio: 0,
            cardRect: null,
            confidence: 0,
            allEdgesVisible: false,
            visibleEdges: { top: false, bottom: false, left: false, right: false },
            aspectRatio: 0,
            rectangularity: 0,
            hint: hint,
            transition: this._transition,
        };
    }

    // =========================================================================
    // Internal: transition tracking
    //
    // Track card state over time:
    //   none -> entering -> centered -> exiting -> none
    //
    // "entering": card detected but not all edges visible (coming into frame)
    // "centered": card detected, centered, all edges visible for N frames
    // "exiting": was centered, now edges disappearing
    // =========================================================================

    _updateTransition(result) {
        this._history.push({
            detected: result.detected,
            centered: result.centered,
            allEdgesVisible: result.allEdgesVisible,
            fillRatio: result.fillRatio,
            cardRect: result.cardRect,
        });
        if (this._history.length > this._historyMax) {
            this._history.shift();
        }

        const prev = this._transition;

        if (!result.detected) {
            // No card visible
            if (prev === 'centered' || prev === 'exiting') {
                this._transition = 'exiting';
                // After a few frames of no detection, go to 'none'
                const recentMisses = this._history.slice(-3).filter(h => !h.detected).length;
                if (recentMisses >= 3) {
                    this._transition = 'none';
                    this._centeredFrames = 0;
                }
            } else {
                this._transition = 'none';
                this._centeredFrames = 0;
            }
        } else if (result.centered) {
            // Card detected AND centered with all edges visible
            this._centeredFrames++;
            if (this._centeredFrames >= this._centeredRequired) {
                this._transition = 'centered';
            } else {
                this._transition = 'entering';
            }
        } else {
            // Card detected but not centered (partially visible or off-center)
            if (prev === 'centered') {
                this._transition = 'exiting';
                this._centeredFrames = 0;
            } else if (prev === 'exiting') {
                // Still exiting
                const recentCentered = this._history.slice(-4).filter(h => h.centered).length;
                if (recentCentered === 0) {
                    // Hasn't been centered recently, might be entering a new card
                    this._transition = 'entering';
                }
            } else {
                this._transition = 'entering';
                this._centeredFrames = 0;
            }
        }

        result.transition = this._transition;
    }
}
"""
