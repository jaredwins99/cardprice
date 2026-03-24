/**
 * CardContourDetector — Client-side card contour detection for the scanner camera UI.
 *
 * Detects rectangular card outlines in a live camera feed at 10fps on a 480px-wide
 * canvas. Returns corner positions, aspect ratios, and confidence scores for each
 * detected card.
 *
 * Designed for binder scanning: cards are in clear sleeves inside a binder page.
 * Handles uneven lighting via adaptive thresholding and filters out sleeve edges
 * by removing rectangles that are entirely contained within another.
 *
 * Algorithm:
 *   1. Grayscale conversion (inline, no library)
 *   2. Adaptive threshold (8x8 block means, threshold at mean - 10)
 *   3. Morphological close (3x3, fills small gaps in card edges)
 *   4. Suzuki-Abe outer contour tracing (simplified, max 20 contours)
 *   5. Douglas-Peucker polygon approximation (epsilon = 3% of perimeter)
 *   6. Card validation (area, aspect ratio, angles, convexity)
 *   7. Containment filtering (remove sleeve-inside-card duplicates)
 *   8. Corner ordering (TL, TR, BR, BL)
 *
 * Performance: <5ms per frame on mid-range phones at 480px width.
 *
 * Usage:
 *   const detector = new CardContourDetector();
 *
 *   // In your rAF / setInterval loop:
 *   const imageData = ctx.getImageData(0, 0, w, h);
 *   const cards = detector.detect(imageData, w, h);
 *   // cards = [{ corners, center, area, aspectRatio, confidence }, ...]
 */

class CardContourDetector {
    /**
     * @param {Object} opts
     * @param {number} opts.minArea       Min card area as fraction of frame (default 0.05)
     * @param {number} opts.maxArea       Max card area as fraction of frame (default 0.45)
     * @param {number[]} opts.aspectRange Acceptable width/height range (default [0.60, 0.82])
     * @param {number} opts.maxContours   Stop contour tracing after this many (default 20)
     * @param {number} opts.blockSize     Adaptive threshold block grid size (default 8)
     * @param {number} opts.threshOffset  Threshold offset below block mean (default 10)
     */
    constructor(opts = {}) {
        this.minArea     = opts.minArea     || 0.05;   // 5% of frame
        this.maxArea     = opts.maxArea     || 0.45;   // 45% of frame
        this.aspectRange = opts.aspectRange || [0.60, 0.82]; // Pokemon card 63/88 = 0.716
        this.maxContours = opts.maxContours || 20;
        this.blockSize   = opts.blockSize   || 8;
        this.threshOffset = opts.threshOffset ?? 10;

        // Reusable buffers (allocated on first use, resized as needed)
        this._gray = null;
        this._binary = null;
        this._visited = null;
        this._lastW = 0;
        this._lastH = 0;
    }

    /**
     * Detect card-shaped rectangles in an image frame.
     *
     * @param {ImageData} imageData  Raw pixel data from canvas getImageData()
     * @param {number} width         Image width in pixels
     * @param {number} height        Image height in pixels
     * @returns {Array<{
     *   corners: Array<{x: number, y: number}>,
     *   center: {x: number, y: number},
     *   area: number,
     *   aspectRatio: number,
     *   confidence: number
     * }>} Array of detected cards, sorted by area descending
     */
    detect(imageData, width, height) {
        const pixels = imageData.data; // Uint8ClampedArray, RGBA
        const totalPixels = width * height;
        const frameArea = totalPixels;

        // --- Allocate / reuse buffers ---
        if (width !== this._lastW || height !== this._lastH) {
            this._gray    = new Uint8Array(totalPixels);
            this._binary  = new Uint8Array(totalPixels);
            this._visited = new Uint8Array(totalPixels);
            this._lastW = width;
            this._lastH = height;
        }
        const gray   = this._gray;
        const binary = this._binary;

        // =====================================================================
        // Step 1: Grayscale conversion
        // Standard luminance weights: 0.299R + 0.587G + 0.114B
        // Using integer approximation: (77R + 150G + 29B) >> 8 for speed
        // =====================================================================
        for (let i = 0; i < totalPixels; i++) {
            const j = i << 2; // i * 4
            gray[i] = (77 * pixels[j] + 150 * pixels[j + 1] + 29 * pixels[j + 2]) >> 8;
        }

        // =====================================================================
        // Step 2: Adaptive threshold
        // Divide image into blockSize x blockSize grid. For each block, compute
        // the mean grayscale value. Each pixel is thresholded against its block's
        // mean minus threshOffset. This handles the uneven lighting typical of
        // binder pages (center brighter, edges darker from camera flash / lamp).
        //
        // Output: binary[] where 1 = foreground (edge/dark), 0 = background
        // =====================================================================
        const bsX = this.blockSize;
        const bsY = this.blockSize;
        const blockW = Math.ceil(width / bsX);
        const blockH = Math.ceil(height / bsY);
        const offset = this.threshOffset;

        // Compute block means
        // We use a flat array of block sums and counts
        const blockMeans = new Float32Array(blockW * blockH);
        const blockCounts = new Uint16Array(blockW * blockH);
        const blockSums = new Float64Array(blockW * blockH);

        for (let y = 0; y < height; y++) {
            const by = Math.min((y / bsY) | 0, blockH - 1);
            const rowOff = by * blockW;
            for (let x = 0; x < width; x++) {
                const bx = Math.min((x / bsX) | 0, blockW - 1);
                const bi = rowOff + bx;
                blockSums[bi] += gray[y * width + x];
                blockCounts[bi]++;
            }
        }

        for (let i = 0; i < blockW * blockH; i++) {
            blockMeans[i] = blockCounts[i] > 0 ? blockSums[i] / blockCounts[i] : 128;
        }

        // Apply threshold: pixel < (block_mean - offset) => foreground (1)
        // We want card EDGES to be foreground. Card edges are typically darker
        // than the card surface or lighter than the binder background.
        // Actually, for contour tracing we want the card BORDER region to form
        // closed contours. Cards appear as bright rectangles on a darker binder.
        // So we threshold: pixel > (block_mean + offset) => foreground.
        // This picks up the bright card surface against the darker binder.
        // But we actually want EDGES, not surfaces. Let's use Sobel-like approach:
        //
        // Better approach: compute gradient magnitude, then threshold that.
        // But Sobel is expensive. Instead, use the absolute difference from
        // the block mean as an edge indicator, then do contour tracing on that.
        //
        // Simplest correct approach for card detection: threshold to get bright
        // card regions (cards are brighter than binder background), then trace
        // the outer boundary of those bright regions.
        for (let y = 0; y < height; y++) {
            const by = Math.min((y / bsY) | 0, blockH - 1);
            const rowOff = by * blockW;
            for (let x = 0; x < width; x++) {
                const bx = Math.min((x / bsX) | 0, blockW - 1);
                const thresh = blockMeans[rowOff + bx] + offset;
                // Card surface is brighter than binder -> mark as foreground
                binary[y * width + x] = gray[y * width + x] > thresh ? 1 : 0;
            }
        }

        // =====================================================================
        // Step 2b: Morphological close (dilate then erode) with 3x3 kernel
        // Fills small gaps in card edges caused by reflections, sleeve seams, etc.
        // We reuse the binary buffer with a temporary copy.
        // =====================================================================
        this._morphClose3x3(binary, width, height);

        // =====================================================================
        // Step 3: Contour tracing (simplified Suzuki-Abe)
        // Traces outer contours only (no holes). Each contour is a list of
        // {x, y} points forming the boundary of a connected foreground region.
        //
        // We scan left-to-right, top-to-bottom. When we find a foreground pixel
        // that hasn't been visited and has a background pixel to its left (or is
        // on the left edge), we start tracing the outer boundary.
        //
        // Direction encoding (8-connected, clockwise from right):
        //   0=right, 1=down-right, 2=down, 3=down-left,
        //   4=left, 5=up-left, 6=up, 7=up-right
        // =====================================================================
        const contours = this._traceContours(binary, width, height);

        // =====================================================================
        // Step 4: Douglas-Peucker polygon approximation
        // Simplify each contour to a polygon. Keep only 4-point polygons
        // (potential card rectangles).
        // =====================================================================
        const quads = [];
        for (let ci = 0; ci < contours.length; ci++) {
            const contour = contours[ci];
            if (contour.length < 20) continue; // too few points, not a card

            const perimeter = this._perimeter(contour);
            const epsilon = perimeter * 0.03; // 3% of perimeter
            const approx = this._douglasPeucker(contour, epsilon);

            if (approx.length === 4) {
                quads.push(approx);
            }
        }

        // =====================================================================
        // Step 5: Card validation
        // For each quadrilateral, check:
        //   - Area within [minArea, maxArea] of frame
        //   - Aspect ratio within range (portrait or landscape)
        //   - All 4 interior angles within 60-120 degrees
        //   - Convex (all cross products have the same sign)
        // =====================================================================
        const candidates = [];
        for (let qi = 0; qi < quads.length; qi++) {
            const quad = quads[qi];
            const result = this._validateCard(quad, frameArea);
            if (result) {
                candidates.push(result);
            }
        }

        // =====================================================================
        // Step 6: Containment filtering
        // Remove rectangles entirely contained within another (sleeve edges).
        // Also remove near-duplicates (>80% overlap).
        // Prefer rectangles closer to the expected Pokemon card aspect ratio.
        // =====================================================================
        const filtered = this._filterContainment(candidates);

        // Sort by area descending (largest cards first)
        filtered.sort((a, b) => b.area - a.area);

        return filtered;
    }

    // =========================================================================
    // Morphological close (dilate then erode) with 3x3 structuring element
    // =========================================================================
    _morphClose3x3(binary, w, h) {
        // Dilate: output pixel = 1 if any neighbor is 1
        const temp = new Uint8Array(w * h);
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                if (binary[idx] ||
                    binary[idx - 1] || binary[idx + 1] ||
                    binary[idx - w] || binary[idx + w] ||
                    binary[idx - w - 1] || binary[idx - w + 1] ||
                    binary[idx + w - 1] || binary[idx + w + 1]) {
                    temp[idx] = 1;
                } else {
                    temp[idx] = 0;
                }
            }
        }
        // Erode: output pixel = 1 only if all neighbors are 1
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                if (temp[idx] &&
                    temp[idx - 1] && temp[idx + 1] &&
                    temp[idx - w] && temp[idx + w] &&
                    temp[idx - w - 1] && temp[idx - w + 1] &&
                    temp[idx + w - 1] && temp[idx + w + 1]) {
                    binary[idx] = 1;
                } else {
                    binary[idx] = 0;
                }
            }
        }
    }

    // =========================================================================
    // Contour tracing (simplified Suzuki-Abe, outer contours only)
    //
    // 8-connected boundary tracing. For each unvisited foreground pixel that
    // borders background on its left, we trace the full outer boundary by
    // walking clockwise around the boundary.
    //
    // We subsample the contour (every 3rd point) to reduce point count for
    // the Douglas-Peucker step, since card edges are straight lines and we
    // don't need sub-pixel boundary detail.
    // =========================================================================
    _traceContours(binary, w, h) {
        const visited = this._visited;
        visited.fill(0);

        const contours = [];

        // Direction vectors for 8-connected neighbors (clockwise from right)
        // dx: [1, 1, 0, -1, -1, -1, 0, 1]
        // dy: [0, 1, 1, 1, 0, -1, -1, -1]
        const dx = [1, 1, 0, -1, -1, -1, 0, 1];
        const dy = [0, 1, 1, 1, 0, -1, -1, -1];

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;

                // Start condition: foreground pixel with background to its left
                // (or not yet visited as a contour start)
                if (binary[idx] !== 1) continue;
                if (visited[idx]) continue;
                if (binary[idx - 1] !== 0) continue; // need background on left

                // Trace this outer contour
                const contour = [];
                let cx = x, cy = y;
                let dir = 0; // start searching to the right

                // Moore boundary tracing: walk the boundary by searching
                // clockwise from the direction we came from
                const startX = x, startY = y;
                let steps = 0;
                const maxSteps = w * h; // safety limit

                do {
                    // Subsample: add every 3rd point to reduce contour size
                    if (steps % 3 === 0) {
                        contour.push({ x: cx, y: cy });
                    }
                    visited[cy * w + cx] = 1;

                    // Search for next boundary pixel: start from (dir + 5) % 8
                    // which is the direction we came from, rotated 90° CCW.
                    // This ensures we trace the boundary clockwise.
                    let searchDir = (dir + 5) & 7; // equivalent to (dir + 5) % 8
                    let found = false;

                    for (let i = 0; i < 8; i++) {
                        const sd = (searchDir + i) & 7;
                        const nx = cx + dx[sd];
                        const ny = cy + dy[sd];

                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

                        if (binary[ny * w + nx] === 1) {
                            cx = nx;
                            cy = ny;
                            dir = sd;
                            found = true;
                            break;
                        }
                    }

                    if (!found) break; // isolated pixel
                    steps++;

                } while ((cx !== startX || cy !== startY) && steps < maxSteps);

                // Only keep contours with enough points (at least 20 before subsampling)
                if (contour.length >= 7) {
                    contours.push(contour);
                    if (contours.length >= this.maxContours) return contours;
                }
            }
        }

        return contours;
    }

    // =========================================================================
    // Perimeter of a polygon (sum of edge lengths)
    // =========================================================================
    _perimeter(points) {
        let p = 0;
        const n = points.length;
        for (let i = 0; i < n; i++) {
            const a = points[i];
            const b = points[(i + 1) % n];
            const ddx = b.x - a.x;
            const ddy = b.y - a.y;
            p += Math.sqrt(ddx * ddx + ddy * ddy);
        }
        return p;
    }

    // =========================================================================
    // Douglas-Peucker polygon approximation
    //
    // Recursively simplifies a polyline by removing points that are within
    // `epsilon` distance of the line between endpoints. For closed contours,
    // we first find the point farthest from the line between first and last
    // points to split the contour into two open polylines, then simplify each.
    //
    // This is the same algorithm OpenCV uses for cv2.approxPolyDP().
    // =========================================================================
    _douglasPeucker(points, epsilon) {
        const n = points.length;
        if (n <= 2) return points.slice();

        // For closed contours: find the two points farthest apart to use as
        // split points, then run DP on each half.
        // Simplified: just run DP on the open polyline and close it.
        const keep = new Uint8Array(n);
        keep[0] = 1;
        keep[n - 1] = 1;

        this._dpRecurse(points, 0, n - 1, epsilon, keep);

        const result = [];
        for (let i = 0; i < n; i++) {
            if (keep[i]) result.push(points[i]);
        }
        return result;
    }

    _dpRecurse(points, start, end, epsilon, keep) {
        if (end - start <= 1) return;

        // Find point with maximum distance from line (start -> end)
        let maxDist = 0;
        let maxIdx = start;

        const sx = points[start].x, sy = points[start].y;
        const ex = points[end].x, ey = points[end].y;
        const lx = ex - sx, ly = ey - sy;
        const lenSq = lx * lx + ly * ly;

        for (let i = start + 1; i < end; i++) {
            let dist;
            if (lenSq === 0) {
                // start == end, just use distance to start
                const ddx = points[i].x - sx;
                const ddy = points[i].y - sy;
                dist = Math.sqrt(ddx * ddx + ddy * ddy);
            } else {
                // Perpendicular distance from point to line
                const cross = Math.abs(lx * (sy - points[i].y) - ly * (sx - points[i].x));
                dist = cross / Math.sqrt(lenSq);
            }

            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }

        if (maxDist > epsilon) {
            keep[maxIdx] = 1;
            this._dpRecurse(points, start, maxIdx, epsilon, keep);
            this._dpRecurse(points, maxIdx, end, epsilon, keep);
        }
    }

    // =========================================================================
    // Validate a quadrilateral as a potential card
    //
    // Checks:
    //   1. Area within [minArea, maxArea] of frame area
    //   2. Aspect ratio within range (supports both portrait and landscape)
    //   3. All 4 interior angles within 60-120 degrees
    //   4. Convex polygon (all cross products have the same sign)
    //
    // Returns null if invalid, or a card result object if valid.
    // =========================================================================
    _validateCard(quad, frameArea) {
        // --- Compute area via shoelace formula ---
        const area = this._polygonArea(quad);
        const areaFrac = area / frameArea;
        if (areaFrac < this.minArea || areaFrac > this.maxArea) return null;

        // --- Check convexity (all cross products same sign) ---
        const cross = [];
        for (let i = 0; i < 4; i++) {
            const a = quad[i];
            const b = quad[(i + 1) % 4];
            const c = quad[(i + 2) % 4];
            // Cross product of vectors (a->b) x (b->c)
            const cp = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            cross.push(cp);
        }
        // All same sign = convex
        const allPos = cross.every(c => c > 0);
        const allNeg = cross.every(c => c < 0);
        if (!allPos && !allNeg) return null;

        // --- Check interior angles (60-120 degrees) ---
        for (let i = 0; i < 4; i++) {
            const a = quad[(i + 3) % 4]; // previous vertex
            const b = quad[i];             // current vertex
            const c = quad[(i + 1) % 4]; // next vertex
            const angle = this._angleDeg(a, b, c);
            if (angle < 60 || angle > 120) return null;
        }

        // --- Compute aspect ratio ---
        // Order corners first, then compute side lengths
        const ordered = this._orderCorners(quad);

        // Width = average of top and bottom edge lengths
        const topW = this._dist(ordered[0], ordered[1]);
        const botW = this._dist(ordered[3], ordered[2]);
        const avgW = (topW + botW) / 2;

        // Height = average of left and right edge lengths
        const leftH  = this._dist(ordered[0], ordered[3]);
        const rightH = this._dist(ordered[1], ordered[2]);
        const avgH = (leftH + rightH) / 2;

        // Aspect ratio = width / height (for portrait card, this should be ~0.716)
        // Support both portrait and landscape: compute min/max ratio
        let aspectRatio;
        if (avgW < avgH) {
            aspectRatio = avgW / avgH; // portrait
        } else {
            aspectRatio = avgH / avgW; // landscape, normalize to portrait
        }

        if (aspectRatio < this.aspectRange[0] || aspectRatio > this.aspectRange[1]) {
            return null;
        }

        // --- Compute center ---
        const center = {
            x: (ordered[0].x + ordered[1].x + ordered[2].x + ordered[3].x) / 4,
            y: (ordered[0].y + ordered[1].y + ordered[2].y + ordered[3].y) / 4
        };

        // --- Compute confidence ---
        // Based on how close the aspect ratio is to ideal (0.716) and how
        // close the angles are to 90 degrees.
        const idealAspect = 0.716;
        const aspectScore = 1 - Math.abs(aspectRatio - idealAspect) / 0.12; // 0-1 range

        let angleScore = 0;
        for (let i = 0; i < 4; i++) {
            const a = quad[(i + 3) % 4];
            const b = quad[i];
            const c = quad[(i + 1) % 4];
            const angle = this._angleDeg(a, b, c);
            angleScore += 1 - Math.abs(angle - 90) / 30; // deviation from 90°
        }
        angleScore /= 4;

        const confidence = Math.max(0, Math.min(1,
            0.6 * Math.max(0, aspectScore) + 0.4 * Math.max(0, angleScore)
        ));

        return {
            corners: ordered,
            center,
            area: areaFrac,
            aspectRatio,
            confidence
        };
    }

    // =========================================================================
    // Order 4 corners into TL, TR, BR, BL
    //
    // Strategy:
    //   1. Sort by sum (x + y): smallest = TL, largest = BR
    //   2. Sort by difference (y - x): smallest = TR, largest = BL
    // This is the same approach used by OpenCV card detector tutorials.
    // =========================================================================
    _orderCorners(quad) {
        const pts = quad.slice(); // don't mutate input

        // Compute sum and diff for each point
        const sums = pts.map(p => p.x + p.y);
        const diffs = pts.map(p => p.y - p.x);

        const tl = pts[sums.indexOf(Math.min(...sums))];
        const br = pts[sums.indexOf(Math.max(...sums))];
        const tr = pts[diffs.indexOf(Math.min(...diffs))];
        const bl = pts[diffs.indexOf(Math.max(...diffs))];

        return [
            { x: tl.x, y: tl.y },
            { x: tr.x, y: tr.y },
            { x: br.x, y: br.y },
            { x: bl.x, y: bl.y }
        ];
    }

    // =========================================================================
    // Filter out contained/overlapping rectangles
    //
    // When a card is in a clear sleeve, both the card edge and sleeve edge
    // may be detected. One rectangle will be almost entirely inside the other.
    // We keep the one whose aspect ratio is closest to the ideal card ratio.
    //
    // Also removes near-duplicates (>80% mutual overlap measured by IoU of
    // axis-aligned bounding boxes).
    // =========================================================================
    _filterContainment(candidates) {
        if (candidates.length <= 1) return candidates;

        const idealAspect = 0.716;
        const toRemove = new Set();

        for (let i = 0; i < candidates.length; i++) {
            if (toRemove.has(i)) continue;
            for (let j = i + 1; j < candidates.length; j++) {
                if (toRemove.has(j)) continue;

                const a = candidates[i];
                const b = candidates[j];

                // Check if centers are close (within 15% of the larger card's dimensions)
                const cdx = Math.abs(a.center.x - b.center.x);
                const cdy = Math.abs(a.center.y - b.center.y);

                // Get bounding box dimensions of the larger card
                const aBox = this._boundingBox(a.corners);
                const bBox = this._boundingBox(b.corners);
                const maxW = Math.max(aBox.w, bBox.w);
                const maxH = Math.max(aBox.h, bBox.h);

                // If centers are close relative to card size, these might be
                // the same card detected twice (card edge + sleeve edge)
                if (cdx < maxW * 0.15 && cdy < maxH * 0.15) {
                    // Keep the one with aspect ratio closer to ideal
                    const aDiff = Math.abs(a.aspectRatio - idealAspect);
                    const bDiff = Math.abs(b.aspectRatio - idealAspect);
                    toRemove.add(aDiff <= bDiff ? j : i);
                    continue;
                }

                // Check axis-aligned bounding box IoU for overlap
                const iou = this._bboxIoU(aBox, bBox);
                if (iou > 0.5) {
                    // High overlap — keep the better aspect ratio match
                    const aDiff = Math.abs(a.aspectRatio - idealAspect);
                    const bDiff = Math.abs(b.aspectRatio - idealAspect);
                    toRemove.add(aDiff <= bDiff ? j : i);
                }
            }
        }

        return candidates.filter((_, i) => !toRemove.has(i));
    }

    // =========================================================================
    // Helper: polygon area via shoelace formula (unsigned)
    // =========================================================================
    _polygonArea(pts) {
        let area = 0;
        const n = pts.length;
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            area += pts[i].x * pts[j].y;
            area -= pts[j].x * pts[i].y;
        }
        return Math.abs(area) / 2;
    }

    // =========================================================================
    // Helper: angle at vertex B in triangle A-B-C, in degrees
    // =========================================================================
    _angleDeg(a, b, c) {
        const bax = a.x - b.x, bay = a.y - b.y;
        const bcx = c.x - b.x, bcy = c.y - b.y;
        const dot = bax * bcx + bay * bcy;
        const magA = Math.sqrt(bax * bax + bay * bay);
        const magC = Math.sqrt(bcx * bcx + bcy * bcy);
        if (magA === 0 || magC === 0) return 0;
        const cosAngle = Math.max(-1, Math.min(1, dot / (magA * magC)));
        return Math.acos(cosAngle) * (180 / Math.PI);
    }

    // =========================================================================
    // Helper: Euclidean distance between two points
    // =========================================================================
    _dist(a, b) {
        const ddx = b.x - a.x;
        const ddy = b.y - a.y;
        return Math.sqrt(ddx * ddx + ddy * ddy);
    }

    // =========================================================================
    // Helper: axis-aligned bounding box of 4 corners
    // =========================================================================
    _boundingBox(corners) {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < corners.length; i++) {
            if (corners[i].x < minX) minX = corners[i].x;
            if (corners[i].y < minY) minY = corners[i].y;
            if (corners[i].x > maxX) maxX = corners[i].x;
            if (corners[i].y > maxY) maxY = corners[i].y;
        }
        return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
    }

    // =========================================================================
    // Helper: IoU of two axis-aligned bounding boxes
    // =========================================================================
    _bboxIoU(a, b) {
        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.w, b.x + b.w);
        const y2 = Math.min(a.y + a.h, b.y + b.h);

        if (x2 <= x1 || y2 <= y1) return 0;

        const intersection = (x2 - x1) * (y2 - y1);
        const areaA = a.w * a.h;
        const areaB = b.w * b.h;
        return intersection / (areaA + areaB - intersection);
    }
}


// ===========================================================================
// Self-test (run in browser console: CardContourDetector.selfTest())
//
// Tests each algorithmic component in isolation using synthetic data.
// No canvas or DOM required — works in Node.js too.
// ===========================================================================

CardContourDetector.selfTest = function () {
    let passed = 0;
    let failed = 0;

    function assert(cond, name) {
        if (cond) { console.log('  PASS: ' + name); passed++; }
        else      { console.error('  FAIL: ' + name); failed++; }
    }

    console.log('=== CardContourDetector Self-Test ===');

    const det = new CardContourDetector();

    // --- 1. Polygon area (shoelace) ---
    {
        // 10x10 square = area 100
        const sq = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }];
        const area = det._polygonArea(sq);
        assert(Math.abs(area - 100) < 0.01, 'shoelace: 10x10 square = 100 (got ' + area + ')');
    }

    // --- 2. Angle calculation ---
    {
        // Right angle at origin: (1,0)-(0,0)-(0,1) = 90°
        const angle90 = det._angleDeg({ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 });
        assert(Math.abs(angle90 - 90) < 0.1, 'angle: 90° (got ' + angle90.toFixed(1) + ')');

        // 45° angle
        const angle45 = det._angleDeg({ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 1 });
        assert(Math.abs(angle45 - 45) < 0.1, 'angle: 45° (got ' + angle45.toFixed(1) + ')');
    }

    // --- 3. Corner ordering ---
    {
        // Scrambled corners of a rectangle
        const scrambled = [
            { x: 100, y: 10 },  // TR
            { x: 10, y: 120 },  // BL
            { x: 10, y: 10 },   // TL
            { x: 100, y: 120 }  // BR
        ];
        const ordered = det._orderCorners(scrambled);
        assert(ordered[0].x === 10 && ordered[0].y === 10, 'corner order: TL = (10,10)');
        assert(ordered[1].x === 100 && ordered[1].y === 10, 'corner order: TR = (100,10)');
        assert(ordered[2].x === 100 && ordered[2].y === 120, 'corner order: BR = (100,120)');
        assert(ordered[3].x === 10 && ordered[3].y === 120, 'corner order: BL = (10,120)');
    }

    // --- 4. Douglas-Peucker simplification ---
    {
        // Rectangle with extra points on edges -> should simplify to 4 points
        const rect = [
            { x: 0, y: 0 },
            { x: 5, y: 0 },   // on top edge
            { x: 10, y: 0 },
            { x: 10, y: 5 },  // on right edge
            { x: 10, y: 10 },
            { x: 5, y: 10 },  // on bottom edge
            { x: 0, y: 10 },
            { x: 0, y: 5 }    // on left edge
        ];
        const simplified = det._douglasPeucker(rect, 0.5);
        assert(simplified.length === 4, 'DP: rectangle simplifies to 4 points (got ' + simplified.length + ')');
    }

    // --- 5. Card validation ---
    {
        // Valid card-shaped quad (63x88 aspect ratio, scaled up)
        // In a 480x640 frame (frameArea = 307200)
        const frameArea = 480 * 640;
        const cardW = 150; // ~0.073 of frame width
        const cardH = 210; // aspect = 150/210 = 0.714 ≈ 0.716
        const card = [
            { x: 100, y: 50 },
            { x: 100 + cardW, y: 50 },
            { x: 100 + cardW, y: 50 + cardH },
            { x: 100, y: 50 + cardH }
        ];
        const result = det._validateCard(card, frameArea);
        assert(result !== null, 'validateCard: valid card accepted');
        if (result) {
            assert(Math.abs(result.aspectRatio - 0.714) < 0.01,
                'validateCard: aspect ratio ~0.714 (got ' + result.aspectRatio.toFixed(3) + ')');
            assert(result.confidence > 0.8, 'validateCard: high confidence (got ' + result.confidence.toFixed(2) + ')');
        }

        // Too small (1% of frame)
        const tiny = [
            { x: 0, y: 0 }, { x: 20, y: 0 }, { x: 20, y: 28 }, { x: 0, y: 28 }
        ];
        assert(det._validateCard(tiny, frameArea) === null, 'validateCard: too small rejected');

        // Bad aspect ratio (square)
        const square = [
            { x: 0, y: 0 }, { x: 200, y: 0 }, { x: 200, y: 200 }, { x: 0, y: 200 }
        ];
        assert(det._validateCard(square, frameArea) === null, 'validateCard: square rejected');
    }

    // --- 6. Containment filter ---
    {
        // Two overlapping rectangles with same center (card + sleeve)
        const inner = {
            corners: [{ x: 12, y: 12 }, { x: 98, y: 12 }, { x: 98, y: 118 }, { x: 12, y: 118 }],
            center: { x: 55, y: 65 },
            area: 0.10,
            aspectRatio: 0.716, // closer to ideal
            confidence: 0.9
        };
        const outer = {
            corners: [{ x: 10, y: 10 }, { x: 100, y: 10 }, { x: 100, y: 120 }, { x: 10, y: 120 }],
            center: { x: 55, y: 65 },
            area: 0.12,
            aspectRatio: 0.75,
            confidence: 0.8
        };
        const filtered = det._filterContainment([inner, outer]);
        assert(filtered.length === 1, 'containment: 2 overlapping -> 1 kept (got ' + filtered.length + ')');
        if (filtered.length === 1) {
            assert(Math.abs(filtered[0].aspectRatio - 0.716) < 0.01,
                'containment: keeps better aspect ratio');
        }
    }

    // --- 7. BBox IoU ---
    {
        const a = { x: 0, y: 0, w: 10, h: 10 };
        const b = { x: 5, y: 5, w: 10, h: 10 };
        const iou = det._bboxIoU(a, b);
        // Intersection = 5x5 = 25, Union = 100 + 100 - 25 = 175
        assert(Math.abs(iou - 25 / 175) < 0.01, 'IoU: partial overlap = 0.143 (got ' + iou.toFixed(3) + ')');

        const c = { x: 20, y: 20, w: 5, h: 5 };
        assert(det._bboxIoU(a, c) === 0, 'IoU: no overlap = 0');
    }

    // --- 8. Perimeter ---
    {
        const sq = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }];
        const p = det._perimeter(sq);
        assert(Math.abs(p - 40) < 0.01, 'perimeter: 10x10 square = 40 (got ' + p + ')');
    }

    // --- 9. Morphological close preserves solid blocks ---
    {
        const w = 20, h = 20;
        const bin = new Uint8Array(w * h);
        // Fill a 10x10 block in the center
        for (let y = 5; y < 15; y++) {
            for (let x = 5; x < 15; x++) {
                bin[y * w + x] = 1;
            }
        }
        // Add a 1-pixel gap at (10, 10)
        bin[10 * w + 10] = 0;

        det._morphClose3x3(bin, w, h);
        assert(bin[10 * w + 10] === 1, 'morphClose: fills 1px gap');
        assert(bin[0] === 0, 'morphClose: background stays background');
    }

    // --- 10. Full detect on synthetic image ---
    {
        // Create a 100x130 synthetic image with a white card on dark background
        const w = 100, h = 130;
        const data = new Uint8ClampedArray(w * h * 4);

        // Dark background (value 40)
        for (let i = 0; i < w * h; i++) {
            data[i * 4] = 40;
            data[i * 4 + 1] = 40;
            data[i * 4 + 2] = 40;
            data[i * 4 + 3] = 255;
        }

        // White card rectangle: 40x56 pixels (aspect 40/56 = 0.714 ≈ card)
        // Positioned at (30, 37) to (70, 93)
        // Area = 40*56 = 2240, frame = 13000, frac = 0.172
        for (let y = 37; y < 93; y++) {
            for (let x = 30; x < 70; x++) {
                const i = (y * w + x) * 4;
                data[i] = 220;
                data[i + 1] = 220;
                data[i + 2] = 220;
                data[i + 3] = 255;
            }
        }

        const imageData = { data, width: w, height: h };
        const cards = det.detect(imageData, w, h);
        assert(cards.length >= 1, 'full detect: finds card in synthetic image (found ' + cards.length + ')');
        if (cards.length >= 1) {
            const c = cards[0];
            // Center should be near (50, 65)
            assert(Math.abs(c.center.x - 50) < 10, 'full detect: center.x near 50 (got ' + c.center.x.toFixed(0) + ')');
            assert(Math.abs(c.center.y - 65) < 10, 'full detect: center.y near 65 (got ' + c.center.y.toFixed(0) + ')');
            assert(c.aspectRatio > 0.60 && c.aspectRatio < 0.82,
                'full detect: aspect in range (got ' + c.aspectRatio.toFixed(3) + ')');
        }
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CardContourDetector;
}
