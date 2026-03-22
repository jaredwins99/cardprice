/**
 * perspectiveCorrect — Client-side perspective warp for captured card images.
 *
 * When a card is photographed at close range, slight tilt produces a
 * trapezoidal image instead of a clean rectangle.  This module computes the
 * 8-parameter projective (homography) transform from the detected card
 * corners to a target rectangle and applies it with bilinear interpolation,
 * producing a flat 420x586 image (Pokemon card aspect ratio 63:88).
 *
 * Approach: inverse-mapping with per-pixel homography.  For each pixel in the
 * output we compute the corresponding source coordinate via the inverse
 * homography, then bilinear-interpolate from the source canvas.  This avoids
 * triangle-strip seam artifacts and handles arbitrary quad distortion.
 *
 * Performance: ~15-25 ms on a modern phone for 420x586 output (246k pixels),
 * well within a single-frame budget.  Uses ImageData for direct pixel access.
 *
 * Usage:
 *   const corrected = perspectiveCorrect(sourceCanvas, [
 *       {x: 10, y: 5},    // top-left
 *       {x: 405, y: 12},  // top-right
 *       {x: 410, y: 590}, // bottom-right
 *       {x: 8, y: 583},   // bottom-left
 *   ]);
 *   // corrected is a canvas element with the de-warped 420x586 card image
 */

/**
 * Apply perspective correction to extract a card from a source canvas.
 *
 * @param {HTMLCanvasElement|HTMLImageElement|HTMLVideoElement} source
 *     The source image/canvas/video containing the card.
 * @param {Array<{x: number, y: number}>} cardCorners
 *     Four corner points in source coordinates: [TL, TR, BR, BL].
 * @param {number} [dstW=420] Target output width.
 * @param {number} [dstH=586] Target output height.
 * @returns {HTMLCanvasElement} New canvas with the perspective-corrected card.
 */
function perspectiveCorrect(source, cardCorners, dstW, dstH) {
    dstW = dstW || 420;
    dstH = dstH || 586;

    // --- 1. Get source pixel data ---
    var srcCanvas;
    if (source instanceof HTMLCanvasElement) {
        srcCanvas = source;
    } else {
        // Draw image/video onto a temp canvas to get ImageData
        srcCanvas = document.createElement('canvas');
        srcCanvas.width = source.videoWidth || source.naturalWidth || source.width;
        srcCanvas.height = source.videoHeight || source.naturalHeight || source.height;
        srcCanvas.getContext('2d').drawImage(source, 0, 0);
    }
    var srcW = srcCanvas.width;
    var srcH = srcCanvas.height;
    var srcCtx = srcCanvas.getContext('2d');
    var srcData = srcCtx.getImageData(0, 0, srcW, srcH).data;

    // --- 2. Compute homography: dst -> src (inverse mapping) ---
    // Source corners (the quad in the source image)
    var sx = [cardCorners[0].x, cardCorners[1].x, cardCorners[2].x, cardCorners[3].x];
    var sy = [cardCorners[0].y, cardCorners[1].y, cardCorners[2].y, cardCorners[3].y];

    // Destination corners (the target rectangle)
    var dx = [0, dstW - 1, dstW - 1, 0];
    var dy = [0, 0, dstH - 1, dstH - 1];

    // We want the transform T such that T * dst_point = src_point.
    // This is the inverse mapping: for each output pixel, find where it
    // came from in the source.
    var H = _computeHomography(dx, dy, sx, sy);
    if (!H) {
        // Degenerate corners — return a simple crop fallback
        var fallback = document.createElement('canvas');
        fallback.width = dstW;
        fallback.height = dstH;
        fallback.getContext('2d').drawImage(srcCanvas, 0, 0, dstW, dstH);
        return fallback;
    }

    // --- 3. Apply inverse mapping with bilinear interpolation ---
    var outCanvas = document.createElement('canvas');
    outCanvas.width = dstW;
    outCanvas.height = dstH;
    var outCtx = outCanvas.getContext('2d');
    var outImg = outCtx.createImageData(dstW, dstH);
    var out = outImg.data;

    var h0 = H[0], h1 = H[1], h2 = H[2];
    var h3 = H[3], h4 = H[4], h5 = H[5];
    var h6 = H[6], h7 = H[7];
    // H[8] = 1 (normalized)

    for (var oy = 0; oy < dstH; oy++) {
        // Precompute row-constant terms
        var ry = h1 * oy + h2;
        var ryy = h4 * oy + h5;
        var rw = h7 * oy + 1; // h8 = 1

        for (var ox = 0; ox < dstW; ox++) {
            // Homogeneous transform: dst -> src
            var w = h6 * ox + rw;
            var srcX = (h0 * ox + ry) / w;
            var srcY = (h3 * ox + ryy) / w;

            // Bilinear interpolation
            var ix = srcX | 0; // floor
            var iy = srcY | 0;
            var fx = srcX - ix;
            var fy = srcY - iy;

            // Clamp to source bounds
            if (ix < 0) { ix = 0; fx = 0; }
            if (iy < 0) { iy = 0; fy = 0; }
            if (ix >= srcW - 1) { ix = srcW - 2; fx = 1; }
            if (iy >= srcH - 1) { iy = srcH - 2; fy = 1; }

            // Four neighbor pixel indices (RGBA stride = 4)
            var i00 = (iy * srcW + ix) * 4;
            var i10 = i00 + 4;
            var i01 = i00 + srcW * 4;
            var i11 = i01 + 4;

            var w00 = (1 - fx) * (1 - fy);
            var w10 = fx * (1 - fy);
            var w01 = (1 - fx) * fy;
            var w11 = fx * fy;

            var outIdx = (oy * dstW + ox) * 4;
            out[outIdx]     = (srcData[i00]     * w00 + srcData[i10]     * w10 + srcData[i01]     * w01 + srcData[i11]     * w11 + 0.5) | 0;
            out[outIdx + 1] = (srcData[i00 + 1] * w00 + srcData[i10 + 1] * w10 + srcData[i01 + 1] * w01 + srcData[i11 + 1] * w11 + 0.5) | 0;
            out[outIdx + 2] = (srcData[i00 + 2] * w00 + srcData[i10 + 2] * w10 + srcData[i01 + 2] * w01 + srcData[i11 + 2] * w11 + 0.5) | 0;
            out[outIdx + 3] = 255; // fully opaque
        }
    }

    outCtx.putImageData(outImg, 0, 0);
    return outCanvas;
}


/**
 * Compute 3x3 homography matrix mapping points (srcX,srcY) -> (dstX,dstY).
 *
 * Uses the standard DLT (Direct Linear Transform) with 4 point correspondences.
 * Solves the 8x8 linear system via Gaussian elimination.
 *
 * The homography H maps homogeneous coordinates:
 *   [dstX]     [h0 h1 h2] [srcX]
 *   [dstY]  =  [h3 h4 h5] [srcY]
 *   [ w  ]     [h6 h7  1] [  1 ]
 *
 * Returns flat array [h0..h8] with h8=1, or null if degenerate.
 *
 * @param {number[]} srcX  4 source x-coordinates
 * @param {number[]} srcY  4 source y-coordinates
 * @param {number[]} dstX  4 destination x-coordinates
 * @param {number[]} dstY  4 destination y-coordinates
 * @returns {number[]|null} [h0, h1, h2, h3, h4, h5, h6, h7, 1] or null
 */
function _computeHomography(srcX, srcY, dstX, dstY) {
    // Build the 8x9 augmented matrix for the DLT system Ah = 0,
    // rearranged as Ah' = b (where h' has 8 unknowns, h8 = 1).
    //
    // Each point correspondence gives 2 equations:
    //   sx*h0 + sy*h1 + h2 - sx*dx*h6 - sy*dx*h7 = dx
    //   sx*h3 + sy*h4 + h5 - sx*dy*h6 - sy*dy*h7 = dy

    var A = [];  // 8x8
    var b = [];  // 8x1

    for (var i = 0; i < 4; i++) {
        var sx = srcX[i], sy = srcY[i];
        var ddx = dstX[i], ddy = dstY[i];

        A.push([sx, sy, 1, 0, 0, 0, -sx * ddx, -sy * ddx]);
        b.push(ddx);

        A.push([0, 0, 0, sx, sy, 1, -sx * ddy, -sy * ddy]);
        b.push(ddy);
    }

    // Gaussian elimination with partial pivoting
    var n = 8;
    // Augment A with b
    for (var i = 0; i < n; i++) {
        A[i].push(b[i]);
    }

    for (var col = 0; col < n; col++) {
        // Find pivot
        var maxVal = Math.abs(A[col][col]);
        var maxRow = col;
        for (var row = col + 1; row < n; row++) {
            var v = Math.abs(A[row][col]);
            if (v > maxVal) {
                maxVal = v;
                maxRow = row;
            }
        }
        if (maxVal < 1e-10) return null; // singular

        // Swap rows
        if (maxRow !== col) {
            var tmp = A[col];
            A[col] = A[maxRow];
            A[maxRow] = tmp;
        }

        // Eliminate below
        var pivot = A[col][col];
        for (var row = col + 1; row < n; row++) {
            var factor = A[row][col] / pivot;
            for (var j = col; j <= n; j++) {
                A[row][j] -= factor * A[col][j];
            }
        }
    }

    // Back substitution
    var h = new Array(n);
    for (var i = n - 1; i >= 0; i--) {
        var sum = A[i][n];
        for (var j = i + 1; j < n; j++) {
            sum -= A[i][j] * h[j];
        }
        h[i] = sum / A[i][i];
    }

    h.push(1); // h8 = 1
    return h;
}


/**
 * Estimate how much perspective distortion is present in a quad.
 *
 * Returns a value 0..1 where 0 = perfect rectangle, 1 = extreme distortion.
 * Useful for deciding whether to bother with correction (skip if < 0.02).
 *
 * Measures two things:
 *   1. Edge length ratio deviation from rectangle (opposing sides should match)
 *   2. Angle deviation from 90 degrees at each corner
 *
 * @param {Array<{x: number, y: number}>} corners [TL, TR, BR, BL]
 * @returns {number} Distortion score 0..1
 */
function perspectiveDistortion(corners) {
    var tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];

    // Edge lengths
    var top    = _dist(tl, tr);
    var right  = _dist(tr, br);
    var bottom = _dist(bl, br);
    var left   = _dist(tl, bl);

    // Ratio deviation: how different are opposing sides?
    var hRatio = Math.min(top, bottom) / Math.max(top, bottom);
    var vRatio = Math.min(left, right) / Math.max(left, right);
    var sideDeviation = 1 - (hRatio * vRatio);

    // Angle deviation: measure each corner angle's departure from 90 deg
    var angles = [
        _cornerAngle(bl, tl, tr),
        _cornerAngle(tl, tr, br),
        _cornerAngle(tr, br, bl),
        _cornerAngle(br, bl, tl),
    ];
    var maxAngleDev = 0;
    for (var i = 0; i < 4; i++) {
        var dev = Math.abs(angles[i] - Math.PI / 2) / (Math.PI / 2);
        if (dev > maxAngleDev) maxAngleDev = dev;
    }

    // Combine: weight sides 0.5, angles 0.5
    return Math.min(1, sideDeviation * 0.5 + maxAngleDev * 0.5);
}


/**
 * Euclidean distance between two points.
 * @param {{x:number,y:number}} a
 * @param {{x:number,y:number}} b
 * @returns {number}
 */
function _dist(a, b) {
    var dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}


/**
 * Angle at vertex B in triangle A-B-C (radians).
 * @param {{x:number,y:number}} a
 * @param {{x:number,y:number}} b
 * @param {{x:number,y:number}} c
 * @returns {number}
 */
function _cornerAngle(a, b, c) {
    var ba = { x: a.x - b.x, y: a.y - b.y };
    var bc = { x: c.x - b.x, y: c.y - b.y };
    var dot = ba.x * bc.x + ba.y * bc.y;
    var magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    var magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    var cos = dot / (magBA * magBC + 1e-10);
    return Math.acos(Math.max(-1, Math.min(1, cos)));
}
