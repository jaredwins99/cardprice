/**
 * ScannerCapture — Perspective correction and multi-row card capture.
 *
 * Grabs full-resolution video frames, extracts individual cards using
 * homography-based perspective correction, and manages the 3-row
 * capture lifecycle for a 3x3 binder page.
 *
 * Usage:
 *   const capture = new ScannerCapture(videoEl, {
 *       detectionCanvas: detCanvas,  // the 480px canvas used for detection
 *   });
 *
 *   // When auto-capture fires with 3 locked cards:
 *   const cardDataUrls = capture.captureRow(lockedCards);
 *
 *   // After all 3 rows:
 *   if (capture.isComplete()) {
 *       const results = await capture.submit();
 *   }
 */

class ScannerCapture {
    /**
     * @param {HTMLVideoElement} video  The camera video element
     * @param {Object} opts
     * @param {HTMLCanvasElement} opts.detectionCanvas  Canvas used for card detection (480px wide)
     * @param {number} opts.cardWidth   Output card width in pixels (default 420)
     * @param {number} opts.cardHeight  Output card height in pixels (default 586)
     * @param {number} opts.rows        Total rows to capture (default 3)
     * @param {number} opts.cols        Cards per row (default 3)
     * @param {number} opts.jpegQuality JPEG export quality (default 0.92)
     * @param {string} opts.submitUrl   Endpoint for card identification (default '/slide-scan/identify')
     */
    constructor(video, opts = {}) {
        this.video = video;
        this.detectionCanvas = opts.detectionCanvas || null;
        this.cardWidth = opts.cardWidth || 420;
        this.cardHeight = opts.cardHeight || 586;
        this.totalRows = opts.rows || 3;
        this.cardsPerRow = opts.cols || 3;
        this.jpegQuality = opts.jpegQuality || 0.92;
        this.submitUrl = opts.submitUrl || '/slide-scan/identify';

        this.capturedCards = [];  // flat array of data URL strings, length up to rows*cols
        this.currentRow = 0;

        // Hidden canvas for grabbing full-res video frames
        this._frameCanvas = document.createElement('canvas');
        this._frameCtx = this._frameCanvas.getContext('2d');

        // Hidden canvas for rendering individual corrected cards
        this._cardCanvas = document.createElement('canvas');
        this._cardCanvas.width = this.cardWidth;
        this._cardCanvas.height = this.cardHeight;
        this._cardCtx = this._cardCanvas.getContext('2d');
    }

    /**
     * Capture a row of cards from the current video frame.
     *
     * @param {Array<Object>} lockedCards  Array of detected cards, each with
     *   { corners: [{x, y}, {x, y}, {x, y}, {x, y}] } in detection canvas coordinates.
     * @returns {string[]}  Array of JPEG data URLs for the captured cards (left-to-right order).
     */
    captureRow(lockedCards) {
        if (this.currentRow >= this.totalRows) {
            throw new Error('All rows already captured');
        }

        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;

        // Grab the full-resolution video frame
        this._frameCanvas.width = vw;
        this._frameCanvas.height = vh;
        this._frameCtx.drawImage(this.video, 0, 0, vw, vh);
        const frameData = this._frameCtx.getImageData(0, 0, vw, vh);

        // Compute scale from detection canvas to full video
        const detW = this.detectionCanvas ? this.detectionCanvas.width : 480;
        const detH = this.detectionCanvas ? this.detectionCanvas.height : 270;
        const scaleX = vw / detW;
        const scaleY = vh / detH;

        // Sort cards left-to-right by center X
        const sorted = lockedCards.slice().sort((a, b) => {
            const cx_a = a.corners.reduce((s, c) => s + c.x, 0) / a.corners.length;
            const cx_b = b.corners.reduce((s, c) => s + c.x, 0) / b.corners.length;
            return cx_a - cx_b;
        });

        const dataUrls = [];

        for (const card of sorted) {
            // Scale corners to full video coordinates
            const scaledCorners = card.corners.map(c => ({
                x: c.x * scaleX,
                y: c.y * scaleY,
            }));

            // Order corners: TL, TR, BR, BL
            const ordered = _orderCorners(scaledCorners);

            // Compute homography from output rect to source quad
            const dstCorners = [
                { x: 0, y: 0 },
                { x: this.cardWidth - 1, y: 0 },
                { x: this.cardWidth - 1, y: this.cardHeight - 1 },
                { x: 0, y: this.cardHeight - 1 },
            ];
            const H = _computeHomography(dstCorners, ordered);

            // Apply inverse mapping with bilinear interpolation
            const outData = this._cardCtx.createImageData(this.cardWidth, this.cardHeight);
            _applyHomography(H, frameData, outData, this.cardWidth, this.cardHeight, vw, vh);

            this._cardCtx.putImageData(outData, 0, 0);
            const dataUrl = this._cardCanvas.toDataURL('image/jpeg', this.jpegQuality);
            dataUrls.push(dataUrl);
            this.capturedCards.push(dataUrl);
        }

        this.currentRow++;
        return dataUrls;
    }

    /** Get all captured card data URLs. */
    getAllCards() {
        return this.capturedCards;
    }

    /** True when all rows have been captured. */
    isComplete() {
        return this.currentRow >= this.totalRows;
    }

    /** Total number of cards expected. */
    totalCards() {
        return this.totalRows * this.cardsPerRow;
    }

    /** Reset capture state for a new scan session. */
    reset() {
        this.capturedCards = [];
        this.currentRow = 0;
    }

    /**
     * Build a FormData object with all captured cards.
     * Keys are card_0 through card_N.
     * @returns {Promise<FormData>}
     */
    async buildSubmission() {
        const fd = new FormData();
        for (let i = 0; i < this.capturedCards.length; i++) {
            const blob = await _dataURLtoBlob(this.capturedCards[i]);
            fd.append('card_' + i, blob, 'card_' + i + '.jpg');
        }
        return fd;
    }

    /**
     * Submit all captured cards for identification.
     * @returns {Promise<Object>}  Server response JSON.
     */
    async submit() {
        const fd = await this.buildSubmission();
        const resp = await fetch(this.submitUrl, { method: 'POST', body: fd });
        if (!resp.ok) {
            throw new Error('Submission failed: ' + resp.status + ' ' + resp.statusText);
        }
        return resp.json();
    }
}


// ===================================================================
// Corner ordering
// ===================================================================

/**
 * Order 4 corners into [TL, TR, BR, BL].
 *
 * Strategy: TL has the smallest (x+y), BR has the largest (x+y).
 * TR has the smallest (y-x), BL has the largest (y-x).
 *
 * @param {Array<{x: number, y: number}>} corners  4 corner points
 * @returns {Array<{x: number, y: number}>}  Ordered [TL, TR, BR, BL]
 */
function _orderCorners(corners) {
    if (corners.length !== 4) {
        throw new Error('Expected 4 corners, got ' + corners.length);
    }

    const pts = corners.slice();

    // Sum and difference for classification
    const sums = pts.map(p => p.x + p.y);
    const diffs = pts.map(p => p.y - p.x);

    const tlIdx = sums.indexOf(Math.min(...sums));
    const brIdx = sums.indexOf(Math.max(...sums));
    const trIdx = diffs.indexOf(Math.min(...diffs));
    const blIdx = diffs.indexOf(Math.max(...diffs));

    return [pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]];
}


// ===================================================================
// Homography computation (DLT algorithm)
// ===================================================================

/**
 * Compute a 3x3 homography matrix that maps source points to destination points.
 *
 * Uses the Direct Linear Transform (DLT) with 4 point correspondences,
 * producing 8 equations for the 8 unknowns of the homography (h33 = 1).
 *
 * The homography H maps: dst = H * src (in homogeneous coordinates).
 * For inverse mapping we compute H: output_pixel -> source_pixel.
 *
 * @param {Array<{x,y}>} src  4 source points (output/destination image corners)
 * @param {Array<{x,y}>} dst  4 destination points (source image quad corners)
 * @returns {number[]}  3x3 matrix as flat array [h11,h12,h13,h21,h22,h23,h31,h32,h33]
 */
function _computeHomography(src, dst) {
    // Build the 8x8 system Ah = b where h = [h11..h32] and h33 = 1
    //
    // For each correspondence (sx,sy) -> (dx,dy):
    //   dx = (h11*sx + h12*sy + h13) / (h31*sx + h32*sy + 1)
    //   dy = (h21*sx + h22*sy + h23) / (h31*sx + h32*sy + 1)
    //
    // Rearranging:
    //   h11*sx + h12*sy + h13 - h31*sx*dx - h32*sy*dx = dx
    //   h21*sx + h22*sy + h23 - h31*sx*dy - h32*sy*dy = dy

    const A = [];
    const b = [];

    for (let i = 0; i < 4; i++) {
        const sx = src[i].x, sy = src[i].y;
        const dx = dst[i].x, dy = dst[i].y;

        A.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
        b.push(dx);

        A.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
        b.push(dy);
    }

    // Solve 8x8 system via Gaussian elimination with partial pivoting
    const h = _solveLinearSystem(A, b);

    return [
        h[0], h[1], h[2],
        h[3], h[4], h[5],
        h[6], h[7], 1.0,
    ];
}

/**
 * Solve Ax = b using Gaussian elimination with partial pivoting.
 * @param {number[][]} A  NxN matrix (modified in place)
 * @param {number[]} b    N-vector (modified in place)
 * @returns {number[]}    Solution vector x
 */
function _solveLinearSystem(A, b) {
    const n = b.length;

    // Forward elimination with partial pivoting
    for (let col = 0; col < n; col++) {
        // Find pivot
        let maxVal = Math.abs(A[col][col]);
        let maxRow = col;
        for (let row = col + 1; row < n; row++) {
            const val = Math.abs(A[row][col]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = row;
            }
        }

        // Swap rows
        if (maxRow !== col) {
            const tmpA = A[col]; A[col] = A[maxRow]; A[maxRow] = tmpA;
            const tmpB = b[col]; b[col] = b[maxRow]; b[maxRow] = tmpB;
        }

        // Check for singular matrix
        if (Math.abs(A[col][col]) < 1e-12) {
            throw new Error('Singular matrix in homography computation');
        }

        // Eliminate below
        for (let row = col + 1; row < n; row++) {
            const factor = A[row][col] / A[col][col];
            for (let j = col; j < n; j++) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    const x = new Array(n);
    for (let row = n - 1; row >= 0; row--) {
        let sum = b[row];
        for (let j = row + 1; j < n; j++) {
            sum -= A[row][j] * x[j];
        }
        x[row] = sum / A[row][row];
    }

    return x;
}


// ===================================================================
// Perspective warp via inverse mapping + bilinear interpolation
// ===================================================================

/**
 * Apply homography H (mapping output coords -> source coords) to warp
 * the source image into the output image.
 *
 * For each pixel (ox, oy) in the output:
 *   [sx, sy, sw] = H * [ox, oy, 1]
 *   sourceX = sx / sw
 *   sourceY = sy / sw
 *   sample source with bilinear interpolation
 *
 * @param {number[]} H          3x3 homography (flat array, row-major)
 * @param {ImageData} srcData   Source image data (full video frame)
 * @param {ImageData} dstData   Output image data (card-sized, written in place)
 * @param {number} outW         Output width
 * @param {number} outH         Output height
 * @param {number} srcW         Source width
 * @param {number} srcH         Source height
 */
function _applyHomography(H, srcData, dstData, outW, outH, srcW, srcH) {
    const src = srcData.data;
    const dst = dstData.data;

    const h11 = H[0], h12 = H[1], h13 = H[2];
    const h21 = H[3], h22 = H[4], h23 = H[5];
    const h31 = H[6], h32 = H[7], h33 = H[8];

    const srcWm1 = srcW - 1;
    const srcHm1 = srcH - 1;

    for (let oy = 0; oy < outH; oy++) {
        // Precompute terms that only depend on oy
        const hy1 = h12 * oy + h13;
        const hy2 = h22 * oy + h23;
        const hy3 = h32 * oy + h33;

        for (let ox = 0; ox < outW; ox++) {
            const sw = h31 * ox + hy3;
            const sx = (h11 * ox + hy1) / sw;
            const sy = (h21 * ox + hy2) / sw;

            // Bounds check
            if (sx < 0 || sy < 0 || sx > srcWm1 || sy > srcHm1) {
                const di = (oy * outW + ox) * 4;
                dst[di] = 0;
                dst[di + 1] = 0;
                dst[di + 2] = 0;
                dst[di + 3] = 255;
                continue;
            }

            // Bilinear interpolation
            const x0 = sx | 0;  // floor
            const y0 = sy | 0;
            const x1 = Math.min(x0 + 1, srcWm1);
            const y1 = Math.min(y0 + 1, srcHm1);
            const fx = sx - x0;
            const fy = sy - y0;
            const fx1 = 1 - fx;
            const fy1 = 1 - fy;

            // Weight for each corner
            const w00 = fx1 * fy1;
            const w10 = fx * fy1;
            const w01 = fx1 * fy;
            const w11 = fx * fy;

            // Source pixel indices
            const i00 = (y0 * srcW + x0) * 4;
            const i10 = (y0 * srcW + x1) * 4;
            const i01 = (y1 * srcW + x0) * 4;
            const i11 = (y1 * srcW + x1) * 4;

            const di = (oy * outW + ox) * 4;
            dst[di]     = w00 * src[i00]     + w10 * src[i10]     + w01 * src[i01]     + w11 * src[i11];
            dst[di + 1] = w00 * src[i00 + 1] + w10 * src[i10 + 1] + w01 * src[i01 + 1] + w11 * src[i11 + 1];
            dst[di + 2] = w00 * src[i00 + 2] + w10 * src[i10 + 2] + w01 * src[i01 + 2] + w11 * src[i11 + 2];
            dst[di + 3] = 255;
        }
    }
}


// ===================================================================
// Data URL to Blob conversion
// ===================================================================

/**
 * Convert a data URL string to a Blob.
 * @param {string} dataUrl
 * @returns {Promise<Blob>}
 */
function _dataURLtoBlob(dataUrl) {
    return new Promise((resolve, reject) => {
        try {
            const parts = dataUrl.split(',');
            const mime = parts[0].match(/:(.*?);/)[1];
            const b64 = atob(parts[1]);
            const len = b64.length;
            const u8 = new Uint8Array(len);
            for (let i = 0; i < len; i++) {
                u8[i] = b64.charCodeAt(i);
            }
            resolve(new Blob([u8], { type: mime }));
        } catch (e) {
            reject(e);
        }
    });
}


// ===================================================================
// Self-test
// ===================================================================

ScannerCapture.selfTest = function () {
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

    console.log('=== ScannerCapture Self-Test ===');

    // Test _orderCorners
    {
        const corners = [
            { x: 100, y: 10 },   // TR
            { x: 10, y: 100 },   // BL
            { x: 100, y: 100 },  // BR
            { x: 10, y: 10 },    // TL
        ];
        const ordered = _orderCorners(corners);
        assert(ordered[0].x === 10  && ordered[0].y === 10,  '_orderCorners TL correct');
        assert(ordered[1].x === 100 && ordered[1].y === 10,  '_orderCorners TR correct');
        assert(ordered[2].x === 100 && ordered[2].y === 100, '_orderCorners BR correct');
        assert(ordered[3].x === 10  && ordered[3].y === 100, '_orderCorners BL correct');
    }

    // Test _orderCorners with non-axis-aligned quad
    {
        const corners = [
            { x: 95, y: 5 },
            { x: 5, y: 5 },
            { x: 105, y: 95 },
            { x: 15, y: 95 },
        ];
        const ordered = _orderCorners(corners);
        assert(ordered[0].x === 5   && ordered[0].y === 5,  '_orderCorners skewed TL');
        assert(ordered[1].x === 95  && ordered[1].y === 5,  '_orderCorners skewed TR');
        assert(ordered[2].x === 105 && ordered[2].y === 95, '_orderCorners skewed BR');
        assert(ordered[3].x === 15  && ordered[3].y === 95, '_orderCorners skewed BL');
    }

    // Test _computeHomography with identity-like mapping
    {
        const src = [
            { x: 0, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 100 },
            { x: 0, y: 100 },
        ];
        const dst = [
            { x: 0, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 100 },
            { x: 0, y: 100 },
        ];
        const H = _computeHomography(src, dst);
        // Should be close to identity: H[0]=1, H[4]=1, H[8]=1, rest ~0
        assert(Math.abs(H[0] - 1) < 1e-6, 'identity homography H[0]=1');
        assert(Math.abs(H[4] - 1) < 1e-6, 'identity homography H[4]=1');
        assert(Math.abs(H[8] - 1) < 1e-6, 'identity homography H[8]=1');
        assert(Math.abs(H[1]) < 1e-6, 'identity homography H[1]=0');
        assert(Math.abs(H[3]) < 1e-6, 'identity homography H[3]=0');
        assert(Math.abs(H[6]) < 1e-6, 'identity homography H[6]=0');
        assert(Math.abs(H[7]) < 1e-6, 'identity homography H[7]=0');
    }

    // Test _computeHomography with a known transformation (2x scale)
    {
        const src = [
            { x: 0, y: 0 },
            { x: 50, y: 0 },
            { x: 50, y: 50 },
            { x: 0, y: 50 },
        ];
        const dst = [
            { x: 0, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 100 },
            { x: 0, y: 100 },
        ];
        const H = _computeHomography(src, dst);
        // Mapping (25,25) should give (50,50)
        const w = H[6] * 25 + H[7] * 25 + H[8];
        const mx = (H[0] * 25 + H[1] * 25 + H[2]) / w;
        const my = (H[3] * 25 + H[4] * 25 + H[5]) / w;
        assert(Math.abs(mx - 50) < 1e-4, 'scale homography maps (25,25)->(50,50) x');
        assert(Math.abs(my - 50) < 1e-4, 'scale homography maps (25,25)->(50,50) y');
    }

    // Test _computeHomography with perspective (trapezoid)
    {
        const src = [
            { x: 0, y: 0 },
            { x: 420, y: 0 },
            { x: 420, y: 586 },
            { x: 0, y: 586 },
        ];
        const dst = [
            { x: 110, y: 50 },
            { x: 400, y: 60 },
            { x: 410, y: 500 },
            { x: 100, y: 490 },
        ];
        const H = _computeHomography(src, dst);
        // Verify corners map correctly
        for (let i = 0; i < 4; i++) {
            const w = H[6] * src[i].x + H[7] * src[i].y + H[8];
            const mx = (H[0] * src[i].x + H[1] * src[i].y + H[2]) / w;
            const my = (H[3] * src[i].x + H[4] * src[i].y + H[5]) / w;
            const ok = Math.abs(mx - dst[i].x) < 0.1 && Math.abs(my - dst[i].y) < 0.1;
            assert(ok, 'perspective homography corner ' + i + ' maps correctly');
        }
    }

    // Test _solveLinearSystem with simple 2x2
    {
        // 2x + 3y = 8, x + y = 3 => x=1, y=2
        const A = [[2, 3], [1, 1]];
        const b = [8, 3];
        const x = _solveLinearSystem(A, b);
        assert(Math.abs(x[0] - 1) < 1e-10, 'solve 2x2: x=1');
        assert(Math.abs(x[1] - 2) < 1e-10, 'solve 2x2: y=2');
    }

    // Test ScannerCapture constructor defaults
    {
        const mockVideo = { videoWidth: 1920, videoHeight: 1080 };
        const cap = new ScannerCapture(mockVideo);
        assert(cap.cardWidth === 420, 'default cardWidth = 420');
        assert(cap.cardHeight === 586, 'default cardHeight = 586');
        assert(cap.totalRows === 3, 'default totalRows = 3');
        assert(cap.cardsPerRow === 3, 'default cardsPerRow = 3');
        assert(cap.capturedCards.length === 0, 'initial capturedCards empty');
        assert(cap.currentRow === 0, 'initial currentRow = 0');
        assert(cap.isComplete() === false, 'not complete initially');
        assert(cap.totalCards() === 9, 'totalCards = 9');
    }

    // Test reset
    {
        const mockVideo = { videoWidth: 1920, videoHeight: 1080 };
        const cap = new ScannerCapture(mockVideo);
        cap.capturedCards = ['a', 'b', 'c'];
        cap.currentRow = 1;
        cap.reset();
        assert(cap.capturedCards.length === 0, 'reset clears capturedCards');
        assert(cap.currentRow === 0, 'reset clears currentRow');
    }

    // Test _dataURLtoBlob
    {
        // Minimal valid JPEG data URL
        const tiny = 'data:image/jpeg;base64,/9j/4AAQSkZJRg==';
        _dataURLtoBlob(tiny).then(blob => {
            console.log('  PASS: _dataURLtoBlob produces blob (type=' + blob.type + ', size=' + blob.size + ')');
        }).catch(e => {
            console.error('  FAIL: _dataURLtoBlob threw: ' + e);
        });
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScannerCapture;
}
