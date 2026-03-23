/**
 * ScannerOverlay — Premium card scanner overlay with guide rectangle,
 * dimmed surround, corner brackets, status text, and thumbnail strip.
 *
 * Draws onto a canvas#overlay positioned absolutely over a <video> element.
 * Follows the same canvas setup pattern as condition_camera_ui.py:
 *
 *   canvas#overlay {
 *       position: absolute;
 *       top: 0; left: 0;
 *       width: 100%; height: 100%;
 *       z-index: 2;
 *       pointer-events: none;
 *   }
 *
 * Usage:
 *   const overlay = new ScannerOverlay(canvas, {
 *       numCardsPerRow: 3,
 *       numRows: 3,
 *   });
 *   overlay.resizeCanvas();  // call on init + window resize
 *
 *   // In animation loop:
 *   overlay.setState({ ... });
 *   overlay.draw();
 *
 *   // After capturing a card:
 *   overlay.addThumbnail(dataUrl);
 *
 *   // Moving to next row:
 *   overlay.nextRow();
 */

class ScannerOverlay {
    /**
     * @param {HTMLCanvasElement} canvas  The overlay canvas element
     * @param {Object} opts
     * @param {number} opts.numCardsPerRow  Cards per row (default 3)
     * @param {number} opts.numRows         Total rows (default 3)
     * @param {number} opts.guideHeightPct  Guide rect height as fraction of viewport (default 0.55)
     */
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');

        this.numCardsPerRow = opts.numCardsPerRow ?? 3;
        this.numRows        = opts.numRows        ?? 3;
        this.guideHeightPct = opts.guideHeightPct ?? 0.55;

        // Pokemon card aspect ratio: 63mm x 88mm
        this.CARD_ASPECT = 63 / 88;  // W/H = 0.7159

        // State
        this._state = 'idle';        // idle | detected | ready | capturing | captured
        this._statusText = 'Point at card';
        this._capturedThisRow = 0;
        this._currentRow = 0;
        this._thumbnails = [];       // dataUrl strings for current row
        this._allThumbnails = [];    // all captured thumbnails across rows
        this._pulsePhase = 0;        // for green pulse animation
        this._frameCount = 0;

        // Pre-load thumbnail images
        this._thumbImages = [];

        // Bind resize handler
        this._onResize = () => setTimeout(() => this.resizeCanvas(), 50);
        this._onOrient = () => setTimeout(() => this.resizeCanvas(), 200);
        window.addEventListener('resize', this._onResize);
        window.addEventListener('orientationchange', this._onOrient);
    }

    /**
     * Resize canvas to match CSS layout with devicePixelRatio scaling.
     * Must be called after layout is settled (after video.play()).
     */
    resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        if (w === 0 || h === 0) return;
        this.canvas.width  = w * dpr;
        this.canvas.height = h * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    /**
     * Update overlay state. Call before draw().
     * @param {Object} s
     * @param {string} s.state        One of: idle, detected, ready, capturing, captured
     * @param {string} s.statusText   Status message to display
     * @param {number} s.capturedThisRow  Cards captured in current row (0-based)
     * @param {number} s.currentRow   Current row index (0-based)
     */
    setState(s) {
        if (s.state !== undefined)          this._state = s.state;
        if (s.statusText !== undefined)     this._statusText = s.statusText;
        if (s.capturedThisRow !== undefined) this._capturedThisRow = s.capturedThisRow;
        if (s.currentRow !== undefined)     this._currentRow = s.currentRow;
    }

    /**
     * Add a thumbnail image for the current row.
     * @param {string} dataUrl  JPEG data URL of the captured card
     */
    addThumbnail(dataUrl) {
        this._thumbnails.push(dataUrl);
        this._allThumbnails.push(dataUrl);

        // Preload as Image for canvas drawing
        const img = new Image();
        img.src = dataUrl;
        this._thumbImages.push(img);
    }

    /**
     * Advance to the next row. Clears row thumbnails.
     */
    nextRow() {
        this._currentRow++;
        this._capturedThisRow = 0;
        this._thumbnails = [];
        this._thumbImages = [];
    }

    /**
     * Reset everything (new scan session).
     */
    reset() {
        this._state = 'idle';
        this._statusText = 'Point at card';
        this._capturedThisRow = 0;
        this._currentRow = 0;
        this._thumbnails = [];
        this._allThumbnails = [];
        this._thumbImages = [];
        this._pulsePhase = 0;
        this._frameCount = 0;
    }

    /**
     * Clean up event listeners.
     */
    destroy() {
        window.removeEventListener('resize', this._onResize);
        window.removeEventListener('orientationchange', this._onOrient);
    }

    // ---------------------------------------------------------------
    // Main draw method — call every animation frame
    // ---------------------------------------------------------------

    draw() {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        if (w === 0 || h === 0) return;

        this._frameCount++;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, w, h);

        // Compute guide rectangle (card-shaped, centered)
        const guide = this._computeGuideRect(w, h);

        // 1. Dimmed surround
        this._drawDimmedSurround(ctx, w, h, guide);

        // 2. Guide border
        this._drawGuideBorder(ctx, guide);

        // 3. Corner brackets
        this._drawCornerBrackets(ctx, guide);

        // 4. Status text
        this._drawStatusText(ctx, w, h, guide);

        // 5. Capture counter (top-left)
        this._drawCaptureCounter(ctx, w);

        // 6. Row indicator (top-center)
        this._drawRowIndicator(ctx, w);

        // 7. Thumbnail strip (bottom)
        this._drawThumbnailStrip(ctx, w, h);
    }

    // ---------------------------------------------------------------
    // Guide rectangle computation
    // ---------------------------------------------------------------

    _computeGuideRect(viewW, viewH) {
        const guideH = viewH * this.guideHeightPct;
        const guideW = guideH * this.CARD_ASPECT;
        const x = (viewW - guideW) / 2;
        const y = (viewH - guideH) / 2 - viewH * 0.04; // slightly above center
        return { x, y, w: guideW, h: guideH };
    }

    /**
     * Get the guide rectangle in viewport coordinates (for external hit-testing).
     */
    getGuideRect(viewW, viewH) {
        return this._computeGuideRect(viewW || this.canvas.clientWidth,
                                       viewH || this.canvas.clientHeight);
    }

    // ---------------------------------------------------------------
    // Drawing helpers
    // ---------------------------------------------------------------

    _drawDimmedSurround(ctx, w, h, guide) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillRect(0, 0, w, h);

        // Cut out guide area
        ctx.save();
        ctx.globalCompositeOperation = 'destination-out';
        ctx.beginPath();
        this._roundRect(ctx, guide.x, guide.y, guide.w, guide.h, 12);
        ctx.fill();
        ctx.restore();
    }

    _drawGuideBorder(ctx, guide) {
        const state = this._state;

        if (state === 'capturing') {
            // Green pulsing border
            this._pulsePhase += 0.15;
            const alpha = 0.5 + 0.5 * Math.sin(this._pulsePhase);
            ctx.strokeStyle = `rgba(78, 204, 163, ${alpha})`;
            ctx.lineWidth = 4;
            ctx.setLineDash([]);
        } else if (state === 'ready') {
            // Solid green
            ctx.strokeStyle = '#4ecca3';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
        } else if (state === 'detected') {
            // Solid yellow
            ctx.strokeStyle = '#f1c40f';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
        } else if (state === 'captured') {
            // Bright green, solid
            ctx.strokeStyle = '#4ecca3';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
        } else {
            // Idle: white dashed
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([10, 8]);
        }

        ctx.beginPath();
        this._roundRect(ctx, guide.x, guide.y, guide.w, guide.h, 12);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    _drawCornerBrackets(ctx, guide) {
        const state = this._state;
        const bracketLen = 28;
        const bracketW = 4;
        const inset = 0; // draw right on the guide edge

        // Color matches the border
        if (state === 'capturing') {
            const alpha = 0.6 + 0.4 * Math.sin(this._pulsePhase);
            ctx.strokeStyle = `rgba(78, 204, 163, ${alpha})`;
        } else if (state === 'ready') {
            ctx.strokeStyle = '#4ecca3';
        } else if (state === 'detected') {
            ctx.strokeStyle = '#f1c40f';
        } else if (state === 'captured') {
            ctx.strokeStyle = '#4ecca3';
        } else {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        }

        ctx.lineWidth = bracketW;
        ctx.lineCap = 'round';

        const x1 = guide.x + inset;
        const y1 = guide.y + inset;
        const x2 = guide.x + guide.w - inset;
        const y2 = guide.y + guide.h - inset;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(x1, y1 + bracketLen);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + bracketLen, y1);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(x2 - bracketLen, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + bracketLen);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x1, y2 - bracketLen);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + bracketLen, y2);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x2 - bracketLen, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - bracketLen);
        ctx.stroke();
    }

    _drawStatusText(ctx, w, h, guide) {
        const text = this._statusText;
        if (!text) return;

        const fontSize = Math.max(16, Math.min(22, w * 0.05));
        ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // Position below the guide rect, with some spacing
        const textY = guide.y + guide.h + Math.max(20, h * 0.03);

        // Background pill for readability
        const metrics = ctx.measureText(text);
        const pillW = metrics.width + 32;
        const pillH = fontSize + 16;
        const pillX = (w - pillW) / 2;
        const pillY = textY - 8;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        this._roundRect(ctx, pillX, pillY, pillW, pillH, pillH / 2);
        ctx.fill();

        // Text color based on state
        if (this._state === 'ready' || this._state === 'capturing') {
            ctx.fillStyle = '#4ecca3';
        } else if (this._state === 'detected') {
            ctx.fillStyle = '#f1c40f';
        } else if (this._state === 'captured') {
            ctx.fillStyle = '#4ecca3';
        } else {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        }

        ctx.fillText(text, w / 2, textY);
    }

    _drawCaptureCounter(ctx, w) {
        const current = this._capturedThisRow;
        const total = this.numCardsPerRow;
        const text = `${current}/${total}`;

        ctx.font = '700 18px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';

        // Background pill
        const metrics = ctx.measureText(text);
        const pillW = metrics.width + 20;
        const pillH = 30;
        const pillX = 16;
        const pillY = 14;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.beginPath();
        this._roundRect(ctx, pillX, pillY, pillW, pillH, 8);
        ctx.fill();

        ctx.fillStyle = current >= total ? '#4ecca3' : 'rgba(255, 255, 255, 0.9)';
        ctx.fillText(text, pillX + 10, pillY + 6);
    }

    _drawRowIndicator(ctx, w) {
        const row = this._currentRow + 1;
        const total = this.numRows;
        const text = `Row ${row} of ${total}`;

        ctx.font = '600 15px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // Background pill
        const metrics = ctx.measureText(text);
        const pillW = metrics.width + 24;
        const pillH = 28;
        const pillX = (w - pillW) / 2;
        const pillY = 14;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.beginPath();
        this._roundRect(ctx, pillX, pillY, pillW, pillH, 8);
        ctx.fill();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.fillText(text, w / 2, pillY + 7);
    }

    _drawThumbnailStrip(ctx, w, h) {
        if (this._thumbImages.length === 0 && this._capturedThisRow === 0) return;

        const thumbH = Math.min(72, h * 0.1);
        const thumbW = thumbH * this.CARD_ASPECT;
        const gap = 8;
        const stripH = thumbH + 16;
        const stripY = h - stripH - 8;

        // Background strip
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.beginPath();
        this._roundRect(ctx, 8, stripY, w - 16, stripH, 10);
        ctx.fill();

        // Draw thumbnail slots
        const totalW = this.numCardsPerRow * (thumbW + gap) - gap;
        const startX = (w - totalW) / 2;

        for (let i = 0; i < this.numCardsPerRow; i++) {
            const tx = startX + i * (thumbW + gap);
            const ty = stripY + 8;

            if (i < this._thumbImages.length && this._thumbImages[i].complete) {
                // Filled thumbnail
                ctx.save();
                ctx.beginPath();
                this._roundRect(ctx, tx, ty, thumbW, thumbH, 6);
                ctx.clip();
                ctx.drawImage(this._thumbImages[i], tx, ty, thumbW, thumbH);
                ctx.restore();

                // Green border
                ctx.strokeStyle = '#4ecca3';
                ctx.lineWidth = 2;
                ctx.beginPath();
                this._roundRect(ctx, tx, ty, thumbW, thumbH, 6);
                ctx.stroke();
            } else {
                // Empty slot
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                this._roundRect(ctx, tx, ty, thumbW, thumbH, 6);
                ctx.stroke();
                ctx.setLineDash([]);

                // Slot number
                ctx.font = '500 13px -apple-system, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.fillText(String(i + 1), tx + thumbW / 2, ty + thumbH / 2);
            }
        }
    }

    // ---------------------------------------------------------------
    // Utility: rounded rectangle path
    // ---------------------------------------------------------------

    _roundRect(ctx, x, y, w, h, r) {
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
    }
}


// ---------------------------------------------------------------
// CSS for the overlay canvas (inject via <style> or copy to CSS)
// ---------------------------------------------------------------

ScannerOverlay.CSS = `
/* ---- Overlay canvas (guides + HUD) ---- */
canvas#overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    pointer-events: none;
}
`;


// ---------------------------------------------------------------
// Self-test: verifies core methods don't throw
// ---------------------------------------------------------------

ScannerOverlay.selfTest = function () {
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

    console.log('=== ScannerOverlay Self-Test ===');

    // Test constructor defaults
    {
        // Create a mock canvas (no DOM needed for unit tests)
        const mockCanvas = {
            clientWidth: 375,
            clientHeight: 667,
            width: 0,
            height: 0,
            getContext: function () {
                return {
                    setTransform: function () {},
                    clearRect: function () {},
                    fillRect: function () {},
                    fillText: function () {},
                    fill: function () {},
                    stroke: function () {},
                    beginPath: function () {},
                    moveTo: function () {},
                    lineTo: function () {},
                    quadraticCurveTo: function () {},
                    closePath: function () {},
                    save: function () {},
                    restore: function () {},
                    clip: function () {},
                    drawImage: function () {},
                    measureText: function () { return { width: 50 }; },
                    setLineDash: function () {},
                    globalCompositeOperation: '',
                    strokeStyle: '',
                    fillStyle: '',
                    lineWidth: 1,
                    lineCap: '',
                    font: '',
                    textAlign: '',
                    textBaseline: '',
                };
            },
        };

        const ov = new ScannerOverlay(mockCanvas);
        assert(ov.numCardsPerRow === 3, 'default numCardsPerRow = 3');
        assert(ov.numRows === 3, 'default numRows = 3');
        assert(ov._state === 'idle', 'initial state = idle');
        assert(ov._thumbnails.length === 0, 'initial thumbnails empty');
    }

    // Test guide rect computation
    {
        const mockCanvas = {
            clientWidth: 375,
            clientHeight: 667,
            width: 375,
            height: 667,
            getContext: function () { return {}; },
        };
        const ov = new ScannerOverlay(mockCanvas);
        const rect = ov._computeGuideRect(375, 667);

        assert(rect.w > 0, 'guide width > 0');
        assert(rect.h > 0, 'guide height > 0');
        assert(Math.abs(rect.w / rect.h - 63 / 88) < 0.01, 'guide has card aspect ratio');
        assert(rect.x > 0, 'guide not at left edge');
        assert(rect.y > 0, 'guide not at top edge');

        // Centered horizontally
        const centerX = rect.x + rect.w / 2;
        assert(Math.abs(centerX - 375 / 2) < 1, 'guide centered horizontally');
    }

    // Test setState
    {
        const mockCanvas = {
            clientWidth: 375, clientHeight: 667,
            getContext: function () { return {}; },
        };
        const ov = new ScannerOverlay(mockCanvas);
        ov.setState({ state: 'ready', statusText: 'Hold steady...', capturedThisRow: 2, currentRow: 1 });
        assert(ov._state === 'ready', 'setState updates state');
        assert(ov._statusText === 'Hold steady...', 'setState updates statusText');
        assert(ov._capturedThisRow === 2, 'setState updates capturedThisRow');
        assert(ov._currentRow === 1, 'setState updates currentRow');
    }

    // Test addThumbnail and nextRow
    {
        const mockCanvas = {
            clientWidth: 375, clientHeight: 667,
            getContext: function () { return {}; },
        };
        const ov = new ScannerOverlay(mockCanvas);
        ov.addThumbnail('data:image/jpeg;base64,/9j/test');
        assert(ov._thumbnails.length === 1, 'addThumbnail adds to current row');
        assert(ov._allThumbnails.length === 1, 'addThumbnail adds to all thumbnails');

        ov.nextRow();
        assert(ov._currentRow === 1, 'nextRow increments row');
        assert(ov._thumbnails.length === 0, 'nextRow clears row thumbnails');
        assert(ov._allThumbnails.length === 1, 'nextRow preserves allThumbnails');
    }

    // Test reset
    {
        const mockCanvas = {
            clientWidth: 375, clientHeight: 667,
            getContext: function () { return {}; },
        };
        const ov = new ScannerOverlay(mockCanvas);
        ov.setState({ state: 'ready', currentRow: 2, capturedThisRow: 1 });
        ov.addThumbnail('data:test');
        ov.reset();
        assert(ov._state === 'idle', 'reset restores idle state');
        assert(ov._currentRow === 0, 'reset restores row 0');
        assert(ov._thumbnails.length === 0, 'reset clears thumbnails');
        assert(ov._allThumbnails.length === 0, 'reset clears all thumbnails');
    }

    // Test draw does not throw (smoke test with mock canvas)
    {
        const mockCanvas = {
            clientWidth: 375,
            clientHeight: 667,
            width: 750,
            height: 1334,
            getContext: function () {
                return {
                    setTransform: function () {},
                    clearRect: function () {},
                    fillRect: function () {},
                    fillText: function () {},
                    fill: function () {},
                    stroke: function () {},
                    beginPath: function () {},
                    moveTo: function () {},
                    lineTo: function () {},
                    quadraticCurveTo: function () {},
                    closePath: function () {},
                    save: function () {},
                    restore: function () {},
                    clip: function () {},
                    drawImage: function () {},
                    measureText: function () { return { width: 60 }; },
                    setLineDash: function () {},
                    globalCompositeOperation: '',
                    strokeStyle: '',
                    fillStyle: '',
                    lineWidth: 1,
                    lineCap: '',
                    font: '',
                    textAlign: '',
                    textBaseline: '',
                };
            },
        };

        const ov = new ScannerOverlay(mockCanvas);

        // Test all states
        const states = ['idle', 'detected', 'ready', 'capturing', 'captured'];
        let threw = false;
        for (const s of states) {
            try {
                ov.setState({ state: s, statusText: 'Test ' + s });
                ov.draw();
            } catch (e) {
                console.error('  draw() threw for state=' + s + ':', e);
                threw = true;
            }
        }
        assert(!threw, 'draw() does not throw for any state');

        // Test with thumbnails
        try {
            ov.addThumbnail('data:image/jpeg;base64,/9j/fake');
            ov.setState({ capturedThisRow: 1 });
            ov.draw();
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'draw() does not throw with thumbnails');
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScannerOverlay;
}
