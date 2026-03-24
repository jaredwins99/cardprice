/**
 * ScannerOverlay — Real-time overlay rendering for the binder page scanner camera.
 *
 * Draws card outlines, status indicators, progress ring, row dots, capture
 * flash, transition prompts, and a completion grid on top of the live video
 * feed.  All rendering happens on a single overlay canvas positioned
 * absolutely over the <video> element, driven by requestAnimationFrame.
 *
 * CSS for the overlay canvas:
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
 *   const overlay = new ScannerOverlay(canvasElement);
 *   overlay.resizeCanvas();
 *
 *   // In your rAF loop:
 *   overlay.render(state);
 *
 *   // state = {
 *   //   cards: [{ x, y, w, h, status: 'detecting'|'stabilizing'|'locked' }],
 *   //   progress: 0-100,
 *   //   currentRow: 0-2,
 *   //   rowsCompleted: [true, false, false],
 *   //   capturedCount: 0-9,
 *   //   totalCards: 9,
 *   //   statusText: 'Scanning...',
 *   //   elapsed: 12.3,
 *   //   flash: false,
 *   //   showTransition: false,
 *   //   transitionPausedMs: 0,
 *   //   completed: false,
 *   //   thumbnails: [dataUrl, ...],
 *   // }
 */

class ScannerOverlay {
    /**
     * @param {HTMLCanvasElement} canvas  The overlay canvas element
     * @param {Object} opts
     * @param {number} opts.numCardsPerRow  Cards per row (default 3)
     * @param {number} opts.numRows         Total rows (default 3)
     */
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');

        this.numCardsPerRow = opts.numCardsPerRow ?? 3;
        this.numRows        = opts.numRows        ?? 3;

        // Colors
        this.COLOR_RED    = '#e74c3c';
        this.COLOR_YELLOW = '#f1c40f';
        this.COLOR_GREEN  = '#4ecca3';
        this.COLOR_WHITE  = 'rgba(255, 255, 255, 0.85)';

        // Animation state
        this._frameCount   = 0;
        this._pulsePhase   = 0;
        this._flashAlpha   = 0;        // capture flash opacity (decays)
        this._flashStartMs = 0;
        this._shrinkCards   = [];       // [{x,y,w,h, startMs}] for shrink animation
        this._arrowPhase   = 0;        // transition arrow animation

        // Completion state
        this._completionThumbnails = []; // Image objects for the 3x3 grid
        this._submitCallback = null;

        // Bind resize handler
        this._onResize = () => setTimeout(() => this.resizeCanvas(), 50);
        this._onOrient = () => setTimeout(() => this.resizeCanvas(), 200);
        window.addEventListener('resize', this._onResize);
        window.addEventListener('orientationchange', this._onOrient);
    }

    /**
     * Resize canvas to match CSS layout with devicePixelRatio scaling.
     * Call on init + window resize.
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
     * Trigger a capture flash + card shrink animation.
     * Call this when auto-capture fires.
     * @param {Array} cardRects  Array of {x,y,w,h} for cards to animate
     * @param {number} completedRow  Row index that was just completed (0-based)
     */
    triggerCaptureFlash(cardRects, completedRow) {
        const now = performance.now();
        this._flashAlpha   = 1.0;
        this._flashStartMs = now;
        this._shrinkCards  = (cardRects || []).map(r => ({
            x: r.x, y: r.y, w: r.w, h: r.h, startMs: now
        }));
    }

    /**
     * Set callback for the "Submit for ID" button on completion screen.
     * @param {function} cb  Callback function
     */
    onSubmit(cb) {
        this._submitCallback = cb;
    }

    /**
     * Add a thumbnail for the completion grid.
     * @param {string} dataUrl  JPEG/PNG data URL
     */
    addThumbnail(dataUrl) {
        const img = new Image();
        img.src = dataUrl;
        this._completionThumbnails.push(img);
    }

    /**
     * Reset all overlay state for a new scan session.
     */
    reset() {
        this._frameCount   = 0;
        this._pulsePhase   = 0;
        this._flashAlpha   = 0;
        this._flashStartMs = 0;
        this._shrinkCards  = [];
        this._arrowPhase   = 0;
        this._completionThumbnails = [];
    }

    /**
     * Clean up event listeners.
     */
    destroy() {
        window.removeEventListener('resize', this._onResize);
        window.removeEventListener('orientationchange', this._onOrient);
    }

    // ---------------------------------------------------------------
    // Main render method — call every requestAnimationFrame
    // ---------------------------------------------------------------

    /**
     * Render the full overlay.
     * @param {Object} state  State from ScannerAutoCapture.update() or equivalent
     * @param {Array}  state.cards           [{x,y,w,h, status:'detecting'|'stabilizing'|'locked'}]
     * @param {number} state.progress        Auto-capture progress 0-100
     * @param {number} state.currentRow      Current row index (0-based)
     * @param {Array}  state.rowsCompleted   [bool, bool, bool] per-row completion
     * @param {number} state.capturedCount   Total cards captured so far
     * @param {number} state.totalCards      Total cards to capture (default 9)
     * @param {string} state.statusText      Status message
     * @param {number} state.elapsed         Elapsed seconds
     * @param {boolean} state.flash          Whether flash is active (alternative to triggerCaptureFlash)
     * @param {boolean} state.showTransition Show transition overlay between rows
     * @param {number} state.transitionPausedMs  Ms the user has paused during transition
     * @param {boolean} state.completed      All rows done — show completion screen
     * @param {Array}  state.thumbnails      Array of data URLs for completion grid
     */
    render(state) {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        if (w === 0 || h === 0) return;

        this._frameCount++;
        this._pulsePhase += 0.08;
        this._arrowPhase += 0.04;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, w, h);

        const now = performance.now();
        const s = state || {};

        // Handle completion screen
        if (s.completed) {
            this._drawCompletionOverlay(ctx, w, h, s);
            return;
        }

        // Handle transition overlay
        if (s.showTransition) {
            this._drawTransitionOverlay(ctx, w, h, s);
        }

        // 1. Viewfinder dimming outside card areas
        this._drawViewfinderDim(ctx, w, h, s.cards || []);

        // 2. Card outlines with status colors
        this._drawCardOutlines(ctx, s.cards || [], now);

        // 3. Shrink animations
        this._drawShrinkAnimations(ctx, now);

        // 4. Capture flash
        this._drawCaptureFlash(ctx, w, h, now);

        // 5. Progress ring (top-right corner)
        this._drawProgressRing(ctx, w, s.progress || 0);

        // 6. Row indicator (top center)
        this._drawRowIndicator(ctx, w, s.currentRow || 0, s.rowsCompleted || []);

        // 7. Status bar (bottom)
        this._drawStatusBar(ctx, w, h, s);
    }

    // ---------------------------------------------------------------
    // Viewfinder dimming
    // ---------------------------------------------------------------

    _drawViewfinderDim(ctx, w, h, cards) {
        // Dark overlay everywhere
        ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.fillRect(0, 0, w, h);

        // Cut out card regions (clear those areas)
        if (cards.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = 'destination-out';
            for (const card of cards) {
                ctx.beginPath();
                this._roundRect(ctx, card.x, card.y, card.w, card.h, 6);
                ctx.fill();
            }
            ctx.restore();
        }
    }

    // ---------------------------------------------------------------
    // Card outlines
    // ---------------------------------------------------------------

    _drawCardOutlines(ctx, cards, now) {
        for (const card of cards) {
            const status = card.status || 'detecting';

            if (status === 'detecting') {
                this._drawDetectingOutline(ctx, card);
            } else if (status === 'stabilizing') {
                this._drawStabilizingOutline(ctx, card);
            } else if (status === 'locked') {
                this._drawLockedOutline(ctx, card);
            }
        }
    }

    /**
     * Detecting: thin dashed line, corners only (red)
     */
    _drawDetectingOutline(ctx, card) {
        const { x, y, w, h } = card;
        const cornerLen = Math.min(w, h) * 0.2;

        ctx.strokeStyle = this.COLOR_RED;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.lineCap = 'round';

        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        ctx.stroke();

        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(x + w - cornerLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + cornerLen);
        ctx.stroke();

        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + h - cornerLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + cornerLen, y + h);
        ctx.stroke();

        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(x + w - cornerLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - cornerLen);
        ctx.stroke();

        ctx.setLineDash([]);
    }

    /**
     * Stabilizing: thicker solid line, pulsing opacity (yellow)
     */
    _drawStabilizingOutline(ctx, card) {
        const { x, y, w, h } = card;
        const alpha = 0.5 + 0.5 * Math.sin(this._pulsePhase * 3);

        ctx.strokeStyle = `rgba(241, 196, 15, ${alpha})`;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.lineCap = 'round';

        ctx.beginPath();
        this._roundRect(ctx, x, y, w, h, 6);
        ctx.stroke();
    }

    /**
     * Locked: thick solid line, corner brackets, steady (green)
     */
    _drawLockedOutline(ctx, card) {
        const { x, y, w, h } = card;
        const bracketLen = Math.min(w, h) * 0.22;

        // Solid border
        ctx.strokeStyle = this.COLOR_GREEN;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.beginPath();
        this._roundRect(ctx, x, y, w, h, 6);
        ctx.stroke();

        // Corner brackets (thicker, on top)
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';

        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + bracketLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + bracketLen, y);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + w - bracketLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + bracketLen);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + h - bracketLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + bracketLen, y + h);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + w - bracketLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - bracketLen);
        ctx.stroke();
    }

    // ---------------------------------------------------------------
    // Shrink animation (on capture)
    // ---------------------------------------------------------------

    _drawShrinkAnimations(ctx, now) {
        const SHRINK_DURATION_MS = 300;
        const remaining = [];

        for (const s of this._shrinkCards) {
            const elapsed = now - s.startMs;
            if (elapsed > SHRINK_DURATION_MS) continue;

            const t = elapsed / SHRINK_DURATION_MS;
            // Ease-out: fast start, slow end
            const ease = 1 - Math.pow(1 - t, 3);
            const scale = 1 - ease * 0.15;  // shrink to 85%

            const cx = s.x + s.w / 2;
            const cy = s.y + s.h / 2;
            const nw = s.w * scale;
            const nh = s.h * scale;
            const nx = cx - nw / 2;
            const ny = cy - nh / 2;

            // White border with fading opacity
            const alpha = 1 - ease;
            ctx.strokeStyle = `rgba(78, 204, 163, ${alpha})`;
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.beginPath();
            this._roundRect(ctx, nx, ny, nw, nh, 6);
            ctx.stroke();

            remaining.push(s);
        }

        this._shrinkCards = remaining;
    }

    // ---------------------------------------------------------------
    // Capture flash
    // ---------------------------------------------------------------

    _drawCaptureFlash(ctx, w, h, now) {
        const FLASH_DURATION_MS = 200;

        if (this._flashAlpha > 0) {
            const elapsed = now - this._flashStartMs;
            if (elapsed < FLASH_DURATION_MS) {
                // Fast fade out
                const t = elapsed / FLASH_DURATION_MS;
                this._flashAlpha = 1 - t;
            } else {
                this._flashAlpha = 0;
            }
        }

        if (this._flashAlpha > 0.01) {
            ctx.fillStyle = `rgba(78, 204, 163, ${this._flashAlpha * 0.3})`;
            ctx.fillRect(0, 0, w, h);
        }
    }

    // ---------------------------------------------------------------
    // Progress ring
    // ---------------------------------------------------------------

    _drawProgressRing(ctx, canvasW, progress) {
        const radius  = 22;
        const cx      = canvasW - 20 - radius;
        const cy      = 20 + radius;
        const lineW   = 4;

        // Background ring
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = lineW;
        ctx.stroke();

        // Progress arc
        if (progress > 0) {
            const endAngle = -Math.PI / 2 + (Math.PI * 2 * Math.min(progress, 100) / 100);
            ctx.beginPath();
            ctx.arc(cx, cy, radius, -Math.PI / 2, endAngle);

            // Color transitions: red < 33, yellow < 66, green >= 66
            if (progress < 33) {
                ctx.strokeStyle = this.COLOR_RED;
            } else if (progress < 66) {
                ctx.strokeStyle = this.COLOR_YELLOW;
            } else {
                ctx.strokeStyle = this.COLOR_GREEN;
            }
            ctx.lineWidth = lineW;
            ctx.lineCap = 'round';
            ctx.stroke();
        }

        // Percentage text
        const pctText = Math.round(progress) + '%';
        ctx.font = '600 11px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = this.COLOR_WHITE;
        this._textWithShadow(ctx, pctText, cx, cy);
    }

    // ---------------------------------------------------------------
    // Row indicator
    // ---------------------------------------------------------------

    _drawRowIndicator(ctx, canvasW, currentRow, rowsCompleted) {
        const totalRows = this.numRows;
        const rowLabel = `Row ${currentRow + 1}/${totalRows}`;
        const dotRadius = 6;
        const dotGap = 18;
        const totalDotsW = totalRows * dotRadius * 2 + (totalRows - 1) * (dotGap - dotRadius * 2);

        // Background pill
        ctx.font = '600 14px -apple-system, BlinkMacSystemFont, sans-serif';
        const textMetrics = ctx.measureText(rowLabel);
        const pillW = Math.max(textMetrics.width + 24, totalDotsW + 24);
        const pillH = 50;
        const pillX = (canvasW - pillW) / 2;
        const pillY = 12;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.beginPath();
        this._roundRect(ctx, pillX, pillY, pillW, pillH, 10);
        ctx.fill();

        // Row text
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = this.COLOR_WHITE;
        this._textWithShadow(ctx, rowLabel, canvasW / 2, pillY + 8);

        // Row dots
        const dotsY = pillY + 32;
        const dotsStartX = canvasW / 2 - (totalRows - 1) * dotGap / 2;

        for (let i = 0; i < totalRows; i++) {
            const dx = dotsStartX + i * dotGap;
            ctx.beginPath();
            ctx.arc(dx, dotsY, dotRadius, 0, Math.PI * 2);

            if (rowsCompleted[i]) {
                // Completed: solid green
                ctx.fillStyle = this.COLOR_GREEN;
                ctx.fill();
            } else if (i === currentRow) {
                // Current: yellow outline, pulsing
                const alpha = 0.5 + 0.5 * Math.sin(this._pulsePhase * 2);
                ctx.strokeStyle = `rgba(241, 196, 15, ${alpha})`;
                ctx.lineWidth = 2;
                ctx.stroke();
            } else {
                // Future: dim outline
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }
    }

    // ---------------------------------------------------------------
    // Transition overlay (between rows)
    // ---------------------------------------------------------------

    _drawTransitionOverlay(ctx, w, h, state) {
        // Semi-transparent backdrop
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillRect(0, 0, w, h);

        // Animated downward arrow
        const arrowX = w / 2;
        const arrowBaseY = h * 0.4;
        const bobOffset = Math.sin(this._arrowPhase * 2) * 8;
        const arrowY = arrowBaseY + bobOffset;
        const arrowSize = 30;

        ctx.strokeStyle = this.COLOR_GREEN;
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Arrow shaft
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY - arrowSize);
        ctx.lineTo(arrowX, arrowY + arrowSize);
        ctx.stroke();

        // Arrow head
        ctx.beginPath();
        ctx.moveTo(arrowX - arrowSize * 0.6, arrowY + arrowSize * 0.4);
        ctx.lineTo(arrowX, arrowY + arrowSize);
        ctx.lineTo(arrowX + arrowSize * 0.6, arrowY + arrowSize * 0.4);
        ctx.stroke();

        // "Move to next row" text — only show if user has paused > 2s
        const pausedMs = state.transitionPausedMs || 0;
        if (pausedMs > 2000) {
            // Fade in over 500ms
            const fadeT = Math.min(1, (pausedMs - 2000) / 500);

            ctx.globalAlpha = fadeT;
            ctx.font = '600 20px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            this._textWithShadow(ctx, 'Move to next row', w / 2, arrowY + arrowSize + 30);
            ctx.globalAlpha = 1;
        }
    }

    // ---------------------------------------------------------------
    // Completion overlay
    // ---------------------------------------------------------------

    _drawCompletionOverlay(ctx, w, h, state) {
        // Full dark background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.fillRect(0, 0, w, h);

        const thumbnails = state.thumbnails || [];
        const totalCards = state.totalCards || this.numCardsPerRow * this.numRows;

        // Title
        ctx.font = '700 22px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = this.COLOR_GREEN;
        this._textWithShadow(ctx, 'Scan Complete', w / 2, 30);

        // Card count
        ctx.font = '500 16px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = this.COLOR_WHITE;
        this._textWithShadow(ctx, `${thumbnails.length} card${thumbnails.length !== 1 ? 's' : ''} captured`, w / 2, 60);

        // 3x3 thumbnail grid
        const gridCols = this.numCardsPerRow;
        const gridRows = this.numRows;
        const cardAspect = 63 / 88;  // Pokemon card W/H

        // Compute grid dimensions to fit in available space
        const gridAreaW = w * 0.8;
        const gridAreaH = h * 0.55;
        const gridTop = 90;
        const gridGap = 8;

        const cellW = (gridAreaW - (gridCols - 1) * gridGap) / gridCols;
        const cellH = cellW / cardAspect;
        const actualGridH = gridRows * cellH + (gridRows - 1) * gridGap;

        // If grid is too tall, scale by height instead
        let scale = 1;
        if (actualGridH > gridAreaH) {
            scale = gridAreaH / actualGridH;
        }
        const finalCellW = cellW * scale;
        const finalCellH = cellH * scale;
        const finalGridW = gridCols * finalCellW + (gridCols - 1) * gridGap;
        const finalGridH = gridRows * finalCellH + (gridRows - 1) * gridGap;

        const gridLeft = (w - finalGridW) / 2;
        const gridY = gridTop;

        // Ensure we have Image objects for drawing
        this._ensureCompletionThumbs(thumbnails);

        for (let row = 0; row < gridRows; row++) {
            for (let col = 0; col < gridCols; col++) {
                const idx = row * gridCols + col;
                const cx = gridLeft + col * (finalCellW + gridGap);
                const cy = gridY + row * (finalCellH + gridGap);

                // Card background
                ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
                ctx.beginPath();
                this._roundRect(ctx, cx, cy, finalCellW, finalCellH, 6);
                ctx.fill();

                if (idx < this._completionThumbnails.length && this._completionThumbnails[idx].complete) {
                    // Draw thumbnail
                    ctx.save();
                    ctx.beginPath();
                    this._roundRect(ctx, cx, cy, finalCellW, finalCellH, 6);
                    ctx.clip();
                    ctx.drawImage(this._completionThumbnails[idx], cx, cy, finalCellW, finalCellH);
                    ctx.restore();

                    // Green border
                    ctx.strokeStyle = this.COLOR_GREEN;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    this._roundRect(ctx, cx, cy, finalCellW, finalCellH, 6);
                    ctx.stroke();
                } else {
                    // Empty slot
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    this._roundRect(ctx, cx, cy, finalCellW, finalCellH, 6);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Slot number
                    ctx.font = '400 12px -apple-system, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                    ctx.fillText(String(idx + 1), cx + finalCellW / 2, cy + finalCellH / 2);
                }
            }
        }

        // "Submit for ID" button
        const btnW = Math.min(240, w * 0.6);
        const btnH = 48;
        const btnX = (w - btnW) / 2;
        const btnY = gridY + finalGridH + 30;

        // Button background
        ctx.fillStyle = this.COLOR_GREEN;
        ctx.beginPath();
        this._roundRect(ctx, btnX, btnY, btnW, btnH, btnH / 2);
        ctx.fill();

        // Button text
        ctx.font = '700 17px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#1a1a2e';
        ctx.fillText('Submit for ID', w / 2, btnY + btnH / 2);

        // Store button rect for hit testing
        this._submitBtnRect = { x: btnX, y: btnY, w: btnW, h: btnH };
    }

    /**
     * Ensure completion thumbnail Image objects are synced with data URLs.
     * Only creates new Image objects for URLs not yet loaded.
     */
    _ensureCompletionThumbs(thumbnails) {
        for (let i = this._completionThumbnails.length; i < thumbnails.length; i++) {
            const img = new Image();
            img.src = thumbnails[i];
            this._completionThumbnails.push(img);
        }
    }

    /**
     * Hit-test the "Submit for ID" button.
     * Call this from a click/tap handler on the overlay canvas.
     * @param {number} x  Click x in CSS pixels
     * @param {number} y  Click y in CSS pixels
     * @returns {boolean} True if the submit button was clicked
     */
    hitTestSubmit(x, y) {
        const r = this._submitBtnRect;
        if (!r) return false;
        return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
    }

    // ---------------------------------------------------------------
    // Status bar
    // ---------------------------------------------------------------

    _drawStatusBar(ctx, w, h, state) {
        const barH = 52;
        const barY = h - barH;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        this._roundRect(ctx, 8, barY, w - 16, barH - 4, 10);
        ctx.fill();

        const innerY = barY + barH / 2;

        // Left: status text
        const statusText = state.statusText || 'Scanning...';
        ctx.font = '600 14px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';

        // Status text color based on content
        if (statusText.includes('done') || statusText.includes('Done') || statusText.includes('Captured')) {
            ctx.fillStyle = this.COLOR_GREEN;
        } else if (statusText.includes('Hold') || statusText.includes('steady')) {
            ctx.fillStyle = this.COLOR_YELLOW;
        } else {
            ctx.fillStyle = this.COLOR_WHITE;
        }
        this._textWithShadow(ctx, statusText, 22, innerY);

        // Right: cards captured + elapsed time
        const captured = state.capturedCount || 0;
        const total = state.totalCards || (this.numCardsPerRow * this.numRows);
        const elapsed = state.elapsed || 0;
        const elapsedStr = this._formatElapsed(elapsed);

        const rightText = `${captured}/${total}  ${elapsedStr}`;
        ctx.font = '500 13px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        this._textWithShadow(ctx, rightText, w - 22, innerY);
    }

    // ---------------------------------------------------------------
    // Utility methods
    // ---------------------------------------------------------------

    /**
     * Draw text with a dark shadow for readability over any background.
     */
    _textWithShadow(ctx, text, x, y) {
        ctx.save();
        ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.fillText(text, x, y);
        ctx.restore();
    }

    /**
     * Format elapsed seconds as "M:SS".
     */
    _formatElapsed(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }

    /**
     * Draw a rounded rectangle path (does NOT stroke or fill).
     */
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

/* Enable pointer-events on completion screen for Submit button */
canvas#overlay.interactive {
    pointer-events: auto;
}
`;


// ---------------------------------------------------------------
// Self-test: exercises all rendering paths with a mock canvas
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

    // Create a mock canvas context
    function mockCanvas() {
        return {
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
                    arc: function () {},
                    closePath: function () {},
                    save: function () {},
                    restore: function () {},
                    clip: function () {},
                    drawImage: function () {},
                    measureText: function () { return { width: 60 }; },
                    setLineDash: function () {},
                    globalCompositeOperation: '',
                    globalAlpha: 1,
                    strokeStyle: '',
                    fillStyle: '',
                    lineWidth: 1,
                    lineCap: '',
                    lineJoin: '',
                    font: '',
                    textAlign: '',
                    textBaseline: '',
                    shadowColor: '',
                    shadowBlur: 0,
                    shadowOffsetX: 0,
                    shadowOffsetY: 0,
                };
            },
        };
    }

    // Test constructor defaults
    {
        const ov = new ScannerOverlay(mockCanvas());
        assert(ov.numCardsPerRow === 3, 'default numCardsPerRow = 3');
        assert(ov.numRows === 3, 'default numRows = 3');
        assert(ov._frameCount === 0, 'initial frameCount = 0');
        assert(ov._flashAlpha === 0, 'initial flashAlpha = 0');
        assert(ov._completionThumbnails.length === 0, 'initial thumbnails empty');
    }

    // Test reset
    {
        const ov = new ScannerOverlay(mockCanvas());
        ov._frameCount = 100;
        ov._flashAlpha = 0.5;
        ov.addThumbnail('data:test');
        ov.reset();
        assert(ov._frameCount === 0, 'reset clears frameCount');
        assert(ov._flashAlpha === 0, 'reset clears flashAlpha');
        assert(ov._completionThumbnails.length === 0, 'reset clears thumbnails');
    }

    // Test render with detecting cards (no throw)
    {
        const ov = new ScannerOverlay(mockCanvas());
        let threw = false;
        try {
            ov.render({
                cards: [
                    { x: 10, y: 10, w: 80, h: 112, status: 'detecting' },
                    { x: 100, y: 10, w: 80, h: 112, status: 'stabilizing' },
                    { x: 190, y: 10, w: 80, h: 112, status: 'locked' },
                ],
                progress: 45,
                currentRow: 1,
                rowsCompleted: [true, false, false],
                capturedCount: 3,
                totalCards: 9,
                statusText: 'Hold steady...',
                elapsed: 15.5,
            });
        } catch (e) {
            console.error('  render() threw:', e);
            threw = true;
        }
        assert(!threw, 'render() with cards does not throw');
    }

    // Test render with transition overlay
    {
        const ov = new ScannerOverlay(mockCanvas());
        let threw = false;
        try {
            ov.render({
                cards: [],
                progress: 0,
                currentRow: 1,
                rowsCompleted: [true, false, false],
                capturedCount: 3,
                totalCards: 9,
                statusText: 'Captured! Move down',
                elapsed: 20,
                showTransition: true,
                transitionPausedMs: 3000,
            });
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'render() with transition does not throw');
    }

    // Test render completion screen
    {
        const ov = new ScannerOverlay(mockCanvas());
        let threw = false;
        try {
            ov.render({
                completed: true,
                thumbnails: ['data:img1', 'data:img2', 'data:img3'],
                capturedCount: 9,
                totalCards: 9,
                statusText: 'All done!',
                elapsed: 45,
            });
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'render() completion screen does not throw');
    }

    // Test render with empty state
    {
        const ov = new ScannerOverlay(mockCanvas());
        let threw = false;
        try {
            ov.render({});
            ov.render(null);
            ov.render();
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'render() with empty/null state does not throw');
    }

    // Test capture flash trigger
    {
        const ov = new ScannerOverlay(mockCanvas());
        ov.triggerCaptureFlash([{ x: 10, y: 10, w: 80, h: 112 }], 0);
        assert(ov._flashAlpha === 1.0, 'triggerCaptureFlash sets flashAlpha to 1');
        assert(ov._shrinkCards.length === 1, 'triggerCaptureFlash creates shrink cards');
    }

    // Test hitTestSubmit
    {
        const ov = new ScannerOverlay(mockCanvas());
        ov._submitBtnRect = { x: 100, y: 400, w: 200, h: 48 };
        assert(ov.hitTestSubmit(200, 420) === true, 'hitTestSubmit inside returns true');
        assert(ov.hitTestSubmit(50, 420) === false, 'hitTestSubmit outside returns false');
        assert(ov.hitTestSubmit(200, 200) === false, 'hitTestSubmit above returns false');
    }

    // Test _formatElapsed
    {
        const ov = new ScannerOverlay(mockCanvas());
        assert(ov._formatElapsed(0) === '0:00', 'format 0s');
        assert(ov._formatElapsed(5) === '0:05', 'format 5s');
        assert(ov._formatElapsed(65) === '1:05', 'format 65s');
        assert(ov._formatElapsed(125) === '2:05', 'format 125s');
    }

    // Test addThumbnail
    {
        const ov = new ScannerOverlay(mockCanvas());
        ov.addThumbnail('data:image/jpeg;base64,/9j/test');
        assert(ov._completionThumbnails.length === 1, 'addThumbnail adds one image');
    }

    // Test _ensureCompletionThumbs only adds new
    {
        const ov = new ScannerOverlay(mockCanvas());
        ov._ensureCompletionThumbs(['a', 'b', 'c']);
        assert(ov._completionThumbnails.length === 3, 'ensureCompletionThumbs creates 3');
        ov._ensureCompletionThumbs(['a', 'b', 'c', 'd']);
        assert(ov._completionThumbnails.length === 4, 'ensureCompletionThumbs adds only new');
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScannerOverlay;
}
