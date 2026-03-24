/**
 * ScannerAutoCapture -- Stability tracking and auto-capture for the card scanner.
 *
 * The scanner auto-captures when cards are stable and sharp -- no button press
 * needed.  Each frame, the caller passes detected card contours.  This class
 * tracks per-card stability (corner movement over time), checks sharpness on
 * locked cards, and fires a capture once all expected cards are locked and sharp.
 *
 * After capture, it monitors for row transitions: the old cards leaving the
 * frame and new cards entering, then resets for the next row.
 *
 * State machine:
 *
 *   SCANNING ──(cards appear)──> STABILIZING ──(all locked+sharp)──> READY
 *       ^                                                              │
 *       │                                                         (capture)
 *       │                                                              │
 *       │                                                              v
 *   TRANSITIONING <──(old cards leave)───────────────────────── CAPTURED
 *       │
 *       └──(new cards stabilize)──> SCANNING (next row)
 *
 * Per-card status progression:
 *   detecting  (0-3 stable frames)
 *   stabilizing (4 to stabilityFrames-1)
 *   locked     (stabilityFrames+)
 *
 * Progress indicator:
 *   0.0  = no cards detected
 *   0.33 = 1 of 3 cards locked
 *   0.66 = 2 of 3 locked
 *   0.9  = all 3 locked, checking sharpness
 *   1.0  = capture!
 *
 * Usage:
 *   const autoCapture = new ScannerAutoCapture({ expectedCards: 3 });
 *
 *   // In your rAF loop, after detecting card contours:
 *   const result = autoCapture.update(detectedCards);
 *   // result.state       -- 'scanning' | 'stabilizing' | 'ready' | 'captured' | 'transitioning'
 *   // result.cards       -- per-card tracking info
 *   // result.progress    -- 0-1 progress toward capture
 *   // result.captureReady -- true when capture just fired
 *
 *   // After capture fires:
 *   autoCapture.recordCapture(rowIndex);
 *
 *   // Row transition detection:
 *   const transitioning = autoCapture.detectRowTransition(detectedCards);
 */

class ScannerAutoCapture {
    /**
     * @param {Object} opts
     * @param {number} opts.stabilityFrames    Consecutive stable frames to reach "locked" (default 12, ~1.2s at 10fps)
     * @param {number} opts.sharpnessThreshold Laplacian variance threshold for sharp (default 80)
     * @param {number} opts.movementThreshold  Max corner movement in px to count as stable (default 8)
     * @param {number} opts.expectedCards      Number of cards expected per row (default 3)
     * @param {number} opts.matchRadius        Max px between card centers to match across frames (default 50)
     * @param {number} opts.transitionFrames   Frames with <1 visible card before transitioning (default 5)
     * @param {number} opts.detectingMax       Stable frames for "detecting" status (default 3)
     */
    constructor(opts = {}) {
        this.stabilityFrames    = opts.stabilityFrames    ?? 12;
        this.sharpnessThreshold = opts.sharpnessThreshold ?? 80;
        this.movementThreshold  = opts.movementThreshold  ?? 8;
        this.expectedCards      = opts.expectedCards      ?? 3;
        this.matchRadius        = opts.matchRadius        ?? 50;
        this.transitionFrames   = opts.transitionFrames   ?? 5;
        this.detectingMax       = opts.detectingMax       ?? 3;

        // Internal state
        this._state             = 'scanning';   // scanning | stabilizing | ready | captured | transitioning
        this._trackedCards      = [];            // array of TrackedCard objects
        this._capturedRows      = [];            // recorded captures
        this._capturedCardIds   = null;          // tracked card IDs at capture time (for transition detection)
        this._lowVisibilityRun  = 0;             // consecutive frames with <1 matched card from capture set
        this._hasFiredCapture   = false;         // prevents double-fire within same lock cycle
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Call every frame with the current detected card contours.
     *
     * @param {Array<{corners: Array<{x: number, y: number}>, imageData?: ImageData}>} detectedCards
     *   Each card has:
     *     - corners: array of 4 corner points [{x, y}, ...]
     *     - imageData: (optional) ImageData of the card region for sharpness check
     *
     * @returns {{
     *   state: string,
     *   cards: Array<{corners: Array, status: string, stabilityCount: number, sharpness: number}>,
     *   progress: number,
     *   captureReady: boolean
     * }}
     */
    update(detectedCards) {
        if (this._state === 'captured') {
            // In captured state, just return current status with green outlines
            return this._buildResult(false);
        }

        if (this._state === 'transitioning') {
            // Waiting for new row -- check if new cards are stabilizing
            this._matchAndTrack(detectedCards);
            const lockedCount = this._countLocked();
            if (lockedCount >= this.expectedCards) {
                // New row is ready
                this._state = 'scanning';
                this._hasFiredCapture = false;
            }
            return this._buildResult(false);
        }

        // Active tracking: scanning, stabilizing, or ready
        this._matchAndTrack(detectedCards);
        this._updateState();

        // Check for auto-capture trigger
        let captureReady = false;
        if (this._state === 'ready' && !this._hasFiredCapture) {
            captureReady = true;
            this._hasFiredCapture = true;
            this._state = 'captured';
            this._capturedCardIds = this._trackedCards.map(c => c.id);
        }

        return this._buildResult(captureReady);
    }

    /**
     * Call when capture fires -- records the captured row.
     *
     * @param {number} rowIndex  Which row was captured (0-based)
     */
    recordCapture(rowIndex) {
        this._capturedRows.push({
            rowIndex,
            timestamp: Date.now(),
            cardCount: this._trackedCards.length,
        });
    }

    /**
     * Call every frame after capture to detect row transitions.
     * Returns true when old row has left the frame and new cards are entering.
     *
     * @param {Array<{corners: Array<{x: number, y: number}>}>} detectedCards
     * @returns {boolean}  True when transitioning from old row to new
     */
    detectRowTransition(detectedCards) {
        if (this._state !== 'captured') {
            return false;
        }

        if (!this._capturedCardIds || this._capturedCardIds.length === 0) {
            return false;
        }

        // Count how many of the captured cards are still visible
        const stillVisible = this._countCapturedCardsVisible(detectedCards);

        if (stillVisible < 1) {
            this._lowVisibilityRun++;
        } else {
            this._lowVisibilityRun = 0;
        }

        if (this._lowVisibilityRun >= this.transitionFrames) {
            this._state = 'transitioning';
            this._trackedCards = [];
            this._capturedCardIds = null;
            this._lowVisibilityRun = 0;
            this._hasFiredCapture = false;
            return true;
        }

        return false;
    }

    /**
     * Reset all state (new scan session).
     */
    reset() {
        this._state             = 'scanning';
        this._trackedCards      = [];
        this._capturedRows      = [];
        this._capturedCardIds   = null;
        this._lowVisibilityRun  = 0;
        this._hasFiredCapture   = false;
    }

    /** Current state. */
    get state() {
        return this._state;
    }

    /** Number of rows captured so far. */
    get capturedRowCount() {
        return this._capturedRows.length;
    }

    // -----------------------------------------------------------------------
    // Internal: card matching and tracking
    // -----------------------------------------------------------------------

    /**
     * Match current frame's detected cards to previously tracked cards by
     * nearest center distance, then update or create tracked cards.
     *
     * @param {Array} detectedCards  Cards detected this frame
     */
    _matchAndTrack(detectedCards) {
        const currentCenters = detectedCards.map(c => ScannerAutoCapture._cardCenter(c.corners));
        const prevCards = this._trackedCards;

        // Build cost matrix: distance from each detected card to each tracked card
        const matched = new Set();       // indices into prevCards that got matched
        const usedDetected = new Set();  // indices into detectedCards that got matched

        // Greedy nearest-neighbor matching (good enough for 3-9 cards)
        const pairs = [];
        for (let di = 0; di < currentCenters.length; di++) {
            for (let ti = 0; ti < prevCards.length; ti++) {
                const dist = ScannerAutoCapture._dist(currentCenters[di], prevCards[ti].center);
                pairs.push({ di, ti, dist });
            }
        }
        pairs.sort((a, b) => a.dist - b.dist);

        const assignments = []; // { detectedIdx, trackedIdx }
        for (const { di, ti, dist } of pairs) {
            if (usedDetected.has(di) || matched.has(ti)) continue;
            if (dist > this.matchRadius) continue;
            assignments.push({ di, ti, dist });
            usedDetected.add(di);
            matched.add(ti);
        }

        // Update matched tracked cards
        for (const { di, ti } of assignments) {
            const tracked = prevCards[ti];
            const card = detectedCards[di];
            const newCenter = currentCenters[di];

            // Check corner movement for stability
            const maxCornerMove = this._maxCornerMovement(tracked.corners, card.corners);

            if (maxCornerMove <= this.movementThreshold) {
                tracked.stabilityCount++;
            } else {
                tracked.stabilityCount = 0;
            }

            tracked.corners = card.corners;
            tracked.center = newCenter;
            tracked.visible = true;
            tracked.lastSeen = this._frameCount();

            // Update sharpness if card provides imageData and is locked
            if (card.imageData && this._getCardStatus(tracked.stabilityCount) === 'locked') {
                tracked.sharpness = ScannerAutoCapture._computeSharpness(card.imageData);
            }
        }

        // Mark unmatched tracked cards as not visible
        for (let ti = 0; ti < prevCards.length; ti++) {
            if (!matched.has(ti)) {
                prevCards[ti].visible = false;
                prevCards[ti].stabilityCount = 0;
            }
        }

        // Create new tracked cards for unmatched detections
        for (let di = 0; di < detectedCards.length; di++) {
            if (!usedDetected.has(di)) {
                this._trackedCards.push({
                    id: ScannerAutoCapture._nextId(),
                    corners: detectedCards[di].corners,
                    center: currentCenters[di],
                    stabilityCount: 0,
                    sharpness: 0,
                    visible: true,
                    lastSeen: this._frameCount(),
                });
            }
        }

        // Remove tracked cards that haven't been seen for 10+ frames
        this._trackedCards = this._trackedCards.filter(
            c => c.visible || (this._frameCount() - c.lastSeen) < 10
        );
    }

    /**
     * Compute maximum corner-to-corner movement between two sets of corners.
     * Corners must be in the same order (TL, TR, BR, BL).
     *
     * @param {Array<{x: number, y: number}>} prevCorners
     * @param {Array<{x: number, y: number}>} currCorners
     * @returns {number}  Max distance any corner moved, in pixels
     */
    _maxCornerMovement(prevCorners, currCorners) {
        if (!prevCorners || !currCorners) return Infinity;
        const len = Math.min(prevCorners.length, currCorners.length);
        let maxDist = 0;
        for (let i = 0; i < len; i++) {
            const dx = currCorners[i].x - prevCorners[i].x;
            const dy = currCorners[i].y - prevCorners[i].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > maxDist) maxDist = dist;
        }
        return maxDist;
    }

    // -----------------------------------------------------------------------
    // Internal: state machine
    // -----------------------------------------------------------------------

    /**
     * Update the overall state based on per-card tracking.
     */
    _updateState() {
        const visibleCards = this._trackedCards.filter(c => c.visible);
        const lockedCount = this._countLocked();
        const allLocked = lockedCount >= this.expectedCards;

        if (visibleCards.length === 0) {
            this._state = 'scanning';
            return;
        }

        if (allLocked) {
            // All expected cards are locked -- check sharpness
            const allSharp = this._allLockedSharp();
            if (allSharp) {
                this._state = 'ready';
            } else {
                this._state = 'stabilizing';
            }
            return;
        }

        if (lockedCount > 0 || visibleCards.length > 0) {
            this._state = 'stabilizing';
            return;
        }

        this._state = 'scanning';
    }

    /**
     * Get card status from stability count.
     *
     * @param {number} count  Consecutive stable frames
     * @returns {string}  'detecting' | 'stabilizing' | 'locked'
     */
    _getCardStatus(count) {
        if (count >= this.stabilityFrames) return 'locked';
        if (count > this.detectingMax) return 'stabilizing';
        return 'detecting';
    }

    /**
     * Count how many tracked cards are "locked".
     * @returns {number}
     */
    _countLocked() {
        return this._trackedCards.filter(
            c => c.visible && this._getCardStatus(c.stabilityCount) === 'locked'
        ).length;
    }

    /**
     * Check if all locked cards pass the sharpness threshold.
     * @returns {boolean}
     */
    _allLockedSharp() {
        const locked = this._trackedCards.filter(
            c => c.visible && this._getCardStatus(c.stabilityCount) === 'locked'
        );
        if (locked.length < this.expectedCards) return false;
        return locked.every(c => c.sharpness >= this.sharpnessThreshold);
    }

    // -----------------------------------------------------------------------
    // Internal: transition detection
    // -----------------------------------------------------------------------

    /**
     * Count how many of the captured cards are still visible in the current
     * detections.  Matches by center proximity to tracked cards.
     *
     * @param {Array} detectedCards  Current frame's detected cards
     * @returns {number}  Count of captured cards still visible
     */
    _countCapturedCardsVisible(detectedCards) {
        if (!this._capturedCardIds) return 0;

        const capturedCards = this._trackedCards.filter(
            c => this._capturedCardIds.includes(c.id)
        );

        const detectedCenters = detectedCards.map(
            c => ScannerAutoCapture._cardCenter(c.corners)
        );

        let count = 0;
        for (const tracked of capturedCards) {
            for (const center of detectedCenters) {
                if (ScannerAutoCapture._dist(tracked.center, center) < this.matchRadius) {
                    count++;
                    break;
                }
            }
        }
        return count;
    }

    // -----------------------------------------------------------------------
    // Internal: progress and result building
    // -----------------------------------------------------------------------

    /**
     * Compute progress value (0-1).
     * @returns {number}
     */
    _computeProgress() {
        if (this._state === 'captured') return 1.0;
        if (this._state === 'transitioning') return 0.0;

        const visibleCards = this._trackedCards.filter(c => c.visible);
        if (visibleCards.length === 0) return 0.0;

        const lockedCount = this._countLocked();

        if (lockedCount >= this.expectedCards) {
            // All locked -- checking sharpness
            const allSharp = this._allLockedSharp();
            return allSharp ? 1.0 : 0.9;
        }

        // Progress based on locked fraction
        const lockFraction = lockedCount / this.expectedCards;
        // Scale to 0.0 - 0.89 range (0.9+ reserved for sharpness check)
        return Math.min(0.89, lockFraction * 0.9);
    }

    /**
     * Build the return value for update().
     *
     * @param {boolean} captureReady  Whether capture just fired this frame
     * @returns {Object}
     */
    _buildResult(captureReady) {
        const cards = this._trackedCards
            .filter(c => c.visible)
            .map(c => ({
                corners:        c.corners,
                status:         this._getCardStatus(c.stabilityCount),
                stabilityCount: c.stabilityCount,
                sharpness:      Math.round(c.sharpness * 10) / 10,
            }));

        return {
            state:        this._state,
            cards:        cards,
            progress:     Math.round(this._computeProgress() * 100) / 100,
            captureReady: captureReady,
        };
    }

    // -----------------------------------------------------------------------
    // Internal: frame counter (monotonic)
    // -----------------------------------------------------------------------

    _frameCount() {
        // Use a simple counter incremented via closure
        if (this.__frameCounter === undefined) {
            this.__frameCounter = 0;
        }
        return this.__frameCounter++;
    }

    // -----------------------------------------------------------------------
    // Static utilities
    // -----------------------------------------------------------------------

    /**
     * Compute the center point of a card from its corners.
     *
     * @param {Array<{x: number, y: number}>} corners
     * @returns {{x: number, y: number}}
     */
    static _cardCenter(corners) {
        let sx = 0, sy = 0;
        for (const c of corners) {
            sx += c.x;
            sy += c.y;
        }
        return { x: sx / corners.length, y: sy / corners.length };
    }

    /**
     * Euclidean distance between two points.
     *
     * @param {{x: number, y: number}} a
     * @param {{x: number, y: number}} b
     * @returns {number}
     */
    static _dist(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Compute Laplacian variance (sharpness) from ImageData.
     * Same algorithm as FrameAnalyzer and SharpnessDetector.
     *
     * @param {ImageData} imageData  RGBA pixel data of the card region
     * @returns {number}  Laplacian variance (higher = sharper)
     */
    static _computeSharpness(imageData) {
        const w = imageData.width;
        const h = imageData.height;
        const rgba = imageData.data;

        // Convert to grayscale
        const gray = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
            gray[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
        }

        // Laplacian variance: kernel [0 1 0; 1 -4 1; 0 1 0]
        let sum = 0;
        let sumSq = 0;
        let n = 0;

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                const lap = gray[idx - w] + gray[idx + w]
                          + gray[idx - 1] + gray[idx + 1]
                          - 4 * gray[idx];
                sum   += lap;
                sumSq += lap * lap;
                n++;
            }
        }

        if (n === 0) return 0;
        const mean = sum / n;
        return (sumSq / n) - (mean * mean);
    }

    /** Auto-incrementing ID generator for tracked cards. */
    static _nextId() {
        if (ScannerAutoCapture.__idCounter === undefined) {
            ScannerAutoCapture.__idCounter = 0;
        }
        return ++ScannerAutoCapture.__idCounter;
    }
}


// ---------------------------------------------------------------------------
// Self-test (run in browser console: ScannerAutoCapture.selfTest())
// ---------------------------------------------------------------------------

ScannerAutoCapture.selfTest = function () {
    let passed = 0;
    let failed = 0;

    function assert(cond, name) {
        if (cond) { console.log('  PASS: ' + name); passed++; }
        else      { console.error('  FAIL: ' + name); failed++; }
    }

    console.log('=== ScannerAutoCapture Self-Test ===');

    // --- Constructor defaults ---
    {
        const ac = new ScannerAutoCapture();
        assert(ac.stabilityFrames === 12, 'default stabilityFrames = 12');
        assert(ac.sharpnessThreshold === 80, 'default sharpnessThreshold = 80');
        assert(ac.movementThreshold === 8, 'default movementThreshold = 8');
        assert(ac.expectedCards === 3, 'default expectedCards = 3');
        assert(ac.state === 'scanning', 'initial state = scanning');
    }

    // --- Static: _cardCenter ---
    {
        const corners = [
            { x: 0, y: 0 }, { x: 100, y: 0 },
            { x: 100, y: 100 }, { x: 0, y: 100 }
        ];
        const center = ScannerAutoCapture._cardCenter(corners);
        assert(center.x === 50 && center.y === 50, '_cardCenter computes centroid');
    }

    // --- Static: _dist ---
    {
        const d = ScannerAutoCapture._dist({ x: 0, y: 0 }, { x: 3, y: 4 });
        assert(d === 5, '_dist(0,0 -> 3,4) = 5');
    }

    // --- Static: _computeSharpness ---
    {
        // Uniform gray image -> sharpness = 0
        const w = 10, h = 10;
        const rgba = new Uint8ClampedArray(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            rgba[i * 4] = 128;
            rgba[i * 4 + 1] = 128;
            rgba[i * 4 + 2] = 128;
            rgba[i * 4 + 3] = 255;
        }
        const uniformSharp = ScannerAutoCapture._computeSharpness({ data: rgba, width: w, height: h });
        assert(uniformSharp === 0, 'uniform image -> sharpness 0');

        // Checkerboard pattern -> high sharpness
        const checker = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const val = ((x + y) % 2 === 0) ? 50 : 200;
                const idx = (y * w + x) * 4;
                checker[idx] = val;
                checker[idx + 1] = val;
                checker[idx + 2] = val;
                checker[idx + 3] = 255;
            }
        }
        const checkerSharp = ScannerAutoCapture._computeSharpness({ data: checker, width: w, height: h });
        assert(checkerSharp > 100, 'checkerboard -> high sharpness (got ' + Math.round(checkerSharp) + ')');
    }

    // --- Corner movement detection ---
    {
        const ac = new ScannerAutoCapture({ movementThreshold: 8 });
        const c1 = [
            { x: 10, y: 10 }, { x: 110, y: 10 },
            { x: 110, y: 160 }, { x: 10, y: 160 }
        ];
        // Barely moved (2px) -> should be stable
        const c2 = [
            { x: 11, y: 11 }, { x: 111, y: 11 },
            { x: 111, y: 161 }, { x: 11, y: 161 }
        ];
        const move1 = ac._maxCornerMovement(c1, c2);
        assert(move1 < 8, 'small movement (' + move1.toFixed(1) + ') < threshold');

        // Large movement (20px) -> not stable
        const c3 = [
            { x: 30, y: 30 }, { x: 130, y: 30 },
            { x: 130, y: 180 }, { x: 30, y: 180 }
        ];
        const move2 = ac._maxCornerMovement(c1, c3);
        assert(move2 > 8, 'large movement (' + move2.toFixed(1) + ') > threshold');
    }

    // --- Card status progression ---
    {
        const ac = new ScannerAutoCapture({ stabilityFrames: 12, detectingMax: 3 });
        assert(ac._getCardStatus(0) === 'detecting', '0 frames -> detecting');
        assert(ac._getCardStatus(3) === 'detecting', '3 frames -> detecting');
        assert(ac._getCardStatus(4) === 'stabilizing', '4 frames -> stabilizing');
        assert(ac._getCardStatus(11) === 'stabilizing', '11 frames -> stabilizing');
        assert(ac._getCardStatus(12) === 'locked', '12 frames -> locked');
        assert(ac._getCardStatus(100) === 'locked', '100 frames -> locked');
    }

    // --- update() with no cards -> scanning ---
    {
        const ac = new ScannerAutoCapture();
        const result = ac.update([]);
        assert(result.state === 'scanning', 'no cards -> scanning');
        assert(result.progress === 0, 'no cards -> progress 0');
        assert(result.captureReady === false, 'no cards -> not capture ready');
        assert(result.cards.length === 0, 'no cards -> empty cards array');
    }

    // --- update() tracks new cards ---
    {
        const ac = new ScannerAutoCapture({ expectedCards: 1, stabilityFrames: 3, detectingMax: 1 });

        const card = {
            corners: [
                { x: 10, y: 10 }, { x: 110, y: 10 },
                { x: 110, y: 160 }, { x: 10, y: 160 }
            ]
        };

        const r1 = ac.update([card]);
        assert(r1.cards.length === 1, 'one card tracked');
        assert(r1.cards[0].status === 'detecting', 'first frame -> detecting');
        assert(r1.cards[0].stabilityCount === 0, 'first frame -> count 0');
    }

    // --- Stability tracking across frames ---
    {
        const ac = new ScannerAutoCapture({
            expectedCards: 1,
            stabilityFrames: 3,
            detectingMax: 1,
            movementThreshold: 8,
        });

        // Same card, barely moving
        const card = {
            corners: [
                { x: 100, y: 100 }, { x: 200, y: 100 },
                { x: 200, y: 250 }, { x: 100, y: 250 }
            ]
        };

        // Frame 1: new card
        ac.update([card]);

        // Frame 2: same position -> stability +1
        const r2 = ac.update([card]);
        assert(r2.cards[0].stabilityCount === 1, 'frame 2 -> count 1');
        assert(r2.cards[0].status === 'detecting', 'count 1 -> detecting');

        // Frame 3: same position -> stability +1
        const r3 = ac.update([card]);
        assert(r3.cards[0].stabilityCount === 2, 'frame 3 -> count 2');
        assert(r3.cards[0].status === 'stabilizing', 'count 2 -> stabilizing');
    }

    // --- Stability resets on movement ---
    {
        const ac = new ScannerAutoCapture({
            expectedCards: 1,
            stabilityFrames: 5,
            detectingMax: 1,
            movementThreshold: 8,
        });

        const card1 = {
            corners: [
                { x: 100, y: 100 }, { x: 200, y: 100 },
                { x: 200, y: 250 }, { x: 100, y: 250 }
            ]
        };
        const card2 = {
            corners: [
                { x: 130, y: 130 }, { x: 230, y: 130 },
                { x: 230, y: 280 }, { x: 130, y: 280 }
            ]
        };

        ac.update([card1]);
        ac.update([card1]); // stability = 1
        ac.update([card1]); // stability = 2

        // Big move -> reset
        const rMoved = ac.update([card2]);
        assert(rMoved.cards[0].stabilityCount === 0, 'movement resets stability to 0');
    }

    // --- Full auto-capture: cards lock and become sharp ---
    {
        const ac = new ScannerAutoCapture({
            expectedCards: 1,
            stabilityFrames: 3,
            detectingMax: 1,
            movementThreshold: 8,
            sharpnessThreshold: 50,
        });

        // Create a "sharp" imageData (checkerboard)
        const w = 20, h = 20;
        const sharpData = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const val = ((x + y) % 2 === 0) ? 30 : 220;
                const idx = (y * w + x) * 4;
                sharpData[idx] = val;
                sharpData[idx + 1] = val;
                sharpData[idx + 2] = val;
                sharpData[idx + 3] = 255;
            }
        }
        const sharpImageData = { data: sharpData, width: w, height: h };

        const card = {
            corners: [
                { x: 100, y: 100 }, { x: 200, y: 100 },
                { x: 200, y: 250 }, { x: 100, y: 250 }
            ],
            imageData: sharpImageData,
        };

        // Pump frames until locked
        let captureResult = null;
        for (let i = 0; i < 10; i++) {
            const r = ac.update([card]);
            if (r.captureReady) {
                captureResult = r;
                break;
            }
        }

        assert(captureResult !== null, 'auto-capture fires after stability + sharpness');
        assert(captureResult.state === 'captured', 'state transitions to captured');
        assert(captureResult.progress === 1.0, 'progress = 1.0 on capture');
    }

    // --- Progress indicator ---
    {
        const ac = new ScannerAutoCapture({
            expectedCards: 3,
            stabilityFrames: 2,
            detectingMax: 0,
            movementThreshold: 100, // generous threshold
        });

        // No cards
        let r = ac.update([]);
        assert(r.progress === 0, 'progress 0 with no cards');

        // 1 card, not yet locked
        const c1 = {
            corners: [
                { x: 50, y: 50 }, { x: 100, y: 50 },
                { x: 100, y: 120 }, { x: 50, y: 120 }
            ]
        };
        r = ac.update([c1]);
        // First frame, stability=0, not locked yet
        assert(r.progress < 0.33, 'progress < 0.33 with 1 unlocked card');
    }

    // --- recordCapture ---
    {
        const ac = new ScannerAutoCapture();
        ac.recordCapture(0);
        assert(ac.capturedRowCount === 1, 'recordCapture increments count');
        ac.recordCapture(1);
        assert(ac.capturedRowCount === 2, 'second recordCapture');
    }

    // --- reset ---
    {
        const ac = new ScannerAutoCapture();
        ac._state = 'captured';
        ac.recordCapture(0);
        ac.reset();
        assert(ac.state === 'scanning', 'reset -> scanning');
        assert(ac.capturedRowCount === 0, 'reset clears captures');
    }

    // --- detectRowTransition returns false when not captured ---
    {
        const ac = new ScannerAutoCapture();
        assert(ac.detectRowTransition([]) === false, 'not captured -> no transition');
    }

    // --- Multiple card matching ---
    {
        const ac = new ScannerAutoCapture({ expectedCards: 2, matchRadius: 50 });

        const cards = [
            {
                corners: [
                    { x: 10, y: 10 }, { x: 60, y: 10 },
                    { x: 60, y: 80 }, { x: 10, y: 80 }
                ]
            },
            {
                corners: [
                    { x: 200, y: 10 }, { x: 250, y: 10 },
                    { x: 250, y: 80 }, { x: 200, y: 80 }
                ]
            },
        ];

        ac.update(cards);
        const r = ac.update(cards);
        assert(r.cards.length === 2, 'tracks 2 cards independently');
        // Both should have incremented stability
        assert(r.cards[0].stabilityCount === 1, 'card 0 stability incremented');
        assert(r.cards[1].stabilityCount === 1, 'card 1 stability incremented');
    }

    // --- Card matching across frames (nearest center) ---
    {
        const ac = new ScannerAutoCapture({ matchRadius: 50 });

        // Frame 1: card at (50, 50)
        ac.update([{
            corners: [
                { x: 25, y: 25 }, { x: 75, y: 25 },
                { x: 75, y: 75 }, { x: 25, y: 75 }
            ]
        }]);

        // Frame 2: card moved slightly (within threshold)
        const r = ac.update([{
            corners: [
                { x: 27, y: 27 }, { x: 77, y: 27 },
                { x: 77, y: 77 }, { x: 27, y: 77 }
            ]
        }]);

        // Should match to existing card, not create a new one
        assert(ac._trackedCards.length === 1, 'slight move -> same tracked card (not duplicated)');
    }

    // --- Unmatched detected cards create new tracks ---
    {
        const ac = new ScannerAutoCapture({ matchRadius: 50 });

        ac.update([{
            corners: [
                { x: 10, y: 10 }, { x: 60, y: 10 },
                { x: 60, y: 80 }, { x: 10, y: 80 }
            ]
        }]);

        // Frame 2: completely different position -> new card
        const r = ac.update([{
            corners: [
                { x: 500, y: 500 }, { x: 550, y: 500 },
                { x: 550, y: 570 }, { x: 500, y: 570 }
            ]
        }]);

        // Should have 2 tracked cards (one not visible, one new)
        assert(ac._trackedCards.length === 2, 'far-away card creates new track');
    }

    console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
    return failed === 0;
};


// Export for module usage; also works as inline <script>
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScannerAutoCapture;
}
