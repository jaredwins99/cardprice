# Approach: Row-Based Binder Scanner

Last updated: 2026-03-23

## Overview

The scanner is a third card capture method (after binder-page photo and slide-scan) that uses a live camera feed to automatically detect, track, and capture rows of 3 cards at once. The user holds their phone above a binder page and slowly moves it row by row. The system auto-captures when it detects 3 stable, well-framed cards -- zero taps required during the scan itself.

**Key difference from slide-scan**: The slide-scan captures individual cards one at a time as the phone slides across. The scanner captures an entire row (3 cards) in a single frame, then the user moves to the next row. This is closer to the binder-page approach in terms of capture unit (row vs card) but uses live video for quality control and auto-capture instead of a manual shutter press.

---

## Architecture Diagram

```
Phone Camera (1920x1080, 30fps)
    |
    v
[Video Frame]
    |
    +--[every frame]--> Downsample to 480px height
    |                       |
    |                       +--[every 30th frame]--> BinderDetector.detectBackground()
    |                       |                            - HSV histogram of frame edges
    |                       |                            - Identifies binder color (orange, blue, black, etc.)
    |                       |                            - Returns { color, hsvRange, confidence }
    |                       |                            - Used by CardContourDetector to exclude binder
    |                       |
    |                       +--[every frame]-----------> CardContourDetector.detect(downsampled, binderInfo)
    |                       |                            - Canny edges + contour finding
    |                       |                            - Filter for card-shaped rectangles (aspect ~0.716)
    |                       |                            - Exclude contours matching binder background color
    |                       |                            - Returns array of { corners, center, area, score }
    |                       |
    |                       +--[every frame]-----------> ScannerAutoCapture.update(contours, timestamp)
    |                       |                            - Track contour stability across frames
    |                       |                            - Count consecutive frames with 3 stable cards
    |                       |                            - Check: all 3 contours within guide region?
    |                       |                            - Check: area variance < threshold? (similar sizes)
    |                       |                            - Returns { state, stableFrames, readyToCapture }
    |                       |
    |                       +--[every frame]-----------> ScannerOverlay.setState(state) + .draw()
    |                                                    - Render colored outlines per contour
    |                                                    - Guide rectangle, corner brackets
    |                                                    - Status text ("Align 3 cards", "Hold steady...")
    |                                                    - Row indicator, capture counter, thumbnail strip
    |
    +--[when readyToCapture]---> ScannerCapture.captureRow(fullResFrame, contours)
    |                            - Extract from FULL resolution frame (not downsampled)
    |                            - Scale contour coordinates from 480px back to 1080px
    |                            - Perspective-correct each of 3 cards using 4-corner homography
    |                            - Quality check: blur detection (Laplacian variance)
    |                            - Returns 3 card images as JPEG data URLs
    |
    +--[after 3 rows captured]--> ScannerCapture.buildSubmission(allCards[9])
    |                              - Package 9 card images as FormData
    |                              - POST to /scanner/identify
    |
    +--[server response]--------> Display results
                                   - Same 3x3 grid as binder-page results
                                   - Card name, set, price, confidence
                                   - Tap to expand, add to inventory
```

---

## State Machine

```
                    +-----------+
                    |   INIT    |  Camera loading, permissions
                    +-----+-----+
                          |
                          v  camera ready
                    +-----+-----+
               +--->| SCANNING  |  Looking for 3 card contours
               |    +-----+-----+
               |          |
               |          v  3 contours detected, within guide
               |    +-----+-----+
               |    |STABILIZING|  Counting stable frames (need ~10 = 1s at 10fps)
               |    +-----+-----+
               |          |
               |     lost |          stable for 1s
               |  contours|              |
               |          v              v
               |    +-----+-----+  +-----+-----+
               +----+ SCANNING  |  | CAPTURING  |  Extracting 3 cards from full-res frame
                    +-----------+  +-----+-----+
                                         |
                                         v  3 cards extracted successfully
                                   +-----+-----+
                                   | ROW_DONE   |  Show thumbnails, flash confirmation
                                   +-----+-----+
                                         |
                                         v  (if rows < 3)
                                   +-----+-------+
                                   |TRANSITIONING |  "Move to next row" prompt
                                   +-----+-------+  2-second delay, then auto-advance
                                         |
                                         v
                                   +-----+-----+
                                   | SCANNING   |  (next row)
                                   +-----+-----+
                                         |
                                    ... repeat ...
                                         |
                                         v  (rows == 3, all 9 cards captured)
                                   +-----+-----+
                                   | ALL_DONE   |  Show all 9 thumbnails
                                   +-----+-----+
                                         |
                                         v  auto-submit (or user taps "Submit")
                                   +-----+-----+
                                   |SUBMITTING  |  POST /scanner/identify, show spinner
                                   +-----+-----+
                                         |
                                         v  server response
                                   +-----+-----+
                                   |  RESULTS   |  Display 3x3 grid with prices
                                   +-----------+
```

### State Descriptions

| State | Overlay color | Status text | Duration |
|-------|--------------|-------------|----------|
| INIT | -- | "Starting camera..." | 1-3s |
| SCANNING | White dashed guide | "Align 3 cards in frame" | Until detected |
| STABILIZING | Yellow outlines | "Hold steady... (N/10)" | ~1s (10 frames) |
| CAPTURING | Green pulsing | "Capturing..." | <200ms |
| ROW_DONE | Green solid | "Row N captured!" | 1s |
| TRANSITIONING | White dashed | "Move to next row" | 2s |
| ALL_DONE | Green solid | "All rows captured!" | 1s |
| SUBMITTING | -- | Spinner + "Identifying cards..." | 5-15s |
| RESULTS | -- | 3x3 result grid | Until user action |

---

## Component Inventory

### 1. `CardContourDetector` (client-side JS)

**Purpose**: Finds card-shaped rectangles in a video frame.

**Input**: Downsampled video frame (480px height), optional binder color info.

**Output**: Array of detected card contours with corners, center, area, confidence score.

**Key logic**:
- Convert to grayscale, apply Gaussian blur
- Canny edge detection (adaptive thresholds based on frame brightness)
- `findContours` + `approxPolyDP` to find quadrilaterals
- Filter by: aspect ratio (~0.716 +/- 20%), minimum area (>2% of frame), convexity
- If binder color is known, reject contours whose interior matches the binder HSV range
- Sort left-to-right by center x-coordinate
- Return top 3 candidates (or fewer if not enough qualify)

**File**: `cardprice/scanner_camera_ui.py` (inline JS class)

### 2. `ScannerAutoCapture` (client-side JS)

**Purpose**: Tracks contour stability across frames and decides when to trigger capture.

**Input**: Array of contours from `CardContourDetector`, current timestamp.

**Output**: State object indicating readiness to capture.

**Key logic**:
- Match contours across frames by proximity (nearest-neighbor matching, max displacement threshold)
- Track "stable frames" counter: incremented when all 3 contours move less than 5px between frames
- Reset counter if any contour moves too much, or contour count changes
- Trigger capture when `stableFrames >= STABILITY_THRESHOLD` (default 10 frames = 1s at 10fps)
- Also check: all contours within the guide region, similar areas (max 2x ratio between largest/smallest)

**File**: `cardprice/scanner_camera_ui.py` (inline JS class)

### 3. `ScannerOverlay` (client-side JS)

**Purpose**: Draws UI elements on a transparent canvas overlaying the video feed.

**Input**: State from `ScannerAutoCapture`, contour positions, captured thumbnails.

**Output**: Visual overlay on canvas.

**Already built**: `/home/godli/cardprice/cardprice/scanner_overlay.js`

**Key features**:
- Card-shaped guide rectangle with corner brackets
- Dimmed surround (focuses attention on guide area)
- State-dependent colors: white (idle), yellow (detected), green (ready/capturing)
- Capture counter (e.g., "2/3"), row indicator ("Row 1 of 3")
- Thumbnail strip at bottom showing captured cards
- Pulsing animation during capture

### 4. `ScannerCapture` (client-side JS)

**Purpose**: Extracts perspective-corrected card images from a full-resolution video frame.

**Input**: Full-res video frame, array of contour corners (scaled from 480px coords to full-res coords).

**Output**: Array of 3 JPEG data URLs (one per card).

**Key logic**:
- Scale contour coordinates: `corners * (fullResHeight / 480)`
- For each card's 4 corners, compute perspective transform matrix
- Apply `drawImage` with canvas transforms to warp card to standard rectangle (1008x1530)
- Alternatively, use a simplified crop-and-scale if perspective distortion is mild
- Encode each card canvas as JPEG data URL (quality 0.92)
- Run blur detection: compute Laplacian variance of center region; reject if below threshold
- If any card fails quality check, return to SCANNING state instead of ROW_DONE

**File**: `cardprice/scanner_camera_ui.py` (inline JS class)

### 5. `BinderDetector` (client-side JS)

**Purpose**: Identifies the binder background color to help `CardContourDetector` separate cards from binder.

**Input**: Downsampled video frame.

**Output**: Binder color profile (HSV range, dominant color name, confidence).

**Key logic**:
- Sample pixels from frame edges (top/bottom/left/right 10% strips)
- Build HSV histogram of edge pixels
- Find dominant hue cluster
- Classify: orange (most common binder), blue, black, white, green
- Return HSV range that defines "binder background" for contour filtering
- Only runs every 30th frame (~1/second) since binder color changes rarely

**File**: `cardprice/scanner_camera_ui.py` (inline JS class)

---

## Integration Guide: Assembling `scanner_camera_ui.py`

### File Structure

```python
# cardprice/scanner_camera_ui.py
"""Row-based binder scanner with live camera, auto-detection, and auto-capture.

Captures 3 cards per row, 3 rows per page = 9 cards total.
Zero taps during scanning -- auto-captures when 3 stable cards detected.

Served at GET /scanner, submits to POST /scanner/identify.
"""

SCANNER_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Card Scanner</title>
<style>
  /* ... CSS (copy pattern from condition_camera_ui.py) ... */
</style>
</head>
<body>
<div class="camera-wrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
  <canvas id="capture" style="display:none"></canvas>
  <canvas id="downsample" style="display:none"></canvas>
</div>

<script>
// === Component 1: BinderDetector ===
class BinderDetector { ... }

// === Component 2: CardContourDetector ===
class CardContourDetector { ... }

// === Component 3: ScannerAutoCapture ===
class ScannerAutoCapture { ... }

// === Component 4: ScannerOverlay ===
// (paste from scanner_overlay.js, or inline)
class ScannerOverlay { ... }

// === Component 5: ScannerCapture ===
class ScannerCapture { ... }

// === Main Controller ===
class ScannerController {
  constructor() { ... }
  async init() { ... }
  startLoop() { ... }
  processFrame() { ... }
  onCapture(cards) { ... }
  async submit() { ... }
  showResults(data) { ... }
}

// Boot
const scanner = new ScannerController();
scanner.init();
</script>
</body>
</html>
"""
```

### The Main Loop (`ScannerController.processFrame`)

This is the heart of the scanner. Called by `requestAnimationFrame`, throttled to every 3rd frame (~10fps).

```javascript
processFrame(timestamp) {
    // 1. Throttle to 10fps
    if (timestamp - this._lastProcess < 100) {
        requestAnimationFrame(ts => this.processFrame(ts));
        return;
    }
    this._lastProcess = timestamp;

    // 2. Skip if not in a scanning state
    if (this.state === 'SUBMITTING' || this.state === 'RESULTS') {
        requestAnimationFrame(ts => this.processFrame(ts));
        return;
    }

    const video = this.video;
    if (video.readyState < 2) {
        requestAnimationFrame(ts => this.processFrame(ts));
        return;
    }

    // 3. Downsample video to 480px height
    const scale = 480 / video.videoHeight;
    const dw = Math.round(video.videoWidth * scale);
    this.dsCanvas.width = dw;
    this.dsCanvas.height = 480;
    this.dsCtx.drawImage(video, 0, 0, dw, 480);

    // 4. Binder detection (every 30th frame)
    this._frameCount++;
    if (this._frameCount % 30 === 0) {
        this.binderInfo = this.binderDetector.detectBackground(this.dsCanvas);
    }

    // 5. Card contour detection
    const contours = this.contourDetector.detect(this.dsCanvas, this.binderInfo);

    // 6. Stability tracking
    const captureState = this.autoCapture.update(contours, timestamp);

    // 7. Update overlay
    this.overlay.setState({
        state: captureState.overlayState,  // idle|detected|ready|capturing
        statusText: captureState.statusText,
        capturedThisRow: this.capturedRows.length > 0
            ? this.capturedRows[this.capturedRows.length - 1].length : 0,
        currentRow: this.currentRow,
    });
    // Draw contour outlines on overlay (colored per state)
    this.overlay.drawContours(contours, captureState.contourColors);
    this.overlay.draw();

    // 8. Trigger capture if stable
    if (captureState.readyToCapture && this.state === 'SCANNING') {
        this.state = 'CAPTURING';
        const cards = this.capture.captureRow(video, contours, scale);

        if (cards && cards.length === 3) {
            // Success
            cards.forEach(c => this.overlay.addThumbnail(c));
            this.capturedRows.push(cards);
            this.state = 'ROW_DONE';

            // Haptic feedback
            if (navigator.vibrate) navigator.vibrate(100);

            // Transition to next row or submit
            setTimeout(() => {
                if (this.currentRow < 2) {
                    this.currentRow++;
                    this.overlay.nextRow();
                    this.autoCapture.reset();
                    this.state = 'SCANNING';
                } else {
                    this.state = 'ALL_DONE';
                    setTimeout(() => this.submit(), 1000);
                }
            }, 2000);
        } else {
            // Quality check failed, retry
            this.autoCapture.reset();
            this.state = 'SCANNING';
        }
    }

    requestAnimationFrame(ts => this.processFrame(ts));
}
```

### Camera Initialization

```javascript
async init() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width:  { ideal: 1920 },
                height: { ideal: 1080 },
                frameRate: { ideal: 30 },
            },
            audio: false,
        });
        this.video.srcObject = stream;
        await this.video.play();

        // Wait for video dimensions to stabilize
        await new Promise(r => setTimeout(r, 500));
        this.overlay.resizeCanvas();
        this.state = 'SCANNING';
        this.startLoop();
    } catch (err) {
        this.showError('Camera access denied: ' + err.message);
    }
}
```

### Submission to Server

```javascript
async submit() {
    this.state = 'SUBMITTING';

    const formData = new FormData();
    const allCards = this.capturedRows.flat();

    for (let i = 0; i < allCards.length; i++) {
        const blob = await (await fetch(allCards[i])).blob();
        formData.append('card_' + i, blob, `card_${String(i).padStart(2, '0')}.jpg`);
    }

    try {
        const resp = await fetch('/scanner/identify', { method: 'POST', body: formData });
        const data = await resp.json();
        this.showResults(data);
    } catch (err) {
        this.showError('Identification failed: ' + err.message);
    }
}
```

### Event Handling

| Event | Handler | Action |
|-------|---------|--------|
| Camera permission denied | `init()` catch | Show error message with instructions |
| Camera stream interrupted | `video.onended` | Show "Camera disconnected" error, offer retry |
| Window resize / orientation change | `ScannerOverlay._onResize` | Resize overlay canvas |
| Contours lost during stabilization | `ScannerAutoCapture.update()` | Reset stable frame counter |
| Blur detected during capture | `ScannerCapture.captureRow()` | Return null, state goes back to SCANNING |
| Server error on /scanner/identify | `submit()` catch | Show error with retry button |
| User taps "Re-scan row" | Button handler | Pop last captured row, decrement currentRow, state = SCANNING |
| User taps "Start over" | Button handler | Reset all state, state = SCANNING |

### Error States

| Error | Detection | Recovery |
|-------|-----------|----------|
| No camera permission | `getUserMedia` rejects | Show instructions for browser settings |
| Camera resolution too low | `video.videoWidth < 1280` | Warn but continue (degraded quality) |
| No cards detected for >10s | Timer in SCANNING state | Show "No cards found. Check lighting and distance." |
| Repeated capture failures | Counter > 3 consecutive | Show "Try moving closer / improving lighting" |
| Server unreachable | `fetch()` rejects | Show "Server offline" with retry button |
| Server returns error JSON | `resp.ok === false` | Show error message, offer re-submit |

---

## Comparison: Capture Methods

| Dimension | Binder Photo | Slide-Scan (v1) | Scanner (this) |
|-----------|-------------|-----------------|----------------|
| **User taps during capture** | 1 (shutter) | 9+ (broken) | 0 (auto-capture) |
| **Capture unit** | Full page (9 cards) | 1 card at a time | 1 row (3 cards) |
| **Time per page** | ~13s total | Untested (broken) | ~30s (10s/row x 3) |
| **Segmentation needed** | Yes (contour + grid) | No (pre-cropped) | No (pre-cropped per row) |
| **Resolution per card** | ~1008x1530 | ~3000x4000 | ~1500x2200 (estimated) |
| **Quality control** | None (post-hoc) | Motion blur rejection | Live: stability + blur check |
| **Accuracy** | 98.1% (eval) / ~88% (wild) | Untested | TBD (expected 93-97%) |
| **Failure feedback** | After the fact | During scan | During scan (real-time overlay) |
| **Edge clipping** | Common (segmentation) | None | Rare (perspective correction) |
| **Glare handling** | None | None | Can retry immediately |
| **Holo detection** | Not possible (static) | Not possible (static) | Future: multi-frame analysis |
| **Implementation status** | Production | Broken | In development |

### Why "row at a time" instead of "card at a time"?

1. **3x faster than individual cards**: 3 captures vs 9 captures per page.
2. **Natural binder viewing**: Users naturally see rows when flipping pages.
3. **Page context preserved**: Having 3 cards from the same row enables page-context reranking (same pass 2/3 as binder-page scanning).
4. **Simpler motion**: "Position and hold" is easier than "slide smoothly across."
5. **Better perspective**: At row distance (~20cm), cards are large enough for OCR but all 3 fit in frame.

### Why not just improve binder-page photo?

The binder-page photo approach hits a fundamental resolution ceiling. At the distance needed to see all 9 cards, each card gets ~1008x1530 pixels. This is adequate for name OCR in good conditions but insufficient for:
- Stamp detection (translucent overlays need 2x more pixels)
- Attack OCR on small text
- Holo shimmer analysis
- Condition grading (surface detail)

The scanner captures at ~1.5-2x the resolution of binder-page scanning per card, bridging the gap between binder-page (fast, lower quality) and slide-scan (slow, highest quality).

---

## Testing Guide

### Testing Without Physical Cards

**Method 1: Monitor as binder (recommended for development)**
1. Display a binder page image on a monitor or tablet (use one of `data/inbox/page_*.jpg`)
2. Point your phone camera at the monitor
3. The scanner should detect card contours from the displayed image
4. Caveat: Monitor introduces moire patterns and color shift. Contour detection thresholds may need to be more lenient for monitor testing.

**Method 2: Printed test sheets**
1. Print a 3x3 grid of Pokemon cards on photo paper
2. Place in a binder sleeve or lay flat
3. More realistic than monitor but requires a printer

**Method 3: Synthetic frame injection (unit testing)**
```javascript
// Instead of reading from video, inject a static image:
class MockVideo {
    constructor(imagePath) {
        this.img = new Image();
        this.img.src = imagePath;
    }
    get videoWidth() { return this.img.width; }
    get videoHeight() { return this.img.height; }
    get readyState() { return 4; } // HAVE_ENOUGH_DATA
    // Canvas drawImage accepts Image elements just like Video elements
}

// In test harness:
const mockVideo = new MockVideo('/test/binder_page.jpg');
const contours = detector.detect(mockCanvas, null);
assert(contours.length === 3, 'Should detect 3 cards in top row');
```

**Method 4: Replay recorded frames**
```javascript
// Record frames from a real scanning session:
// In processFrame(), save frames to an array:
//   this.recordedFrames.push(dsCanvas.toDataURL('image/jpeg', 0.5));
// Then replay them in a test loop without a live camera.
```

### Edge Cases to Test

| Edge Case | Expected Behavior | How to Test |
|-----------|-------------------|-------------|
| **2 cards visible** (end of row) | Show "Align 3 cards" message, do NOT capture | Cover one card slot with finger |
| **4 cards visible** (between rows) | Detect only 3 best-aligned, or wait for user to reposition | Position phone between two rows |
| **Empty sleeve** | Ignore empty slot, treat as 2 visible cards | Remove a card from the binder |
| **Card back showing** | Detect contour but flag during capture (use `is_card_back` from segmenter) | Flip one card face-down |
| **Reflection / glare on sleeve** | May break contour detection; show "Adjust angle" hint | Use overhead light source |
| **Binder ring shadow** | Shadow creates false edge in middle of row | Use ring binder, position phone over center |
| **Phone too close** (1-2 cards fill frame) | Show "Move back" hint | Hold phone 5cm from binder |
| **Phone too far** (6+ cards visible) | Show "Move closer" hint | Hold phone 50cm from binder |
| **Tilted phone** (perspective) | Perspective correction handles mild tilt; reject extreme tilt | Hold phone at 45+ degrees |
| **Low light** | Contour detection fails; show "Improve lighting" hint | Dim room lights |
| **Fast movement** | Stability counter resets, prevents capture until steady | Shake phone during stabilization |
| **Mixed card sizes** | Reject if area variance > 2x between cards | Mix Pokemon cards with MTG cards |

### Performance Profiling

**Target**: Process at 10fps on a mid-range phone (2022-era, e.g. Pixel 6a, iPhone SE 3rd gen).

**Budget per frame (100ms)**:
| Operation | Target | Measurement |
|-----------|--------|-------------|
| Downsample video to 480px | <5ms | `performance.now()` around `drawImage` |
| BinderDetector (every 30th frame) | <10ms | Profile HSV histogram computation |
| CardContourDetector.detect | <30ms | Profile edge detection + contour finding |
| ScannerAutoCapture.update | <2ms | Simple arithmetic, no image processing |
| ScannerOverlay.draw | <15ms | Canvas 2D drawing, profile on target device |
| **Total per frame** | **<52ms** | Should leave 48ms headroom |
| ScannerCapture.captureRow (once/row) | <200ms | Perspective warp + JPEG encode x3 |

**How to profile**:
```javascript
// Add timing to processFrame:
const t0 = performance.now();
// ... downsample ...
const t1 = performance.now();
// ... contour detect ...
const t2 = performance.now();
// ... auto capture ...
const t3 = performance.now();
// ... overlay ...
const t4 = performance.now();

console.log(`ds=${(t1-t0).toFixed(1)} contour=${(t2-t1).toFixed(1)} ` +
            `auto=${(t3-t2).toFixed(1)} overlay=${(t4-t3).toFixed(1)} ` +
            `total=${(t4-t0).toFixed(1)}ms`);
```

**Performance escape hatches (if too slow)**:
1. Reduce processing to every 4th or 5th frame (7.5fps or 6fps) -- still smooth enough for visual feedback
2. Skip binder detection entirely if not needed (hardcode orange binder)
3. Reduce Canny edge detection resolution to 360px instead of 480px
4. Use `OffscreenCanvas` + Web Worker for contour detection (Chrome only)
5. Simplify overlay: remove dimmed surround, simplify to outlines only

### Integration Test Checklist

- [ ] Camera opens on first load (permissions prompt shown)
- [ ] Overlay canvas matches video dimensions after rotation
- [ ] Contour detector finds 3 cards in well-lit binder photo
- [ ] Stability counter increments when phone is held steady
- [ ] Auto-capture triggers after 1s of stability
- [ ] Captured thumbnails appear in strip at bottom
- [ ] Row counter advances (1/3 -> 2/3 -> 3/3)
- [ ] After 3 rows, submission fires to /scanner/identify
- [ ] Server returns valid JSON with 9 card identifications
- [ ] Results display in 3x3 grid with names and prices
- [ ] "Start over" resets all state
- [ ] Works in landscape orientation
- [ ] Works in portrait orientation
- [ ] No memory leak after 5 minutes of continuous scanning (check DevTools memory)
- [ ] Frame rate stays above 8fps on target phone

---

## Known Challenges

### 1. Detecting exactly 3 cards (not 2 or 4)

**Problem**: The phone may be positioned between rows, showing cards from two rows, or at the edge of a row where only 2 cards are visible.

**Mitigation**:
- Guide rectangle hints the user to position 3 cards within the marked area
- Only accept contours whose centers fall within the guide rectangle
- If 4+ contours detected, pick the 3 with most consistent y-coordinates (same row)
- If 2 contours detected, show "Align 3 cards in frame" and wait
- Allow user override: "Capture 2 cards" button for pages with empty slots

### 2. Cards at different heights in sleeves

**Problem**: Cards may not be perfectly centered in their binder sleeve pockets, creating inconsistent vertical positions within a row.

**Mitigation**:
- Use generous aspect ratio tolerance (20%) during contour detection
- Stability check uses center-of-contour proximity, not pixel-perfect alignment
- Perspective correction handles mild vertical offsets
- Do NOT require all 3 cards to have identical y-coordinates; allow up to 15% height variance

### 3. Orange binder contaminating edge detection

**Problem**: Many binders are orange/red. Pokemon card borders are also yellow/orange. Canny edge detection produces false edges at binder-card transitions.

**Mitigation**:
- `BinderDetector` identifies the dominant binder color
- `CardContourDetector` masks out pixels matching the binder HSV range before edge detection
- Alternative: Use card interior features (artwork region is distinct from binder) rather than card border edges
- Fallback: If binder color is similar to card borders, switch to brightness-based detection (cards are brighter than binder pockets)

### 4. Binder ring shadow in the middle of a row

**Problem**: In 3-ring binders, the metal ring mechanism casts a shadow across the middle column. This shadow can:
- Break the middle card's contour (split into two contours)
- Reduce contrast, making edge detection fail for the middle card
- Create a false vertical edge through the center of the frame

**Mitigation**:
- If exactly 2 contours found and they are in left/right positions (no center), infer the middle card exists and estimate its position from the gap between the two detected cards
- Apply local contrast enhancement to the center 30% of the frame before edge detection
- Accept contours with one "broken" edge (3 clean sides + 1 partial side)

### 5. Plastic sleeve reflections

**Problem**: Overhead lights create bright reflections on the plastic sleeve surface, washing out card details and breaking contour detection.

**Mitigation**:
- Quality check during capture: detect large bright regions (>20% of card area above brightness 240)
- Show "Tilt phone slightly to avoid glare" hint when glare detected
- Stability requirement naturally helps: user must hold steady, encouraging slight repositioning
- Post-capture: CLAHE normalization can partially recover detail from moderate glare

### 6. Client-side contour detection accuracy

**Problem**: JavaScript canvas-based edge detection is less capable than OpenCV's optimized C++ implementation used in `card_segmenter.py`. No access to `cv2.findContours`, `approxPolyDP`, or adaptive thresholding.

**Mitigation**:
- Use a simplified detection approach: instead of full contour detection, look for rectangular regions that differ from the binder background in color/brightness
- Consider using OpenCV.js (WASM build of OpenCV, ~8MB) for proper contour detection -- trades download size for accuracy
- Alternative: Send downsampled frames to the server for contour detection (adds ~50ms network latency per frame, may be acceptable at 10fps)
- Start with simple brightness-based detection; upgrade to OpenCV.js if accuracy is insufficient

---

## Server-Side Endpoint

### `POST /scanner/identify`

Receives 9 card images captured by the scanner, runs the identification pipeline.

```python
def _handle_scanner_identify(self):
    """Handle scanner row capture submission.

    Receives up to 9 card images as multipart form data.
    Each image is a perspective-corrected card crop from the scanner.
    Runs identify_page_v2() on the images (skips segmentation).

    Same flow as _handle_slide_scan but with scanner-specific metadata.
    """
    # Parse multipart form data
    # Save images to data/inbox/scanner_{timestamp}_cards/card_XX.png
    # Run identify_page_v2() with skip_segmentation=True
    # Return JSON response (same format as /scan-page)
```

The server handler should reuse `_handle_slide_scan_identify()` logic since both receive pre-cropped card images. The only difference is the save directory prefix (`scanner_` vs `slide_`).

---

## File Dependencies

```
scanner_camera_ui.py (new)
    imports: nothing (self-contained HTML string)
    includes: scanner_overlay.js (inline, not as import)
    submits to: /scanner/identify

server.py (existing)
    GET /scanner -> serves SCANNER_HTML from scanner_camera_ui.py
    POST /scanner/identify -> _handle_scanner_identify() (new method)
    _handle_scanner_identify -> reuses slide-scan identification logic

scanner_overlay.js (existing)
    standalone JS class
    inlined into scanner_camera_ui.py

row_scanner.js (existing, NOT USED)
    this is from slide-scan v1, uses brightness-based 1D signal detection
    scanner uses contour-based 2D detection instead
    kept for reference but not integrated
```

---

## Implementation Order

1. **CardContourDetector** -- core detection, testable with static images
2. **ScannerAutoCapture** -- stability logic, testable with synthetic contour sequences
3. **ScannerCapture** -- perspective correction, testable with known contour coordinates
4. **BinderDetector** -- background analysis, testable with binder images
5. **ScannerController** -- main loop assembly, requires all above
6. **scanner_camera_ui.py** -- HTML wrapper with inline JS
7. **server.py `_handle_scanner_identify`** -- server endpoint
8. **Integration testing** -- full end-to-end on phone

Components 1-4 can be developed and unit-tested independently. The `ScannerOverlay` (component 3 in the architecture) is already complete.
