# Slide-Scan Approach

Last updated: 2026-03-22

## Overview

Slide-scan is an alternative card capture method where the user holds their phone
close to a binder page and slowly slides it across rows of cards, capturing
individual card images from a live video feed. This contrasts with the existing
binder-page scanning approach where the user takes a single wide-angle photo of
the entire 3x3 page and the server segments it into 9 cards.

The key insight: most pipeline failures trace back to low image quality from
photographing 9 cards at once. At binder-page distance, each card occupies
roughly 1008x1530 pixels -- insufficient for reliable name OCR, stamp detection,
or holo shimmer analysis. Slide-scan captures each card at 2-4x higher effective
resolution by bringing the phone closer.

---

## User Flow

1. **Open slide-scan UI**: User navigates to `/slide-scan` on their phone browser.
2. **Position phone**: Hold phone 10-15cm above the binder page, close enough
   that 1-2 cards fill the frame.
3. **Slide across row**: Move the phone steadily across a row of cards (left to
   right). The live camera feed detects card edges as they enter and leave the
   frame.
4. **Automatic capture**: When a card is centered and in focus, the client-side
   JS captures a still frame. A brief haptic/visual pulse confirms capture.
5. **Move to next row**: After completing a row (3 captures), slide down to the
   next row. Repeat for all 3 rows.
6. **Review results**: After 9 cards are captured, the client sends all images
   to the server for identification. Results appear in the same grid layout as
   the binder-page scan results.

The user does not need to tap a shutter button. Capture is automatic based on
card detection, motion stability, and focus quality.

---

## Technical Architecture

### Client-Side (JS in `slide_scan_ui.py`)

**Video processing:**
- Access rear camera via `getUserMedia` with constraints for autofocus, high
  resolution (1920x1080 minimum), and rear-facing lens.
- Process frames from the video stream at 15-30 fps using a `<canvas>` element.
- Run lightweight card detection on each frame (edge detection or color
  contrast against the binder sleeve background).

**Card detection:**
- Detect card boundaries by looking for the characteristic yellow/silver/white
  Pokemon card border against the dark binder sleeve background.
- Use simple edge contrast: card borders create strong horizontal and vertical
  edges that form a rectangle.
- Track the detected rectangle's position and size across frames to determine
  when a card is centered and stable.

**Motion blur rejection:**
- Compare consecutive frames using pixel difference in a central region.
  High inter-frame difference indicates motion -- suppress capture.
- Require N consecutive low-motion frames (e.g., 3 frames at 15fps = 200ms
  of stability) before triggering capture.
- Laplacian variance on the capture region provides a sharpness score. Reject
  frames below a minimum sharpness threshold.

**Duplicate prevention:**
- After capturing a card, compute a simple fingerprint (downscaled grayscale
  thumbnail, e.g., 8x8 pixels) and compare against previously captured cards.
- Reject captures that are too similar to an already-captured card (same card
  still in frame after capture).
- Track card position movement: after capture, require the detected card
  rectangle to move significantly (>30% of frame width) before allowing the
  next capture. This prevents double-capturing the same card.

**Perspective correction:**
- The phone is held at an angle to the binder surface, producing mild
  perspective distortion (trapezoidal card shape).
- Apply a 4-point perspective warp on the detected card corners to produce a
  rectangular card image, similar to `_perspective_crop` in `card_segmenter.py`.
- This correction is lightweight and runs client-side using canvas transforms
  or a small WASM module.

**Capture flow:**
- Maintain a capture queue of up to 9 card images.
- Display a 3x3 grid showing captured card thumbnails (filled slots) and
  empty slots (remaining).
- Allow the user to tap a captured card to delete and re-capture.
- When 9 cards are captured (or user taps "Done"), send all images to the
  server.

### Server-Side

**Endpoint: `GET /slide-scan`**
- Serves the slide-scan UI HTML/CSS/JS from `slide_scan_ui.py`.

**Endpoint: `POST /slide-scan/identify`**
- Receives up to 9 card images as multipart form data.
- Each image is a pre-cropped individual card -- no segmentation needed.
  This skips the entire `card_segmenter.py` step, eliminating Problems 4
  (edge clipping), 11 (partially filled pages), and 14 (ring shadow).
- Runs the standard identification pipeline (`identify_page_v2`) on the
  submitted images. The pipeline still performs:
  - Name OCR (top 25% crop, upscale 3x, unsharp mask)
  - Attack OCR fallback
  - DINOv2 reference matching among candidates
  - Page context reranking (Pass 2/3)
  - Variant detection (stamp, holo, 1st edition, shadowless)
- Returns the same JSON response format as `/scan-page` for UI compatibility.

**No segmentation step:**
- The most significant architectural difference. Binder-page scanning requires
  `segment_cards()` to find and extract 9 cards from one photo. Slide-scan
  receives pre-isolated card images, so segmentation is skipped entirely.
- This eliminates an entire class of failure modes: contour detection failures,
  grid fallback inaccuracies, edge clipping, perspective distortion from
  wide-angle capture, and card back detection for empty slots.

### Results Display

- Results are displayed in the same 3x3 grid format as binder-page scans.
- Each cell shows: card thumbnail (captured image), identified card name,
  set name, market price, confidence score, and detected variant.
- Cards can be tapped to expand and show the reference image side-by-side
  with the captured image for visual verification.
- "Add to inventory" and "Add to cart" actions are available per-card.

---

## Expected Accuracy Improvement

### Why close-range capture produces better images

At binder-page distance (~40cm), the phone captures all 9 cards in one frame.
A 4032x3024 photo yields roughly 1008x1530 pixels per card after segmentation.
At slide-scan distance (~12cm), each card fills most of the frame, yielding
approximately 3000x4000 pixels per card -- a 3-4x resolution increase.

This higher resolution directly improves:
- **Name OCR**: Card name text goes from ~15px tall to ~45px tall. OCR
  accuracy on 45px text is near-perfect; at 15px, character confusion
  (n/h, l/i, o/0) is common.
- **Attack OCR**: Attack text is even smaller than name text. At binder
  resolution, attacks are often unreadable. At slide-scan resolution,
  attack text becomes reliably readable.
- **Stamp detection**: EX-era stamps are translucent overlays that require
  fine texture analysis. At 3-4x resolution, stamp text ("DRAGON FRONTIERS",
  "CRYSTAL GUARDIANS") becomes OCR-readable, potentially replacing the
  current DINOv2+logistic regression classifier (68.8% on binder scans).

### Expected DINOv2 score improvement

DINOv2 cosine similarity between a captured card image and its reference
image depends heavily on image quality:

| Capture method     | Typical DINOv2 score | Notes                               |
|--------------------|----------------------|-------------------------------------|
| Binder page scan   | 0.30 - 0.60          | Plastic sleeve, glare, low res      |
| Slide-scan         | 0.65 - 0.85          | Close-up, still through sleeve      |
| Out-of-sleeve      | 0.80 - 0.95          | Direct card photo, no plastic       |
| Digital reference   | 1.00                 | Identical image                     |

The 0.30-0.60 range for binder scans means DINOv2 often cannot discriminate
between similar cards (correct card at 0.55, wrong card at 0.53). At 0.65-0.85,
the margin between correct and incorrect increases significantly, making DINOv2
a reliable discriminator among name-matched candidates.

### Failure modes this solves

| Problem | Binder scan | Slide-scan | Why                                    |
|---------|-------------|------------|----------------------------------------|
| P1: Possessive prefix | Partial | Solved | Full name visible at close range |
| P2: Stamp detection | 68.8% | ~90%+ | Stamp text OCR-readable |
| P3: Holo detection | ~0% | ~0% (static), possible (video) | Still single frame |
| P4: Edge clipping | Mitigated | Eliminated | No segmentation needed |
| P6: Art variant | Partial | Improved | Higher DINOv2 discrimination |
| P7: OCR noise | Mitigated | Reduced | Cleaner text at higher res |
| P13: Glare | Common | Reduced | Smaller area, easier to avoid |
| P14: Ring shadow | Present | Eliminated | No full-page capture |
| P16: Blur | 0/6 | Rare | Motion blur rejection, auto-capture |

### Failure modes that remain

- **Wrong variant when multiple exist (P6)**: If two cards share identical
  artwork and differ only in holo treatment, slide-scan (static capture) cannot
  distinguish them. Only video-based holo detection or out-of-sleeve inspection
  can resolve this.
- **Holo detection through sleeves (P3)**: A single static frame through
  plastic still cannot detect holographic shimmer. This requires multi-frame
  video analysis (see Future Extensions).
- **Japanese/foreign text (P8)**: Higher resolution helps OCR but does not
  solve the fundamental lack of Japanese OCR model.
- **DINOv2 global fallback (P9)**: When both OCR paths fail, DINOv2 global
  is still useless regardless of image quality.

---

## Trade-offs vs Binder-Page Scanning

| Dimension              | Binder-page scan      | Slide-scan                |
|------------------------|-----------------------|---------------------------|
| **Time per page**      | ~15s (1 photo + processing) | ~30s (slide 3 rows + processing) |
| **Accuracy (eval GT)** | 98.1% (clean photos)  | Expected 95%+ on external images |
| **Accuracy (external)**| ~88% (estimated)      | Expected 95%+             |
| **Re-scan rate**       | ~10% (blurry/glare)   | ~2% (auto quality check)  |
| **Effective time**     | ~17s (including re-scans) | ~31s (rarely needs re-scan) |
| **Segmentation errors**| Present (P4, P11, P14)| None (no segmentation)    |
| **UX complexity**      | Point and shoot       | Guided sliding motion     |
| **Server load**        | Segmentation + 9x ID  | 9x ID only (no segmentation) |
| **Image quality**      | ~1008x1530 per card   | ~3000x4000 per card       |
| **Variant detection**  | Limited by resolution  | Significantly improved    |
| **Holo detection**     | Impossible (static)   | Impossible (static), possible (video extension) |
| **Works in sleeves**   | Yes                   | Yes                       |

**When to use binder-page scan:**
- Speed is the priority (large collection, hundreds of pages)
- Cards are in good lighting with no glare
- Variant detection is not critical (just need card identity)
- User prefers minimal interaction

**When to use slide-scan:**
- Accuracy is the priority (high-value cards, pricing)
- Binder-page scan produced errors that need correction
- Variant/stamp detection matters for pricing
- External/unknown images that may not segment cleanly

---

## File Structure

```
cardprice/
  slide_scan_ui.py        # HTML/CSS/JS for the slide-scan UI
                          # Returns the full page as a string, served by server.py

cardprice/
  server.py               # Adds two new routes:
                          #   GET  /slide-scan          -> serves UI from slide_scan_ui.py
                          #   POST /slide-scan/identify  -> receives 9 card images,
                          #                                 runs identify_page_v2,
                          #                                 returns JSON results
```

**`cardprice/slide_scan_ui.py`:**
- Single file containing the complete client-side application as an HTML string.
- Follows the same pattern as the existing binder scan UI (inline JS/CSS).
- Handles camera access, card detection, capture logic, duplicate prevention,
  perspective correction, and result display.
- Communicates with the server via `fetch()` to `/slide-scan/identify`.

**`GET /slide-scan`:**
- Serves the slide-scan UI HTML page.
- No authentication or session required.
- Mobile-optimized layout (designed for phone use).

**`POST /slide-scan/identify`:**
- Accepts `multipart/form-data` with up to 9 image files.
- Each image is an individual card crop (JPEG or PNG).
- Saves images to `data/inbox/slide_{timestamp}_cards/card_XX.png`.
- Runs `identify_page_v2()` on the list of image paths.
- Returns JSON with the same structure as `/scan-page` response:
  ```json
  {
    "status": "ok",
    "page_image": null,
    "cards": [
      {
        "position": 0,
        "row": 0,
        "col": 0,
        "card_id": "base1-4/holofoil",
        "confidence": 0.92,
        "method": "name_path",
        "detected_variant": "holofoil",
        "card_name": "Charizard",
        "market_price": 245.00,
        "local_image_url": "/card-image/base1-4/holofoil",
        "segment_image_url": "/segment-image/slide_20260322_143000_cards/card_00.png"
      }
    ]
  }
  ```

---

## Future Extensions

### Video-Based Holo Detection

The most promising extension. Instead of capturing a single still frame per
card, record a 1-2 second video clip while the phone is moving over each card.
Holographic cards produce color shifts across frames as the viewing angle
changes; non-holo cards remain static.

**Approach:**
- Extract 10-20 frames from the video clip.
- Compute hue histograms for the artwork region in each frame.
- Measure hue distribution variance across frames.
- Holo cards will show high inter-frame hue variance (prismatic color shifts).
- Non-holo cards will show low inter-frame hue variance (consistent colors).

This directly addresses Problem 3 (holo detection through sleeves), which is
currently unsolvable with single-frame capture. The sliding motion inherent
to slide-scan naturally provides the angle variation needed.

### Automatic Row Counting

Currently assumes a 3x3 binder page (standard 9-pocket page). Could be
extended to support other layouts:

**Approach:**
- Before sliding, take one wide-angle "overview" shot of the full page.
- Detect the grid structure (number of rows and columns) from the overview.
- Guide the user to slide across the correct number of rows.
- Support 4x3 (12-pocket), 3x3 (9-pocket), 2x2 (4-pocket), and 1x1 (top
  loader) layouts.

### Integration with Batch Scanning Session State

Link slide-scan sessions to the existing batch scanning workflow:

- Maintain a session across multiple binder pages (slide-scan page 1, page 2,
  etc.).
- Track which pages have been scanned and their results.
- Allow mixing binder-page scan and slide-scan within the same session
  (e.g., binder-page scan for quick overview, slide-scan for pages with
  errors).
- Cumulative price totals and inventory updates across the session.

### Hybrid Approach: Binder Scan + Slide Re-scan

Combine both methods for optimal speed and accuracy:

1. Quick binder-page scan of all pages (~15s each).
2. Identify low-confidence cards from the results.
3. Prompt user to slide-scan only the pages or cards that need re-scanning.
4. Merge high-confidence results from binder scan with improved results from
   slide-scan.

This gives binder-scan speed for the ~90% of cards that identify correctly
on the first pass, and slide-scan accuracy for the ~10% that need it.
