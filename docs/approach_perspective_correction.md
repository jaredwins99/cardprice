# Approach: Post-Segmentation Perspective Correction

## Overview

Post-segmentation image correction to reduce the domain gap between binder scans and reference images. After `card_segmenter.py` extracts individual card crops from a binder page photo, each crop is corrected for perspective distortion, color cast, and lighting inconsistency before being passed to DINOv2 matching or OCR.

## The Domain Gap Problem

The identification pipeline compares binder-scanned card images against a reference database of 20,026 clean card images. These two sources look very different:

**Binder scans (query images):**
- Perspective distortion from camera angle (cards are rarely photographed perfectly head-on)
- Color cast from ambient lighting (warm indoor light, fluorescent tint)
- Sleeve reflections and glare hotspots
- Uneven lighting across the binder page (corners darker than center)
- Slight barrel/pincushion distortion from phone lenses

**Reference images (database):**
- Flat, front-facing, no perspective warp
- Uniform studio lighting, neutral white balance
- No sleeves, no reflections
- Consistent color reproduction

**Impact on matching:**
- DINOv2 cosine similarity on binder scans: typically 0.3-0.6
- DINOv2 cosine similarity on clean/flat scans: 0.85+
- This gap means DINOv2 can only reliably pick among a small candidate set (2-20 cards pre-filtered by OCR), not search the full 20k database directly

## Correction Pipeline

Applied per-card after segmentation, before identification:

### Step 1: Perspective Warp (4-Corner Homography)

Detect the four corners of the card within the crop (the segmenter includes some background margin). Compute a homography matrix mapping those four points to a standard rectangle matching the Pokemon card aspect ratio (2.5" x 3.5", or 63mm x 88mm). Apply `cv2.warpPerspective` to produce a front-facing, de-skewed card image.

Corner detection options:
- Hough line intersection (find card edges as lines, intersect for corners)
- Contour-based (find the card contour, approximate to 4 points)
- Harris/Shi-Tomasi corner detection with geometric filtering

### Step 2: CLAHE Contrast Normalization

Apply Contrast Limited Adaptive Histogram Equalization to the L channel in LAB color space. This corrects uneven lighting across the card surface (e.g., one side brighter due to light angle) without distorting colors. Use a clip limit of 2.0-3.0 and tile grid of 8x8.

### Step 3: White Balance Correction (Gray World)

Apply the gray world assumption: the average color of the image should be neutral gray. Scale each channel so its mean equals the global mean across all channels. This removes color casts from warm/cool ambient lighting.

### Step 4: Optional Histogram Matching Against Reference

Match the color histogram of the corrected card image to a "typical" reference image histogram. This pulls the overall brightness and contrast distribution closer to what DINOv2 was trained on (or at least closer to the reference embeddings). Optional because it is the most aggressive step and risks destroying distinguishing features.

## Expected Improvements

- **DINOv2 similarity scores** should increase by 0.1-0.3 on typical binder scans, bringing them closer to the 0.85+ range seen on clean scans
- **DINOv2-only fast path** may become viable: if corrected images consistently score 0.7+ against the correct card, it could be possible to skip OCR entirely for high-confidence matches, reducing per-card time
- **Stamp and variant detection** should improve from cleaner, more uniform images (stamps like "1st Edition" and "PRERELEASE" are small and sensitive to perspective/lighting)
- **Attack OCR accuracy** may improve slightly from better contrast and de-skewing

## Trade-offs

**Processing time:**
- Adds approximately 50ms per card (perspective warp + color corrections)
- For a 3x3 binder page (9 cards), this adds ~450ms total
- Current pipeline is ~2-4 seconds per page, so this is a 10-20% increase

**Risk of over-correction:**
- Aggressive histogram matching could destroy subtle visual features that distinguish card variants (holo patterns, reverse holo sheen, shadowless border differences)
- CLAHE with too-high clip limit can amplify noise in dark regions
- White balance correction assumes no intentionally colored lighting in the card art, which is always violated (card art is not gray on average) -- must apply to border regions or full image carefully

**Uncorrectable distortions:**
- Severe motion blur (camera shake during capture)
- Extreme viewing angles (>30 degrees from normal)
- Heavy glare covering card text or art
- Cards partially obscured by sleeve edges or binder ring shadows

**Tuning sensitivity:**
- CLAHE clip limit, tile size, white balance strength all need tuning
- Different binder page photos may need different correction strengths
- Over-tuning to one set of test images may regress on others

## Files

| File | Purpose |
|------|---------|
| `cardprice/ml/card_corrector.py` | Perspective warp (homography) + normalization pipeline |
| `cardprice/ml/color_normalizer.py` | Color correction techniques (CLAHE, gray world, histogram matching) |
| `scripts/benchmark_correction.py` | Before/after DINOv2 score comparison on test images |

## Comparison with Other Approaches

| | Approach 1 (Current) | Approach 2 (Slide-Scan) | Approach 3 (This) |
|---|---|---|---|
| **Method** | Binder page photo, segment, identify | Close-range video capture, one card at a time | Same as Approach 1 + image correction |
| **Input** | Single photo of 3x3 page | Video stream, ~1 card/second | Single photo of 3x3 page |
| **Image quality** | Moderate (perspective, lighting issues) | High (close-up, controlled framing) | Improved (corrected perspective/color) |
| **Speed** | ~3s per page (9 cards) | ~10-15s per page equivalent | ~3.5s per page (9 cards) |
| **External accuracy** | 88% | Higher per-card quality, untested at scale | Potentially 90-95% (estimated) |
| **User effort** | Low (one photo per page) | High (manually slide each card) | Low (same as Approach 1) |
| **Implementation** | Done | Requires new capture UI | Incremental addition to existing pipeline |

Approach 3 is the lowest-effort improvement path: same user workflow, same segmentation, just better image quality before identification. It does not replace OCR-based candidate narrowing but should make DINOv2's final pick more reliable and may reduce the number of cases that fall through to the attack-OCR fallback path.
