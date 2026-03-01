# Condition Grading from Photos: Computer Vision Research

**Date:** 2026-02-28
**Goal:** Detect card condition (NM, LP, MP, HP, DMG) from phone photos for inventory valuation.

---

## 1. What Features Indicate Wear?

Professional grading (PSA, BGS, CGC) evaluates four sub-grades that map directly to
computer-vision targets:

| Sub-grade | Visual signals | CV detection approach |
|-----------|---------------|----------------------|
| **Corners** | Whitening, rounding, dings, fraying, peeling layers | Crop 4 corner ROIs, classify sharpness via CNN or measure radius of curvature |
| **Edges** | White specks/lines along borders, chipping, nicks | Edge detection (Canny/Sobel) on card perimeter, compare border color uniformity |
| **Surface** | Scratches, print lines, creases, stains, holo-bleeding, indentations | Texture analysis under controlled lighting; contrast enhancement to reveal micro-scratches |
| **Centering** | Border width asymmetry (front: 60/40 for PSA 10, back: 75/25) | Detect card art bounding box, measure pixel distances to each edge, compute L/R and T/B ratios |

### Condition-to-grade mapping (TCGPlayer scale)

| TCGPlayer Condition | PSA Equivalent | Key defects |
|---------------------|----------------|-------------|
| Near Mint (NM) | PSA 7-10 | Minimal to no wear; corners sharp; surface clean |
| Lightly Played (LP) | PSA 5-6 | Slight edge whitening; minor corner wear; light surface marks |
| Moderately Played (MP) | PSA 3-4 | Noticeable whitening on edges/corners; light creases; visible scratches |
| Heavily Played (HP) | PSA 1-2 | Heavy whitening; rounded corners; creases; surface damage |
| Damaged (DMG) | PSA 1 (Authentic) | Tears, bends, water damage, missing pieces, heavy creases |

---

## 2. Industry Landscape (2025-2026)

### Commercial services

| Service | Claimed accuracy | Approach | Notes |
|---------|-----------------|----------|-------|
| **BinderAI** | 87% vs PSA grades | Phone photo upload (front+back), ML model trained on graded cards | iOS app; pre-grading tool, not a replacement for PSA |
| **TCG AI Pro** | 95% (self-reported) | Pixel-distance centering, contrast analysis for micro-scratches | Supports Pokemon, MTG, sports cards |
| **TCGrader** | "high accuracy" (unspecified) | Continuously trained on thousands of professionally graded cards | Pokemon-focused marketplace integration |
| **CardCondition** | Not disclosed | 2-3 photos, simulates PSA/BGS/CGC/SGC grades in 60s | Multi-grader simulation |
| **Ximilar** | Not disclosed | API-based; full grading + lightweight condition endpoint (NM/Played/Damaged) | Supports TCGPlayer/eBay/Cardmarket scales; 0.5 credits for condition vs 1.0 for full grade |
| **TAG Grading** | Patented | Photometric Stereoscopic Imaging; 1000-point scale; 800% zoom | Physical grading company using CV, not just pre-grading |
| **AGS** | Not disclosed | Fully automated AI grading (no humans); 10x faster than manual | Positions as replacement for traditional grading |

### How BinderAI achieves 87% accuracy

BinderAI's approach is straightforward but effective:
- User photographs front and back of card on a dark, shadow-free, glare-free background
- ML models trained specifically on sports/TCG card grading criteria analyze the four sub-grades
- The system outputs a predicted PSA grade range
- 87% of predictions match the actual PSA grade (likely within +/-1 grade point)
- Limitation: phone camera quality varies; no controlled lighting means surface defects
  are harder to detect than with dedicated scanning hardware

### Academic research

1. **"Automated corner grading of trading cards: Defect identification and confidence
   calibration through deep learning"** (ScienceDirect, 2024)
   - Transfer learning model for corner grading
   - Confidence calibration techniques to improve reliability
   - Rule-based method using confidence scores for final grade assignment

2. **"A Multistage Hybrid AI Framework for Explainable Automated Trading Card Grading"**
   (ResearchGate, 2025)
   - Multi-stage pipeline: preprocessing -> defect detection -> grading
   - YOLOv8 with oriented bounding boxes for defect types (corner whitening, edge wear,
     scratches, creases)
   - Flat Field Correction (FFC) for illumination normalization
   - Explainable AI emphasis: shows which defects drove the grade

3. **"Edge Grading in Trading Cards Using Transfer Learning"** (ResearchGate, 2025)
   - Focused specifically on edge sub-grade
   - Transfer learning methods with evaluation

---

## 3. Image Preprocessing Pipeline

Based on the literature and commercial implementations, the preprocessing pipeline is:

```
Raw phone photo
    |
    v
[1] Card detection & extraction
    - YOLO or contour detection to find card boundaries
    - Perspective correction (de-skew) using corner points
    - Crop to card only, normalize to standard size (e.g., 224x224 or higher)
    |
    v
[2] Illumination normalization
    - Flat Field Correction (FFC) to remove uneven lighting
    - White balance correction
    - Histogram equalization or CLAHE for contrast enhancement
    |
    v
[3] Region segmentation
    - Crop 4 corners (e.g., 64x64px each)
    - Crop 4 edges (thin strips along each border)
    - Extract center/art region for surface analysis
    - Detect card border for centering measurement
    |
    v
[4] Feature extraction per region
    - Corners: edge sharpness, whitening (color deviation from expected border color)
    - Edges: uniformity analysis, white pixel detection along border
    - Surface: scratch detection via directional filters, crease detection via line detection
    - Centering: border width ratios (left/right, top/bottom)
    |
    v
[5] Grade prediction
    - Per-subgrade CNN classifiers, or
    - Single multi-output model, or
    - Defect detection (YOLO) + rule-based grade mapping
```

### Key preprocessing insights

- **Dark background is essential** for edge/corner whitening detection (white-on-dark contrast)
- **Even lighting matters more than resolution** for surface defect detection
- **Front AND back photos needed** for centering (back centering standards are looser: 75/25)
- **Image augmentation** during training: vary blur, exposure, saturation to handle phone camera variance
- **OpenCV edge detection** (Canny/Sobel) on card perimeter to verify card is flat and unskewed

---

## 4. Open-Source Tools and Resources

### GitHub repositories

1. **crimsonthinker/psa_pokemon_cards**
   - Pokemon card grading using deep learning
   - VGG16 backbone with dual network flows for different card aspects
   - U-Net for card content cropping (preprocessing)
   - Average prediction error: ~0.5 on PSA 1-10 scale
   - Grades all 4 sub-aspects: Centering, Corners, Edges, Surface
   - https://github.com/crimsonthinker/psa_pokemon_cards

2. **rthorst/mint_condition**
   - Automatic sports card grading
   - https://github.com/rthorst/mint_condition

3. **u-siri-ous/KYC (Know Your Cards)**
   - Card grader and classifier
   - https://github.com/u-siri-ous/KYC

4. **NickPiscitelli/pokemon-card-analyzer**
   - Pokemon card centering analysis
   - https://github.com/NickPiscitelli/pokemon-card-analyzer

### Roboflow models (pre-trained, deployable)

- **Card Grader** (Group 6 Major Project): 632 images, detects Edge Wear, Scratch, Corner Wear
  - https://universe.roboflow.com/group-6-major-project/card-grader
- **Card Grading** (Jason Brenan): Object detection for grading defects
  - https://universe.roboflow.com/jason-brenan/card-grading
- **AI Card Grading** workspace on Roboflow Universe
  - https://universe.roboflow.com/ai-card-grading

### Commercial APIs (if we want to avoid building from scratch)

- **Ximilar Card Grading API**: `POST /card-grader/v2/condition` returns NM/Played/Damaged
  with marketplace-specific labels (TCGPlayer, eBay, Cardmarket modes). Lower cost than
  full grading endpoint. Good for bulk inventory processing.

---

## 5. Centering Analysis Deep Dive

Centering is the most straightforward sub-grade to automate because it is purely geometric.

### Algorithm

```python
# Pseudocode for centering measurement
def measure_centering(card_image):
    # 1. Detect outer card boundary (physical edge of card)
    card_contour = detect_card_edge(card_image)  # Canny + findContours

    # 2. Detect inner art/frame boundary
    #    For Pokemon cards: the yellow/colored border meets the card art
    art_contour = detect_art_boundary(card_image)  # Color segmentation or edge detection

    # 3. Measure border widths
    left_border = art_contour.left - card_contour.left
    right_border = card_contour.right - art_contour.right
    top_border = art_contour.top - card_contour.top
    bottom_border = card_contour.bottom - art_contour.bottom

    # 4. Calculate ratios
    lr_ratio = f"{min(left_border, right_border)}/{max(left_border, right_border)}"
    tb_ratio = f"{min(top_border, bottom_border)}/{max(top_border, bottom_border)}"

    # 5. Grade centering
    # PSA 10: 60/40 front, 75/25 back
    # PSA 9: 65/35 front, 80/20 back
    lr_pct = min(left_border, right_border) / (left_border + right_border) * 100
    tb_pct = min(top_border, bottom_border) / (top_border + bottom_border) * 100
    worst_pct = min(lr_pct, tb_pct)

    if worst_pct >= 45:    return 10  # 55/45 or better
    elif worst_pct >= 40:  return 9   # 60/40
    elif worst_pct >= 35:  return 8   # 65/35
    elif worst_pct >= 30:  return 7   # 70/30
    elif worst_pct >= 25:  return 6   # 75/25
    else:                  return 5   # worse than 75/25
```

### Challenges
- Pokemon cards have varying border styles across eras (Base Set vs modern V/VSTAR)
- Full-art and alternate-art cards may lack traditional borders
- Back centering requires a separate photo
- Camera angle distortion must be corrected before measuring

---

## 6. Recommended Approach for Cardprice

### Goal
We do NOT need PSA-level precision. We need TCGPlayer condition bins (NM/LP/MP/HP/DMG)
for inventory valuation. This is a 5-class classification problem, not a 1-10 regression.

### Phased approach

#### Phase A: Quick wins with existing tools (1-2 days)
1. **Centering measurement** via OpenCV
   - Detect card contour, detect art boundary, compute border ratios
   - Pure geometry, no ML needed
   - Output: centering sub-grade (perfect/acceptable/off-center/severely-off)

2. **Corner/edge whitening detection** via color analysis
   - Crop corners and edges after perspective correction
   - Compare border region color to expected card border color
   - Threshold white pixel percentage for each region
   - No ML needed for a rough cut (NM vs not-NM)

#### Phase B: ML condition classifier (3-5 days)
1. **Data collection strategy**
   - Scrape eBay sold listings that include condition in title ("NM", "LP", "HP", etc.)
   - Use our existing eBay scraper infrastructure
   - Target: 500+ images per condition class
   - Alternative: use Ximilar API to label our own photos (bootstrap labeling)

2. **Model architecture** (pick one)
   - **Option 1 - Transfer learning CNN**: ResNet-50 or EfficientNet, fine-tuned on
     card condition dataset. Multi-output for sub-grades or single output for overall condition.
     Based on crimsonthinker's results, expect ~0.5 grade error on PSA scale.
   - **Option 2 - Defect detection + rules**: YOLOv8 trained to detect specific defects
     (whitening, scratches, creases, rounding). Count and severity of defects maps to
     condition via rules. More explainable. Based on 2025 research paper approach.
   - **Option 3 - Claude Vision**: Send card photo to Claude with a structured prompt
     asking it to evaluate corners, edges, surface, centering. No training needed.
     Good for bootstrapping labels and as a fallback.

3. **Recommended: Option 2 (defect detection) + Option 3 (Claude fallback)**
   - YOLOv8 defect detector is explainable ("LP because: edge whitening detected on left
     and bottom edges")
   - Claude Vision handles edge cases and provides second opinion
   - We already use Claude Haiku for card identification; same infrastructure

#### Phase C: Integration with inventory (1 day)
- Add condition column to inventory scan workflow
- Condition-adjusted pricing: apply TCGPlayer condition multipliers to market price
- Already have `condition_pricing.py` -- wire it to actual detected condition

### What we should NOT do
- Build a PSA 1-10 predictor (we don't need that precision)
- Invest in special hardware (phone photos are sufficient for 5-class condition)
- Try to detect authentication/fakes (different problem, requires physical inspection)

### Image capture requirements for users
- Dark/black background (essential for edge whitening detection)
- Even, diffuse lighting (no direct flash, no harsh shadows)
- Both front and back photos
- Card fills most of the frame
- No glare on holofoil (or take a second non-glare photo)
- Our existing scanner UI already handles photo capture from phone

### Expected accuracy
- Centering: ~95%+ (purely geometric)
- Overall 5-class condition: 75-85% with transfer learning CNN, improving with more data
- With Claude Vision fallback: potentially 85-90% for the combined system
- BinderAI's 87% on PSA 1-10 scale suggests 90%+ is achievable for our coarser 5-class problem

---

## Sources

- [CardGrading.app - AI Pokemon Card Grading](https://www.cardgrading.app/blog/ai-pokemon-card-grading/)
- [TCGrader - How AI Card Grading Works](https://www.tcgrader.com/blog/how-ai-card-grading-works-complete-guide)
- [Roboflow - Using CV to Make Card Grading Faster and Cheaper](https://blog.roboflow.com/using-computer-vision-to-make-card-grading-faster-and-cheaper/)
- [Roboflow Universe - Card Grader Model](https://universe.roboflow.com/group-6-major-project/card-grader)
- [Ximilar - AI Card Grading via API](https://www.ximilar.com/blog/ai-card-grading-automate-sports-cards-pre-grading/)
- [Ximilar API Docs - Card Grading](https://docs.ximilar.com/collectibles/card-grading)
- [ScienceDirect - Automated Corner Grading of Trading Cards (2024)](https://www.sciencedirect.com/science/article/pii/S0166361524001155)
- [ResearchGate - Multistage Hybrid AI Framework for Card Grading (2025)](https://www.researchgate.net/publication/400864149_A_Multistage_Hybrid_Artificial_Intelligence_Framework_for_Explainable_Automated_Trading_Card_Grading)
- [ResearchGate - Edge Grading via Transfer Learning (2025)](https://www.researchgate.net/publication/388093386_Edge_Grading_in_Trading_Cards_Using_Transfer_Learning_Methods_Experiments_and_Evaluation)
- [GitHub - crimsonthinker/psa_pokemon_cards](https://github.com/crimsonthinker/psa_pokemon_cards)
- [GitHub - rthorst/mint_condition](https://github.com/rthorst/mint_condition)
- [GitHub - NickPiscitelli/pokemon-card-analyzer](https://github.com/NickPiscitelli/pokemon-card-analyzer)
- [BinderAI](https://www.binder-ai.com/)
- [TCG AI Pro](https://tcgai.pro/)
- [TAG Grading](https://taggrading.com/)
- [AGS Card Grading](https://agscard.com/)
- [CardCondition](https://cardcondition.com/)
- [CardGrader.AI](https://cardgrader.ai/)
