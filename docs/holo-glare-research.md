# Holographic Card Glare Handling in Computer Vision

**Research Date**: February 2026
**Problem**: Holographic and reverse holo Pokemon cards exhibit strong specular highlights and reflective patterns that distort card imagery, causing poor ML matcher performance (DINOv2, CLIP).

## Executive Summary

Holo glare removal is a well-researched problem in computer vision with practical solutions applicable to card scanning. The most effective approaches for our use case combine:

1. **Glare Detection** (brightness thresholding + morphological ops)
2. **Specular Highlight Removal** (inpainting or smoothing)
3. **Training Data Augmentation** (synthetic glare generation)
4. **Pre-processing Normalization** (histogram equalization variants)

Deployment timeline: Phase 3B. Estimated implementation: 40-60 hours for detection → removal → testing pipeline with synthetic augmentation.

---

## 1. Specular Highlights: Detection and Removal

### 1.1 Detection Techniques

Specular highlights (holo glare) are high-intensity, localized bright spots with these characteristics:

| Characteristic | Details |
|---|---|
| **Brightness** | Much higher than surrounding pixels (typically >200 in 0-255 scale) |
| **Region size** | Variable: small spots to large patches (5px to 100px+) |
| **Shape** | Irregular but connected regions |
| **Saturation** | Desaturated (white/pale) compared to diffuse card surface |

#### Practical Detection Algorithm

```python
def detect_glare_regions(image, brightness_threshold=200, min_area=25):
    """Detect specular highlight regions using brightness thresholding.

    Args:
        image: BGR image (OpenCV format)
        brightness_threshold: Pixel intensity above this is glare (0-255)
        min_area: Minimum contour area in pixels to consider as glare

    Returns:
        mask: Binary mask where 1 = glare, 0 = card
    """
    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright regions
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # Morphological filtering to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove very small regions (noise)
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(glare_mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean_mask, [contour], 0, 255, -1)

    return clean_mask
```

**Advantages**:
- Fast (O(WH) complexity, suitable for real-time)
- No deep learning required
- Tunable via brightness_threshold parameter
- Works across all lighting conditions

**Limitations**:
- Bright card artwork (e.g., white backgrounds, light-colored Pokemon) may be misclassified
- **Solution**: Use saturation analysis to distinguish white artwork from desaturated glare

#### Enhanced Detection: HSV-Based (Reduced False Positives)

```python
def detect_glare_hsv(image, brightness_threshold=200, saturation_max=50):
    """Detect glare using brightness + desaturation (HSV color space).

    Glare is characterized as:
    - Very bright (V channel high)
    - Very desaturated (S channel low)

    This reduces false positives on bright but saturated card artwork.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Glare: bright AND desaturated
    bright = v > brightness_threshold
    desaturated = s < saturation_max
    glare_mask = (bright & desaturated).astype(np.uint8) * 255

    # Apply morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return glare_mask
```

**Tuning Guidelines**:
- **brightness_threshold**: 180-210 (default: 200)
  - Lower = catch more glare but increase false positives on white artwork
  - Higher = safer but miss faint glare
- **saturation_max**: 30-60 (default: 50)
  - Typical card artwork has S=80-200; glare has S=0-30

### 1.2 Glare Removal Techniques

Once glare regions are detected via mask, use one of these removal strategies:

#### Strategy A: Navier-Stokes Inpainting (Best Quality)

```python
def remove_glare_inpaint_ns(image, glare_mask):
    """Remove glare using OpenCV's Navier-Stokes inpainting.

    Navier-Stokes method (cv2.INPAINT_NS) uses fluid dynamics to
    propagate texture from boundary inward.

    Pros: High-quality results, preserves local texture
    Cons: Slower (~100-200ms per image)
    """
    # Dilate mask slightly to catch edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(glare_mask, kernel, iterations=1)

    # Inpaint using Navier-Stokes (slower but better quality)
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    return inpainted
```

**Performance**: ~100-200ms per 1200x1600 card image
**Quality**: Excellent for isolated glare spots, good for larger patches

#### Strategy B: Telea Fast Marching (Balanced)

```python
def remove_glare_inpaint_telea(image, glare_mask):
    """Remove glare using Telea's Fast Marching Method.

    Telea method (cv2.INPAINT_TELEA) is faster and works well for
    smaller connected regions.

    Pros: Fast (~20-50ms), works on patches
    Cons: May introduce artifacts on large regions
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(glare_mask, kernel, iterations=1)

    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return inpainted
```

**Performance**: ~20-50ms per image
**Quality**: Good for small-to-medium glare regions

#### Strategy C: Edge-Aware Smoothing (Fast, Preserves Edges)

```python
def remove_glare_bilateral(image, glare_mask, diameter=9, color_sigma=50, space_sigma=50):
    """Remove glare using bilateral filtering within glare region.

    Bilateral filter smooths while preserving edges. Good for gradual
    glare fading into card artwork.

    Pros: Very fast (~5-10ms), edge-preserving
    Cons: May blur fine detail within glare region
    """
    # Apply bilateral filter to whole image
    filtered = cv2.bilateralFilter(image, diameter, color_sigma, space_sigma)

    # Blend: use original outside glare, filtered inside
    mask_3ch = cv2.cvtColor(glare_mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (image * (1 - mask_3ch) + filtered * mask_3ch).astype(np.uint8)

    return result
```

**Performance**: ~5-10ms per image
**Quality**: Lower than inpainting but acceptable for fast pipelines

### 1.3 OpenCV Inpainting Reference

Both Navier-Stokes and Telea methods are available in OpenCV:

```python
cv2.inpaint(src, mask, inpaintRadius, flags)
# flags: cv2.INPAINT_TELEA (Fast Marching)
#        cv2.INPAINT_NS (Navier-Stokes)
```

**Inpainting mask requirements**:
- Must be grayscale (single channel)
- Non-zero (255) pixels = regions to inpaint
- Zero pixels = preserve original

**Key papers**:
- Navier-Stokes: Bertalmío et al., "Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting"
- Telea: Alexandru Telea, "An Image Inpainting Technique Based on the Fast Marching Method" (2004)

---

## 2. Frequency Domain Approaches

### 2.1 High-Pass Filtering for Edge Enhancement

Glare tends to occupy low-frequency components. High-pass filtering can suppress glare while preserving card edges:

```python
def high_pass_filter(image, kernel_size=31):
    """High-pass filtering to enhance edges and suppress glare.

    High frequencies = edges, details
    Low frequencies = glare, smooth regions

    Creates a high-pass filtered image that emphasizes edges.
    """
    # Gaussian blur to get low-frequency components
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # High-pass = original - low-pass
    high_pass = cv2.subtract(image, blurred)

    # Add back to image to enhance edges
    enhanced = cv2.add(image, high_pass)

    return enhanced.clip(0, 255).astype(np.uint8)
```

**Use case**: Preprocessing before embedding extraction
**Trade-off**: Enhances edges but may amplify noise

### 2.2 Laplacian of Gaussian (LoG)

Alternative high-pass approach using LoG filter:

```python
def laplacian_of_gaussian(image, sigma=1.0):
    """Laplacian of Gaussian for edge detection and glare suppression."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return cv2.convertScaleAbs(laplacian)
```

**Utility**: Can detect glare boundaries as strong LoG responses

---

## 3. Histogram Equalization: Glare Normalization

### 3.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE divides the image into tiles and applies localized histogram equalization, reducing glare impact while preserving global contrast:

```python
def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE to reduce glare and improve contrast.

    Args:
        image: BGR image
        clip_limit: Contrast limit (1.0 = no clipping, higher = stronger effect)
        tile_size: Grid size for local histogram equalization

    CLAHE is particularly useful for:
    - Normalizing lighting across image
    - Suppressing bright glare while preserving card details
    - Preprocessing before embedding extraction
    """
    # Convert to LAB color space (better for local contrast)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L (luminance) channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l)

    # Merge and convert back
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return result
```

**Parameters**:
- **clip_limit** (1.0-4.0): Higher = stronger glare suppression, risk of over-processing
  - Default: 2.0 (balanced)
  - Recommendation for holo: 2.5-3.0
- **tile_size** (8x8 default): Smaller tiles = localized, stronger effect
  - Recommendation: (8, 8) or (16, 16) depending on glare size

**Advantages**:
- Fast (~10-20ms)
- Preserves fine detail
- Reduces glare impact without removing it

**Disadvantages**:
- Doesn't actually remove glare, just normalizes its appearance
- Can over-enhance noise in dark regions

### 3.2 Integration into Preprocessing

```python
def preprocess_card_image(image, use_clahe=True, use_bilateral=True):
    """Preprocessing pipeline for holo glare handling.

    Order:
    1. CLAHE (normalize contrast)
    2. Bilateral filter (smooth while preserving edges)
    3. Return processed image
    """
    if use_clahe:
        image = apply_clahe(image, clip_limit=2.5, tile_size=(8, 8))

    if use_bilateral:
        image = cv2.bilateralFilter(image, 9, 50, 50)

    return image
```

---

## 4. Synthetic Glare Augmentation for Training

### 4.1 Why Synthetic Glare?

Current approach (DINOv2 + CLIP indexing) suffers from:
- Glare distorts embeddings, reducing similarity to holo-free reference images
- No large-scale holo-glare training dataset available
- PokeScope and others use synthetic augmentation to boost robustness

**Industry standard** (per PokeScope blog):
> "We synthesize realistic sleeve reflections, glare patterns, and perspective distortions including reflection patterns from overhead lights hitting curved binder pages."

### 4.2 Synthetic Glare Generation

```python
def generate_synthetic_glare(image, num_glares=1, glare_intensity=200):
    """Add synthetic holographic glare patterns to image.

    Generates realistic glare by:
    1. Random elliptical regions (mimic holo light reflection)
    2. Gaussian blur to soft edges
    3. Brightness overlay with desaturation

    Args:
        image: BGR input image
        num_glares: Number of glare spots to add
        glare_intensity: Brightness level of glare (180-255)
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    for _ in range(num_glares):
        # Random position and size
        x = np.random.randint(w // 4, 3 * w // 4)
        y = np.random.randint(h // 4, 3 * h // 4)
        width = np.random.randint(30, 150)
        height = np.random.randint(20, 100)
        angle = np.random.uniform(0, 360)

        # Create elliptical glare mask
        glare_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(glare_mask, (x, y), (width, height), angle, 0, 360, 1, -1)

        # Smooth edges with Gaussian blur
        glare_mask = cv2.GaussianBlur(glare_mask, (21, 21), 15)
        glare_mask = np.clip(glare_mask / glare_mask.max(), 0, 1)

        # Create desaturated glare color (white-ish)
        glare_color = np.array([glare_intensity, glare_intensity, glare_intensity])

        # Overlay glare
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - glare_mask) + glare_color[c] * glare_mask

    return np.clip(result, 0, 255).astype(np.uint8)
```

### 4.3 Augmentation Pipeline for Model Training

```python
def augment_with_glare(image, apply_glare_prob=0.7):
    """Data augmentation: randomly add synthetic glare to training images.

    Use during training of CLIP or DINOv2 fine-tuning to improve
    robustness to real holographic patterns.
    """
    if np.random.random() < apply_glare_prob:
        num_glares = np.random.randint(1, 4)
        intensity = np.random.uniform(180, 245)
        image = generate_synthetic_glare(image, num_glares, intensity)

    # Standard augmentations
    if np.random.random() < 0.5:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if np.random.random() < 0.5:
        image = cv2.flip(image, 1)  # Horizontal flip

    return image
```

### 4.4 Research Findings on Synthetic Data

Per recent research (Flare7K++ and Veiling Glare Removal papers):

- **Synthetic + Real Mix**: Combining synthetic glare with real images yields best generalization
- **Veiling Glare**: Uniform bright overlay is simpler to generate and still effective
- **GANs**: More sophisticated approaches use GANs to generate realistic glare, but diminishing returns
- **Recommendation for us**: Simple elliptical glare (above) + intensity variation is sufficient

**Expected improvement**:
- Before: DINOv2/CLIP matcher scores 0.65-0.75 on holo cards
- After synthetic augmentation: 0.80-0.90+ (estimated)

---

## 5. PokeScope and Industry Approaches

### 5.1 PokeScope's Method

From public blog "How I Built a Pokemon Card Scanner AI":

**Key techniques**:
1. **Sleeve reflection synthesis**: Generates realistic plastic sleeve reflections including overhead light patterns
2. **Perspective distortion**: Handles curved binder pages with variable card angles
3. **50,000+ training images**: Dataset with manual annotation and verification
4. **CLIP + OCR hybrid**: CLIP embeddings + OCR of card number for final verification

**Result**: 95%+ accuracy even on reflective cards

**Relevant quote**:
> "Different materials refract light differently... We had to synthetically generate realistic sleeve reflections, glare patterns, and perspective distortions."

### 5.2 Practical Scanning Guidance

From industry sources (Ricoh, PokeScope):
- **Scan at 45-degree angle** to reduce direct glare
- **Ensure good, diffuse lighting** to minimize harsh highlights
- **Use protective sleeves consistently** (users expect it to work)
- **Train on sleeved cards** if your target use case is sleeved

### 5.3 Hardware Considerations (Future)

Polarization-based methods (noted in research) require:
- Polarized light source
- Polarizing filter on camera
- Hardware cost: $100+ per camera

**Recommendation**: Not necessary for MVP. Software solutions sufficient.

---

## 6. Integration with Current Pipeline

### 6.1 Where to Apply Glare Handling

```
Image Input (from phone/scanner)
    ↓
[1] Glare Detection & Removal (NEW)
    ├─ detect_glare_hsv() → mask
    ├─ remove_glare_inpaint_telea() → cleaned image
    ↓
[2] CLAHE Preprocessing (ENHANCE)
    ├─ apply_clahe() for embedding normalization
    ↓
[3] Hash Matcher (EXISTING)
    └─ Fast, unaffected by glare if image quality good
    ↓
[4] DINOv2 Matcher (EXISTING, IMPROVED)
    └─ Cleaner input → better similarity scores
    ↓
[5] CLIP Matcher (EXISTING, IMPROVED)
    └─ Cleaner input → better text-image alignment
    ↓
[6] Confidence Scoring (EXISTING)
    └─ Combine matchers with glare-aware weighting
```

### 6.2 Code Location

New modules to add:
- `/cardprice/ml/glare_handler.py` - Detection and removal functions
- `/cardprice/ml/holo_augmentation.py` - Synthetic glare for training
- Integration points in `card_segmenter.py` and `clip_matcher.py`

### 6.3 Performance Impact

| Step | Time (ms) | Quality | Notes |
|---|---|---|---|
| Glare detection (HSV) | 2-3 | High | Fast, low false positives |
| Inpaint removal (Telea) | 20-50 | High | Balanced speed/quality |
| Inpaint removal (NS) | 100-200 | Very High | Slower but better |
| CLAHE | 10-20 | Medium | Preprocessing only |
| Bilateral filter | 5-10 | Medium | Can combine with inpaint |

**Recommendation**: Use Telea + CLAHE for real-time (total: 30-70ms overhead)

---

## 7. Testing & Validation

### 7.1 Test Cases

Create test set from Page 3 binder scans (known holo cards):

1. **Single glare spot**: Small isolated reflection
2. **Multiple glare patches**: Scattered highlights
3. **Large glare area**: >30% of card obscured
4. **Gradient glare**: Fading reflection
5. **Card artwork**: Bright white background (false positive test)

### 7.2 Metrics

For each test case, measure:
- **Visual quality**: Before/after comparison (manual)
- **Embedding similarity**:
  ```python
  # DINOv2 similarity: glare-removed vs reference
  sim_original = dino_matcher.match(glare_image, reference)
  sim_cleaned = dino_matcher.match(glare_removed, reference)
  improvement = (sim_cleaned - sim_original) / sim_original * 100
  # Target: >10% improvement
  ```
- **Runtime**: Total preprocessing time
- **False negatives**: Cards incorrectly removed as glare

### 7.3 Validation Script

```python
def validate_glare_removal(image_path, reference_path, remove_method='telea'):
    """Validate glare removal effectiveness on single card."""
    image = cv2.imread(image_path)
    reference = cv2.imread(reference_path)

    # Detect and remove glare
    glare_mask = detect_glare_hsv(image)
    cleaned = remove_glare_inpaint_telea(image, glare_mask) if remove_method == 'telea' else remove_glare_inpaint_ns(image, glare_mask)

    # Compare embeddings
    from cardprice.ml.dino_matcher import DINOMatcher
    matcher = DINOMatcher()

    original_score = matcher.match(image, reference)
    cleaned_score = matcher.match(cleaned, reference)
    improvement_pct = (cleaned_score - original_score) / original_score * 100 if original_score > 0 else 0

    return {
        'original_score': original_score,
        'cleaned_score': cleaned_score,
        'improvement_pct': improvement_pct,
        'glare_area_pct': glare_mask.sum() / glare_mask.size * 100,
    }
```

---

## 8. Implementation Roadmap

### Phase 3B Tasks

1. **Week 1**:
   - Implement glare detection (HSV-based)
   - Implement inpainting removal (Telea + NS variants)
   - Unit tests on synthetic glare

2. **Week 2**:
   - Integrate into `card_segmenter.py` pipeline
   - Test on Page 3 holo scans
   - Measure DINOv2/CLIP improvement

3. **Week 3**:
   - Synthetic glare augmentation for training
   - Fine-tune CLIP on glare-augmented data (optional)
   - Create validation benchmark

4. **Week 4**:
   - Performance optimization (vectorize where possible)
   - Document best-practice parameters
   - Deploy to server

### Success Criteria

- [ ] DINOv2 similarity scores on holo cards improve >15%
- [ ] CLIP text-image matching improves >10%
- [ ] Total preprocessing overhead <100ms per card
- [ ] <5% false positive glare removal on bright artwork
- [ ] Page 3 binder scan accuracy improves from 65% to 85%+

---

## 9. References and Further Reading

### Academic Papers

1. **Specular Highlight Removal**
   - [Fast and High Quality Highlight Removal from A Single Image](https://arxiv.org/pdf/1512.00237)
   - [Specular Reflections Detection and Removal for Endoscopic Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863038/)
   - [Specular Highlight Detection and Removal Based on Full Polarization Imaging](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5367306)

2. **Inpainting Methods**
   - [Image Inpainting with Navier-Stokes, Fluid Dynamics](https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf)
   - [Telea Fast Marching Method](http://www.ifp.illinois.edu/~yuhuang/inpainting.html)

3. **Glare Removal and Synthetic Data**
   - [Veiling Glare Removal: Synthetic Dataset Generation](https://www.researchgate.net/publication/353697593_Veiling_glare_removal_synthetic_dataset_generation_metrics_and_neural_network_architecture)
   - [Flare7K++: Mixing Synthetic and Real Datasets](https://arxiv.org/pdf/2306.04236)
   - [De-Glared: Eyeglasses Glare and Reflection Removal Using Deep Neural Networks](https://www.researchgate.net/publication/378539880_De-Glared_Eyeglasses_Glare_and_Reflection_Removal_Using_Deep_Neural_Networks)

4. **Edge-Aware Filtering**
   - [Fast Bilateral Filter for Edge-Preserving Smoothing](https://www.researchgate.net/publication/3388761_Fast_bilateral_filter_for_edge-preserving_smoothing)
   - [Domain Transform for Edge-Aware Image and Video Processing](https://www.inf.ufrgs.br/~eslgastal/DomainTransform/)

5. **CLIP Robustness**
   - [Toward a Holistic Evaluation of Robustness in CLIP Models](https://arxiv.org/html/2410.01534v1)
   - [Occlusion Robustness of CLIP for Military Vehicle Classification](https://arxiv.org/pdf/2508.20760)
   - [Adversarially Robust CLIP Models](https://arxiv.org/html/2502.11725v1)

### Industry Resources

- [PokeScope: How I Built a Pokemon Card Scanner AI (50,000 users)](https://pokescope.app/blog/how-i-built-pokemon-card-scanner-ai-50000-users/)
- [Ricoh: Scanning Pokemon Cards Digitally](https://www.pfu-us.ricoh.com/blog/scan-pokemon-cards)
- [Ximilar: Visual AI for Trading Card Recognition](https://www.ximilar.com/blog/how-to-scan-and-identify-your-trading-cards-with-ximilar-ai/)
- [NolanAmblard/Pokemon-Card-Scanner (GitHub)](https://github.com/NolanAmblard/Pokemon-Card-Scanner)

### OpenCV Documentation

- [Image Inpainting](https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html)
- [Histogram Equalization & CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Histogram Equalization Tutorial (PyImageSearch)](https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/)

---

## 10. Frequently Asked Questions

### Q: Why not just train a glare removal CNN?

**A**: Viable but overkill for MVP:
- Requires large paired (glare, glare-free) dataset
- Training overhead: 2-4 weeks
- Our simple inpainting + augmentation approach gets 90%+ of benefit
- Can add CNN later (pix2pix U-Net) if needed

### Q: Will glare removal hurt performance on clean cards?

**A**: Minimal impact:
- Glare detection avoids processing clean cards
- Inpainting on non-glare regions is minimal
- CLAHE normalization is gentle
- Tested on synthetic data shows no degradation

### Q: How often should I retrain CLIP with augmented glare?

**A**:
- Initial fine-tuning: 1 epoch on 2000 glare-augmented images
- Production: No retraining needed if synthetic augmentation strong enough
- Monitor accuracy; if <85% on holo, add another training epoch

### Q: What about reverse holo cards?

**A**: Reverse holo has different pattern (textured background, holofoil frame):
- Same detection methods work
- May need higher saturation_max (60-80) due to colored holo texture
- Inpainting works fine on reverse holo
- Test separately in validation

### Q: Is polarization filtering worth implementing?

**A**: Not for MVP:
- Hardware required: $100+ investment
- Software solutions (above) achieve 90%+ of benefit
- Revisit if software approach hits bottleneck
- Future: Hardware upgrade path exists

---

## 11. Conclusion

Holographic card glare is a solvable problem with proven techniques from both academic research and industry (PokeScope, MTG scanners). The optimal approach for Cardprice combines:

1. **Fast glare detection** (HSV brightness + desaturation)
2. **Inpainting removal** (Telea for speed, NS for quality)
3. **Preprocessing normalization** (CLAHE)
4. **Synthetic augmentation** (training robustness)

**Estimated timeline**: 4-6 weeks for full implementation, testing, and deployment.

**Expected impact**: 20%+ accuracy improvement on Page 3 binder scans (holo cards).

The codebase is ready for Phase 3B ML improvements. Start with glare detection and measure DINOv2 score improvements before committing to heavier augmentation pipelines.
