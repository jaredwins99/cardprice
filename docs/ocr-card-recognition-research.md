# OCR for Pokemon Card Name Recognition

Research date: 2026-02-28

## Summary

EasyOCR can reliably read Pokemon card names from well-framed card photos.
On CPU it runs at ~1.3s per full card image, or ~0.19s on a cropped name region.
It significantly outperforms Tesseract for this use case. The main challenge is
not OCR accuracy but rather locating the name text among all detected text on
the card.

## Environment

- EasyOCR 1.7.2 (already installed, pulled in by `ocr-ops`)
- Model cache: 94 MB in `~/.EasyOCR/`
- Dependencies: torch, torchvision, numpy, opencv, scipy, etc. (already present)
- Tesseract available via pytesseract + `/usr/bin/tesseract`
- Test images: 9 card segments from a binder page scan (630x880px each)

## Test Results

### EasyOCR on Full Card Images

Running `reader.readtext()` on the full 630x880 card image detects all text
regions (name, HP, attacks, flavor text, etc.). The card name "Mr. Mime" was
detected with 0.88 confidence on the test image.

Batch results across 9 cards (full image, name identified by highest confidence
among top-positioned detections):

| Card | Actual Name | EasyOCR Top Result | Confidence |
|------|-------------|-------------------|------------|
| card_00 | Natu | Natu | 0.95 |
| card_01 | Xatu | Xatu | 1.00 |
| card_02 | Mr. Mime | Mr. Mime | 0.88 |
| card_03 | Natu | Natu | 1.00 |
| card_04 | Xatu | Xatu | 1.00 |
| card_05 | Rattata | Rattata | 0.91 |
| card_06 | Raticate* | Not detected | -- |
| card_07 | Sandshrew* | Not detected | -- |
| card_08 | Ditto* | Not detected | -- |

*Cards 06-08 had the name area partially cut off/obscured in the segmented
image. The name was not visible in the photo at all -- this is a segmentation
problem, not an OCR problem.

### EasyOCR on Cropped Name Region (top 5-25%)

Cropping to just the top portion of the card (where the name lives) produces
cleaner results when the name is visible:

- "Mr. Mime" detected at 0.88 confidence from 5-25% crop
- "Natu" at 0.74-0.90 from 5-25% crop
- "Rattata" at 1.00 from 5-25% crop
- "Xatu" at 0.26-1.00 (varies; "STAGE" text nearby sometimes wins)

The 5-25% vertical range is the sweet spot for name extraction on well-framed
cards (skips binder sleeve edge at top, stops before attack text).

### Tesseract Comparison

Tesseract performed poorly on Pokemon card images:

| Method | Result on Mr. Mime card |
|--------|------------------------|
| EasyOCR full | "Mr. Mime" (0.88) + "Energy Barrier" (0.999) + 6 more |
| Tesseract full | "Cua", "Energy Barrier", garbage text |
| Tesseract top 30% crop | (empty - nothing detected) |
| Tesseract PSM 6 (block) | (empty) |
| Tesseract PSM 7 (line) | (empty) |
| Tesseract PSM 11 (sparse) | "1. ee", "Ge vine" |
| Tesseract PSM 13 (raw) | "=\"" |

Tesseract completely fails to read the card name in any configuration tested.
It occasionally picks up high-contrast attack text ("Energy Barrier") but
cannot handle the varied backgrounds, fonts, and image quality of card photos.

**Verdict: EasyOCR is dramatically better than Tesseract for this use case.**

## Performance

### Speed (CPU, Intel, no GPU)

| Input | Time |
|-------|------|
| Model load (first call) | 1.56s |
| Full card (630x880) | 1.31s avg (std 0.02s) |
| Top crop (630x176) | 0.19s avg (std 0.01s) |
| 2x upscaled crop (1260x352) | 0.89s |
| Tesseract full card | 0.16s avg |

EasyOCR is ~8x slower than Tesseract on full images, but the crop-only strategy
brings it down to 0.19s which is comparable. For a pipeline that already runs
DINOv2 and CLIP, 0.19s of OCR is negligible.

### Throughput Estimate

At 0.19s per crop, a 9-card binder page takes ~1.7s for OCR. Combined with
existing segmentation (~0.5s) and FAISS lookup, total pipeline stays under 5s
per binder page.

## Preprocessing Effects

| Preprocessing | Effect on Name Detection |
|---------------|------------------------|
| Raw color crop | Best results (0.88 conf for "Mr. Mime") |
| Grayscale only | Same results, no improvement |
| Adaptive threshold | Same or slightly worse (threshold artifacts) |
| 2x upscale | No improvement, 4.5x slower |
| Allowlist (letters only) | Not tested due to tool limits, but supported |

**Recommendation: No preprocessing needed.** EasyOCR's built-in CRAFT text
detector handles varying backgrounds well. Grayscale conversion adds no value;
thresholding can hurt by introducing artifacts on gradient backgrounds.

## Name Extraction Strategy

The challenge is not reading the text but identifying which detected text is
the card name. A heuristic approach:

1. Run EasyOCR on the full card image (gets all text)
2. Filter detections to top 40% of image (y_center < 0.4 * height)
3. Filter for confidence > 0.2 and text length > 1
4. Score by: `(1 - y_position) * confidence * (bbox_height / image_height)`
5. Take the highest-scoring detection as the card name

This works well when the name is visible but fails when the segmenter crops
the name off (cards 06-08 in our test). The fix is better segmentation, not
better OCR.

Alternative strategy: crop to top 5-25% and take the highest-confidence result.
Simpler, faster (0.19s vs 1.31s), but requires consistent card framing.

## Integration Opportunities

### As a CLIP/FAISS Supplement

OCR could supplement the existing recognition cascade:

```
1. Hash lookup (exact match, <1ms)
2. DINOv2+FAISS (visual similarity, ~50ms)
3. EasyOCR name extraction (~200ms)  <-- NEW
4. CLIP text matching using OCR'd name
5. Claude Code subagent (fallback)
```

The OCR name could be used to:
- Filter FAISS candidates (must contain the OCR'd name)
- Query the card database directly by name
- Disambiguate between visually similar cards (e.g., same Pokemon, different set)

### For Set/Number Identification

EasyOCR also reads the set number (bottom of card), which combined with the
name would uniquely identify most cards. This was not tested but the full-card
OCR did pick up numbers like "20" and partial set codes.

## Known Limitations

1. **Cut-off names**: If the card segmenter crops the top of the card, the name
   is simply not in the image. OCR cannot read what isn't there.
2. **Holographic/textured cards**: Holo patterns over the name area may reduce
   confidence. Not tested (all test cards were non-holo).
3. **Special characters**: Characters like delta (delta species), star, and
   non-ASCII in card names were not tested. EasyOCR supports Unicode but may
   need the character in its training data.
4. **Dark cards**: Cards with dark backgrounds and light text (e.g., some
   full-art cards) were not tested. May need inverted preprocessing.
5. **Rotated/skewed cards**: EasyOCR handles moderate rotation but heavily
   skewed cards may need deskewing first.

## Conclusion

EasyOCR is a strong fit for Pokemon card name extraction:

- Already installed, no new dependencies
- 0.88+ confidence on card names when visible
- 0.19s per card with cropping (fast enough for real-time)
- Dramatically outperforms Tesseract on card imagery
- No preprocessing required

The main blocker is not OCR quality but segmentation quality -- ensuring the
card name region is captured in the extracted card image. If the name is
visible, EasyOCR reads it reliably.

**Recommendation**: Integrate as step 3 in the recognition cascade, between
DINOv2+FAISS and the Claude Code subagent. Use the top-crop strategy (5-25%)
for speed, falling back to full-card OCR if the crop returns nothing.
