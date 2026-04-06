#!/usr/bin/env python3
"""Test metallic text OCR - V2 with better matching, more upscale, Tesseract, combined approaches."""

import cv2
import numpy as np
import easyocr
import subprocess
import os
import sys
from pathlib import Path

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_v2"
os.makedirs(OUT_DIR, exist_ok=True)

GROUND_TRUTH = {
    0: ("ex11-41", "Dragonair"),
    1: ("ex14-41", "Skitty"),
    2: ("ex15-68", "Trapinch"),
    3: ("ex15-24", "Vibrava"),
    4: ("ex14-91", "Delcatty"),
    5: ("ex5-101", "Wigglytuff"),
    6: ("ex14-98", "Swampert"),
    7: ("ex14-94", "Jirachi"),
    8: ("ex15-92", "Flygon"),
}

reader = easyocr.Reader(['en'], gpu=False)

# Check if tesseract is available
try:
    subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
    HAS_TESSERACT = True
    print("Tesseract available")
except:
    HAS_TESSERACT = False
    print("Tesseract NOT available")

# Try pytesseract
try:
    import pytesseract
    HAS_PYTESSERACT = True
    print("pytesseract available")
except ImportError:
    HAS_PYTESSERACT = False
    print("pytesseract NOT available")


def crop_name_region(img):
    """Crop top 18% of card image (name region, slightly tighter)."""
    h = img.shape[0]
    # Skip top 2% (card border) and take next 16%
    y_start = int(h * 0.02)
    y_end = int(h * 0.18)
    # Also trim left/right borders (5% each side)
    w = img.shape[1]
    x_start = int(w * 0.05)
    x_end = int(w * 0.65)  # Name is in left 65% of card
    return img[y_start:y_end, x_start:x_end]


def check_match(ocr_texts, expected_name):
    """Strict match: OCR text must be at least 3 chars and contain significant portion of name."""
    expected_lower = expected_name.lower()
    for text in ocr_texts:
        text_lower = text.lower().strip()
        if len(text_lower) < 3:
            continue
        # Check if significant substring match
        if expected_lower in text_lower:
            return True
        if text_lower in expected_lower and len(text_lower) >= 4:
            return True
        # First 4 chars match
        if len(text_lower) >= 4 and expected_lower[:4] == text_lower[:4]:
            return True
        # Levenshtein-like: check if >60% of expected chars present in order
        matches = 0
        pos = 0
        for c in expected_lower:
            idx = text_lower.find(c, pos)
            if idx >= 0:
                matches += 1
                pos = idx + 1
        if matches >= len(expected_lower) * 0.6 and len(text_lower) >= 4:
            return True
    return False


def run_easyocr(img_bgr):
    """Run EasyOCR, return list of (text, confidence)."""
    if len(img_bgr.shape) == 3:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_bgr
    results = reader.readtext(rgb, paragraph=False)
    return [(text, conf) for (_, text, conf) in results]


def run_tesseract(img_bgr, psm=7):
    """Run Tesseract OCR, return list of (text, confidence)."""
    if not HAS_PYTESSERACT:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    # psm 7 = single line, psm 6 = single block
    config = f'--psm {psm} --oem 3'
    text = pytesseract.image_to_string(gray, config=config).strip()
    if text:
        return [(text, 0.5)]
    return []


def preprocess_variants(img):
    """Generate many preprocessed variants of the name region."""
    variants = {}
    h, w = img.shape[:2]

    # Scale factors to try
    for scale in [3, 4]:
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # 1. Raw grayscale
        variants[f"gray_{scale}x"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. Heavy CLAHE
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(gray)
        variants[f"clahe8_{scale}x"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # 3. Super heavy CLAHE
        clahe16 = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(2, 2))
        enhanced16 = clahe16.apply(gray)
        variants[f"clahe16_{scale}x"] = cv2.cvtColor(enhanced16, cv2.COLOR_GRAY2BGR)

        # 4. CLAHE + Otsu threshold
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"clahe_otsu_{scale}x"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

        # 5. Negative + CLAHE
        neg = 255 - gray
        neg_clahe = clahe.apply(neg)
        variants[f"neg_clahe_{scale}x"] = cv2.cvtColor(neg_clahe, cv2.COLOR_GRAY2BGR)

        # 6. Adaptive threshold various block sizes
        for bs in [15, 25, 41]:
            at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, bs, 5)
            variants[f"adapt_g{bs}_{scale}x"] = cv2.cvtColor(at, cv2.COLOR_GRAY2BGR)

        # 7. Bilateral + CLAHE
        bilateral = cv2.bilateralFilter(scaled, 9, 75, 75)
        bil_gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        bil_clahe = clahe.apply(bil_gray)
        variants[f"bil_clahe_{scale}x"] = cv2.cvtColor(bil_clahe, cv2.COLOR_GRAY2BGR)

        # 8. Bilateral + unsharp mask
        blurred = cv2.GaussianBlur(bilateral, (0, 0), 3)
        sharpened = cv2.addWeighted(bilateral, 3.0, blurred, -2.0, 0)
        variants[f"bil_sharp_{scale}x"] = sharpened

        # 9. Morphological gradient (highlights text edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        variants[f"morph_grad_{scale}x"] = cv2.cvtColor(morph_grad, cv2.COLOR_GRAY2BGR)

        # 10. Morph gradient + threshold
        _, mg_thresh = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"morph_otsu_{scale}x"] = cv2.cvtColor(mg_thresh, cv2.COLOR_GRAY2BGR)

        # 11. Black hat (reveals dark text on light background)
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
        # Threshold the blackhat
        _, bh_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"blackhat_{scale}x"] = cv2.cvtColor(bh_thresh, cv2.COLOR_GRAY2BGR)

        # 12. White hat (tophat - reveals light text on dark background)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_bh)
        _, th_thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"tophat_{scale}x"] = cv2.cvtColor(th_thresh, cv2.COLOR_GRAY2BGR)

        # 13. Channel isolation: try each channel with CLAHE
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        for ch_name, ch_img in [("sat", hsv[:,:,1]), ("val", hsv[:,:,2]),
                                 ("lab_L", lab[:,:,0]), ("lab_a", lab[:,:,1])]:
            ch_clahe = clahe.apply(ch_img)
            variants[f"{ch_name}_clahe_{scale}x"] = cv2.cvtColor(ch_clahe, cv2.COLOR_GRAY2BGR)

        # 14. Inverted saturation + CLAHE (metallic = low saturation)
        inv_sat = 255 - hsv[:,:,1]
        inv_sat_clahe = clahe.apply(inv_sat)
        variants[f"inv_sat_clahe_{scale}x"] = cv2.cvtColor(inv_sat_clahe, cv2.COLOR_GRAY2BGR)

        # 15. Difference of Gaussians (band-pass filter for text-sized features)
        blur1 = cv2.GaussianBlur(gray, (3, 3), 1)
        blur2 = cv2.GaussianBlur(gray, (15, 15), 5)
        dog = cv2.subtract(blur1, blur2)
        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        variants[f"dog_{scale}x"] = cv2.cvtColor(dog_norm, cv2.COLOR_GRAY2BGR)

        # 16. DoG + threshold
        _, dog_thresh = cv2.threshold(dog_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"dog_otsu_{scale}x"] = cv2.cvtColor(dog_thresh, cv2.COLOR_GRAY2BGR)

        # 17. Combine: bilateral + CLAHE16 + adaptive threshold
        bil_clahe16 = clahe16.apply(bil_gray)
        bil_adapt = cv2.adaptiveThreshold(bil_clahe16, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 25, 3)
        variants[f"combo_bil_clahe16_adapt_{scale}x"] = cv2.cvtColor(bil_adapt, cv2.COLOR_GRAY2BGR)

        # 18. CLAHE + morphological closing to connect text strokes
        clahe_closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        variants[f"clahe_close_{scale}x"] = cv2.cvtColor(clahe_closed, cv2.COLOR_GRAY2BGR)

    return variants


def main():
    results_summary = {}  # variant_name -> list of (card_idx, matched, texts)
    card_found_by = {i: [] for i in range(9)}  # card_idx -> list of method names that found it

    for card_idx in range(9):
        card_id, expected_name = GROUND_TRUTH[card_idx]
        img_path = os.path.join(IMG_DIR, f"card_{card_idx:02d}.png")
        img = cv2.imread(img_path)

        # Crop name region
        name_crop = crop_name_region(img)

        # Save original crop for reference
        cv2.imwrite(os.path.join(OUT_DIR, f"card{card_idx:02d}_original_crop.png"), name_crop)

        # Generate all variants
        variants = preprocess_variants(name_crop)

        print(f"\ncard_{card_idx:02d} [{expected_name}] ({card_id}):")
        print(f"  Name crop size: {name_crop.shape}")

        for var_name, var_img in variants.items():
            # Save debug image
            debug_path = os.path.join(OUT_DIR, f"card{card_idx:02d}_{var_name}.png")
            cv2.imwrite(debug_path, var_img)

            # Run EasyOCR
            ocr_results = run_easyocr(var_img)
            texts = [t for t, c in ocr_results]
            matched = check_match(texts, expected_name)

            # Run Tesseract too
            tess_results = run_tesseract(var_img, psm=7)
            tess_texts = [t for t, c in tess_results]
            tess_matched = check_match(tess_texts, expected_name)

            # Also try Tesseract psm=6
            tess6_results = run_tesseract(var_img, psm=6)
            tess6_texts = [t for t, c in tess6_results]
            tess6_matched = check_match(tess6_texts, expected_name)

            any_matched = matched or tess_matched or tess6_matched

            if var_name not in results_summary:
                results_summary[var_name] = []

            if any_matched:
                card_found_by[card_idx].append(var_name)
                engine = []
                if matched:
                    engine.append(f"easyocr:{texts}")
                if tess_matched:
                    engine.append(f"tess7:{tess_texts}")
                if tess6_matched:
                    engine.append(f"tess6:{tess6_texts}")
                print(f"  MATCH {var_name}: {', '.join(engine)}")

            results_summary[var_name].append((card_idx, any_matched, texts + tess_texts + tess6_texts))

    # Print summary
    print(f"\n{'='*70}")
    print(f"VARIANT ACCURACY RANKING (of 9 cards)")
    print(f"{'='*70}")

    variant_scores = []
    for var_name, card_results in results_summary.items():
        matches = sum(1 for _, m, _ in card_results if m)
        if matches > 0:
            variant_scores.append((matches, var_name))

    variant_scores.sort(reverse=True)
    for score, name in variant_scores[:30]:
        print(f"  {score}/9  {name}")

    if not variant_scores:
        print("  No variants found any cards!")

    print(f"\n{'='*70}")
    print(f"PER-CARD RECOVERY")
    print(f"{'='*70}")
    total = 0
    for card_idx in range(9):
        _, expected_name = GROUND_TRUTH[card_idx]
        methods = card_found_by[card_idx]
        if methods:
            total += 1
            print(f"  card_{card_idx:02d} [{expected_name:12s}]: FOUND by {len(methods)} methods")
            for m in methods[:3]:
                print(f"    - {m}")
        else:
            print(f"  card_{card_idx:02d} [{expected_name:12s}]: NOT FOUND")

    print(f"\nTotal recoverable: {total}/9")


if __name__ == "__main__":
    main()
