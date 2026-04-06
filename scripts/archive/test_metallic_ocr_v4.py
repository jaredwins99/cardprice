#!/usr/bin/env python3
"""Test metallic text OCR - V4: targeted approaches, unbuffered output."""

import cv2
import numpy as np
import easyocr
import os
import sys

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_v4"
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

def p(msg):
    print(msg)
    sys.stdout.flush()

p("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
p("EasyOCR loaded.")


def crop_name_region(img):
    """Multiple crop regions to try."""
    h, w = img.shape[:2]
    crops = {}
    # Standard: top 2-15%, left 65%
    crops["top_standard"] = img[int(h*0.02):int(h*0.15), int(w*0.05):int(w*0.65)]
    # Wider: top 2-15%, full width
    crops["top_wide"] = img[int(h*0.02):int(h*0.15), int(w*0.03):int(w*0.85)]
    # Taller: top 0-18%, left 70%
    crops["top_tall"] = img[0:int(h*0.18), int(w*0.03):int(w*0.70)]
    # Very top: just 0-10%
    crops["top_narrow"] = img[0:int(h*0.10), int(w*0.05):int(w*0.65)]
    return crops


def strict_match(ocr_results, expected_name):
    """Very strict: require 4+ char match."""
    expected = expected_name.lower()
    for text, conf in ocr_results:
        t = text.lower().strip()
        if len(t) < 3:
            continue
        if len(t) >= 4 and (expected in t or (t in expected and len(t) >= 4)):
            return True, t, conf
        if len(t) >= 4 and len(expected) >= 4 and expected[:4] == t[:4]:
            return True, t, conf
        # Positional match: at least 60% of chars match in order
        if len(t) >= 4 and len(expected) >= 5:
            common = sum(1 for a, b in zip(expected, t) if a == b)
            if common >= len(expected) * 0.6:
                return True, t, conf
    return False, "", 0.0


def run_ocr(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
    return [(text, conf) for (_, text, conf) in reader.readtext(rgb, paragraph=False)]


def preprocess(img, scale=4):
    """Generate a focused set of preprocessed variants."""
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    variants = {}

    # 1. Just grayscale upscaled
    variants["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. CLAHE at various clip limits
    for clip in [4.0, 8.0, 16.0, 32.0]:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(2, 2))
        variants[f"clahe{int(clip)}"] = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    # 3. CLAHE + Otsu
    c8 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(gray)
    _, otsu = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["clahe8_otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

    # 4. Negative + CLAHE
    neg = 255 - gray
    neg_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(neg)
    variants["neg_clahe"] = cv2.cvtColor(neg_c, cv2.COLOR_GRAY2BGR)

    # 5. Bilateral + CLAHE
    bilateral = cv2.bilateralFilter(scaled, 9, 75, 75)
    bil_gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    bil_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(bil_gray)
    variants["bil_clahe"] = cv2.cvtColor(bil_c, cv2.COLOR_GRAY2BGR)

    # 6. Adaptive threshold
    for bs in [11, 21, 31]:
        at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, 3)
        variants[f"adapt{bs}"] = cv2.cvtColor(at, cv2.COLOR_GRAY2BGR)

    # 7. Inverted saturation + CLAHE (metallic = low saturation)
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    inv_sat = 255 - hsv[:, :, 1]
    inv_sat_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(inv_sat)
    variants["inv_sat_clahe"] = cv2.cvtColor(inv_sat_c, cv2.COLOR_GRAY2BGR)

    # 8. Gamma correction (brighten dark metallic text)
    for gamma in [0.3, 0.5]:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_img = cv2.LUT(gray, table)
        variants[f"gamma{gamma}"] = cv2.cvtColor(gamma_img, cv2.COLOR_GRAY2BGR)

    # 9. Color equalization per channel
    b, g, r = cv2.split(scaled)
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    color_eq = cv2.merge([clahe4.apply(b), clahe4.apply(g), clahe4.apply(r)])
    variants["color_clahe"] = color_eq

    # 10. DoG (difference of Gaussians)
    b1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
    b2 = cv2.GaussianBlur(gray, (11, 11), 3.0)
    dog = cv2.normalize(cv2.subtract(b1, b2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    variants["dog"] = cv2.cvtColor(dog, cv2.COLOR_GRAY2BGR)

    # 11. Top-hat (bright features on dark bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    tophat_n = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    variants["tophat"] = cv2.cvtColor(tophat_n, cv2.COLOR_GRAY2BGR)

    # 12. Bilateral + sharp
    blurred = cv2.GaussianBlur(bilateral, (0, 0), 3)
    sharpened = cv2.addWeighted(bilateral, 3.0, blurred, -2.0, 0)
    variants["bil_sharp"] = sharpened

    return variants


def main():
    card_found = {i: [] for i in range(9)}
    variant_scores = {}  # variant_name -> count

    for card_idx in range(9):
        card_id, expected = GROUND_TRUTH[card_idx]
        img = cv2.imread(os.path.join(IMG_DIR, f"card_{card_idx:02d}.png"))
        p(f"\n--- card_{card_idx:02d} [{expected}] ---")

        crops = crop_name_region(img)

        for crop_name, crop_img in crops.items():
            cv2.imwrite(os.path.join(OUT_DIR, f"c{card_idx:02d}_{crop_name}_raw4x.png"),
                       cv2.resize(crop_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))

            for scale in [4, 5]:
                variants = preprocess(crop_img, scale=scale)

                for var_name, var_img in variants.items():
                    full_name = f"{crop_name}/{var_name}/{scale}x"
                    ocr = run_ocr(var_img)
                    matched, match_text, conf = strict_match(ocr, expected)

                    if full_name not in variant_scores:
                        variant_scores[full_name] = 0

                    if matched:
                        variant_scores[full_name] += 1
                        card_found[card_idx].append((full_name, match_text, conf))
                        cv2.imwrite(os.path.join(OUT_DIR, f"c{card_idx:02d}_MATCH_{crop_name}_{var_name}_{scale}x.png"), var_img)
                        p(f"  MATCH [{full_name}]: '{match_text}' (conf={conf:.2f}) all={[(t,f'{c:.2f}') for t,c in ocr]}")

        # Debug output for failed cards
        if not card_found[card_idx]:
            # Show what clahe8 saw at 4x on the top_wide crop
            crop = crops["top_wide"]
            scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            c8 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(gray)
            debug_img = cv2.cvtColor(c8, cv2.COLOR_GRAY2BGR)
            ocr = run_ocr(debug_img)
            p(f"  FAILED. Best effort (clahe8/wide/4x): {[(t,f'{c:.2f}') for t,c in ocr]}")

            # Also try the color_clahe and inv_sat on the wide crop
            hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
            inv_sat = 255 - hsv[:,:,1]
            inv_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2,2)).apply(inv_sat)
            debug2 = cv2.cvtColor(inv_c, cv2.COLOR_GRAY2BGR)
            ocr2 = run_ocr(debug2)
            p(f"  FAILED. inv_sat_clahe/wide/4x: {[(t,f'{c:.2f}') for t,c in ocr2]}")

    # Summary
    p(f"\n{'='*70}")
    p(f"PER-CARD RESULTS")
    p(f"{'='*70}")
    total = 0
    for card_idx in range(9):
        _, expected = GROUND_TRUTH[card_idx]
        hits = card_found[card_idx]
        if hits:
            total += 1
            p(f"  card_{card_idx:02d} [{expected:12s}]: FOUND ({len(hits)} variants)")
            for v, t, c in hits[:3]:
                p(f"         {v}: '{t}' (conf={c:.2f})")
        else:
            p(f"  card_{card_idx:02d} [{expected:12s}]: FAILED")

    p(f"\n  Recoverable: {total}/9")

    # Top variants
    p(f"\n{'='*70}")
    p(f"TOP VARIANTS BY CARD COUNT")
    p(f"{'='*70}")
    ranked = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)
    for name, count in ranked[:15]:
        if count > 0:
            p(f"  {count}/9  {name}")


if __name__ == "__main__":
    main()
