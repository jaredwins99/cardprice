#!/usr/bin/env python3
"""Test metallic text OCR - V6: Try BOTH rotations for each card."""

import cv2
import numpy as np
import easyocr
import os
import sys

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_v6"
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
    print(msg); sys.stdout.flush()

p("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
p("EasyOCR loaded.")


def crop_name_region(img):
    """Crop name region from correctly oriented card (landscape ~880x630)."""
    h, w = img.shape[:2]
    crops = {}
    crops["name_wide"] = img[int(h*0.01):int(h*0.14), int(w*0.02):int(w*0.75)]
    crops["name_std"] = img[int(h*0.01):int(h*0.12), int(w*0.02):int(w*0.55)]
    return crops


def strict_match(ocr_results, expected_name):
    expected = expected_name.lower()
    for box, text, conf in ocr_results:
        t = text.lower().strip()
        if len(t) < 3:
            continue
        if len(t) >= 4 and (expected in t or (t in expected and len(t) >= 4)):
            return True, text, conf
        if len(t) >= 4 and len(expected) >= 4 and expected[:4] == t[:4]:
            return True, text, conf
        if len(t) >= 4 and len(expected) >= 5:
            common = sum(1 for a, b in zip(expected, t) if a == b)
            if common >= len(expected) * 0.6:
                return True, text, conf
    return False, "", 0.0


def preprocess_variants(crop, scale=3):
    """Key preprocessing variants."""
    scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    variants = {}
    variants["raw"] = scaled
    variants["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for clip in [4.0, 8.0, 16.0]:
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(2, 2)).apply(gray)
        variants[f"clahe{int(clip)}"] = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    # Bilateral + CLAHE
    bil = cv2.bilateralFilter(scaled, 9, 75, 75)
    bil_gray = cv2.cvtColor(bil, cv2.COLOR_BGR2GRAY)
    bil_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(bil_gray)
    variants["bil_clahe"] = cv2.cvtColor(bil_c, cv2.COLOR_GRAY2BGR)
    # Negative + CLAHE
    neg = 255 - gray
    neg_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(neg)
    variants["neg_clahe"] = cv2.cvtColor(neg_c, cv2.COLOR_GRAY2BGR)
    # Adaptive threshold
    at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 3)
    variants["adapt21"] = cv2.cvtColor(at, cv2.COLOR_GRAY2BGR)
    # Inv saturation
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    inv_sat = 255 - hsv[:, :, 1]
    inv_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(inv_sat)
    variants["inv_sat"] = cv2.cvtColor(inv_c, cv2.COLOR_GRAY2BGR)
    # Color CLAHE
    b, g, r = cv2.split(scaled)
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    variants["color_clahe"] = cv2.merge([clahe4.apply(b), clahe4.apply(g), clahe4.apply(r)])
    # Bilateral + sharpen
    blurred = cv2.GaussianBlur(bil, (0, 0), 3)
    sharp = cv2.addWeighted(bil, 3.0, blurred, -2.0, 0)
    variants["bil_sharp"] = sharp
    # CLAHE + Otsu
    _, otsu = cv2.threshold(cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(gray),
                            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["clahe_otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    return variants


def main():
    card_found = {i: [] for i in range(9)}
    variant_scores = {}

    for card_idx in range(9):
        card_id, expected = GROUND_TRUTH[card_idx]
        img = cv2.imread(os.path.join(IMG_DIR, f"card_{card_idx:02d}.png"))

        p(f"\n--- card_{card_idx:02d} [{expected}] ---")

        # Try both rotations
        rotations = {
            "rot_ccw": cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            "rot_cw": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        }

        for rot_name, img_rot in rotations.items():
            cv2.imwrite(os.path.join(OUT_DIR, f"c{card_idx:02d}_{rot_name}.png"), img_rot)
            crops = crop_name_region(img_rot)

            for crop_name, crop_img in crops.items():
                for scale in [3, 4]:
                    variants = preprocess_variants(crop_img, scale)

                    for var_name, var_img in variants.items():
                        full_key = f"{rot_name}/{crop_name}/{var_name}/{scale}x"
                        rgb = cv2.cvtColor(var_img, cv2.COLOR_BGR2RGB) if len(var_img.shape) == 3 else var_img
                        results = reader.readtext(rgb, paragraph=False)
                        matched, match_text, conf = strict_match(results, expected)

                        if full_key not in variant_scores:
                            variant_scores[full_key] = 0

                        if matched:
                            variant_scores[full_key] += 1
                            card_found[card_idx].append((full_key, match_text, conf))

                    # Print baseline for name_wide/raw/3x
                    if var_name == "bil_sharp" and scale == 3 and crop_name == "name_wide":
                        # Run raw OCR for debug
                        raw_scaled = cv2.resize(crop_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        raw_results = reader.readtext(cv2.cvtColor(raw_scaled, cv2.COLOR_BGR2RGB), paragraph=False)
                        texts = [(t, round(float(c), 2)) for (_, t, c) in raw_results]
                        if texts:
                            p(f"  {rot_name}/name_wide/raw/3x: {texts}")

        if card_found[card_idx]:
            first = card_found[card_idx][0]
            p(f"  MATCHED by {len(card_found[card_idx])} variants! Best: {first[0]} -> '{first[1]}' ({float(first[2]):.2f})")
        else:
            p(f"  FAILED - no variant matched")

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
            # Show unique texts found
            seen = set()
            for v, t, c in hits:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    p(f"         '{t}' (conf={float(c):.2f}) via {v}")
                    if len(seen) >= 5:
                        break
        else:
            p(f"  card_{card_idx:02d} [{expected:12s}]: FAILED")

    p(f"\n  Recoverable: {total}/9 ({total*100//9}%)")

    # Top variants
    p(f"\n{'='*70}")
    p(f"TOP VARIANTS (by # cards matched)")
    p(f"{'='*70}")
    ranked = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)
    for name, count in ranked[:15]:
        if count > 0:
            p(f"  {count}/9  {name}")


if __name__ == "__main__":
    main()
