#!/usr/bin/env python3
"""Test metallic text OCR - V5: CORRECT ROTATION + preprocessing."""

import cv2
import numpy as np
import easyocr
import os
import sys

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_v5"
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

p("Loading EasyOCR (en + ja)...")
reader = easyocr.Reader(['en'], gpu=False)
p("EasyOCR loaded.")


def crop_name_region(img_rotated):
    """Crop the name region from a CORRECTLY ORIENTED card (landscape, 880x630)."""
    h, w = img_rotated.shape[:2]  # ~630 x 880
    crops = {}
    # Standard name region: top 12%, left 55%
    crops["name_std"] = img_rotated[int(h*0.01):int(h*0.12), int(w*0.02):int(w*0.55)]
    # Wider name region (include HP area)
    crops["name_wide"] = img_rotated[int(h*0.01):int(h*0.14), int(w*0.02):int(w*0.75)]
    # Tighter top strip
    crops["name_tight"] = img_rotated[int(h*0.02):int(h*0.10), int(w*0.03):int(w*0.50)]
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


def preprocess(img, scale=3):
    """Focused preprocessing variants."""
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    variants = {}

    # Raw
    variants["raw"] = scaled

    # Grayscale
    variants["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # CLAHE
    for clip in [4.0, 8.0, 16.0]:
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(2, 2)).apply(gray)
        variants[f"clahe{int(clip)}"] = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)

    # CLAHE + Otsu
    c8 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(gray)
    _, otsu = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["clahe8_otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

    # Negative + CLAHE
    neg = 255 - gray
    neg_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(neg)
    variants["neg_clahe"] = cv2.cvtColor(neg_c, cv2.COLOR_GRAY2BGR)

    # Bilateral + CLAHE
    bilateral = cv2.bilateralFilter(scaled, 9, 75, 75)
    bil_gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    bil_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(bil_gray)
    variants["bil_clahe"] = cv2.cvtColor(bil_c, cv2.COLOR_GRAY2BGR)

    # Adaptive threshold
    for bs in [21, 31]:
        at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, 3)
        variants[f"adapt{bs}"] = cv2.cvtColor(at, cv2.COLOR_GRAY2BGR)

    # Inverted saturation + CLAHE
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    inv_sat = 255 - hsv[:, :, 1]
    inv_sat_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(inv_sat)
    variants["inv_sat_clahe"] = cv2.cvtColor(inv_sat_c, cv2.COLOR_GRAY2BGR)

    # Gamma 0.5 (brighten)
    table = np.array([((i / 255.0) ** 0.5) * 255 for i in range(256)]).astype("uint8")
    gamma_img = cv2.LUT(gray, table)
    variants["gamma05"] = cv2.cvtColor(gamma_img, cv2.COLOR_GRAY2BGR)

    # Color CLAHE (per channel)
    b, g, r = cv2.split(scaled)
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    variants["color_clahe"] = cv2.merge([clahe4.apply(b), clahe4.apply(g), clahe4.apply(r)])

    # Bilateral + sharpen
    blurred = cv2.GaussianBlur(bilateral, (0, 0), 3)
    sharp = cv2.addWeighted(bilateral, 3.0, blurred, -2.0, 0)
    variants["bil_sharp"] = sharp

    return variants


def main():
    card_found = {i: [] for i in range(9)}
    variant_scores = {}

    for card_idx in range(9):
        card_id, expected = GROUND_TRUTH[card_idx]
        img = cv2.imread(os.path.join(IMG_DIR, f"card_{card_idx:02d}.png"))

        # ROTATE to correct orientation!
        img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(OUT_DIR, f"c{card_idx:02d}_corrected.png"), img_rot)

        p(f"\n--- card_{card_idx:02d} [{expected}] (rotated: {img_rot.shape}) ---")

        crops = crop_name_region(img_rot)

        for crop_name, crop_img in crops.items():
            for scale in [3, 4]:
                variants = preprocess(crop_img, scale=scale)

                for var_name, var_img in variants.items():
                    full_key = f"{crop_name}/{var_name}/{scale}x"

                    results = reader.readtext(
                        cv2.cvtColor(var_img, cv2.COLOR_BGR2RGB) if len(var_img.shape) == 3 else var_img,
                        paragraph=False
                    )

                    matched, match_text, conf = strict_match(results, expected)

                    if full_key not in variant_scores:
                        variant_scores[full_key] = 0

                    if matched:
                        variant_scores[full_key] += 1
                        card_found[card_idx].append((full_key, match_text, conf))
                        cv2.imwrite(os.path.join(OUT_DIR, f"c{card_idx:02d}_MATCH_{crop_name}_{var_name}_{scale}x.png"), var_img)

                    # Print all OCR output for the "raw/3x" baseline
                    if var_name == "raw" and scale == 3 and crop_name == "name_wide":
                        texts = [(t, round(float(c), 2)) for (_, t, c) in results]
                        p(f"  baseline (name_wide/raw/3x): {texts}")

        if card_found[card_idx]:
            first = card_found[card_idx][0]
            p(f"  MATCHED by {len(card_found[card_idx])} variants! First: {first[0]} -> '{first[1]}' ({first[2]:.2f})")
        else:
            # Show debug for all wide crops
            crop = crops["name_wide"]
            for scale in [3, 4]:
                for var_name in ["clahe8", "bil_clahe", "neg_clahe"]:
                    scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                    if var_name == "clahe8":
                        proc = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(gray)
                    elif var_name == "bil_clahe":
                        bil = cv2.bilateralFilter(scaled, 9, 75, 75)
                        proc = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(
                            cv2.cvtColor(bil, cv2.COLOR_BGR2GRAY))
                    else:
                        proc = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(255 - gray)
                    proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                    results = reader.readtext(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB), paragraph=False)
                    texts = [(t, round(float(c), 2)) for (_, t, c) in results]
                    if texts:
                        p(f"  debug ({var_name}/{scale}x): {texts}")

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
            for v, t, c in hits[:5]:
                p(f"         {v}: '{t}' (conf={float(c):.2f})")
        else:
            p(f"  card_{card_idx:02d} [{expected:12s}]: FAILED")

    p(f"\n  Recoverable: {total}/9")

    p(f"\n{'='*70}")
    p(f"TOP VARIANTS")
    p(f"{'='*70}")
    ranked = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)
    for name, count in ranked[:20]:
        if count > 0:
            p(f"  {count}/9  {name}")


if __name__ == "__main__":
    main()
