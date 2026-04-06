#!/usr/bin/env python3
"""Test metallic text OCR - V3: focused, EasyOCR only, strict matching, visual debug."""

import cv2
import numpy as np
import easyocr
import os

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_v3"
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


def crop_name_region(img):
    """Crop the name region. Cards are 880x630."""
    h, w = img.shape[:2]
    # Name region: top ~12-18% of card, left 65%
    y1 = int(h * 0.02)
    y2 = int(h * 0.15)
    x1 = int(w * 0.05)
    x2 = int(w * 0.65)
    return img[y1:y2, x1:x2]


def strict_match(ocr_texts, expected_name):
    """Strict match requiring at least 4 chars and significant overlap."""
    expected = expected_name.lower()
    for text in ocr_texts:
        t = text.lower().strip()
        if len(t) < 3:
            continue
        # Direct containment (either direction, but text must be 4+ chars)
        if len(t) >= 4 and (expected in t or t in expected):
            return True, t
        # First 4 chars match
        if len(t) >= 4 and len(expected) >= 4 and expected[:4] == t[:4]:
            return True, t
        # Edit distance based: for names >= 5 chars, allow 1-2 edits
        if len(expected) >= 5 and len(t) >= 4:
            # Simple: count matching chars in sequence
            common = sum(1 for a, b in zip(expected, t) if a == b)
            if common >= len(expected) * 0.6 and common >= 3:
                return True, t
    return False, ""


def run_ocr(img_bgr, allow_list=None):
    """Run EasyOCR."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
    kwargs = {"paragraph": False}
    if allow_list:
        kwargs["allowlist"] = allow_list
    results = reader.readtext(rgb, **kwargs)
    return [(text, conf) for (_, text, conf) in results]


def generate_preprocessed(name_crop):
    """Generate preprocessed variants. Returns dict of name -> BGR image."""
    variants = {}

    for scale in [3, 4, 5]:
        scaled = cv2.resize(name_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # === Group 1: Contrast Enhancement ===

        # CLAHE variants
        for clip in [4.0, 8.0, 16.0, 32.0]:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(2, 2))
            enhanced = clahe.apply(gray)
            variants[f"clahe{int(clip)}_{scale}x"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # CLAHE + Otsu
        clahe8 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        enhanced8 = clahe8.apply(gray)
        _, otsu = cv2.threshold(enhanced8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"clahe8_otsu_{scale}x"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

        # === Group 2: Negative / Inversion ===
        neg = 255 - gray
        clahe_neg = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        neg_enhanced = clahe_neg.apply(neg)
        variants[f"neg_clahe_{scale}x"] = cv2.cvtColor(neg_enhanced, cv2.COLOR_GRAY2BGR)

        # === Group 3: Bilateral + enhancement ===
        bilateral = cv2.bilateralFilter(scaled, 9, 75, 75)
        bil_gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        bil_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(bil_gray)
        variants[f"bil_clahe_{scale}x"] = cv2.cvtColor(bil_clahe, cv2.COLOR_GRAY2BGR)

        # Bilateral + sharp + CLAHE
        blurred = cv2.GaussianBlur(bilateral, (0, 0), 3)
        sharpened = cv2.addWeighted(bilateral, 3.0, blurred, -2.0, 0)
        sharp_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        sharp_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(sharp_gray)
        variants[f"bil_sharp_clahe_{scale}x"] = cv2.cvtColor(sharp_clahe, cv2.COLOR_GRAY2BGR)

        # === Group 4: Adaptive threshold ===
        for bs in [11, 21, 31, 51]:
            at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, bs, 3)
            variants[f"adapt{bs}_{scale}x"] = cv2.cvtColor(at, cv2.COLOR_GRAY2BGR)

        # Adaptive on CLAHE-enhanced
        for bs in [21, 31]:
            at_clahe = cv2.adaptiveThreshold(enhanced8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, bs, 3)
            variants[f"clahe_adapt{bs}_{scale}x"] = cv2.cvtColor(at_clahe, cv2.COLOR_GRAY2BGR)

        # === Group 5: Color channel tricks ===
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Saturation channel (metallic = low sat, card art = high sat)
        sat = hsv[:, :, 1]
        # Invert: metallic text becomes bright
        inv_sat = 255 - sat
        inv_sat_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(inv_sat)
        variants[f"inv_sat_clahe_{scale}x"] = cv2.cvtColor(inv_sat_clahe, cv2.COLOR_GRAY2BGR)

        # Threshold on inverted saturation
        _, inv_sat_thresh = cv2.threshold(inv_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants[f"inv_sat_otsu_{scale}x"] = cv2.cvtColor(inv_sat_thresh, cv2.COLOR_GRAY2BGR)

        # LAB L channel + CLAHE
        lab_L = lab[:, :, 0]
        lab_L_clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(2, 2)).apply(lab_L)
        variants[f"lab_L_clahe16_{scale}x"] = cv2.cvtColor(lab_L_clahe, cv2.COLOR_GRAY2BGR)

        # === Group 6: Difference of Gaussians ===
        blur1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        blur2 = cv2.GaussianBlur(gray, (11, 11), 3.0)
        dog = cv2.subtract(blur1, blur2)
        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        variants[f"dog_{scale}x"] = cv2.cvtColor(dog_norm, cv2.COLOR_GRAY2BGR)

        # DoG + CLAHE
        dog_clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(dog_norm)
        variants[f"dog_clahe_{scale}x"] = cv2.cvtColor(dog_clahe, cv2.COLOR_GRAY2BGR)

        # === Group 7: Morphological ===
        # Top-hat for bright text on darker background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        variants[f"tophat_{scale}x"] = cv2.cvtColor(tophat_norm, cv2.COLOR_GRAY2BGR)

        # Black-hat for dark text on lighter background
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        variants[f"blackhat_{scale}x"] = cv2.cvtColor(blackhat_norm, cv2.COLOR_GRAY2BGR)

        # === Group 8: Combined best approaches ===
        # Bilateral + CLAHE32 + adaptive
        bil_clahe32 = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(2, 2)).apply(bil_gray)
        variants[f"bil_clahe32_{scale}x"] = cv2.cvtColor(bil_clahe32, cv2.COLOR_GRAY2BGR)

        # CLAHE on each RGB channel separately, then merge
        b, g, r = cv2.split(scaled)
        clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
        b_c, g_c, r_c = clahe4.apply(b), clahe4.apply(g), clahe4.apply(r)
        color_clahe = cv2.merge([b_c, g_c, r_c])
        variants[f"color_clahe_{scale}x"] = color_clahe

        # Gamma correction (bright)
        for gamma in [0.3, 0.5, 2.0, 3.0]:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
            gamma_img = cv2.LUT(gray, table)
            variants[f"gamma{gamma}_{scale}x"] = cv2.cvtColor(gamma_img, cv2.COLOR_GRAY2BGR)

    return variants


def main():
    # Track results per variant and per card
    variant_hits = {}  # variant_name -> set of card indices matched
    card_best = {}     # card_idx -> list of (variant, text, conf)

    for card_idx in range(9):
        card_id, expected_name = GROUND_TRUTH[card_idx]
        img = cv2.imread(os.path.join(IMG_DIR, f"card_{card_idx:02d}.png"))
        name_crop = crop_name_region(img)

        card_best[card_idx] = []

        # Save raw crop
        cv2.imwrite(os.path.join(OUT_DIR, f"card{card_idx:02d}_raw_crop.png"),
                     cv2.resize(name_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))

        variants = generate_preprocessed(name_crop)
        print(f"\ncard_{card_idx:02d} [{expected_name}] - testing {len(variants)} variants...")

        for var_name, var_img in variants.items():
            ocr_results = run_ocr(var_img)
            texts = [t for t, c in ocr_results]
            matched, match_text = strict_match(texts, expected_name)

            if var_name not in variant_hits:
                variant_hits[var_name] = set()

            if matched:
                variant_hits[var_name].add(card_idx)
                card_best[card_idx].append((var_name, match_text, texts))

            # Save debug image only for interesting results
            if matched or "clahe8_4x" in var_name or "bil_clahe_4x" in var_name:
                cv2.imwrite(os.path.join(OUT_DIR, f"card{card_idx:02d}_{var_name}.png"), var_img)

        # Report
        if card_best[card_idx]:
            first = card_best[card_idx][0]
            print(f"  FOUND by {len(card_best[card_idx])} variants! First: {first[0]} -> '{first[1]}' (all: {first[2]})")
        else:
            # Show what the best CLAHE variant found (for debugging)
            best_key = f"clahe8_4x"
            if best_key in variants:
                ocr = run_ocr(variants[best_key])
                print(f"  NOT FOUND. clahe8_4x saw: {[(t,f'{c:.2f}') for t,c in ocr]}")
            best_key2 = f"bil_clahe_4x"
            if best_key2 in variants:
                ocr = run_ocr(variants[best_key2])
                print(f"  NOT FOUND. bil_clahe_4x saw: {[(t,f'{c:.2f}') for t,c in ocr]}")
            best_key3 = f"clahe32_4x"
            if best_key3 in variants:
                ocr = run_ocr(variants[best_key3])
                print(f"  NOT FOUND. clahe32_4x saw: {[(t,f'{c:.2f}') for t,c in ocr]}")
            best_key4 = f"neg_clahe_4x"
            if best_key4 in variants:
                ocr = run_ocr(variants[best_key4])
                print(f"  NOT FOUND. neg_clahe_4x saw: {[(t,f'{c:.2f}') for t,c in ocr]}")
            best_key5 = f"inv_sat_clahe_4x"
            if best_key5 in variants:
                ocr = run_ocr(variants[best_key5])
                print(f"  NOT FOUND. inv_sat_clahe_4x saw: {[(t,f'{c:.2f}') for t,c in ocr]}")

    # Summary
    print(f"\n{'='*70}")
    print(f"TOP PREPROCESSING VARIANTS (by # cards matched out of 9)")
    print(f"{'='*70}")

    ranked = sorted(variant_hits.items(), key=lambda x: len(x[1]), reverse=True)
    for name, cards in ranked[:20]:
        if len(cards) > 0:
            print(f"  {len(cards)}/9  {name}  (cards: {sorted(cards)})")

    print(f"\n{'='*70}")
    print(f"PER-CARD SUMMARY")
    print(f"{'='*70}")
    total = 0
    for card_idx in range(9):
        _, expected_name = GROUND_TRUTH[card_idx]
        hits = card_best[card_idx]
        if hits:
            total += 1
            print(f"  card_{card_idx:02d} [{expected_name:12s}]: FOUND ({len(hits)} variants)")
            for v, t, _ in hits[:3]:
                print(f"         -> {v}: '{t}'")
        else:
            print(f"  card_{card_idx:02d} [{expected_name:12s}]: FAILED")

    print(f"\n  Total: {total}/9 cards recoverable by at least one preprocessing variant")

    # Union of all cards found
    all_found = set()
    for cards in variant_hits.values():
        all_found.update(cards)
    print(f"  Union across all methods: {len(all_found)}/9")
    if all_found != set(range(9)):
        missing = set(range(9)) - all_found
        for idx in sorted(missing):
            print(f"    Missing: card_{idx:02d} ({GROUND_TRUTH[idx][1]})")


if __name__ == "__main__":
    main()
