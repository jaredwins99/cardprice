#!/usr/bin/env python3
"""Final metallic OCR test: correct rotation + wider crop search."""

import cv2
import numpy as np
import easyocr
import os
import sys

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_final"
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


def main():
    card_found = {i: [] for i in range(9)}

    for card_idx in range(9):
        card_id, expected = GROUND_TRUTH[card_idx]
        img = cv2.imread(os.path.join(IMG_DIR, f"card_{card_idx:02d}.png"))

        p(f"\n--- card_{card_idx:02d} [{expected}] ---")

        # Try both rotations
        for rot_name, rot_code in [("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
                                    ("cw", cv2.ROTATE_90_CLOCKWISE)]:
            rot = cv2.rotate(img, rot_code)
            h, w = rot.shape[:2]

            # Multiple Y-offset crops to handle varying border thickness
            # The name appears somewhere between Y=5% and Y=18%
            crop_configs = [
                # (y_start%, y_end%, x_start%, x_end%, name)
                (5, 18, 2, 70, "y5_18_wide"),
                (8, 20, 2, 70, "y8_20_wide"),
                (5, 15, 2, 55, "y5_15_std"),
                (8, 18, 2, 55, "y8_18_std"),
                (10, 22, 2, 70, "y10_22_wide"),
                (12, 24, 2, 70, "y12_24_wide"),
                (5, 22, 2, 80, "y5_22_full"),
            ]

            for y1p, y2p, x1p, x2p, crop_name in crop_configs:
                y1, y2 = int(h * y1p / 100), int(h * y2p / 100)
                x1, x2 = int(w * x1p / 100), int(w * x2p / 100)
                crop = rot[y1:y2, x1:x2]

                if crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                # Preprocessing variants
                for scale in [3, 4]:
                    scaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

                    preps = {
                        "raw": scaled,
                        "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                    }

                    # CLAHE
                    for clip in [4.0, 8.0, 16.0]:
                        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(2, 2)).apply(gray)
                        preps[f"clahe{int(clip)}"] = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)

                    # Bilateral + CLAHE
                    bil = cv2.bilateralFilter(scaled, 9, 75, 75)
                    bil_gray = cv2.cvtColor(bil, cv2.COLOR_BGR2GRAY)
                    bil_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(bil_gray)
                    preps["bil_clahe"] = cv2.cvtColor(bil_c, cv2.COLOR_GRAY2BGR)

                    # Negative + CLAHE
                    neg_c = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(255 - gray)
                    preps["neg_clahe"] = cv2.cvtColor(neg_c, cv2.COLOR_GRAY2BGR)

                    # Inv saturation
                    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
                    inv_sat = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2)).apply(255 - hsv[:,:,1])
                    preps["inv_sat"] = cv2.cvtColor(inv_sat, cv2.COLOR_GRAY2BGR)

                    # Color CLAHE
                    b, g, r = cv2.split(scaled)
                    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
                    preps["color_clahe"] = cv2.merge([clahe4.apply(b), clahe4.apply(g), clahe4.apply(r)])

                    for prep_name, prep_img in preps.items():
                        full_key = f"{rot_name}/{crop_name}/{prep_name}/{scale}x"
                        rgb = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)
                        results = reader.readtext(rgb, paragraph=False)
                        matched, match_text, conf = strict_match(results, expected)

                        if matched:
                            card_found[card_idx].append((full_key, match_text, conf, results))
                            # Save the first match image
                            if len(card_found[card_idx]) == 1:
                                cv2.imwrite(os.path.join(OUT_DIR, f"c{card_idx:02d}_FIRST_MATCH.png"), prep_img)

                        # Debug: print raw text for the raw/3x variant
                        if prep_name == "raw" and scale == 3 and "wide" in crop_name:
                            texts = [(t, round(float(c), 2)) for (_, t, c) in results]
                            if texts and any(len(t) >= 3 for t, c in texts):
                                p(f"  {rot_name}/{crop_name}/raw/3x: {texts}")

        if card_found[card_idx]:
            first = card_found[card_idx][0]
            p(f"  >>> MATCHED by {len(card_found[card_idx])} configs! Best: '{first[1]}' ({float(first[2]):.2f}) via {first[0]}")
        else:
            p(f"  >>> FAILED")

    # Summary
    p(f"\n{'='*70}")
    p(f"FINAL RESULTS")
    p(f"{'='*70}")
    total = 0
    for card_idx in range(9):
        _, expected = GROUND_TRUTH[card_idx]
        hits = card_found[card_idx]
        if hits:
            total += 1
            # Show unique OCR texts
            seen = set()
            p(f"  card_{card_idx:02d} [{expected:12s}]: FOUND ({len(hits)} configs)")
            for key, text, conf, _ in hits:
                tl = text.lower()
                if tl not in seen:
                    seen.add(tl)
                    p(f"         '{text}' (conf={float(conf):.2f}) via {key}")
                    if len(seen) >= 5:
                        break
        else:
            p(f"  card_{card_idx:02d} [{expected:12s}]: FAILED")

    p(f"\n  Total recoverable: {total}/9 ({total*100//9}%)")
    p(f"  Page 0 accuracy IF we fix rotation + crop: {(2 + total)}/9 ({(2+total)*100//9}%)")
    p(f"  (2 cards already worked in baseline eval, these {total} are new)")


if __name__ == "__main__":
    main()
