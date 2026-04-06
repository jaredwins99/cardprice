#!/usr/bin/env python3
"""Fast stamp OCR - RapidOCR only, focused on best variants."""

import cv2
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz import fuzz

STAMP_CROPS = {
    "Chikorita (DRAGON FRONTIERS)": {
        "path": "data/inbox/page_20260305_094228_cards/card_00.png",
        "expected": "DRAGON FRONTIERS",
        "crop": (500, 680, 900, 760),
    },
    "Meganium (DRAGON FRONTIERS)": {
        "path": "data/inbox/page_20260305_094228_cards/card_02.png",
        "expected": "DRAGON FRONTIERS",
        "crop": (450, 700, 900, 790),
    },
    "Skitty (CRYSTAL GUARDIANS)": {
        "path": "data/inbox/page_20260228_174819_cards/card_01.png",
        "expected": "CRYSTAL GUARDIANS",
        "crop": (550, 580, 870, 740),
    },
    "Vibrava (DRAGON FRONTIERS)": {
        "path": "data/inbox/page_20260228_174819_cards/card_05.png",
        "expected": "DRAGON FRONTIERS",
        "crop": (440, 660, 850, 750),
    },
}

KNOWN_STAMPS = [
    "DRAGON FRONTIERS", "CRYSTAL GUARDIANS", "DELTA SPECIES",
    "POWER KEEPERS", "LEGEND MAKER", "HOLON PHANTOMS",
    "UNSEEN FORCES", "DEOXYS", "EMERALD",
    "FIRE RED LEAF GREEN", "TEAM ROCKET RETURNS",
    "HIDDEN LEGENDS", "TEAM MAGMA VS TEAM AQUA",
]

engine = RapidOCR()

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

def preprocess(crop_bgr, scale=3):
    h, w = crop_bgr.shape[:2]
    up = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray.astype(float) - p2) / max(p98 - p2, 1) * 255, 0, 255).astype(np.uint8)
    return {"color": up, "gray": gray, "enhanced": enhanced, "sharp": sharp,
            "otsu": otsu, "otsu_inv": otsu_inv, "stretched": stretched}

def ocr(img):
    result, _ = engine(img)
    if not result:
        return []
    return [(text, float(conf)) for (_, text, conf) in result]

def best_stamp_match(text):
    text_upper = text.upper().strip()
    if len(text_upper) < 4:
        return None, 0
    best_stamp, best_score = None, 0
    for stamp in KNOWN_STAMPS:
        score = fuzz.partial_ratio(text_upper, stamp)
        if score > best_score:
            best_score = score
            best_stamp = stamp
    return best_stamp, best_score

def main():
    rotations = [0, -5, -10, -15, -20, -25, -30, 5, 10, 15]
    scales = [3, 4, 5]

    for card_name, info in STAMP_CROPS.items():
        print(f"\n{'='*70}")
        print(f"Card: {card_name}")
        print(f"Expected: {info['expected']}")
        print(f"{'='*70}")

        img = cv2.imread(info["path"])
        x1, y1, x2, y2 = info["crop"]
        crop = img[y1:y2, x1:x2]

        all_results = []  # (text, conf, stamp, fuzzy_score, variant_info)

        for scale in scales:
            variants = preprocess(crop, scale)
            for vname, vimg in variants.items():
                for angle in rotations:
                    rotated = rotate_image(vimg, angle) if angle != 0 else vimg
                    texts = ocr(rotated)
                    for text, conf in texts:
                        if len(text.strip()) < 4:
                            continue
                        stamp, fscore = best_stamp_match(text)
                        all_results.append((text, conf, stamp, fscore, f"s={scale},{vname},rot={angle:+d}"))

        # Sort by fuzzy score descending
        all_results.sort(key=lambda x: x[3], reverse=True)

        # Print top 20 by fuzzy match
        print(f"\n  Top matches by fuzzy score (total detections: {len(all_results)}):")
        seen_texts = set()
        count = 0
        for text, conf, stamp, fscore, vinfo in all_results:
            if text in seen_texts:
                continue
            seen_texts.add(text)
            marker = " <<<" if fscore >= 75 else ""
            print(f"    [{vinfo}] '{text}' -> {stamp} (fuzzy={fscore}, conf={conf:.3f}){marker}")
            count += 1
            if count >= 25:
                break

        # Overall best match
        if all_results:
            best = all_results[0]
            print(f"\n  BEST MATCH: '{best[0]}' -> {best[2]} (fuzzy={best[3]}, conf={best[1]:.3f})")
            correct = best[2] == info["expected"]
            print(f"  CORRECT: {correct}")

if __name__ == "__main__":
    main()
