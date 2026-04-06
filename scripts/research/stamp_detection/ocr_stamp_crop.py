#!/usr/bin/env python3
"""OCR stamp text from focused crops of the stamp region at full resolution.

The stamps (e.g. DRAGON FRONTIERS, CRYSTAL GUARDIANS) sit in the lower-right
of the artwork area on EX-era Pokemon cards. This script crops just the stamp,
upscales + preprocesses, tries multiple rotations, and runs OCR.
"""

import cv2
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

# Stamp region coordinates (relative to 1008x1530 card images)
# Found by visual inspection of each card
STAMP_CROPS = {
    "Chikorita (DRAGON FRONTIERS)": {
        "path": "data/inbox/page_20260305_094228_cards/card_00.png",
        "expected": "DRAGON FRONTIERS",
        "crop": (500, 680, 900, 760),  # tight around the stamp banner
    },
    "Meganium (DRAGON FRONTIERS)": {
        "path": "data/inbox/page_20260305_094228_cards/card_02.png",
        "expected": "DRAGON FRONTIERS",
        "crop": (450, 700, 900, 790),
    },
    "Skitty (CRYSTAL GUARDIANS)": {
        "path": "data/inbox/page_20260228_174819_cards/card_01.png",
        "expected": "CRYSTAL GUARDIANS",
        "crop": (550, 580, 870, 740),  # two-line stamp, taller crop
    },
    "Vibrava (DRAGON FRONTIERS)": {
        "path": "data/inbox/page_20260228_174819_cards/card_05.png",
        "expected": "DRAGON FRONTIERS",
        "crop": (440, 660, 850, 750),
    },
}

KNOWN_STAMPS = [
    "DRAGON FRONTIERS",
    "CRYSTAL GUARDIANS",
    "DELTA SPECIES",
    "POWER KEEPERS",
    "LEGEND MAKER",
    "HOLON PHANTOMS",
    "UNSEEN FORCES",
    "DEOXYS",
    "EMERALD",
    "FIRE RED LEAF GREEN",
    "TEAM ROCKET RETURNS",
    "HIDDEN LEGENDS",
    "TEAM MAGMA VS TEAM AQUA",
]


def preprocess_variants(crop_bgr, scale=3):
    """Generate multiple preprocessed variants for OCR."""
    h, w = crop_bgr.shape[:2]
    upscaled = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Unsharp mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)

    # Binary threshold variants
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive threshold
    adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    adapt_inv = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)

    # High contrast: stretch histogram
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray.astype(float) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

    return {
        "upscaled_color": upscaled,
        "gray": gray,
        "enhanced": enhanced,
        "sharp": sharp,
        "otsu": otsu,
        "otsu_inv": otsu_inv,
        "adapt": adapt,
        "adapt_inv": adapt_inv,
        "stretched": stretched,
    }


def rotate_image(img, angle):
    """Rotate image by angle degrees around center."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def run_rapidocr(img):
    """Run RapidOCR on image, return list of (text, conf) tuples."""
    engine = RapidOCR()
    result, _ = engine(img)
    if result is None:
        return []
    return [(text, float(conf)) for (_, text, conf) in result]


def run_easyocr(img, reader):
    """Run EasyOCR on image, return list of (text, conf) tuples."""
    results = reader.readtext(img)
    return [(text, float(conf)) for (_, text, conf) in results]


def fuzzy_match_stamp(texts, known_stamps):
    """Check if any OCR text fuzzy-matches a known stamp."""
    from rapidfuzz import fuzz
    matches = []
    for text, conf in texts:
        text_upper = text.upper().strip()
        if len(text_upper) < 3:
            continue
        for stamp in known_stamps:
            score = fuzz.partial_ratio(text_upper, stamp)
            if score >= 50:
                matches.append((stamp, text, score, conf))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def main():
    import easyocr

    output_dir = Path("data/stamp_ocr_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    rotations = [0, -5, -10, -15, -20, -30, 5, 10]
    scales = [3, 4]

    # Initialize EasyOCR once
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    for card_name, info in STAMP_CROPS.items():
        print(f"\n{'='*70}")
        print(f"Card: {card_name}")
        print(f"Expected stamp: {info['expected']}")
        print(f"{'='*70}")

        img = cv2.imread(info["path"])
        if img is None:
            print(f"  ERROR: Could not load {info['path']}")
            continue

        x1, y1, x2, y2 = info["crop"]
        crop = img[y1:y2, x1:x2]
        safe_name = card_name.split("(")[0].strip().replace(" ", "_").lower()
        cv2.imwrite(str(output_dir / f"{safe_name}_stamp_crop.png"), crop)

        print(f"  Crop size: {crop.shape[1]}x{crop.shape[0]}")

        all_rapid_texts = []
        all_easy_texts = []

        for scale in scales:
            variants = preprocess_variants(crop, scale=scale)

            # Save all variants for debugging
            for vname, vimg in variants.items():
                cv2.imwrite(str(output_dir / f"{safe_name}_s{scale}_{vname}.png"), vimg)

            # RapidOCR on all variants x rotations
            for vname, vimg in variants.items():
                for angle in rotations:
                    rotated = rotate_image(vimg, angle) if angle != 0 else vimg
                    texts = run_rapidocr(rotated)
                    if texts:
                        for text, conf in texts:
                            if len(text.strip()) >= 3:
                                print(f"  RapidOCR [s={scale}, {vname}, rot={angle:+d}°]: '{text}' (conf={conf:.3f})")
                                all_rapid_texts.append((text, conf))

            # EasyOCR on key variants x rotations
            for vname in ["upscaled_color", "enhanced", "sharp", "stretched"]:
                vimg = variants[vname]
                for angle in rotations:
                    rotated = rotate_image(vimg, angle) if angle != 0 else vimg
                    texts = run_easyocr(rotated, reader)
                    if texts:
                        for text, conf in texts:
                            if len(text.strip()) >= 3:
                                print(f"  EasyOCR  [s={scale}, {vname}, rot={angle:+d}°]: '{text}' (conf={conf:.3f})")
                                all_easy_texts.append((text, conf))

        # Summary
        print(f"\n  --- SUMMARY for {card_name} ---")
        print(f"  Total RapidOCR detections: {len(all_rapid_texts)}")
        print(f"  Total EasyOCR detections: {len(all_easy_texts)}")

        if all_rapid_texts:
            matches = fuzzy_match_stamp(all_rapid_texts, KNOWN_STAMPS)
            if matches:
                print(f"  Best RapidOCR stamp matches:")
                seen = set()
                for stamp, text, score, conf in matches[:10]:
                    key = (stamp, text)
                    if key not in seen:
                        seen.add(key)
                        print(f"    '{text}' -> {stamp} (fuzzy={score}, ocr_conf={conf:.3f})")

        if all_easy_texts:
            matches = fuzzy_match_stamp(all_easy_texts, KNOWN_STAMPS)
            if matches:
                print(f"  Best EasyOCR stamp matches:")
                seen = set()
                for stamp, text, score, conf in matches[:10]:
                    key = (stamp, text)
                    if key not in seen:
                        seen.add(key)
                        print(f"    '{text}' -> {stamp} (fuzzy={score}, ocr_conf={conf:.3f})")

        # Unique texts across both engines
        all_texts = set(t for t, _ in all_rapid_texts + all_easy_texts)
        print(f"  All unique OCR texts: {sorted(all_texts)}")


if __name__ == "__main__":
    main()
