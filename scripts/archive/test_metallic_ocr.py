#!/usr/bin/env python3
"""Test multiple preprocessing approaches for metallic/holographic text OCR."""

import cv2
import numpy as np
import easyocr
import os
import json
from pathlib import Path

IMG_DIR = "data/inbox/page_20260228_174819_cards_v4"
OUT_DIR = "/tmp/metallic_ocr_debug"
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

# Initialize EasyOCR once
reader = easyocr.Reader(['en'], gpu=False)


def crop_name_region(img):
    """Crop top 20% of card image (name region)."""
    h = img.shape[0]
    return img[:int(h * 0.20), :]


def upscale_2x(img):
    """Upscale image 2x for better OCR."""
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def approach_1_heavy_clahe(img):
    """Heavy contrast enhancement: grayscale + aggressive CLAHE + threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
    enhanced = clahe.apply(gray)
    # Also try without threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "clahe_only": cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        "clahe_otsu": cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
    }


def approach_2_channel_isolation(img):
    """Color channel isolation: blue, saturation, LAB a* channel."""
    results = {}

    # Blue channel
    b, g, r = cv2.split(img)
    results["blue_ch"] = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

    # Green channel (metallic gold/silver may show here)
    results["green_ch"] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # Red channel
    results["red_ch"] = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    # Saturation channel (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    results["saturation"] = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)

    # Value channel (HSV)
    results["value"] = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

    # LAB L* channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    results["lab_L"] = cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR)

    # LAB a* channel
    results["lab_a"] = cv2.cvtColor(lab[:, :, 1], cv2.COLOR_GRAY2BGR)

    # LAB b* channel
    results["lab_b"] = cv2.cvtColor(lab[:, :, 2], cv2.COLOR_GRAY2BGR)

    # Inverted saturation (low-sat metallic text becomes white)
    inv_sat = 255 - hsv[:, :, 1]
    results["inv_saturation"] = cv2.cvtColor(inv_sat, cv2.COLOR_GRAY2BGR)

    return results


def approach_3_edge_morphology(img):
    """Edge detection + morphology to form text blobs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    results["canny_dilate"] = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    # Sobel X (horizontal text edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.uint8(np.abs(sobelx) / np.abs(sobelx).max() * 255) if sobelx.max() > 0 else np.uint8(sobelx)
    results["sobel_x"] = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.abs(lap) / max(np.abs(lap).max(), 1) * 255)
    results["laplacian"] = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    return results


def approach_4_negative(img):
    """Invert the image before OCR."""
    results = {}
    inverted = 255 - img
    results["negative"] = inverted

    # Also try negative of grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    results["negative_gray"] = cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR)

    # Negative + CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    inv_clahe = clahe.apply(inv_gray)
    results["negative_clahe"] = cv2.cvtColor(inv_clahe, cv2.COLOR_GRAY2BGR)

    return results


def approach_5_bilateral_sharpen(img):
    """Bilateral filter + heavy sharpening."""
    results = {}

    # Bilateral filter
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    # Unsharp mask: original + alpha*(original - blurred)
    blurred = cv2.GaussianBlur(bilateral, (0, 0), 3)
    sharpened = cv2.addWeighted(bilateral, 2.5, blurred, -1.5, 0)
    results["bilateral_sharpen"] = sharpened

    # Even more aggressive
    sharpened2 = cv2.addWeighted(bilateral, 3.5, blurred, -2.5, 0)
    results["bilateral_sharpen_heavy"] = sharpened2

    # Bilateral + grayscale + CLAHE
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    results["bilateral_clahe"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return results


def approach_6_adaptive_threshold(img):
    """Adaptive thresholding with different block sizes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    for block_size in [11, 21, 31, 51]:
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 5
        )
        results[f"adaptive_b{block_size}"] = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    # Mean adaptive
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 21, 5
    )
    results["adaptive_mean_b21"] = cv2.cvtColor(adaptive_mean, cv2.COLOR_GRAY2BGR)

    # Inverted adaptive (white text on black)
    adaptive_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    results["adaptive_inv_b21"] = cv2.cvtColor(adaptive_inv, cv2.COLOR_GRAY2BGR)

    return results


def run_ocr(reader, img_bgr):
    """Run EasyOCR on a BGR image, return list of (text, confidence)."""
    # EasyOCR expects RGB or grayscale
    if len(img_bgr.shape) == 3:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_bgr
    results = reader.readtext(rgb, paragraph=False)
    return [(text, conf) for (_, text, conf) in results]


def check_match(ocr_results, expected_name):
    """Check if any OCR result contains the expected Pokemon name (fuzzy)."""
    expected_lower = expected_name.lower()
    for text, conf in ocr_results:
        text_lower = text.lower().strip()
        # Exact substring
        if expected_lower in text_lower or text_lower in expected_lower:
            return True
        # Check first 4+ chars match (handle partial reads)
        if len(expected_lower) >= 4 and len(text_lower) >= 4:
            if expected_lower[:4] == text_lower[:4]:
                return True
    return False


def main():
    approaches = {
        "1_heavy_clahe": approach_1_heavy_clahe,
        "2_channel_isolation": approach_2_channel_isolation,
        "3_edge_morphology": approach_3_edge_morphology,
        "4_negative": approach_4_negative,
        "5_bilateral_sharpen": approach_5_bilateral_sharpen,
        "6_adaptive_threshold": approach_6_adaptive_threshold,
    }

    # Results: approach_name -> sub_method -> card_idx -> {ocr_texts, matched}
    all_results = {}
    # Also track per-card best
    card_best = {i: {"text": "", "method": "none", "matched": False} for i in range(9)}

    for approach_name, approach_fn in approaches.items():
        print(f"\n{'='*70}")
        print(f"APPROACH: {approach_name}")
        print(f"{'='*70}")

        for card_idx in range(9):
            card_id, expected_name = GROUND_TRUTH[card_idx]
            img_path = os.path.join(IMG_DIR, f"card_{card_idx:02d}.png")
            img = cv2.imread(img_path)

            # Crop name region (top 20%)
            name_crop = crop_name_region(img)

            # Upscale for better OCR
            name_crop = upscale_2x(name_crop)

            # Apply approach
            sub_results = approach_fn(name_crop)

            for sub_name, processed_img in sub_results.items():
                # Save debug image
                debug_path = os.path.join(OUT_DIR, f"card{card_idx:02d}_{approach_name}_{sub_name}.png")
                cv2.imwrite(debug_path, processed_img)

                # Run OCR
                ocr_results = run_ocr(reader, processed_img)
                matched = check_match(ocr_results, expected_name)

                key = f"{approach_name}/{sub_name}"
                if key not in all_results:
                    all_results[key] = {}
                all_results[key][card_idx] = {
                    "ocr": ocr_results,
                    "matched": matched,
                }

                if matched and not card_best[card_idx]["matched"]:
                    card_best[card_idx] = {
                        "text": str(ocr_results),
                        "method": key,
                        "matched": True,
                    }

                # Print if OCR found anything
                texts = [f"{t}({c:.2f})" for t, c in ocr_results]
                status = "MATCH" if matched else "miss"
                if ocr_results:
                    print(f"  card_{card_idx:02d} [{expected_name:12s}] {sub_name:25s}: {status} | {', '.join(texts)}")

    # Also run baseline (no preprocessing, just upscaled crop)
    print(f"\n{'='*70}")
    print(f"BASELINE (no preprocessing, just upscaled name crop)")
    print(f"{'='*70}")
    for card_idx in range(9):
        card_id, expected_name = GROUND_TRUTH[card_idx]
        img_path = os.path.join(IMG_DIR, f"card_{card_idx:02d}.png")
        img = cv2.imread(img_path)
        name_crop = upscale_2x(crop_name_region(img))
        ocr_results = run_ocr(reader, name_crop)
        matched = check_match(ocr_results, expected_name)
        texts = [f"{t}({c:.2f})" for t, c in ocr_results]
        status = "MATCH" if matched else "miss"
        print(f"  card_{card_idx:02d} [{expected_name:12s}]: {status} | {', '.join(texts)}")

        key = "baseline"
        if key not in all_results:
            all_results[key] = {}
        all_results[key][card_idx] = {"ocr": ocr_results, "matched": matched}

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Method accuracy (how many of 9 cards matched)")
    print(f"{'='*70}")

    method_scores = []
    for method_key, card_results in sorted(all_results.items()):
        matches = sum(1 for cr in card_results.values() if cr["matched"])
        method_scores.append((matches, method_key))

    method_scores.sort(reverse=True)
    for score, method in method_scores:
        print(f"  {score}/9  {method}")

    print(f"\n{'='*70}")
    print(f"PER-CARD BEST RESULT")
    print(f"{'='*70}")
    total_matched = 0
    for card_idx in range(9):
        _, expected_name = GROUND_TRUTH[card_idx]
        best = card_best[card_idx]
        if best["matched"]:
            total_matched += 1
            print(f"  card_{card_idx:02d} [{expected_name:12s}]: FOUND via {best['method']}")
        else:
            # Show what OCR found across all methods
            print(f"  card_{card_idx:02d} [{expected_name:12s}]: NOT FOUND by any method")

    print(f"\nTotal cards recoverable: {total_matched}/9")


if __name__ == "__main__":
    main()
