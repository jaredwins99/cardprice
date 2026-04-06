#!/usr/bin/env python3
"""Detect WotC-era PRERELEASE stamps on Pokemon cards.

WotC prerelease stamps appear as embossed/printed "PRERELEASE" text overlaid
on the card artwork. The stamp position varies -- sometimes bottom-right
(Misty's Seadra, Aerodactyl), sometimes center/upper area (Dragonite).

Approach:
1. Crop the artwork region (full and right-half)
2. Run RapidOCR with 6 preprocessing strategies (raw, unsharp, CLAHE,
   adaptive threshold, inverted threshold, Otsu)
3. Match OCR text against "PRERELEASE" using fuzzy ratio with strict
   length and character-composition guards to avoid false positives

Results on 45 WotC binder cards (3 prerelease, 42 normal):
- Precision: 100% (0 false positives)
- Recall: 66.7% (2/3 detected; Dragonite missed -- stamp too faint/transparent)
- Accuracy: 97.8%

Limitation: Very faint embossed stamps (like the Dragonite promo where the
stamp is nearly invisible against the sky background) cannot be detected by
OCR even with aggressive preprocessing. These would need either higher-res
photos or a DINOv2-based visual approach trained specifically on WotC stamps.

Usage:
    python scripts/research/variant/detect_prerelease.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz import fuzz

# Known prerelease cards for validation
KNOWN_PRERELEASE = {
    "page_20260307_014406_cards/card_02.png",
    "page_20260307_020047_cards/card_08.png",
    "page_20260307_015320_cards/card_05.png",
}

# WotC binder page prefixes to scan
WOTC_PAGES = [
    "page_20260307_014406_cards",
    "page_20260307_015320_cards",
    "page_20260307_020047_cards",
    "page_20260307_120653_cards",
    "page_20260307_123235_cards",
]

INBOX = Path("data/inbox")
OUTPUT_DIR = Path("data/prerelease_crops")


def get_artwork_region(img: np.ndarray) -> np.ndarray:
    """Crop the full artwork area of a WotC-era card.

    WotC card layout (approximate percentages):
    - Name bar: top ~12-15%
    - Artwork: ~14% to ~52% height, ~8% to ~92% width
    """
    h, w = img.shape[:2]
    art_top = int(h * 0.14)
    art_bottom = int(h * 0.53)
    art_left = int(w * 0.08)
    art_right = int(w * 0.92)
    return img[art_top:art_bottom, art_left:art_right]


def get_artwork_right_half(img: np.ndarray) -> np.ndarray:
    """Crop the right half of the artwork (stamp is usually right-of-center)."""
    h, w = img.shape[:2]
    art_top = int(h * 0.14)
    art_bottom = int(h * 0.53)
    art_left = int(w * 0.40)  # right 60% of card
    art_right = int(w * 0.92)
    return img[art_top:art_bottom, art_left:art_right]


def is_prerelease_text(text: str) -> tuple[bool, float, str]:
    """Check if OCR text matches "PRERELEASE" using multiple criteria.

    Returns (is_match, score, method).

    We use strict matching to avoid false positives from random card text
    like "PowereEvolutlenary" which has high fuzzy ratio but is clearly
    not a prerelease stamp.
    """
    clean = text.strip().upper().replace(" ", "")

    # Reject very short text -- can't be "PRERELEASE" (10 chars)
    # Short fragments like "el", "er" get perfect partial_ratio but are noise
    if len(clean) < 6:
        ratio = fuzz.ratio(clean, "PRERELEASE")
        return False, ratio, "too_short"

    # Method 1: Direct fuzzy ratio (full string)
    ratio = fuzz.ratio(clean, "PRERELEASE")

    # Method 2: Check if text is primarily composed of PRERELEASE letters
    # "PRERELEASE" has letters: P, R, E, L, A, S
    # The key discriminator: prerelease stamp text should be SHORT (8-12 chars)
    # and dominated by these specific letters
    prerelease_chars = set("PRERELEASE")
    if len(clean) > 0:
        char_overlap = sum(1 for c in clean if c in prerelease_chars) / len(clean)
    else:
        char_overlap = 0

    # Method 3: Check for "PRE" + "RELEASE" or "PRE" + "RELE" substrings
    has_pre = "PRE" in clean or "PBE" in clean or "PIE" in clean
    has_rele = "RELE" in clean or "RELF" in clean or "NELE" in clean or "RELI" in clean

    # Decision logic:
    # High confidence: fuzzy ratio >= 75 AND text is short (likely just the stamp)
    # AND minimum 7 chars (partial stamp reads like "PRERELE" are fine)
    if ratio >= 75 and 7 <= len(clean) <= 15:
        return True, ratio, "fuzzy_short"

    # Medium confidence: fuzzy ratio >= 70 AND high character overlap
    if ratio >= 70 and char_overlap >= 0.7 and len(clean) >= 7:
        return True, ratio, "fuzzy_overlap"

    # Substring match: contains both PRE and RELE-like substrings, short text
    if has_pre and has_rele and 8 <= len(clean) <= 16:
        return True, ratio, "substring"

    # Partial ratio: text >= 8 chars, high partial match, high char overlap
    partial = fuzz.partial_ratio(clean, "PRERELEASE")
    if partial >= 90 and len(clean) >= 8 and len(clean) <= 20 and char_overlap >= 0.6:
        return True, partial, "partial"

    return False, max(ratio, partial), "none"


def ocr_on_crop(crop: np.ndarray, ocr_engine: RapidOCR, scale: int = 3) -> list[tuple[str, float, str]]:
    """Run OCR with multiple preprocessing strategies on a crop.

    Returns list of (text, ocr_confidence, strategy_name).
    """
    h, w = crop.shape[:2]
    upscaled = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    results = []

    # Strategy 1: Raw color
    result, _ = ocr_engine(upscaled)
    if result:
        for line in result:
            results.append((line[1], line[2], "raw"))

    # Strategy 2: Grayscale + unsharp mask
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
    result2, _ = ocr_engine(sharp)
    if result2:
        for line in result2:
            results.append((line[1], line[2], "sharp"))

    # Strategy 3: CLAHE (contrast-limited adaptive histogram equalization)
    # Good for embossed/low-contrast text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    result3, _ = ocr_engine(enhanced)
    if result3:
        for line in result3:
            results.append((line[1], line[2], "clahe"))

    # Strategy 4: Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 5)
    result4, _ = ocr_engine(thresh)
    if result4:
        for line in result4:
            results.append((line[1], line[2], "thresh"))

    # Strategy 5: Inverted threshold
    thresh_inv = cv2.bitwise_not(thresh)
    result5, _ = ocr_engine(thresh_inv)
    if result5:
        for line in result5:
            results.append((line[1], line[2], "thresh_inv"))

    # Strategy 6: Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result6, _ = ocr_engine(otsu)
    if result6:
        for line in result6:
            results.append((line[1], line[2], "otsu"))

    return results


def detect_prerelease(img_path: Path, ocr_engine: RapidOCR,
                      save_crops: bool = False, verbose: bool = False) -> dict:
    """Detect if a card image has a WotC PRERELEASE stamp.

    Scans multiple regions of the artwork with multiple OCR strategies.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {"error": f"Could not read {img_path}", "is_prerelease": False,
                "ocr_text": "", "ocr_score": 0, "ocr_strategy": "", "path": str(img_path)}

    artwork_full = get_artwork_region(img)
    artwork_right = get_artwork_right_half(img)

    # Save crops for inspection
    if save_crops:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rel = img_path.relative_to(INBOX)
        base = str(rel).replace("/", "_").replace(".png", "")
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_artwork.png"), artwork_full)
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_right.png"), artwork_right)

    # Run OCR on both regions
    best_match = False
    best_score = 0
    best_text = ""
    best_strategy = ""
    best_method = ""
    all_texts = []

    for region_name, region_crop in [("right", artwork_right), ("full", artwork_full)]:
        ocr_results = ocr_on_crop(region_crop, ocr_engine, scale=3)

        for text, conf, strategy in ocr_results:
            is_match, score, method = is_prerelease_text(text)
            tag = f"{region_name}/{strategy}"
            all_texts.append((text, score, tag, method, is_match))

            if is_match and score > best_score:
                best_match = True
                best_score = score
                best_text = text
                best_strategy = tag
                best_method = method
            elif not best_match and score > best_score:
                best_score = score
                best_text = text
                best_strategy = tag
                best_method = method

    if verbose and all_texts:
        # Print all OCR texts sorted by score
        print(f"    All OCR texts for {img_path.name}:")
        for text, score, tag, method, is_match in sorted(all_texts, key=lambda x: -x[1])[:5]:
            flag = " ***" if is_match else ""
            print(f"      [{tag:20s}] '{text}' -> score={score:.1f} ({method}){flag}")

    return {
        "path": str(img_path),
        "is_prerelease": best_match,
        "ocr_text": best_text,
        "ocr_score": best_score,
        "ocr_strategy": best_strategy,
        "ocr_method": best_method,
    }


def main():
    print("=" * 80)
    print("WotC PRERELEASE Stamp Detector")
    print("=" * 80)

    ocr_engine = RapidOCR()

    # Phase 1: Analyze known prerelease cards (verbose)
    print("\n--- Phase 1: Known prerelease cards (verbose) ---\n")
    for rel_path in sorted(KNOWN_PRERELEASE):
        full_path = INBOX / rel_path
        if not full_path.exists():
            print(f"  MISSING: {full_path}")
            continue

        result = detect_prerelease(full_path, ocr_engine, save_crops=True, verbose=True)
        status = "PRERELEASE" if result["is_prerelease"] else "normal"
        print(f"  [{status:>10}] {rel_path}")
        print(f"             Best OCR: '{result['ocr_text']}' (score={result['ocr_score']:.1f}, via={result['ocr_strategy']}, method={result['ocr_method']})")
        print()

    # Phase 2: Test all WotC binder cards
    print("\n--- Phase 2: All WotC binder cards ---\n")

    tp, fp, tn, fn = 0, 0, 0, 0
    all_results = []

    for page_dir in sorted(WOTC_PAGES):
        page_path = INBOX / page_dir
        if not page_path.exists():
            print(f"  MISSING: {page_path}")
            continue

        for card_file in sorted(page_path.glob("card_*.png")):
            rel = card_file.relative_to(INBOX)
            rel_str = str(rel)
            is_known = rel_str in KNOWN_PRERELEASE

            result = detect_prerelease(card_file, ocr_engine,
                                        save_crops=is_known,
                                        verbose=(is_known or False))
            all_results.append(result)

            detected = result["is_prerelease"]
            actual = is_known

            if actual and detected:
                tp += 1
                label = "TP"
            elif actual and not detected:
                fn += 1
                label = "FN"
            elif not actual and detected:
                fp += 1
                label = "FP"
            else:
                tn += 1
                label = "TN"

            # Print interesting cases
            if detected or actual:
                print(f"  [{label}] {rel_str}")
                print(f"       OCR: '{result['ocr_text']}' (score={result['ocr_score']:.1f}, via={result['ocr_strategy']}, method={result['ocr_method']})")

    # Phase 3: Summary
    total = tp + fp + tn + fn
    print("\n--- Phase 3: Summary ---\n")
    print(f"  Total cards tested: {total}")
    print(f"  True positives:  {tp} (prerelease correctly detected)")
    print(f"  False positives: {fp} (non-prerelease flagged)")
    print(f"  True negatives:  {tn} (non-prerelease correctly ignored)")
    print(f"  False negatives: {fn} (prerelease missed)")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision: {precision:.1%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall: {recall:.1%}")
    if total > 0:
        accuracy = (tp + tn) / total
        print(f"  Accuracy: {accuracy:.1%}")

    # Phase 4: Score distribution
    print("\n--- Phase 4: OCR Score Distribution ---\n")
    prerelease_scores = []
    normal_scores = []
    for r in all_results:
        rel = Path(r["path"]).relative_to(INBOX)
        if str(rel) in KNOWN_PRERELEASE:
            prerelease_scores.append((r["ocr_score"], r["ocr_text"], r["ocr_method"]))
        else:
            normal_scores.append((r["ocr_score"], r["ocr_text"], r["ocr_method"]))

    if prerelease_scores:
        print(f"  Prerelease cards:")
        for score, text, method in sorted(prerelease_scores, key=lambda x: -x[0]):
            print(f"    score={score:.1f} '{text}' ({method})")

    if normal_scores:
        print(f"  Normal cards (top 10 scores):")
        for score, text, method in sorted(normal_scores, key=lambda x: -x[0])[:10]:
            print(f"    score={score:.1f} '{text}' ({method})")


if __name__ == "__main__":
    main()
