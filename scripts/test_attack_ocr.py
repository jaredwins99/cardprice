#!/usr/bin/env python3
"""Evaluate attack name OCR accuracy on all eval binder cards.

For each card in data/eval/binder_eval.json:
1. Load segment image
2. Crop to attack region (~40-80% of card height, center 70% width)
3. Run EasyOCR on the cropped region
4. Look up ground truth attacks from data/attack_index.pkl
5. Fuzzy match OCR text against expected attack names
6. Report per-card and overall accuracy

Results saved to data/eval/attack_ocr_results.json
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
ATTACK_INDEX = PROJECT_ROOT / "data" / "attack_index.pkl"
RESULTS_OUT = PROJECT_ROOT / "data" / "eval" / "attack_ocr_results.json"

# Fuzzy match threshold — ratio above which we consider a match
FUZZY_THRESHOLD = 0.70


def fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio between two lowercased strings."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def best_fuzzy_match(ocr_text: str, attack_names: list[str]) -> tuple[str | None, float]:
    """Find best fuzzy match for ocr_text among attack_names. Return (name, ratio)."""
    best_name = None
    best_ratio = 0.0
    for name in attack_names:
        r = fuzzy_ratio(ocr_text, name)
        if r > best_ratio:
            best_ratio = r
            best_name = name
    return best_name, best_ratio


def crop_attack_region(img: np.ndarray) -> np.ndarray:
    """Crop to the attack text region of a Pokemon card.

    Attack names are typically in the middle portion of the card:
    - Vertically: roughly 40%-80% of card height
    - Horizontally: center 80% to avoid border/energy symbols
    """
    h, w = img.shape[:2]
    y_start = int(h * 0.38)
    y_end = int(h * 0.82)
    x_start = int(w * 0.10)
    x_end = int(w * 0.90)
    return img[y_start:y_end, x_start:x_end]


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Enhance the attack region crop for better OCR.

    - Convert to grayscale
    - Upscale if small
    - Apply CLAHE for contrast
    - Light sharpening
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Upscale small images (binder segments are ~630x880)
    h, w = gray.shape[:2]
    if h < 400:
        scale = 2
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def run_easyocr(img: np.ndarray, reader) -> list[dict]:
    """Run EasyOCR on an image, return list of {text, confidence, bbox}."""
    results = reader.readtext(img, detail=1, paragraph=False)
    parsed = []
    for bbox, text, conf in results:
        parsed.append({
            "text": text,
            "confidence": float(conf),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })
    return parsed


def evaluate_card(
    card_info: dict,
    segments_dir: str,
    card_to_attacks: dict,
    reader,
) -> dict:
    """Evaluate attack OCR for a single card."""
    card_id = card_info["card_id"]
    name = card_info["name"]
    segment_file = card_info["segment"]

    result = {
        "card_id": card_id,
        "name": name,
        "segment": segment_file,
        "expected_attacks": [],
        "ocr_texts": [],
        "matches": [],
        "attacks_found": 0,
        "attacks_expected": 0,
        "skipped": False,
    }

    # Skip null card_id (empty slots)
    if card_id is None:
        result["skipped"] = True
        result["skip_reason"] = "null card_id (empty slot)"
        logger.info(f"  SKIP {name}: empty slot")
        return result

    # Look up expected attacks
    # The index uses card_id with variant as key
    expected = card_to_attacks.get(card_id, [])
    if not expected:
        # Try without variant
        base_id = card_id.split("/")[0]
        for k, v in card_to_attacks.items():
            if k.startswith(base_id + "/") or k == base_id:
                expected = v
                break

    if not expected:
        result["skipped"] = True
        result["skip_reason"] = f"no attacks in index for {card_id}"
        logger.info(f"  SKIP {name} ({card_id}): no attacks in index")
        return result

    result["expected_attacks"] = expected
    result["attacks_expected"] = len(expected)

    # Load segment image
    img_path = Path(PROJECT_ROOT) / segments_dir / segment_file
    if not img_path.exists():
        result["skipped"] = True
        result["skip_reason"] = f"segment not found: {img_path}"
        logger.warning(f"  SKIP {name}: segment not found at {img_path}")
        return result

    img = cv2.imread(str(img_path))
    if img is None:
        result["skipped"] = True
        result["skip_reason"] = f"failed to read image: {img_path}"
        return result

    # Crop and preprocess
    attack_crop = crop_attack_region(img)
    processed = preprocess_for_ocr(attack_crop)

    # Run OCR
    ocr_results = run_easyocr(processed, reader)
    ocr_texts = [r["text"] for r in ocr_results]
    result["ocr_texts"] = ocr_texts
    result["ocr_details"] = ocr_results

    # Match each expected attack against OCR texts
    found_attacks = []
    for attack in expected:
        best_text = None
        best_ratio = 0.0
        for ocr_text in ocr_texts:
            r = fuzzy_ratio(ocr_text, attack)
            if r > best_ratio:
                best_ratio = r
                best_text = ocr_text
        matched = best_ratio >= FUZZY_THRESHOLD
        match_info = {
            "expected": attack,
            "best_ocr_text": best_text,
            "fuzzy_ratio": round(best_ratio, 3),
            "matched": matched,
        }
        result["matches"].append(match_info)
        if matched:
            found_attacks.append(attack)

    result["attacks_found"] = len(found_attacks)

    status = f"{len(found_attacks)}/{len(expected)}"
    logger.info(f"  {name} ({card_id}): {status} attacks found | OCR: {ocr_texts}")
    for m in result["matches"]:
        flag = "OK" if m["matched"] else "MISS"
        logger.info(
            f"    [{flag}] '{m['expected']}' -> '{m['best_ocr_text']}' "
            f"(ratio={m['fuzzy_ratio']:.3f})"
        )

    return result


def main():
    t0 = time.time()

    # Load eval data
    logger.info(f"Loading eval data from {EVAL_JSON}")
    with open(EVAL_JSON) as f:
        eval_data = json.load(f)

    # Load attack index
    logger.info(f"Loading attack index from {ATTACK_INDEX}")
    with open(ATTACK_INDEX, "rb") as f:
        attack_index = pickle.load(f)
    card_to_attacks = attack_index["card_to_attacks"]
    logger.info(f"Attack index has {len(card_to_attacks)} cards")

    # Initialize EasyOCR (slow first time — downloads model)
    logger.info("Initializing EasyOCR reader...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    logger.info("EasyOCR ready")

    # Process all cards
    all_results = []
    total_expected = 0
    total_found = 0
    total_cards = 0
    skipped_cards = 0

    for page_idx, page in enumerate(eval_data["pages"]):
        segments_dir = page["segments_dir"]
        logger.info(f"\n=== Page {page_idx + 1}: {page.get('image', 'unknown')} ===")

        for card_info in page["cards"]:
            result = evaluate_card(card_info, segments_dir, card_to_attacks, reader)
            all_results.append(result)

            if result["skipped"]:
                skipped_cards += 1
            else:
                total_cards += 1
                total_expected += result["attacks_expected"]
                total_found += result["attacks_found"]

    # Compute summary
    accuracy = total_found / total_expected if total_expected > 0 else 0.0
    cards_perfect = sum(
        1 for r in all_results
        if not r["skipped"] and r["attacks_found"] == r["attacks_expected"]
    )

    elapsed = time.time() - t0

    summary = {
        "total_cards_evaluated": total_cards,
        "total_cards_skipped": skipped_cards,
        "total_attacks_expected": total_expected,
        "total_attacks_found": total_found,
        "attack_recall": round(accuracy, 4),
        "cards_with_all_attacks_found": cards_perfect,
        "card_perfect_rate": round(cards_perfect / total_cards, 4) if total_cards > 0 else 0.0,
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "elapsed_seconds": round(elapsed, 1),
    }

    output = {
        "summary": summary,
        "per_card": all_results,
    }

    # Save results
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("ATTACK OCR EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Cards evaluated:    {total_cards}")
    logger.info(f"Cards skipped:      {skipped_cards}")
    logger.info(f"Attacks expected:   {total_expected}")
    logger.info(f"Attacks found:      {total_found}")
    logger.info(f"Attack recall:      {accuracy:.1%}")
    logger.info(f"Cards all correct:  {cards_perfect}/{total_cards} ({cards_perfect/total_cards:.1%})" if total_cards > 0 else "N/A")
    logger.info(f"Fuzzy threshold:    {FUZZY_THRESHOLD}")
    logger.info(f"Time elapsed:       {elapsed:.1f}s")
    logger.info(f"Results saved to:   {RESULTS_OUT}")


if __name__ == "__main__":
    main()
