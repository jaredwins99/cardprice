#!/usr/bin/env python3
"""Test the integrated DINOv2 1st Edition stamp detector.

Tests:
1. Reference images (unlimited) -- should NOT detect 1st edition
2. Synthesized 1st edition images -- SHOULD detect 1st edition
3. Integration test via detect_stamps() API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/godli/cardprice/data")
REF_DIR = DATA_DIR / "card_images"

FIRST_EDITION_SETS = ["base1", "base2", "base3", "base5",
                      "gym1", "gym2", "neo1", "neo2", "neo3", "neo4"]


def synthesize_stamp(card_bgr):
    """Quick stamp synthesis for testing."""
    result = card_bgr.copy()
    h, w = result.shape[:2]
    cx, cy = int(w * 0.08), int(h * 0.58)
    r = max(int(h * 0.035), 4)

    # Load and overlay the template stamp
    stamp_path = DATA_DIR / "1st_edition_stamps" / "1st_edition_english_stamp.png"
    if stamp_path.exists():
        stamp = cv2.imread(str(stamp_path), cv2.IMREAD_UNCHANGED)
        if stamp is not None:
            if len(stamp.shape) == 3:
                stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
            else:
                stamp_gray = stamp
            target_h = max(int(h * 0.10), 10)
            scale = target_h / stamp_gray.shape[0]
            target_w = int(stamp_gray.shape[1] * scale)
            stamp_resized = cv2.resize(stamp_gray, (target_w, target_h))
            _, alpha = cv2.threshold(stamp_resized, 128, 255, cv2.THRESH_BINARY_INV)

            x1 = cx - target_w // 2
            y1 = cy - target_h // 2
            x2, y2 = x1 + target_w, y1 + target_h
            sx1, sy1 = max(0, -x1), max(0, -y1)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

            if x2 > x1 and y2 > y1:
                alpha_crop = alpha[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
                for c in range(3):
                    result[y1:y2, x1:x2, c] = (
                        result[y1:y2, x1:x2, c] * (1 - alpha_crop)
                    ).astype(np.uint8)
                return result

    # Fallback: simple circle
    cv2.circle(result, (cx, cy), r, (0, 0, 0), -1)
    return result


def test_dino_detector_direct():
    """Test _check_1st_edition_dino directly."""
    from cardprice.ml.stamp_detection import _check_1st_edition_dino

    print("=" * 70)
    print("Test 1: Direct DINOv2 1st Edition detector")
    print("=" * 70)

    tp = tn = fp = fn = 0

    for set_id in ["base1", "base3", "gym1", "neo1"]:
        set_dir = REF_DIR / set_id
        if not set_dir.exists():
            continue

        images = sorted(set_dir.glob("*.png"))[:3]
        for img_path in images:
            card_name = img_path.stem
            # card_id like "base1-4/normal" from filename "base1-4_normal.png"
            parts = card_name.rsplit("_", 1)
            card_id = f"{parts[0]}/{parts[1]}" if len(parts) == 2 else card_name

            ref_bgr = cv2.imread(str(img_path))
            if ref_bgr is None:
                continue

            # Test unlimited (should NOT detect)
            result = _check_1st_edition_dino(ref_bgr, card_id, set_id)
            if result is None:
                print(f"  {card_name:30s}  unlimited: SKIP (no ref)")
                continue

            if result["detected"]:
                fp += 1
                label = "FALSE POSITIVE"
            else:
                tn += 1
                label = "correct negative"
            print(f"  {card_name:30s}  unlimited: {label:20s}  "
                  f"diff={result['differential']:+.4f}")

            # Test synthesized 1st edition (SHOULD detect)
            stamped = synthesize_stamp(ref_bgr)
            result2 = _check_1st_edition_dino(stamped, card_id, set_id)
            if result2 is None:
                continue

            if result2["detected"]:
                tp += 1
                label = "correct positive"
            else:
                fn += 1
                label = "FALSE NEGATIVE"
            print(f"  {card_name:30s}  1st_ed:    {label:20s}  "
                  f"diff={result2['differential']:+.4f}")

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    print(f"\nResults: TP={tp} TN={tn} FP={fp} FN={fn} accuracy={acc:.1%}")


def test_detect_stamps_integration():
    """Test via the full detect_all_variants() API."""
    from cardprice.ml.stamp_detection import detect_all_variants

    print("\n" + "=" * 70)
    print("Test 2: Integration via detect_all_variants()")
    print("=" * 70)

    # Test on a few base1 cards
    test_cards = [
        ("base1", "base1-4_normal.png", "base1-4/normal"),    # Charizard
        ("base1", "base1-25_normal.png", "base1-25/normal"),   # Pikachu
        ("base1", "base1-100_normal.png", "base1-100/normal"), # Common
    ]

    for set_id, fname, card_id in test_cards:
        img_path = REF_DIR / set_id / fname
        if not img_path.exists():
            print(f"  Skipping {fname} (not found)")
            continue

        # Test unlimited reference image
        result = detect_all_variants(str(img_path), card_id, fast=False)
        has_1st = result["variant_flags"].get("1st_edition", False)
        stamps = result["stamps_detected"]
        detail = result["stamp_details"].get("1st_edition", {})
        dino_detail = result["stamp_details"].get("1st_edition_dino", {})

        print(f"\n  {fname} (unlimited):")
        print(f"    1st_edition detected: {has_1st}")
        print(f"    stamps: {stamps}")
        if dino_detail:
            print(f"    dino diff: {dino_detail.get('differential', 'N/A'):.4f}")

        # Test with synthesized stamp
        ref_bgr = cv2.imread(str(img_path))
        stamped = synthesize_stamp(ref_bgr)

        # Save to temp file for detect_all_variants
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, stamped)

        result2 = detect_all_variants(temp_path, card_id, fast=False)
        has_1st2 = result2["variant_flags"].get("1st_edition", False)
        stamps2 = result2["stamps_detected"]
        detail2 = result2["stamp_details"].get("1st_edition", {})

        print(f"  {fname} (1st edition synth):")
        print(f"    1st_edition detected: {has_1st2}")
        print(f"    stamps: {stamps2}")
        if detail2:
            print(f"    evidence: {detail2.get('evidence', 'N/A')}")
            if 'differential' in detail2:
                print(f"    dino diff: {detail2['differential']:.4f}")

        os.unlink(temp_path)


if __name__ == "__main__":
    test_dino_detector_direct()
    test_detect_stamps_integration()
