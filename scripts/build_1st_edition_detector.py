#!/usr/bin/env python3
"""Build and test a DINOv2-based 1st Edition stamp detector.

Strategy: DINOv2 differential comparison (same as EX stamp detector).
  1. Crop stamp region from scan (x:3-15%, y:53-67%)
  2. Crop same region from reference image (unlimited, no stamp)
  3. Crop a control region (artwork center, same on both editions)
  4. DINOv2 similarity: stamp_sim and ctrl_sim
  5. Differential = ctrl_sim - stamp_sim
  6. If differential > threshold => 1st edition detected

For testing, we SYNTHESIZE 1st edition stamps onto reference images:
  - The stamp template comes from Bulbapedia (1st_edition_english_stamp.png)
  - We overlay it at the correct position on unlimited reference card crops

Then we test:
  - Unlimited reference images (negative, diff should be ~0)
  - Synthesized 1st edition images (positive, diff should be high)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path("/home/godli/cardprice/data")
REF_DIR = DATA_DIR / "card_images"
STAMP_DIR = DATA_DIR / "1st_edition_stamps"

# 1st Edition stamp region (normalized coords on card)
# Same as STAMP_REGIONS["1st_edition"]["tight"] in stamp_detection.py
STAMP_REGION = (0.03, 0.53, 0.15, 0.67)

# Control region: artwork center (never has a stamp)
CONTROL_REGION = (0.15, 0.15, 0.85, 0.45)

# WotC sets that had 1st edition print runs
FIRST_EDITION_SETS = ["base1", "base2", "base3", "base5",
                      "gym1", "gym2", "neo1", "neo2", "neo3", "neo4"]


def _extract_region(img: np.ndarray, x0: float, y0: float,
                    x1: float, y1: float) -> np.ndarray:
    """Extract a rectangular region using fractional coords."""
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def synthesize_1st_edition_stamp(card_bgr: np.ndarray) -> np.ndarray:
    """Overlay a synthesized 1st edition stamp onto a card image.

    The stamp is a black circle with white "1" inside and "EDITION" text
    arced above it. We draw it at approximately the correct position and
    size for a WotC-era Pokemon card.

    Parameters
    ----------
    card_bgr : np.ndarray
        Original unlimited card image in BGR.

    Returns
    -------
    np.ndarray
        Card image with synthetic 1st edition stamp overlay.
    """
    result = card_bgr.copy()
    h, w = result.shape[:2]

    # Stamp center position: approximately x=8%, y=58% of card
    cx = int(w * 0.08)
    cy = int(h * 0.58)

    # Stamp circle radius: approximately 3.5% of card height
    r = max(int(h * 0.035), 4)

    # Draw black filled circle
    cv2.circle(result, (cx, cy), r, (0, 0, 0), -1)

    # Draw white "1" inside the circle
    font_scale = r / 12.0
    thickness = max(int(r / 6), 1)

    # "1" text centered in circle
    text_size = cv2.getTextSize("1", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(result, "1", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # "EDITION" text arced above the circle
    # For simplicity, draw it as straight text above the circle
    ed_scale = font_scale * 0.35
    ed_thick = max(int(thickness * 0.5), 1)
    ed_size = cv2.getTextSize("EDITION", cv2.FONT_HERSHEY_SIMPLEX, ed_scale, ed_thick)[0]
    ex = cx - ed_size[0] // 2
    ey = cy - r - 2  # just above the circle
    cv2.putText(result, "EDITION", (ex, ey), cv2.FONT_HERSHEY_SIMPLEX,
                ed_scale, (0, 0, 0), ed_thick, cv2.LINE_AA)

    return result


def synthesize_from_template(card_bgr: np.ndarray) -> np.ndarray:
    """Overlay the actual Bulbapedia 1st edition stamp template onto a card.

    Uses the downloaded stamp PNG (black on white, high-res) scaled down
    and alpha-blended onto the correct position.
    """
    result = card_bgr.copy()
    h, w = result.shape[:2]

    stamp_path = STAMP_DIR / "1st_edition_english_stamp.png"
    if not stamp_path.exists():
        print(f"  Warning: stamp template not found at {stamp_path}, using synthesized")
        return synthesize_1st_edition_stamp(card_bgr)

    # Load stamp template
    stamp = cv2.imread(str(stamp_path), cv2.IMREAD_UNCHANGED)
    if stamp is None:
        return synthesize_1st_edition_stamp(card_bgr)

    # Convert to grayscale if needed
    if len(stamp.shape) == 3:
        stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    else:
        stamp_gray = stamp

    # Target stamp height: about 10% of card height (matches real stamp proportions)
    target_h = max(int(h * 0.10), 10)
    scale = target_h / stamp_gray.shape[0]
    target_w = int(stamp_gray.shape[1] * scale)
    stamp_resized = cv2.resize(stamp_gray, (target_w, target_h),
                               interpolation=cv2.INTER_AREA)

    # Create alpha mask: stamp is black on white background
    # Black pixels (< 128) are the stamp, white pixels are background
    _, alpha = cv2.threshold(stamp_resized, 128, 255, cv2.THRESH_BINARY_INV)

    # Position: center at x=8%, y=58% of card
    cx = int(w * 0.08)
    cy = int(h * 0.58)
    x1 = cx - target_w // 2
    y1 = cy - target_h // 2
    x2 = x1 + target_w
    y2 = y1 + target_h

    # Clip to card bounds
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return synthesize_1st_edition_stamp(card_bgr)

    # Alpha blend: where stamp is black, paint black on card
    alpha_crop = alpha[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
    for c in range(3):
        result[y1:y2, x1:x2, c] = (
            result[y1:y2, x1:x2, c] * (1 - alpha_crop)
        ).astype(np.uint8)

    return result


def compute_dino_differential(scan_bgr: np.ndarray,
                               ref_bgr: np.ndarray) -> dict:
    """Compute DINOv2 differential between scan and reference.

    Returns dict with stamp_sim, ctrl_sim, differential, and detection result.
    """
    from cardprice.ml.stamp_detection import (
        _extract_region as extract_region,
        _dino_crop_similarity_batch,
    )

    # Crop stamp region from both
    scan_stamp = extract_region(scan_bgr, *STAMP_REGION)
    ref_stamp = extract_region(ref_bgr, *STAMP_REGION)

    # Crop control region from both
    scan_ctrl = extract_region(scan_bgr, *CONTROL_REGION)
    ref_ctrl = extract_region(ref_bgr, *CONTROL_REGION)

    if (scan_stamp.size == 0 or ref_stamp.size == 0 or
            scan_ctrl.size == 0 or ref_ctrl.size == 0):
        return {"error": "empty region"}

    # Compute similarities for both regions
    stamp_sim, ctrl_sim = _dino_crop_similarity_batch(
        [scan_stamp, scan_ctrl],
        [ref_stamp, ref_ctrl],
    )

    diff = ctrl_sim - stamp_sim

    return {
        "stamp_sim": stamp_sim,
        "ctrl_sim": ctrl_sim,
        "differential": diff,
    }


def test_on_reference_images(n_per_set: int = 5):
    """Test the detector on reference images.

    For each card:
    - Original ref vs itself => diff ~0 (negative: no stamp)
    - Synthesized 1st ed vs original ref => diff > 0 (positive: has stamp)
    """
    print("=" * 70)
    print("DINOv2 1st Edition Stamp Detector - Calibration")
    print("=" * 70)

    unlimited_diffs = []
    stamped_diffs = []
    results_table = []

    for set_id in FIRST_EDITION_SETS:
        set_dir = REF_DIR / set_id
        if not set_dir.exists():
            continue

        images = sorted(set_dir.glob("*.png"))[:n_per_set]
        if not images:
            continue

        print(f"\n--- {set_id} ({len(images)} cards) ---")

        for img_path in images:
            ref_bgr = cv2.imread(str(img_path))
            if ref_bgr is None:
                continue

            card_name = img_path.stem

            # Test 1: Unlimited (ref vs itself) => should give diff ~0
            # Simulate slight variation by adding noise
            noisy = ref_bgr.copy()
            noise = np.random.normal(0, 3, noisy.shape).astype(np.int16)
            noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            result_unl = compute_dino_differential(noisy, ref_bgr)
            if "error" in result_unl:
                print(f"  {card_name}: ERROR - {result_unl['error']}")
                continue

            unlimited_diffs.append(result_unl["differential"])

            # Test 2: Synthesized 1st edition vs ref => should give high diff
            stamped = synthesize_from_template(ref_bgr)
            result_1st = compute_dino_differential(stamped, ref_bgr)
            if "error" in result_1st:
                continue

            stamped_diffs.append(result_1st["differential"])

            results_table.append({
                "card": card_name,
                "unl_diff": result_unl["differential"],
                "1st_diff": result_1st["differential"],
                "unl_stamp_sim": result_unl["stamp_sim"],
                "unl_ctrl_sim": result_unl["ctrl_sim"],
                "1st_stamp_sim": result_1st["stamp_sim"],
                "1st_ctrl_sim": result_1st["ctrl_sim"],
            })

            print(f"  {card_name:30s}  unl_diff={result_unl['differential']:+.4f}  "
                  f"1st_diff={result_1st['differential']:+.4f}  "
                  f"(stamp_sim: {result_unl['stamp_sim']:.4f}->{result_1st['stamp_sim']:.4f})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if unlimited_diffs:
        unl_arr = np.array(unlimited_diffs)
        print(f"\nUnlimited (negative) differentials:")
        print(f"  Mean: {unl_arr.mean():.4f}")
        print(f"  Std:  {unl_arr.std():.4f}")
        print(f"  Min:  {unl_arr.min():.4f}")
        print(f"  Max:  {unl_arr.max():.4f}")

    if stamped_diffs:
        st_arr = np.array(stamped_diffs)
        print(f"\n1st Edition (positive) differentials:")
        print(f"  Mean: {st_arr.mean():.4f}")
        print(f"  Std:  {st_arr.std():.4f}")
        print(f"  Min:  {st_arr.min():.4f}")
        print(f"  Max:  {st_arr.max():.4f}")

    if unlimited_diffs and stamped_diffs:
        gap = st_arr.min() - unl_arr.max()
        midpoint = (st_arr.min() + unl_arr.max()) / 2
        print(f"\nGap: {gap:.4f}")
        print(f"Suggested threshold: {midpoint:.4f}")

        if gap > 0:
            print(f"  Clean separation! All 1st ed > all unlimited.")
        else:
            print(f"  WARNING: Overlap detected. Need better approach.")

        # Test accuracy at various thresholds
        print(f"\nAccuracy at various thresholds:")
        for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
            tp = sum(1 for d in stamped_diffs if d > thresh)
            tn = sum(1 for d in unlimited_diffs if d <= thresh)
            fp = len(unlimited_diffs) - tn
            fn = len(stamped_diffs) - tp
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            print(f"  threshold={thresh:.2f}: TP={tp} TN={tn} FP={fp} FN={fn} "
                  f"acc={acc:.1%}")

    # Save synthesized examples for visual inspection
    print("\nSaving sample synthesized images for inspection...")
    out_dir = STAMP_DIR / "samples"
    out_dir.mkdir(exist_ok=True)

    sample_cards = [
        REF_DIR / "base1" / "base1-4_normal.png",   # Charizard
        REF_DIR / "base1" / "base1-15_normal.png",   # Venusaur
        REF_DIR / "base2" / "base2-10_normal.png",
        REF_DIR / "gym1" / "gym1-1_normal.png",
    ]

    for card_path in sample_cards:
        if not card_path.exists():
            continue
        ref = cv2.imread(str(card_path))
        if ref is None:
            continue

        stamped = synthesize_from_template(ref)
        out_path = out_dir / f"{card_path.stem}_1st_ed.png"
        cv2.imwrite(str(out_path), stamped)

        # Also save the stamp region crop for inspection
        stamp_crop = _extract_region(stamped, *STAMP_REGION)
        ref_crop = _extract_region(ref, *STAMP_REGION)
        combined = np.hstack([ref_crop, stamp_crop])
        crop_path = out_dir / f"{card_path.stem}_stamp_compare.png"
        cv2.imwrite(str(crop_path), combined)
        print(f"  Saved: {out_path.name} + {crop_path.name}")


def test_on_binder_scans():
    """Test on actual binder scan card crops (all should be unlimited)."""
    print("\n" + "=" * 70)
    print("Testing on binder scan card crops (should all be unlimited)")
    print("=" * 70)

    import json

    gt_path = DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("  No ground truth file found, skipping binder test")
        return

    with open(gt_path) as f:
        gt = json.load(f)

    from cardprice.ml.ref_matcher import get_reference_image_path

    diffs = []
    for page_key, cards in gt.items():
        if page_key in ("description", "notes"):
            continue
        if not isinstance(cards, list):
            continue

        for i, card_entry in enumerate(cards):
            if not isinstance(card_entry, dict):
                continue
            card_id = card_entry.get("card_id")
            if not card_id:
                continue

            # Only check WotC era cards
            set_id = card_id.split("-")[0]
            if set_id not in FIRST_EDITION_SETS:
                continue

            # Find the scan image
            scan_dir = DATA_DIR / "inbox" / f"{page_key}_cards"
            scan_path = scan_dir / f"card_{i:02d}.png"
            if not scan_path.exists():
                continue

            # Find reference image
            ref_path = get_reference_image_path(card_id)
            if ref_path is None:
                continue

            scan_bgr = cv2.imread(str(scan_path))
            ref_bgr = cv2.imread(str(ref_path))
            if scan_bgr is None or ref_bgr is None:
                continue

            result = compute_dino_differential(scan_bgr, ref_bgr)
            if "error" in result:
                continue

            diffs.append(result["differential"])
            card_name = card_entry.get("name", card_id)
            print(f"  {card_name:30s} diff={result['differential']:+.4f} "
                  f"stamp_sim={result['stamp_sim']:.4f} ctrl_sim={result['ctrl_sim']:.4f}")

    if diffs:
        arr = np.array(diffs)
        print(f"\nBinder scan differentials (all unlimited):")
        print(f"  Mean: {arr.mean():.4f}")
        print(f"  Max:  {arr.max():.4f}")
        print(f"  Min:  {arr.min():.4f}")


if __name__ == "__main__":
    test_on_reference_images(n_per_set=5)
    test_on_binder_scans()
