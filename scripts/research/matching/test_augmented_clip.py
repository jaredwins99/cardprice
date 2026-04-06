#!/usr/bin/env python3
"""Test augmented CLIP index vs standard index on binder page segments.

Builds an augmented index from cards in sets that are relevant to the
test images (based on standard index top matches), then compares matching
accuracy on real phone photos.

Usage:
    python scripts/test_augmented_clip.py
"""

import logging
import pickle
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("test_augmented_clip")

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "card_images"
STANDARD_INDEX = DATA_DIR / "clip_image_index.pkl"
AUGMENTED_INDEX = DATA_DIR / "clip_augmented_index.pkl"
TEST_DIR = DATA_DIR / "inbox" / "page_20260228_195512_cards"


def build_full_augmented_index():
    """Build augmented index from ALL card images."""
    from cardprice.ml.clip_matcher import build_augmented_image_index

    logger.info("=== Building FULL augmented index ===")
    t0 = time.time()
    path = build_augmented_image_index(
        image_dir=str(IMAGE_DIR),
        output_path=str(AUGMENTED_INDEX),
        max_cards=0,  # all cards
        num_augmentations=5,
    )
    elapsed = time.time() - t0
    logger.info("Built in %.1f seconds -> %s", elapsed, path)
    return path


def run_comparison():
    """Compare standard vs augmented index on test images."""
    from cardprice.ml.clip_matcher import identify_card_by_image

    test_images = sorted(TEST_DIR.glob("*.png"))
    if not test_images:
        test_images = sorted(TEST_DIR.glob("*.jpg"))
    if not test_images:
        logger.error("No test images found in %s", TEST_DIR)
        return

    logger.info("=== Comparing on %d test images ===", len(test_images))

    # Load both indexes
    with open(STANDARD_INDEX, "rb") as f:
        std_idx = pickle.load(f)
    logger.info("Standard index: %d cards", len(std_idx["card_ids"]))

    with open(AUGMENTED_INDEX, "rb") as f:
        aug_idx = pickle.load(f)
    logger.info("Augmented index: %d cards (augmented=%s)",
                len(aug_idx["card_ids"]), aug_idx.get("augmented", False))

    # Run each test image against both indexes
    print("\n" + "=" * 110)
    print(f"{'Image':<15} | {'Standard Top-1':<35} {'Score':>6} | {'Augmented Top-1':<35} {'Score':>6} | {'Delta':>7}")
    print("=" * 110)

    std_scores = []
    aug_scores = []
    agreements = 0

    for img_path in test_images:
        # Standard index
        std_results = identify_card_by_image(
            str(img_path), preloaded_index=std_idx, top_k=5
        )
        std_top1 = std_results[0] if std_results else ("N/A", 0.0)

        # Augmented index
        aug_results = identify_card_by_image(
            str(img_path), preloaded_index=aug_idx, top_k=5
        )
        aug_top1 = aug_results[0] if aug_results else ("N/A", 0.0)

        std_scores.append(std_top1[1])
        aug_scores.append(aug_top1[1])

        same = std_top1[0] == aug_top1[0]
        if same:
            agreements += 1

        delta = aug_top1[1] - std_top1[1]
        delta_str = f"{delta:+.4f}"
        marker = " +AUG" if delta > 0.005 else (" -AUG" if delta < -0.005 else " ==")
        agree_str = "SAME" if same else "DIFF"

        print(f"{img_path.name:<15} | {std_top1[0]:<35} {std_top1[1]:>6.4f} | {aug_top1[0]:<35} {aug_top1[1]:>6.4f} | {delta_str} {marker} [{agree_str}]")

        # Print top-3 for each
        for rank in range(1, min(3, len(std_results), len(aug_results))):
            s = std_results[rank]
            a = aug_results[rank]
            print(f"{'':>15}   {s[0]:<35} {s[1]:>6.4f}   {a[0]:<35} {a[1]:>6.4f}")

    print("=" * 110)

    # Summary stats
    import numpy as np
    std_arr = np.array(std_scores)
    aug_arr = np.array(aug_scores)
    delta_arr = aug_arr - std_arr

    print(f"\nSummary (over {len(test_images)} test images):")
    print(f"  Standard  -- mean: {std_arr.mean():.4f}, median: {np.median(std_arr):.4f}, min: {std_arr.min():.4f}, max: {std_arr.max():.4f}")
    print(f"  Augmented -- mean: {aug_arr.mean():.4f}, median: {np.median(aug_arr):.4f}, min: {aug_arr.min():.4f}, max: {aug_arr.max():.4f}")
    print(f"  Delta     -- mean: {delta_arr.mean():+.4f}, median: {np.median(delta_arr):+.4f}")
    print(f"  Top-1 agreement: {agreements}/{len(test_images)}")
    print(f"  Images where augmented scores higher: {(delta_arr > 0.001).sum()}/{len(delta_arr)}")
    print(f"  Images where standard scores higher:  {(delta_arr < -0.001).sum()}/{len(delta_arr)}")

    # Threshold analysis
    for thresh in [0.70, 0.75, 0.80, 0.85]:
        std_above = (std_arr >= thresh).sum()
        aug_above = (aug_arr >= thresh).sum()
        print(f"  Above {thresh:.2f} -- standard: {std_above}/{len(std_arr)}, augmented: {aug_above}/{len(aug_arr)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true",
                        help="Build the full augmented index (takes hours)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only run comparison (index must already exist)")
    args = parser.parse_args()

    if args.compare_only:
        if not AUGMENTED_INDEX.exists():
            logger.error("Augmented index not found at %s. Run with --build first.", AUGMENTED_INDEX)
            sys.exit(1)
        run_comparison()
    elif args.build:
        build_full_augmented_index()
        if STANDARD_INDEX.exists():
            run_comparison()
    else:
        # Default: build if needed, then compare
        if not AUGMENTED_INDEX.exists():
            build_full_augmented_index()
        run_comparison()
