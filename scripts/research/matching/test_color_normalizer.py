#!/usr/bin/env python3
"""Test color normalization techniques on ground truth cards.

Measures DINOv2 cosine similarity between binder-scanned card images and their
reference images, before and after each normalization technique.

Usage:
    python scripts/test_color_normalizer.py [--page PAGE_ID] [--verbose]

Examples:
    # Test all ground truth pages
    python scripts/test_color_normalizer.py

    # Test only the Delta Species page
    python scripts/test_color_normalizer.py --page page_20260305_094228_cards

    # Verbose output per-card
    python scripts/test_color_normalizer.py --verbose
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.ml.color_normalizer import (
    apply_clahe,
    gray_world_white_balance,
    match_histogram,
    normalize_card_colors,
    normalize_with_techniques,
    reduce_sleeve_reflections,
)


def load_ground_truth():
    """Load ground truth data."""
    gt_path = PROJECT_ROOT / "data" / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def find_reference_image(card_id: str) -> Path | None:
    """Find the reference image for a card_id like 'neo1-53/normal'."""
    # card_id format: "setid-num/variant"
    parts = card_id.split("/")
    if len(parts) != 2:
        return None
    base_id, variant = parts[0], parts[1]

    # Try set_id/card_id_variant.png pattern
    set_parts = base_id.rsplit("-", 1)
    if len(set_parts) != 2:
        return None
    set_id = set_parts[0]

    ref_dir = PROJECT_ROOT / "data" / "card_images" / set_id
    if not ref_dir.exists():
        return None

    # Try: base_id_variant.png (e.g., neo1-53_normal.png)
    ref_path = ref_dir / f"{base_id}_{variant}.png"
    if ref_path.exists():
        return ref_path

    # Try without variant
    ref_path = ref_dir / f"{base_id}.png"
    if ref_path.exists():
        return ref_path

    # Try .jpg
    for ext in (".jpg", ".jpeg", ".webp"):
        ref_path = ref_dir / f"{base_id}_{variant}{ext}"
        if ref_path.exists():
            return ref_path

    return None


def compute_dino_similarity(img1_path, img2_path=None, img2_array=None,
                             embedding_cache=None):
    """Compute DINOv2 cosine similarity between two images.

    Can accept either a path or a numpy array for img2.
    Uses embedding_cache dict to avoid recomputing reference embeddings.
    """
    import tempfile
    import os
    from cardprice.ml.dino_matcher import extract_embedding

    # Get embedding for img1 (scan)
    if isinstance(img1_path, np.ndarray):
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp, img1_path)
        try:
            emb1 = extract_embedding(tmp)
        finally:
            os.unlink(tmp)
    else:
        emb1 = extract_embedding(str(img1_path))

    # Get embedding for img2 (reference)
    if img2_array is not None:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp, img2_array)
        try:
            emb2 = extract_embedding(tmp)
        finally:
            os.unlink(tmp)
    elif img2_path is not None:
        cache_key = str(img2_path)
        if embedding_cache is not None and cache_key in embedding_cache:
            emb2 = embedding_cache[cache_key]
        else:
            emb2 = extract_embedding(str(img2_path))
            if embedding_cache is not None:
                embedding_cache[cache_key] = emb2
    else:
        raise ValueError("Must provide img2_path or img2_array")

    return float(np.dot(emb1, emb2))


def compute_similarity_from_array(card_img: np.ndarray, ref_embedding: np.ndarray):
    """Compute DINOv2 similarity between a numpy image and a pre-computed embedding."""
    import tempfile
    import os
    from cardprice.ml.dino_matcher import extract_embedding

    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(tmp, card_img)
    try:
        emb = extract_embedding(tmp)
    finally:
        os.unlink(tmp)

    return float(np.dot(emb, ref_embedding))


def run_test(page_filter=None, verbose=False):
    """Run normalization tests on ground truth cards."""
    gt = load_ground_truth()
    pages = gt["pages"]

    # Technique configurations to test
    techniques = {
        "original":          dict(use_clahe=False, use_gray_world=False,
                                  use_hist_match=False, use_reflection_reduction=False),
        "clahe_only":        dict(use_clahe=True, use_gray_world=False,
                                  use_hist_match=False, use_reflection_reduction=False),
        "gray_world_only":   dict(use_clahe=False, use_gray_world=True,
                                  use_hist_match=False, use_reflection_reduction=False),
        "reflection_only":   dict(use_clahe=False, use_gray_world=False,
                                  use_hist_match=False, use_reflection_reduction=True),
        "hist_match_only":   dict(use_clahe=False, use_gray_world=False,
                                  use_hist_match=True, use_reflection_reduction=False),
        "full_no_histmatch": dict(use_clahe=True, use_gray_world=True,
                                  use_hist_match=False, use_reflection_reduction=True),
        "full_pipeline":     dict(use_clahe=True, use_gray_world=True,
                                  use_hist_match=True, use_reflection_reduction=True),
    }

    # Collect results
    results = {name: [] for name in techniques}
    ref_embedding_cache = {}
    cards_tested = 0
    cards_skipped = 0

    # Pre-load ref embeddings
    from cardprice.ml.ref_matcher import _load_ref_embeddings
    precomputed = _load_ref_embeddings()

    from cardprice.ml.dino_matcher import extract_embedding

    for page_id, page_data in pages.items():
        if page_filter and page_id != page_filter:
            continue

        desc = page_data.get("description", "")
        print(f"\n{'='*70}")
        print(f"Page: {page_id}")
        print(f"  {desc}")
        print(f"{'='*70}")

        for card_key in sorted(page_data.keys()):
            if not card_key.startswith("card_"):
                continue

            card_info = page_data[card_key]
            card_id = card_info.get("card_id", "")
            card_name = card_info.get("name", "unknown")

            # Find scan image
            scan_dir = PROJECT_ROOT / "data" / "inbox" / page_id
            scan_path = scan_dir / f"{card_key}.png"
            if not scan_path.exists():
                if verbose:
                    print(f"  {card_key}: SKIP (scan not found)")
                cards_skipped += 1
                continue

            # Find reference image
            ref_path = find_reference_image(card_id)
            if ref_path is None:
                if verbose:
                    print(f"  {card_key} ({card_name}): SKIP (ref not found for {card_id})")
                cards_skipped += 1
                continue

            # Get reference embedding (from precomputed or extract)
            ref_emb = precomputed.get(card_id)
            if ref_emb is None:
                try:
                    ref_emb = extract_embedding(str(ref_path))
                except Exception as e:
                    if verbose:
                        print(f"  {card_key}: SKIP (ref embedding failed: {e})")
                    cards_skipped += 1
                    continue

            # Load scan image
            scan_img = cv2.imread(str(scan_path))
            if scan_img is None:
                cards_skipped += 1
                continue

            # Load reference image (for histogram matching)
            ref_img = cv2.imread(str(ref_path))

            cards_tested += 1

            if verbose:
                print(f"\n  {card_key}: {card_name} ({card_id})")

            for tech_name, tech_config in techniques.items():
                # Determine if we need reference for this config
                needs_ref = tech_config.get("use_hist_match", False)
                ref_for_tech = ref_img if needs_ref else None

                # Apply normalization
                if tech_name == "original":
                    normalized = scan_img
                else:
                    normalized = normalize_with_techniques(
                        scan_img,
                        reference_img=ref_for_tech,
                        **tech_config,
                    )

                # Compute similarity
                sim = compute_similarity_from_array(normalized, ref_emb)
                results[tech_name].append(sim)

                if verbose:
                    print(f"    {tech_name:25s}: {sim:.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {cards_tested} cards tested, {cards_skipped} skipped")
    print(f"{'='*70}")
    print(f"\n{'Technique':25s} {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'Delta':>8s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    baseline_mean = np.mean(results["original"]) if results["original"] else 0

    for tech_name in techniques:
        scores = results[tech_name]
        if not scores:
            continue
        mean = np.mean(scores)
        median = np.median(scores)
        mn = np.min(scores)
        mx = np.max(scores)
        delta = mean - baseline_mean

        marker = ""
        if delta > 0.005:
            marker = " +"
        elif delta < -0.005:
            marker = " -"

        print(
            f"{tech_name:25s} {mean:8.4f} {median:8.4f} {mn:8.4f} {mx:8.4f} "
            f"{delta:+8.4f}{marker}"
        )

    # Per-card improvement analysis
    if results["original"] and results["full_no_histmatch"]:
        orig = np.array(results["original"])
        full = np.array(results["full_no_histmatch"])
        improved = np.sum(full > orig)
        degraded = np.sum(full < orig)
        unchanged = np.sum(full == orig)
        print(f"\nFull pipeline (no hist match) vs original:")
        print(f"  Improved: {improved}/{len(orig)} cards")
        print(f"  Degraded: {degraded}/{len(orig)} cards")
        print(f"  Unchanged: {unchanged}/{len(orig)} cards")
        print(f"  Avg improvement on improved: {np.mean((full - orig)[full > orig]):+.4f}" if improved else "")
        print(f"  Avg degradation on degraded: {np.mean((full - orig)[full < orig]):+.4f}" if degraded else "")

    if results["original"] and results["full_pipeline"]:
        orig = np.array(results["original"])
        full = np.array(results["full_pipeline"])
        improved = np.sum(full > orig)
        degraded = np.sum(full < orig)
        print(f"\nFull pipeline (with hist match) vs original:")
        print(f"  Improved: {improved}/{len(orig)} cards")
        print(f"  Degraded: {degraded}/{len(orig)} cards")


def main():
    parser = argparse.ArgumentParser(description="Test color normalization techniques")
    parser.add_argument("--page", type=str, default=None,
                        help="Filter to specific page ID")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-card results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    start = time.time()
    run_test(page_filter=args.page, verbose=args.verbose)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
