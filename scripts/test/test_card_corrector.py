#!/usr/bin/env python3
"""Test card_corrector by comparing DINOv2 similarity before/after correction.

Uses ground truth cards to measure whether correction improves embedding
similarity against reference images.
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cardprice.ml.card_corrector import (
    correct_card_image, _apply_clahe, _white_balance,
    CORRECTED_W, CORRECTED_H,
)
from cardprice.ml.ref_matcher import get_reference_image_path
from cardprice.ml.dino_matcher import extract_embedding


def embed_image(img):
    """Extract DINOv2 embedding from a BGR numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)
    emb = extract_embedding(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return emb


def main():
    project_root = Path(__file__).resolve().parent.parent
    gt_path = project_root / "data" / "ground_truth.json"

    with open(gt_path) as f:
        gt = json.load(f)

    # Collect test cases
    test_cases = []
    for page_name, page_data in gt["pages"].items():
        page_dir = project_root / "data" / "inbox" / page_name
        if not page_dir.exists():
            continue
        for key, card_info in page_data.items():
            if not key.startswith("card_"):
                continue
            card_id = card_info.get("card_id")
            if not card_id:
                continue
            card_path = page_dir / f"{key}.png"
            if not card_path.exists():
                continue
            ref_path = get_reference_image_path(card_id)
            if ref_path is None:
                continue
            test_cases.append((card_path, card_id, ref_path))

    print(f"Found {len(test_cases)} test cases with ground truth + reference images")

    # Use all cards for a thorough evaluation
    test_subset = test_cases
    print(f"Testing on {len(test_subset)} cards\n")

    # Pre-compute reference embeddings
    print("Computing reference embeddings...")
    ref_embeddings = {}
    for _, card_id, ref_path in test_subset:
        if card_id not in ref_embeddings:
            ref_embeddings[card_id] = extract_embedding(ref_path)

    # Test configurations
    configs = [
        ("original (resized)", lambda img: cv2.resize(img, (CORRECTED_W, CORRECTED_H), interpolation=cv2.INTER_LANCZOS4)),
        ("WB only", lambda img: _white_balance(cv2.resize(img, (CORRECTED_W, CORRECTED_H), interpolation=cv2.INTER_LANCZOS4))),
        ("CLAHE(1.5,4x4)", lambda img: _apply_clahe(cv2.resize(img, (CORRECTED_W, CORRECTED_H), interpolation=cv2.INTER_LANCZOS4), 1.5, (4, 4))),
        ("CLAHE(1.5,4x4)+WB", lambda img: _white_balance(_apply_clahe(cv2.resize(img, (CORRECTED_W, CORRECTED_H), interpolation=cv2.INTER_LANCZOS4), 1.5, (4, 4)))),
        ("WB+CLAHE(1.5,4x4)", lambda img: _apply_clahe(_white_balance(cv2.resize(img, (CORRECTED_W, CORRECTED_H), interpolation=cv2.INTER_LANCZOS4)), 1.5, (4, 4))),
        ("full correction", lambda img: correct_card_image(img)),
    ]

    results = {name: [] for name, _ in configs}

    for i, (card_path, card_id, ref_path) in enumerate(test_subset):
        card_img = cv2.imread(str(card_path))
        if card_img is None:
            continue

        ref_emb = ref_embeddings[card_id]

        for config_name, transform_fn in configs:
            corrected = transform_fn(card_img)
            emb = embed_image(corrected)
            sim = float(np.dot(emb, ref_emb))
            results[config_name].append(sim)

        orig_sim = results["original (resized)"][-1]
        full_sim = results["full correction"][-1]
        delta = full_sim - orig_sim
        card_name = f"{card_path.parent.name}/{card_path.name}"
        marker = " +++" if delta > 0.02 else (" ---" if delta < -0.02 else "")
        if i < 30 or abs(delta) > 0.02:
            print(f"  [{i:2d}] {card_name:50s} orig={orig_sim:.4f} full={full_sim:.4f} d={delta:+.4f}{marker}")

    # Summary
    print("\n" + "=" * 85)
    print(f"{'Configuration':<25s} {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'Delta':>8s}")
    print("-" * 85)

    baseline_mean = np.mean(results["original (resized)"])
    for config_name, _ in configs:
        scores = results[config_name]
        mean_s = np.mean(scores)
        median_s = np.median(scores)
        min_s = np.min(scores)
        max_s = np.max(scores)
        delta = mean_s - baseline_mean
        print(f"{config_name:<25s} {mean_s:8.4f} {median_s:8.4f} {min_s:8.4f} {max_s:8.4f} {delta:+8.4f}")

    print("=" * 85)

    # Detailed comparison: full correction vs original
    full_scores = results["full correction"]
    orig_scores = results["original (resized)"]
    deltas = [f - o for f, o in zip(full_scores, orig_scores)]
    improved = sum(1 for d in deltas if d > 0.005)
    degraded = sum(1 for d in deltas if d < -0.005)
    neutral = len(deltas) - improved - degraded
    print(f"\nFull correction vs original:")
    print(f"  {improved} improved (>{'+'}0.005), {degraded} degraded (<-0.005), {neutral} neutral")
    print(f"  Mean delta: {np.mean(deltas):+.4f}")
    print(f"  Median delta: {np.median(deltas):+.4f}")

    # Find best overall config
    best_config = max(configs, key=lambda c: np.mean(results[c[0]]))
    print(f"\nBest config: {best_config[0]} (mean={np.mean(results[best_config[0]]):.4f})")


if __name__ == "__main__":
    main()
