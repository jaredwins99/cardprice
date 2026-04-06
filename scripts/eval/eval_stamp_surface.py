#!/usr/bin/env python3
"""Evaluate whether DINOv2 surface detector can distinguish stamped from clean cards.

For each labeled stamp image, compares it against the clean reference of the same
card using DINOv2 patch-level comparison (surface_detector.py).  Reports whether
the stamp region shows up as anomalous patches, and whether this label-free approach
can distinguish stamped from clean cards without any training.

The idea: if a stamped card is compared to a clean reference, the stamp region
(bottom-right of artwork, ~rows 5-8, cols 8-14 in the 16x16 patch grid) should
show lower cosine similarity than the rest of the card.

Usage:
    python scripts/eval/eval_stamp_surface.py
    python scripts/eval/eval_stamp_surface.py --dir data/condition_training/stamps/
    python scripts/eval/eval_stamp_surface.py --threshold 0.90
    python scripts/eval/eval_stamp_surface.py --verbose
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "condition_training", "stamps",
)

# The stamp sits at approximately x: 55-88%, y: 35-55% of card dimensions.
# On a 16x16 patch grid, that maps to:
#   cols: 0.55*16=8.8 to 0.88*16=14.1 -> cols 9-14
#   rows: 0.35*16=5.6 to 0.55*16=8.8  -> rows 6-8
# We use a slightly generous region to account for alignment variance.
STAMP_ROWS = (5, 9)   # rows 5-8 inclusive (0-indexed)
STAMP_COLS = (8, 15)   # cols 8-14 inclusive


def load_labels(label_path: str) -> list[dict]:
    """Load labels.jsonl.

    Expected format per line:
        {"image": "ex11_stamped_001.jpg", "stamped": true, "set_id": "ex11",
         "ref_image": "path/to/clean_reference.png"}

    The 'ref_image' field is the path to the clean reference image for comparison.
    Can be absolute or relative to the stamps directory.
    """
    labels = []
    with open(label_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {lineno}: {e}")
                continue
            if "image" not in obj or "stamped" not in obj:
                print(f"WARNING: Skipping line {lineno}: missing 'image' or 'stamped'")
                continue
            labels.append(obj)
    return labels


def get_stamp_region_stats(anomaly_map: np.ndarray) -> dict:
    """Extract statistics for the stamp region vs the rest of the card.

    Returns dict with:
        stamp_mean: mean similarity in the stamp region
        rest_mean: mean similarity outside the stamp region
        stamp_min: minimum similarity in the stamp region
        delta: rest_mean - stamp_mean (positive = stamp region is more anomalous)
        stamp_defect_ratio: fraction of stamp-region patches below threshold
    """
    stamp_patch = anomaly_map[STAMP_ROWS[0]:STAMP_ROWS[1],
                              STAMP_COLS[0]:STAMP_COLS[1]]

    # Create mask for everything outside stamp region
    mask = np.ones_like(anomaly_map, dtype=bool)
    mask[STAMP_ROWS[0]:STAMP_ROWS[1], STAMP_COLS[0]:STAMP_COLS[1]] = False
    rest_patches = anomaly_map[mask]

    stamp_mean = float(np.mean(stamp_patch))
    rest_mean = float(np.mean(rest_patches))
    stamp_min = float(np.min(stamp_patch))

    return {
        "stamp_mean": stamp_mean,
        "rest_mean": rest_mean,
        "stamp_min": stamp_min,
        "delta": rest_mean - stamp_mean,
    }


def run_eval(labels: list[dict], img_dir: str, patch_threshold: float,
             verbose: bool) -> dict:
    """Run surface detection on all labeled images."""
    from cardprice.ml.surface_detector import (
        extract_patch_tokens,
        compare_patches,
        detect_surface_defects,
    )

    results = []
    skipped = 0

    for entry in labels:
        img_path = os.path.join(img_dir, entry["image"])
        if not os.path.exists(img_path):
            if verbose:
                print(f"  SKIP {entry['image']} (file not found)")
            skipped += 1
            continue

        ref_path = entry.get("ref_image", "")
        if ref_path and not os.path.isabs(ref_path):
            ref_path = os.path.join(img_dir, ref_path)

        if not ref_path or not os.path.exists(ref_path):
            if verbose:
                print(f"  SKIP {entry['image']} (no ref_image or ref not found)")
            skipped += 1
            continue

        gt_stamped = bool(entry["stamped"])
        set_id = entry.get("set_id", "")

        t0 = time.time()
        try:
            query_patches = extract_patch_tokens(img_path)
            ref_patches = extract_patch_tokens(ref_path)
            anomaly_map = compare_patches(query_patches, ref_patches)

            # Full-card defect detection
            defect_result = detect_surface_defects(
                img_path, ref_path,
                patch_threshold=patch_threshold,
                query_patches=query_patches,
                ref_patches=ref_patches,
            )

            # Stamp-region specific analysis
            stamp_stats = get_stamp_region_stats(anomaly_map)

            # Decision: stamp detected if the stamp region is significantly
            # more anomalous than the rest of the card
            # Use two signals:
            #   1. Delta (rest_mean - stamp_mean) > threshold => stamp region
            #      has noticeably lower similarity
            #   2. Stamp region has a higher defect ratio than expected
            stamp_defect_patches = anomaly_map[
                STAMP_ROWS[0]:STAMP_ROWS[1],
                STAMP_COLS[0]:STAMP_COLS[1]
            ]
            stamp_n = stamp_defect_patches.size
            stamp_defects = int(np.sum(stamp_defect_patches < patch_threshold))
            stamp_defect_ratio = stamp_defects / stamp_n if stamp_n > 0 else 0.0

            # Predict stamped if:
            #   - The stamp region has meaningfully lower similarity than rest
            #     (delta > 0.02)
            #   - OR more than 30% of stamp-region patches are flagged
            pred_stamped = (stamp_stats["delta"] > 0.02
                           or stamp_defect_ratio > 0.30)

        except Exception as e:
            if verbose:
                print(f"  ERROR {entry['image']}: {e}")
            pred_stamped = False
            stamp_stats = {"stamp_mean": 0, "rest_mean": 0,
                          "stamp_min": 0, "delta": 0}
            stamp_defect_ratio = 0.0
            defect_result = {"defect_score": 0, "defect_ratio": 0,
                            "mean_similarity": 0}

        elapsed_ms = (time.time() - t0) * 1000
        correct = pred_stamped == gt_stamped

        result = {
            "image": entry["image"],
            "set_id": set_id,
            "gt_stamped": gt_stamped,
            "pred_stamped": pred_stamped,
            "correct": correct,
            "stamp_mean": stamp_stats["stamp_mean"],
            "rest_mean": stamp_stats["rest_mean"],
            "delta": stamp_stats["delta"],
            "stamp_min": stamp_stats["stamp_min"],
            "stamp_defect_ratio": stamp_defect_ratio,
            "overall_defect_score": defect_result["defect_score"],
            "overall_mean_sim": defect_result["mean_similarity"],
            "time_ms": elapsed_ms,
        }
        results.append(result)

        if verbose:
            status = "OK" if correct else "WRONG"
            gt_label = "stamped" if gt_stamped else "clean"
            pred_label = "stamped" if pred_stamped else "clean"
            print(f"  [{status}] {entry['image']:40s}  "
                  f"gt={gt_label:7s}  pred={pred_label:7s}  "
                  f"delta={stamp_stats['delta']:+.4f}  "
                  f"stamp_sim={stamp_stats['stamp_mean']:.3f}  "
                  f"rest_sim={stamp_stats['rest_mean']:.3f}  "
                  f"stamp_defect={stamp_defect_ratio:.2f}  "
                  f"{elapsed_ms:.0f}ms")

    return {"results": results, "skipped": skipped}


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, precision, recall, and confusion matrix."""
    if not results:
        return {"n": 0}

    n = len(results)
    correct = sum(1 for r in results if r["correct"])

    tp = sum(1 for r in results if r["gt_stamped"] and r["pred_stamped"])
    fp = sum(1 for r in results if not r["gt_stamped"] and r["pred_stamped"])
    fn = sum(1 for r in results if r["gt_stamped"] and not r["pred_stamped"])
    tn = sum(1 for r in results if not r["gt_stamped"] and not r["pred_stamped"])

    accuracy = correct / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    mean_time = sum(r["time_ms"] for r in results) / n

    return {
        "n": n, "correct": correct, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_time_ms": mean_time,
    }


def print_report(metrics: dict, per_set: dict[str, dict],
                 results: list[dict], skipped: int,
                 patch_threshold: float) -> None:
    """Print formatted evaluation report."""
    n = metrics["n"]
    if n == 0:
        print("\nNo results to report.")
        return

    print("\n" + "=" * 70)
    print("SURFACE DETECTOR STAMP EVALUATION (label-free DINOv2 approach)")
    print("=" * 70)
    print(f"\nPatch threshold:   {patch_threshold:.2f}")
    print(f"Stamp region:      rows {STAMP_ROWS[0]}-{STAMP_ROWS[1]-1}, "
          f"cols {STAMP_COLS[0]}-{STAMP_COLS[1]-1} (of 16x16 grid)")

    print(f"\nTotal images:      {n}")
    print(f"Skipped:           {skipped}")
    print(f"Accuracy:          {metrics['correct']}/{n} "
          f"({metrics['accuracy']:.1%})")
    print(f"Mean time:         {metrics['mean_time_ms']:.0f}ms per image")

    print(f"\nPrecision:         {metrics['precision']:.3f}")
    print(f"Recall:            {metrics['recall']:.3f}")
    print(f"F1:                {metrics['f1']:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Clean    Stamped")
    print(f"  Actual Clean    {metrics['tn']:5d}    {metrics['fp']:5d}")
    print(f"  Actual Stamped  {metrics['fn']:5d}    {metrics['tp']:5d}")

    # Distribution analysis: show delta values for stamped vs clean
    stamped_deltas = [r["delta"] for r in results if r["gt_stamped"]]
    clean_deltas = [r["delta"] for r in results if not r["gt_stamped"]]

    print(f"\nStamp Region Delta (rest_mean - stamp_mean) Distribution:")
    print(f"  Positive delta = stamp region is more anomalous than rest")
    if stamped_deltas:
        print(f"  Stamped cards:  mean={np.mean(stamped_deltas):+.4f}  "
              f"min={np.min(stamped_deltas):+.4f}  "
              f"max={np.max(stamped_deltas):+.4f}  "
              f"n={len(stamped_deltas)}")
    if clean_deltas:
        print(f"  Clean cards:    mean={np.mean(clean_deltas):+.4f}  "
              f"min={np.min(clean_deltas):+.4f}  "
              f"max={np.max(clean_deltas):+.4f}  "
              f"n={len(clean_deltas)}")

    # Separability analysis
    if stamped_deltas and clean_deltas:
        sep = np.min(stamped_deltas) - np.max(clean_deltas)
        if sep > 0:
            print(f"\n  Separable: gap of {sep:.4f} between worst stamped "
                  f"and best clean")
        else:
            print(f"\n  NOT separable: overlap of {abs(sep):.4f}")

    # Errors
    if metrics["fp"] > 0:
        print(f"\n  False positives (clean detected as stamped): {metrics['fp']}")
        for r in results:
            if not r["gt_stamped"] and r["pred_stamped"]:
                print(f"    - {r['image']} delta={r['delta']:+.4f} "
                      f"stamp_sim={r['stamp_mean']:.3f}")

    if metrics["fn"] > 0:
        print(f"\n  False negatives (stamped detected as clean): {metrics['fn']}")
        for r in results:
            if r["gt_stamped"] and not r["pred_stamped"]:
                print(f"    - {r['image']} delta={r['delta']:+.4f} "
                      f"stamp_sim={r['stamp_mean']:.3f}")

    # Per-set breakdown
    if len(per_set) > 1 or (len(per_set) == 1
                            and list(per_set.keys())[0] != ""):
        print(f"\n{'Set':<8s}  {'N':>4s}  {'Correct':>7s}  {'Acc':>6s}  "
              f"{'Prec':>6s}  {'Rec':>6s}  {'Avg Delta':>10s}")
        print("-" * 60)
        for set_id in sorted(per_set.keys()):
            m = per_set[set_id]
            if m["n"] == 0:
                continue
            set_results = [r for r in results if r["set_id"] == set_id]
            avg_delta = np.mean([r["delta"] for r in set_results])
            print(f"{set_id or '(none)':<8s}  {m['n']:4d}  "
                  f"{m['correct']:4d}/{m['n']:<3d}  "
                  f"{m['accuracy']:5.1%}  "
                  f"{m['precision']:5.3f}  "
                  f"{m['recall']:5.3f}  "
                  f"{avg_delta:+.4f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DINOv2 surface detector for stamp detection.")
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="Directory with images and labels.jsonl")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Patch similarity threshold (default: 0.85)")
    parser.add_argument("--set", dest="filter_set", default=None,
                        help="Evaluate only a single set (e.g. ex11)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each prediction")
    args = parser.parse_args()

    label_path = os.path.join(args.dir, "labels.jsonl")
    if not os.path.exists(label_path):
        print(f"ERROR: labels.jsonl not found at {label_path}")
        print(f"Expected format (one JSON per line):")
        print(f'  {{"image": "ex11_stamped_001.jpg", "stamped": true, '
              f'"set_id": "ex11", "ref_image": "refs/ex11-55_normal.png"}}')
        print(f'  {{"image": "ex11_clean_001.jpg", "stamped": false, '
              f'"set_id": "ex11", "ref_image": "refs/ex11-55_normal.png"}}')
        sys.exit(1)

    labels = load_labels(label_path)
    if not labels:
        print("ERROR: No valid labels found.")
        sys.exit(1)

    # Filter by set if requested
    if args.filter_set:
        labels = [l for l in labels if l.get("set_id") == args.filter_set]
        print(f"Filtering to set: {args.filter_set}")

    # Check how many have ref_image
    with_ref = sum(1 for l in labels if l.get("ref_image"))
    without_ref = len(labels) - with_ref
    print(f"Loaded {len(labels)} labels ({with_ref} with ref_image, "
          f"{without_ref} without)")
    if without_ref > 0:
        print(f"WARNING: {without_ref} entries lack 'ref_image' and will be skipped")

    eval_out = run_eval(labels, args.dir, args.threshold, args.verbose)
    results = eval_out["results"]
    skipped = eval_out["skipped"]

    if not results:
        print("No images evaluated (all skipped or filtered out).")
        sys.exit(1)

    # Overall metrics
    metrics = compute_metrics(results)

    # Per-set breakdown
    per_set_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        per_set_results[r["set_id"]].append(r)

    per_set_metrics = {}
    for set_id, set_results in per_set_results.items():
        per_set_metrics[set_id] = compute_metrics(set_results)

    print_report(metrics, per_set_metrics, results, skipped, args.threshold)


if __name__ == "__main__":
    main()
