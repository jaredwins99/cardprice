#!/usr/bin/env python3
"""Evaluate stamp detection accuracy on labeled EX-era card images.

Reads labels.jsonl from a directory of card images, runs the stamp detector
from variant_detector.py on each image, and reports accuracy, precision/recall,
per-set breakdown, and confusion matrix.

Usage:
    python scripts/eval_stamp_detection.py
    python scripts/eval_stamp_detection.py --dir data/condition_training/stamps/
    python scripts/eval_stamp_detection.py --set ex11
    python scripts/eval_stamp_detection.py --verbose
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "condition_training", "stamps",
)


def load_labels(label_path: str) -> list[dict]:
    """Load labels.jsonl -- one JSON object per line.

    Expected format per line:
        {"image": "ex11_stamped_001.jpg", "stamped": true, "set_id": "ex11"}

    The 'stamped' field is the ground-truth boolean.
    'set_id' is optional but used for per-set reporting.
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


def run_eval(labels: list[dict], img_dir: str, filter_set: str | None,
             verbose: bool) -> dict:
    """Run stamp detection on all labeled images and collect results."""
    from cardprice.ml.variant_detector import detect_stamped

    results = []
    skipped = 0

    for entry in labels:
        set_id = entry.get("set_id", "")
        if filter_set and set_id != filter_set:
            continue

        img_path = os.path.join(img_dir, entry["image"])
        if not os.path.exists(img_path):
            if verbose:
                print(f"  SKIP {entry['image']} (file not found)")
            skipped += 1
            continue

        gt_stamped = bool(entry["stamped"])

        t0 = time.time()
        try:
            pred_stamped, confidence = detect_stamped(img_path, set_id=set_id)
        except Exception as e:
            if verbose:
                print(f"  ERROR {entry['image']}: {e}")
            pred_stamped, confidence = False, 0.0
        elapsed_ms = (time.time() - t0) * 1000

        correct = pred_stamped == gt_stamped
        results.append({
            "image": entry["image"],
            "set_id": set_id,
            "gt_stamped": gt_stamped,
            "pred_stamped": pred_stamped,
            "confidence": confidence,
            "correct": correct,
            "time_ms": elapsed_ms,
        })

        if verbose:
            status = "OK" if correct else "WRONG"
            gt_label = "stamped" if gt_stamped else "clean"
            pred_label = "stamped" if pred_stamped else "clean"
            print(f"  [{status}] {entry['image']:40s}  "
                  f"gt={gt_label:7s}  pred={pred_label:7s}  "
                  f"conf={confidence:.2f}  {elapsed_ms:.0f}ms")

    return {"results": results, "skipped": skipped}


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    if not results:
        return {"n": 0}

    n = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Confusion matrix
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
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_time_ms": mean_time,
    }


def print_report(metrics: dict, per_set: dict[str, dict],
                 results: list[dict], skipped: int) -> None:
    """Print formatted evaluation report."""
    n = metrics["n"]
    if n == 0:
        print("\nNo results to report.")
        return

    print("\n" + "=" * 70)
    print("STAMP DETECTION EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal images:    {n}")
    print(f"Skipped:         {skipped}")
    print(f"Accuracy:        {metrics['correct']}/{n} "
          f"({metrics['accuracy']:.1%})")
    print(f"Mean time:       {metrics['mean_time_ms']:.0f}ms per image")

    print(f"\nPrecision:       {metrics['precision']:.3f}")
    print(f"Recall:          {metrics['recall']:.3f}")
    print(f"F1:              {metrics['f1']:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Clean    Stamped")
    print(f"  Actual Clean    {metrics['tn']:5d}    {metrics['fp']:5d}")
    print(f"  Actual Stamped  {metrics['fn']:5d}    {metrics['tp']:5d}")

    if metrics["fp"] > 0:
        print(f"\n  False positives (clean detected as stamped): {metrics['fp']}")
        for r in results:
            if not r["gt_stamped"] and r["pred_stamped"]:
                print(f"    - {r['image']} (conf={r['confidence']:.2f})")

    if metrics["fn"] > 0:
        print(f"\n  False negatives (stamped detected as clean): {metrics['fn']}")
        for r in results:
            if r["gt_stamped"] and not r["pred_stamped"]:
                print(f"    - {r['image']} (conf={r['confidence']:.2f})")

    # Per-set breakdown
    if len(per_set) > 1 or (len(per_set) == 1
                            and list(per_set.keys())[0] != ""):
        print(f"\n{'Set':<8s}  {'N':>4s}  {'Correct':>7s}  {'Acc':>6s}  "
              f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
        print("-" * 55)
        for set_id in sorted(per_set.keys()):
            m = per_set[set_id]
            if m["n"] == 0:
                continue
            print(f"{set_id or '(none)':<8s}  {m['n']:4d}  "
                  f"{m['correct']:4d}/{m['n']:<3d}  "
                  f"{m['accuracy']:5.1%}  "
                  f"{m['precision']:5.3f}  "
                  f"{m['recall']:5.3f}  "
                  f"{m['f1']:5.3f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stamp detection on labeled card images.")
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="Directory with images and labels.jsonl")
    parser.add_argument("--set", dest="filter_set", default=None,
                        help="Evaluate only a single set (e.g. ex11)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each prediction")
    args = parser.parse_args()

    label_path = os.path.join(args.dir, "labels.jsonl")
    if not os.path.exists(label_path):
        print(f"ERROR: labels.jsonl not found at {label_path}")
        print(f"Expected format (one JSON per line):")
        print(f'  {{"image": "ex11_stamped_001.jpg", "stamped": true, "set_id": "ex11"}}')
        print(f'  {{"image": "ex11_clean_001.jpg", "stamped": false, "set_id": "ex11"}}')
        sys.exit(1)

    labels = load_labels(label_path)
    if not labels:
        print("ERROR: No valid labels found.")
        sys.exit(1)

    print(f"Loaded {len(labels)} labels from {label_path}")
    if args.filter_set:
        print(f"Filtering to set: {args.filter_set}")

    eval_out = run_eval(labels, args.dir, args.filter_set, args.verbose)
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

    print_report(metrics, per_set_metrics, results, skipped)


if __name__ == "__main__":
    main()
