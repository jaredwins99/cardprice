#!/usr/bin/env python3
"""Evaluate the card identification cascade against ground truth binder data.

Loads data/eval/binder_eval.json, runs identify_card (or identify_card_robust)
on every segment, and reports per-tier and overall accuracy.

Usage:
    python scripts/eval/eval_cascade.py [--robust] [--verbose]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"


def load_eval_data():
    with open(EVAL_PATH) as f:
        return json.load(f)


def strip_card_id(raw_cid: str) -> str:
    """Normalize a card_id returned by the cascade.

    The cascade sometimes returns ids with a leading set directory prefix
    (e.g. 'ex15/ex15-92/normal') -- strip that to get 'ex15-92/normal'.
    """
    parts = raw_cid.split("/")
    if len(parts) == 3:
        return "/".join(parts[1:])
    return raw_cid


def run_eval(use_robust: bool = False, verbose: bool = False):
    from cardprice.ml import identify_card, identify_card_robust

    identify = identify_card_robust if use_robust else identify_card

    data = load_eval_data()

    total = 0
    correct = 0
    correct_top5 = 0
    by_method = {}  # method -> {"correct": int, "total": int}
    by_page = []
    results_detail = []

    for page in data["pages"]:
        seg_dir = PROJECT_ROOT / page["segments_dir"]
        page_correct = 0
        page_total = 0

        for card in page["cards"]:
            segment_path = seg_dir / card["segment"]
            if not segment_path.exists():
                print(f"  SKIP {card['segment']} -- file not found")
                continue

            total += 1
            page_total += 1
            expected_id = card["card_id"]
            expected_name = card["name"]

            t0 = time.time()
            result = identify(str(segment_path))
            elapsed = time.time() - t0

            predicted_id = result.get("card_id")
            if predicted_id:
                predicted_id = strip_card_id(predicted_id)

            method = result.get("method") or "none"
            confidence = result.get("confidence", 0.0)
            is_correct = predicted_id == expected_id

            # Check if correct answer appears in top alternatives
            in_top5 = is_correct
            if not in_top5 and result.get("raw_response"):
                alts = result["raw_response"].get("top_alternatives", [])
                top_matches = result["raw_response"].get("top_matches", [])
                alt_ids = set()
                for item in alts:
                    if isinstance(item, (list, tuple)):
                        alt_ids.add(strip_card_id(str(item[0])))
                    elif isinstance(item, str):
                        alt_ids.add(strip_card_id(item))
                for item in top_matches:
                    if isinstance(item, (list, tuple)):
                        alt_ids.add(strip_card_id(str(item[0])))
                if expected_id in alt_ids:
                    in_top5 = True

            if is_correct:
                correct += 1
                page_correct += 1
            if in_top5:
                correct_top5 += 1

            # Track per-method stats
            if method not in by_method:
                by_method[method] = {"correct": 0, "total": 0}
            by_method[method]["total"] += 1
            if is_correct:
                by_method[method]["correct"] += 1

            status = "OK" if is_correct else ("TOP5" if in_top5 else "MISS")
            detail = {
                "segment": card["segment"],
                "expected": expected_id,
                "expected_name": expected_name,
                "predicted": predicted_id,
                "method": method,
                "confidence": confidence,
                "correct": is_correct,
                "in_top5": in_top5,
                "time_s": round(elapsed, 2),
            }
            results_detail.append(detail)

            if verbose or not is_correct:
                print(f"  [{status:4s}] {card['segment']:12s}  "
                      f"expected={expected_id:20s} ({expected_name})"
                      f"  got={str(predicted_id):20s}  "
                      f"method={method:5s} conf={confidence:.2f}  "
                      f"({elapsed:.1f}s)")

        by_page.append({"image": page["image"], "correct": page_correct, "total": page_total})

    # Summary
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    mode = "robust (with rotations)" if use_robust else "standard"
    print(f"Mode:        {mode}")
    print(f"Total cards: {total}")
    print(f"Top-1 acc:   {correct}/{total} = {correct/total:.1%}" if total else "No cards evaluated")
    print(f"Top-5 acc:   {correct_top5}/{total} = {correct_top5/total:.1%}" if total else "")
    print()

    print("Per-page breakdown:")
    for p in by_page:
        acc = p["correct"] / p["total"] if p["total"] else 0
        print(f"  {p['image']:50s}  {p['correct']}/{p['total']} = {acc:.0%}")
    print()

    print("Per-method breakdown:")
    for method, stats in sorted(by_method.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {method:10s}  {stats['correct']}/{stats['total']} = {acc:.0%}")
    print()

    # Show all misses
    misses = [r for r in results_detail if not r["correct"]]
    if misses:
        print(f"Misidentified cards ({len(misses)}):")
        for m in misses:
            in5 = " (in top-5)" if m["in_top5"] else ""
            print(f"  {m['segment']:12s}  expected={m['expected']:20s} ({m['expected_name']})"
                  f"  got={str(m['predicted']):20s}  method={m['method']}  conf={m['confidence']:.2f}{in5}")
    else:
        print("All cards identified correctly!")

    # Write detailed results to JSON
    output_path = PROJECT_ROOT / "data" / "eval" / "cascade_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "mode": mode,
            "total": total,
            "top1_correct": correct,
            "top5_correct": correct_top5,
            "top1_accuracy": round(correct / total, 4) if total else 0,
            "top5_accuracy": round(correct_top5 / total, 4) if total else 0,
            "by_method": by_method,
            "by_page": by_page,
            "details": results_detail,
        }, f, indent=2)
    print(f"\nDetailed results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate card identification cascade")
    parser.add_argument("--robust", action="store_true",
                        help="Use identify_card_robust (tries rotations)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print all results, not just misses")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy library loggers during eval
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    if not EVAL_PATH.exists():
        print(f"ERROR: Eval dataset not found at {EVAL_PATH}")
        sys.exit(1)

    run_eval(use_robust=args.robust, verbose=args.verbose)


if __name__ == "__main__":
    main()
